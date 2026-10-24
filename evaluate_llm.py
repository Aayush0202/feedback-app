import re
import sys
import time
from pathlib import Path
import pandas as pd
from llm import analyse_comment

def check_identifier_leaks(raw_identifiers: str, anonymised_text: str) -> list[str]:
    """
    Returns any identifier phrases from raw_identifiers (semicolon-separated)
    that still appear in anonymised_text (case-insensitive).
    """
    if pd.isna(raw_identifiers) or not str(raw_identifiers).strip():
        return []

    leaks = []
    anon_lower = anonymised_text.lower()
    # Identifiers in comments.csv are separated by ';'
    parts = [p.strip() for p in str(raw_identifiers).split(";")]
    for part in parts:
        if not part:
            continue
        # Check if the exact phrase appears in the anonymised text (case-insensitive)
        if part.lower() in anon_lower:
            leaks.append(part)
    return leaks

def load_or_create_sample(comments_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(comments_csv_path)

    # 1. ALL edge_case comments
    edge = df[df["category"] == "edge_case"]
    # 2. ALL injection comments
    inj = df[df["category"] == "injection"]
    # 3. 15 escalation comments
    esc = df[df["category"] == "escalation"].sample(n=min(15, len(df[df["category"] == "escalation"])), random_state=42)
    # 4. 10 needs_anonymising comments
    needs = df[df["category"] == "needs_anonymising"].sample(n=min(10, len(df[df["category"] == "needs_anonymising"])), random_state=42)
    # 5. 10 routine comments
    rout = df[df["category"] == "routine"].sample(n=min(10, len(df[df["category"] == "routine"])), random_state=42)

    sample_df = pd.concat([edge, inj, esc, needs, rout]).drop_duplicates(subset=["comment_id"]).reset_index(drop=True)
    return sample_df

def run_evaluation():
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    eval_csv_path = results_dir / "llm_eval.csv"
    failures_csv_path = results_dir / "llm_failures.csv"

    comments_csv_path = Path("data") / "comments.csv"
    if not comments_csv_path.exists():
        print(f"Error: {comments_csv_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    sample_df = load_or_create_sample(comments_csv_path)
    total_samples = len(sample_df)
    print(f"Loaded sample of {total_samples} comments:")
    for cat, count in sample_df["category"].value_counts().items():
        print(f"  • {cat}: {count}")
    print()

    # Resume capability: check if eval_csv already has existing rows
    completed_ids = set()
    eval_records = []
    if eval_csv_path.exists():
        try:
            existing_eval = pd.read_csv(eval_csv_path)
            if not existing_eval.empty and "comment_id" in existing_eval.columns:
                completed_ids = set(existing_eval["comment_id"].astype(str))
                eval_records = existing_eval.to_dict(orient="records")
                print(f"Resuming: {len(completed_ids)} comments already evaluated in {eval_csv_path}.")
        except Exception as e:
            print(f"Could not load existing {eval_csv_path} ({e}). Starting fresh.")
            completed_ids = set()
            eval_records = []

    # Run evaluations
    processed_this_run = 0
    for idx, row in sample_df.iterrows():
        cid = str(row["comment_id"])
        if cid in completed_ids:
            continue

        original_text = row["text"]
        cat = row["category"]
        true_theme = str(row["theme"]).lower().strip()
        true_sentiment = str(row["sentiment"]).lower().strip()
        true_severity = str(row["severity"]).lower().strip()
        true_escalate = bool(row["escalate"])
        true_identifiers = str(row["identifiers"]) if pd.notna(row["identifiers"]) else ""

        print(f"[{len(eval_records) + 1}/{total_samples}] Evaluating {cid} ({cat})...", end=" ", flush=True)

        try:
            analysis = analyse_comment(original_text)
        except Exception as ex:
            print(f"FAILED with error: {ex}")
            analysis = {
                "anonymised": original_text,
                "theme": "other",
                "sentiment": "neutral",
                "severity": "low",
                "escalate": False,
                "still_identifiable": False,
                "removed_details": [],
                "injection_attempt": False,
                "needs_human_review": True,
            }

        anonymised_text = analysis.get("anonymised", "")
        pred_theme = str(analysis.get("theme", "other")).lower().strip()
        pred_sentiment = str(analysis.get("sentiment", "neutral")).lower().strip()
        pred_severity = str(analysis.get("severity", "low")).lower().strip()
        pred_escalate = bool(analysis.get("escalate", False))
        pred_injection = bool(analysis.get("injection_attempt", False))
        needs_review = bool(analysis.get("needs_human_review", False))

        # Check for leaks
        leaks = check_identifier_leaks(true_identifiers, anonymised_text)
        has_leak = len(leaks) > 0

        # Matches
        theme_match = (pred_theme == true_theme)
        severity_match = (pred_severity == true_severity)
        escalate_match = (pred_escalate == true_escalate)
        injection_match = (pred_injection is True) if cat == "injection" else (pred_injection is False)

        record = {
            "comment_id": cid,
            "category": cat,
            "original_text": original_text,
            "anonymised_text": anonymised_text,
            "true_theme": true_theme,
            "pred_theme": pred_theme,
            "theme_match": theme_match,
            "true_sentiment": true_sentiment,
            "pred_sentiment": pred_sentiment,
            "true_severity": true_severity,
            "pred_severity": pred_severity,
            "severity_match": severity_match,
            "true_escalate": true_escalate,
            "pred_escalate": pred_escalate,
            "escalate_match": escalate_match,
            "true_identifiers": true_identifiers,
            "has_identifiers": bool(true_identifiers.strip()),
            "leaked_identifiers": "; ".join(leaks),
            "identifier_leak": has_leak,
            "pred_injection_attempt": pred_injection,
            "needs_human_review": needs_review,
        }

        eval_records.append(record)
        completed_ids.add(cid)
        processed_this_run += 1
        print(f"Done. Theme: {pred_theme} | Escalate: {pred_escalate} | Leak: {has_leak}")

        # Save incrementally
        pd.DataFrame(eval_records).to_csv(eval_csv_path, index=False)

        # Wait 4 seconds between calls as requested
        if len(eval_records) < total_samples:
            time.sleep(4)

    # Convert full results to DataFrame
    eval_df = pd.DataFrame(eval_records)

    # Calculate metrics
    n_total = len(eval_df)
    theme_acc = eval_df["theme_match"].mean()
    severity_acc = eval_df["severity_match"].mean()

    # Escalation recall & precision
    # comments that should escalate
    true_esc = eval_df["true_escalate"] == True
    pred_esc = eval_df["pred_escalate"] == True
    tp_esc = (true_esc & pred_esc).sum()
    fn_esc = (true_esc & (~pred_esc)).sum()
    fp_esc = ((~true_esc) & pred_esc).sum()
    tn_esc = ((~true_esc) & (~pred_esc)).sum()

    esc_recall = tp_esc / (tp_esc + fn_esc) if (tp_esc + fn_esc) > 0 else 0.0
    esc_precision = tp_esc / (tp_esc + fp_esc) if (tp_esc + fp_esc) > 0 else 0.0

    # Identifier leak rate
    with_ids_df = eval_df[eval_df["has_identifiers"] == True]
    n_with_ids = len(with_ids_df)
    leaked_count = with_ids_df["identifier_leak"].sum()
    leak_rate = (leaked_count / n_with_ids) if n_with_ids > 0 else 0.0

    # Injection detection rate
    inj_df = eval_df[eval_df["category"] == "injection"]
    n_injections = len(inj_df)
    inj_caught = inj_df["pred_injection_attempt"].sum()
    injection_rate = (inj_caught / n_injections) if n_injections > 0 else 0.0

    # Human review count
    human_review_count = int(eval_df["needs_human_review"].sum())

    # Build failures dataset
    failures = []
    for _, r in eval_df.iterrows():
        reasons = []
        if not r["theme_match"]:
            reasons.append(f"Theme mismatch (expected '{r['true_theme']}', got '{r['pred_theme']}')")
        if not r["severity_match"]:
            reasons.append(f"Severity mismatch (expected '{r['true_severity']}', got '{r['pred_severity']}')")
        if not r["escalate_match"]:
            reasons.append(f"Escalate mismatch (expected {r['true_escalate']}, got {r['pred_escalate']})")
        if r["identifier_leak"]:
            reasons.append(f"Identifier leak: '{r['leaked_identifiers']}' remained in text")
        if r["category"] == "injection" and not r["pred_injection_attempt"]:
            reasons.append("Missed prompt injection attempt")
        if r["needs_human_review"]:
            reasons.append("Output required human review (invalid schema)")

        if reasons:
            failures.append({
                "comment_id": r["comment_id"],
                "category": r["category"],
                "failure_reasons": " | ".join(reasons),
                "original_text": r["original_text"],
                "anonymised_text": r["anonymised_text"],
                "expected_theme": r["true_theme"],
                "predicted_theme": r["pred_theme"],
                "expected_severity": r["true_severity"],
                "predicted_severity": r["pred_severity"],
                "expected_escalate": r["true_escalate"],
                "predicted_escalate": r["pred_escalate"],
                "leaked_identifiers": r["leaked_identifiers"],
            })

    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(failures_csv_path, index=False)
    print(f"\nSaved failures log to: {failures_csv_path} ({len(failures_df)} rows with mismatches/leaks)")

    # Print Report
    print("\n" + "=" * 70)
    print("                 LLM EVALUATION REPORT SUMMARY")
    print("=" * 70)
    print(f"Total Evaluated Comments       : {n_total}")
    print(f"Theme Accuracy                 : {theme_acc:.2%}")
    print(f"Severity Accuracy              : {severity_acc:.2%}")
    print(f"Escalation Recall              : {esc_recall:.2%} ({tp_esc}/{tp_esc + fn_esc} escalated comments caught)")
    print(f"Escalation Precision           : {esc_precision:.2%} (True Escalate: {tp_esc}, False Alarms: {fp_esc})")
    print(f"Identifier Leak Rate           : {leak_rate:.2%} ({leaked_count}/{n_with_ids} leaked among comments with identifiers)")
    print(f"Injection Detection Rate       : {injection_rate:.2%} ({inj_caught}/{n_injections} prompt injections detected)")
    print(f"Required Human Review          : {human_review_count} comments ({human_review_count / n_total:.2%})")
    print("=" * 70)

    # Display 3 most interesting failures
    print("\n" + "=" * 70)
    print("           3 MOST INTERESTING FAILURES / MISMATCHES")
    print("=" * 70)
    if not failures_df.empty:
        # Prioritize leaks or injection misses or interesting edge cases
        prioritized = failures_df.copy()
        def score_failure(row):
            score = 1
            if "Identifier leak" in row["failure_reasons"]:
                score += 10
            if "Missed prompt injection" in row["failure_reasons"]:
                score += 8
            if "Escalate mismatch" in row["failure_reasons"]:
                score += 5
            return score

        prioritized["score"] = prioritized.apply(score_failure, axis=1)
        top_failures = prioritized.sort_values(by="score", ascending=False).head(3)

        for rank, (_, f_row) in enumerate(top_failures.iterrows(), start=1):
            print(f"\n[Failure {rank}] ID: {f_row['comment_id']} ({f_row['category']})")
            print(f"Reasons  : {f_row['failure_reasons']}")
            print(f"Original : \"{f_row['original_text']}\"")
            print(f"Anonymised: \"{f_row['anonymised_text']}\"")
            print(f"Labels   : Theme: exp '{f_row['expected_theme']}' vs pred '{f_row['predicted_theme']}' | Escalate: exp {f_row['expected_escalate']} vs pred {f_row['predicted_escalate']}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    run_evaluation()
