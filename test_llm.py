import sys
import time
from pathlib import Path
import pandas as pd
from llm import analyse_comment

def run_tests():
    data_path = Path("data") / "comments.csv"
    if not data_path.exists():
        print(f"Error: {data_path} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(data_path)
    categories = ["routine", "needs_anonymising", "escalation", "edge_case", "injection"]

    # Pick exactly one comment from each category to conserve Gemini quota
    sample_records = []
    for cat in categories:
        subset = df[df["category"] == cat]
        if not subset.empty:
            sample_records.append(subset.iloc[0])
        else:
            print(f"Warning: No comments found for category '{cat}'")

    print(f"Testing {len(sample_records)} comments across categories: {categories}\n")

    summary_rows = []

    for i, row in enumerate(sample_records, start=1):
        comment_id = row["comment_id"]
        cat = row["category"]
        original_text = row["text"]
        true_theme = str(row["theme"]).lower()
        true_sentiment = str(row["sentiment"]).lower()
        true_severity = str(row["severity"]).lower()
        true_escalate = bool(row["escalate"])

        print("=" * 80)
        print(f"Test Case {i}/5: Category = '{cat}' (ID: {comment_id})")
        print("=" * 80)
        print(f"Original Text:\n  \"{original_text}\"\n")

        # Run analyse_comment
        start_time = time.time()
        analysis = analyse_comment(original_text)
        duration = time.time() - start_time

        print(f"Anonymised Text:\n  \"{analysis['anonymised']}\"\n")
        print("Analysis Labels & Flags:")
        print(f"  • Theme              : {analysis['theme']} (Expected: {true_theme})")
        print(f"  • Sentiment          : {analysis['sentiment']} (Expected: {true_sentiment})")
        print(f"  • Severity           : {analysis['severity']} (Expected: {true_severity})")
        print(f"  • Escalate           : {analysis['escalate']} (Expected: {true_escalate})")
        print(f"  • Still Identifiable : {analysis['still_identifiable']}")
        print(f"  • Injection Attempt  : {analysis['injection_attempt']}")
        print(f"  • Removed Details    : {analysis['removed_details']}")
        print(f"  • Processing Time    : {duration:.2f}s\n")

        # Check matches
        theme_match = analysis["theme"] == true_theme
        severity_match = analysis["severity"] == true_severity
        escalate_match = analysis["escalate"] == true_escalate

        # For injection category, verify injection_attempt was caught
        if cat == "injection":
            cat_flagged = analysis["injection_attempt"] is True
        elif cat in {"escalation"}:
            cat_flagged = analysis["escalate"] is True
        else:
            cat_flagged = True

        print(f"Match Check -> Theme Match: {theme_match} | Severity Match: {severity_match} | Escalate Match: {escalate_match}")
        print("-" * 80 + "\n")

        summary_rows.append({
            "ID": comment_id,
            "Category": cat,
            "True Theme": true_theme,
            "Pred Theme": analysis["theme"],
            "Theme Match": "YES" if theme_match else "NO",
            "True Escalate": true_escalate,
            "Pred Escalate": analysis["escalate"],
            "Escalate Match": "YES" if escalate_match else "NO",
            "True Severity": true_severity,
            "Pred Severity": analysis["severity"],
            "Injection Caught": "YES" if analysis["injection_attempt"] else "N/A" if cat != "injection" else "NO",
        })

    # Print summary table
    summary_df = pd.DataFrame(summary_rows)
    print("\n" + "=" * 95)
    print("TEST SUITE SUMMARY")
    print("=" * 95)
    print(summary_df.to_string(index=False))
    print("=" * 95)

if __name__ == "__main__":
    run_tests()
