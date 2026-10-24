"""
Run human test cases against analyse_comment.
Reads tests/human_test_cases.csv and prints a pass/fail table.
"""
import time
import sys
from pathlib import Path
import pandas as pd
from llm import analyse_comment

TEST_FILE = Path("tests") / "human_test_cases.csv"

def run():
    if not TEST_FILE.exists():
        print(f"ERROR: {TEST_FILE} not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(TEST_FILE)
    rows = []

    for i, row in df.iterrows():
        text        = str(row["text"]).strip()
        exp_theme   = str(row["expected_theme"]).strip().lower()
        exp_sev     = str(row["expected_severity"]).strip().lower()
        exp_esc     = str(row["expected_escalate"]).strip().lower() in {"true", "1", "yes"}
        note        = str(row.get("note", ""))

        print(f"\n[{i+1}/{len(df)}] {note}")
        print(f"  Input : {text!r}")

        result = analyse_comment(text)

        pred_theme = result["theme"]
        pred_sev   = result["severity"]
        pred_esc   = result["escalate"]

        theme_ok = pred_theme == exp_theme
        sev_ok   = pred_sev   == exp_sev
        esc_ok   = pred_esc   == exp_esc
        overall  = "PASS" if (theme_ok and sev_ok and esc_ok) else "FAIL"

        print(f"  Theme : {pred_theme:15s}  exp={exp_theme:15s}  {'OK' if theme_ok else 'FAIL'}")
        print(f"  Sev   : {pred_sev:15s}  exp={exp_sev:15s}  {'OK' if sev_ok   else 'FAIL'}")
        print(f"  Esc   : {str(pred_esc):15s}  exp={str(exp_esc):15s}  {'OK' if esc_ok   else 'FAIL'}")
        print(f"  Anon  : {result['anonymised']!r}")
        print(f"  >>> {overall}")

        rows.append({
            "#"             : i + 1,
            "Note"          : note,
            "Input (short)" : text[:55] + ("…" if len(text) > 55 else ""),
            "Theme"         : f"{pred_theme} vs {exp_theme}",
            "Theme OK"      : "✅" if theme_ok else "❌",
            "Severity"      : f"{pred_sev} vs {exp_sev}",
            "Sev OK"        : "✅" if sev_ok else "❌",
            "Escalate"      : f"{pred_esc} vs {exp_esc}",
            "Esc OK"        : "✅" if esc_ok else "❌",
            "OVERALL"       : "✅ PASS" if overall == "PASS" else "❌ FAIL",
        })

        if i < len(df) - 1:
            time.sleep(4)

    # Summary table
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 110)
    print("PASS / FAIL SUMMARY TABLE")
    print("=" * 110)
    print(summary.to_string(index=False))
    print("=" * 110)
    passed = sum(1 for r in rows if "PASS" in r["OVERALL"])
    print(f"\nResult: {passed}/{len(rows)} passed\n")

if __name__ == "__main__":
    run()
