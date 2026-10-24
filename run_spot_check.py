"""Quick test for two specific comments."""
import time
from llm import analyse_comment

TESTS = [
    {
        "id": 1,
        "text": "As the only person in Finance who uses a wheelchair, the broken lift means I can't reach my desk.",
        "expected_escalate": True,
        "note": "Disability accessibility — must escalate; wheelchair detail must be generalised",
    },
    {
        "id": 2,
        "text": "Senior log mujhe meri jaati ki wajah se alag baithne ko kehte hain.",
        "expected_escalate": True,
        "note": "Hinglish caste discrimination — must escalate",
    },
]

print("=" * 80)
for t in TESTS:
    print(f"\nTest {t['id']}: {t['note']}")
    print(f"  Input    : {t['text']!r}")

    result = analyse_comment(t["text"])

    esc_ok   = result["escalate"] == t["expected_escalate"]
    overall  = "PASS" if esc_ok else "FAIL"

    print(f"  Anonymised : {result['anonymised']!r}")
    print(f"  Theme      : {result['theme']}")
    print(f"  Severity   : {result['severity']}")
    print(f"  Escalate   : {result['escalate']}  (expected {t['expected_escalate']})  {'✅' if esc_ok else '❌'}")
    print(f"  security_flag : {result['security_flag']}")
    print(f"  still_identifiable : {result['still_identifiable']}")
    print(f"  >>> {overall}")
    print("-" * 80)

    if t["id"] < len(TESTS):
        time.sleep(4)
