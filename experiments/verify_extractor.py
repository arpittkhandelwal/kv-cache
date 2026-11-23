"""Quick extractor verification without re-running generation."""
import sys
sys.path.insert(0, "/Users/arpitkhandelwal/kv-cache-lab/platform/LatentMAS")
from utils import extract_gsm8k_answer, normalize_answer
import pandas as pd

# Key test cases from the investigation
test_cases = [
    ("Ex10 — list header bug",
     "(1) original plan contents\n1. First month: 60 downloads.\n2. Second month: 3 x 60 = 180 downloads.\n3. Third month: 180 x 0.7 = 126 downloads.\n4. Total downloads: 60 + 180 + 126 = 366.\n\n(2) constructive feedback\nThe plan correctly calculates each month's downloads.",
     "366"),
    ("Ex3 — plain-text final answer",
     "Feedback: The original plan correctly identifies the multiplication. The final answer is 540 meters.",
     "540"),
    ("Ex24 — price in prose",
     "By dividing $19.50 by 0.75, the original price is $26.",
     "26"),
    ("Ex20 — 'total water is 15 liters'",
     "So total water is 6 + 9 = 15. Yes, that is correct. So the answer should be 15 liters.",
     "15"),
    ("Boxed wins over everything",
     r"The answer might be 42 or 99 but boxed says \boxed{77}",
     "77"),
    ("= sign extractor",
     "2X/3 = 12. Multiply both sides by 3: 2X = 36. Divide by 2: X = 18.",
     "18"),
    ("Bare last-number fallback (no phrase)",
     "He walked 3 miles on day 1 and 7 miles on day 2.",
     "7"),
    ("Should NOT grab (2) from section header",
     "(1) steps\n1. a\n2. b\n(2) feedback\nTotal is 366.",
     "366"),
]

print("=" * 60)
print("EXTRACTOR VERIFICATION")
print("=" * 60)
all_pass = True
for desc, text, expected in test_cases:
    got = normalize_answer(extract_gsm8k_answer(text))
    exp_n = normalize_answer(expected)
    ok = got == exp_n
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {desc}")
    if not ok:
        print(f"           expected={exp_n!r}, got={got!r}")

print()
if all_pass:
    print("All tests passed!")
else:
    print("FAILURES FOUND")
    sys.exit(1)

# Also re-run extraction on the already-saved clean_baseline_full.csv
print()
print("Re-extracting from clean_baseline_full.csv with fixed extractor...")
df = pd.read_csv("/Users/arpitkhandelwal/kv-cache-lab/results/clean_baseline_full.csv")
df["pred_new"] = df["raw_text"].apply(lambda t: normalize_answer(extract_gsm8k_answer(str(t))))
df["gold_n"]   = df["gold"].apply(lambda g: normalize_answer(str(g)))
df["correct_new"] = df.apply(lambda r: r["pred_new"] == r["gold_n"] if r["pred_new"] and r["gold_n"] else False, axis=1)
old_correct = df["correct"].sum()
new_correct = df["correct_new"].sum()
n = len(df)
print(f"  Old accuracy (old extractor): {old_correct}/{n} ({100*old_correct/n:.1f}%)")
print(f"  New accuracy (new extractor): {new_correct}/{n} ({100*new_correct/n:.1f}%)")

changed = df[df["correct"] != df["correct_new"]][["example_idx","gold","pred_new","correct","correct_new","raw_text"]].copy()
if len(changed) == 0:
    print("  No examples changed correctness.")
else:
    print(f"  {len(changed)} example(s) changed:")
    for _, row in changed.iterrows():
        direction = "wrong->correct" if row["correct_new"] else "correct->wrong"
        print(f"    Ex{int(row['example_idx'])}: gold={row['gold']}, old_pred=?, new_pred={row['pred_new']} [{direction}]")

# Spot-check example 10 specifically
ex10 = df[df["example_idx"] == 10].iloc[0]
print(f"\nEx10 specifically:")
print(f"  Gold: {ex10['gold']}")
print(f"  New prediction: {ex10['pred_new']}")
print(f"  New correct: {ex10['correct_new']}")
