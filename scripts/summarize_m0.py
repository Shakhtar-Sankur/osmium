"""Our MMLU (simple-evals, greedy) against what Apple reported for the same open models.

    python scripts/summarize_m0.py [results/m0]
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "m0")
print(f"{'model':28s} {'ours':>7s} {'Apple reported':>15s} {'diff':>6s} {'no answer':>10s} {'truncated':>10s}  questions")
for p in sorted(glob.glob(os.path.join(root, "mmlu_*.json"))):
    r = json.load(open(p))
    ours = 100 * r["accuracy"]
    apple = r.get("apple_reported")
    diff = f"{ours - apple:+.1f}" if apple else ""
    print(f"{r['model']:28s} {ours:6.2f}% {apple if apple else '':>15} {diff:>6s} {r['no_answer']:>10d} {r['truncated']:>10d}  {r['questions']}")
print("Apple's own on-device model (Table 3): 67.8 at 16 bits per weight, 64.4 at 2 bits (shipped).")
