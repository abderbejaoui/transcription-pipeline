#!/usr/bin/env python3
"""Quick analysis of the latest eval results."""
import json, sys, glob, os
sys.stdout.reconfigure(encoding="utf-8")

# Find latest eval results
files = sorted(glob.glob("eval_results_*.json"), reverse=True)
if not files:
    print("No eval results found")
    sys.exit(1)
path = files[0]
print(f"Analyzing: {path}\n")

with open(path, encoding="utf-8") as f:
    data = json.load(f)

# Stage 1: FP cases
print("=== STAGE 1: FALSE POSITIVES (flagging wrong spans) ===")
s1 = next(r for r in data["stage_reports"] if r["stage_name"] == "scoring_and_flagging")
for row in s1["raw"]:
    fp = row.get("fp", 0)
    fn = row.get("fn", 0)
    tp = row.get("tp", 0)
    if fp > 0 or fn > 0:
        print(f"  {row['case_id']}: tp={tp} fp={fp} fn={fn}")
        print(f"    gold: {row.get('gold_spans', [])}")
        print(f"    pred: {row.get('pred_spans', [])}")

print()
print("=== STAGE 2: PHONETIC RETRIEVAL FAILURES ===")
s2 = next(r for r in data["stage_reports"] if r["stage_name"] == "phonetic_retrieval")
for row in s2["raw"]:
    if not row.get("top_1_correct", True) and "gold_term" in row:
        print(f"  {row['case_id']}: span={row.get('span','')} gold={row['gold_term']} top1={row.get('top_1_candidate','?')} all={row.get('all_candidates',[])}")

print()
print("=== STAGE 3: WRONG CORRECTION DECISIONS ===")
s3 = next(r for r in data["stage_reports"] if r["stage_name"] == "correction_decision")
for row in s3["raw"]:
    if not row.get("correct", True) and "gold_chosen" in row:
        print(f"  {row['case_id']}: gold_span={row.get('gold_span','')} gold={row.get('gold_chosen','')} pred={row.get('pred_chosen','')} note={row.get('note','')}")

print()
print("=== STAGE 5: END-TO-END FAILURES ===")
s5 = next(r for r in data["stage_reports"] if r["stage_name"] == "end_to_end")
for row in s5["raw"]:
    if not row.get("exact_match", True):
        print(f"  {row['case_id']}: gold={row['gold_corrected']!r}")
        print(f"    pred={row['pred_corrected']!r}")
