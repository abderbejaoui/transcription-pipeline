"""Print best / worst / random samples from cached predictions, side by side
with both metrics, so a human can judge whether the score makes sense.

Usage:
    python3 scripts/show_samples.py --model qwen3-asr-1.7b --n 5
"""
from __future__ import annotations
import argparse, json, statistics
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_v2 import load_model, score_records  # noqa: E402


def show_one(r, idx=None, prefix=""):
    tag = f"[{idx}] " if idx is not None else ""
    flags = []
    if r.get("broken_ref"):
        flags.append("BROKEN_REF")
    if r.get("length_outlier"):
        flags.append("LEN_OUTLIER")
    flag_s = f"  ({', '.join(flags)})" if flags else ""
    print(f"{prefix}{tag}{r['id']}  WER={r['wer']*100:5.1f}%  CER={r['cer']*100:5.1f}%{flag_s}")
    print(f"{prefix}   ref : {r['ref']}")
    print(f"{prefix}   pred: {r['pred']}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-asr-1.7b")
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    records = load_model(args.model)
    scored = score_records(records)

    # Filter to "trustworthy" rows for the BEST/WORST display
    clean = [r for r in scored if not r["broken_ref"] and not r["length_outlier"]]
    clean.sort(key=lambda r: r["wer"])

    print(f"\n========= {args.model} =========")
    print(f"n_total={len(scored)}  n_clean+aligned={len(clean)}")
    print(f"clean mean WER = {statistics.mean(r['wer'] for r in clean)*100:.1f}%")
    print(f"clean mean CER = {statistics.mean(r['cer'] for r in clean)*100:.1f}%")

    print(f"\n--- TOP {args.n} (lowest WER, clean refs only) ---")
    for i, r in enumerate(clean[: args.n], 1):
        show_one(r, idx=i)

    print(f"\n--- BOTTOM {args.n} (highest WER, clean refs only) ---")
    for i, r in enumerate(reversed(clean[-args.n :]), 1):
        show_one(r, idx=i)

    # Now the ones that were filtered out — to convince you the filter is right
    dropped = [r for r in scored if r["broken_ref"] or r["length_outlier"]]
    print(f"\n--- DROPPED ({len(dropped)} clips) sample (these inflate the WER unfairly) ---")
    for i, r in enumerate(dropped[: args.n], 1):
        show_one(r, idx=i)


if __name__ == "__main__":
    main()
