#!/usr/bin/env python3
"""Deterministically split a manifest into disjoint train / validation sets.

Guards against the single most common fine-tuning flaw: train/val leakage.
The split is:
  * deterministic (seeded shuffle), so reruns reproduce the exact partition;
  * disjoint by construction (each line goes to exactly one side);
  * optionally stratified by a field (e.g. ``source`` or ``dialect``) so every
    bucket is represented in both train and val;
  * optionally deduplicated by transcript text BEFORE splitting, so the same
    sentence cannot land in both sides (near-duplicate leakage).

It writes ``<out-prefix>.train.jsonl`` and ``<out-prefix>.val.jsonl`` and prints
counts plus a sanity assertion that the two sides share zero lines.

Examples
--------
    python scripts/split_manifest.py \
        --in data/preprocessed/sada22/manifest.jsonl \
        --out-prefix data/preprocessed/sada22/sada22 \
        --val-frac 0.05 --stratify-by source

    # Combine several prepared manifests, then split:
    python scripts/split_manifest.py \
        --in data/preprocessed/*/manifest.jsonl \
        --out-prefix data/splits/gulf --val-frac 0.03 \
        --stratify-by dialect --dedup-text
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _norm_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _read_text(rec: Dict) -> str:
    t = rec.get("text") or rec.get("target") or ""
    return t.split("<asr_text>", 1)[1] if "<asr_text>" in t else t


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Writes <prefix>.train.jsonl and <prefix>.val.jsonl.")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="Fraction of clips for validation (default 0.05).")
    ap.add_argument("--stratify-by", default=None,
                    help="Manifest field to stratify on (e.g. source, dialect).")
    ap.add_argument("--dedup-text", action="store_true",
                    help="Drop duplicate transcripts before splitting.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        ap.error("--val-frac must be in (0, 1).")

    rows: List[Dict] = []
    for man in args.inputs:
        if not man.exists():
            print(f"[split] skip missing {man}", file=sys.stderr)
            continue
        for line in man.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        print("[split] no rows read.", file=sys.stderr)
        return 1

    if args.dedup_text:
        seen = set()
        deduped = []
        for r in rows:
            k = _norm_key(_read_text(r))
            if k and k in seen:
                continue
            seen.add(k)
            deduped.append(r)
        print(f"[split] dedup-text: {len(rows)} -> {len(deduped)} rows")
        rows = deduped

    rng = random.Random(args.seed)

    if args.stratify_by:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            groups[str(r.get(args.stratify_by, "unknown"))].append(r)
        train, val = [], []
        for key, items in groups.items():
            rng.shuffle(items)
            n_val = max(1, round(len(items) * args.val_frac)) if len(items) > 1 else 0
            val.extend(items[:n_val])
            train.extend(items[n_val:])
    else:
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * args.val_frac))
        val, train = rows[:n_val], rows[n_val:]

    # Sanity: the two sides must be disjoint at the object level.
    def _sig(r: Dict) -> str:
        return (r.get("audio_path") or r.get("audio") or r.get("path") or "") + "\x00" + _read_text(r)

    train_sigs = {_sig(r) for r in train}
    overlap = sum(1 for r in val if _sig(r) in train_sigs)
    if overlap:
        print(f"[split] FATAL: {overlap} val clips also appear in train "
              f"(leakage). Aborting.", file=sys.stderr)
        return 2

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.out_prefix.parent / f"{args.out_prefix.name}.train.jsonl"
    val_path = args.out_prefix.parent / f"{args.out_prefix.name}.val.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n",
        encoding="utf-8")

    print(f"[split] train={len(train)}  val={len(val)}  "
          f"(val_frac={len(val)/(len(train)+len(val)):.3f}, leakage=0)")
    print(f"[split] -> {train_path}")
    print(f"[split] -> {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
