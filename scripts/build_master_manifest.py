"""Build the master training manifest for the v2 medical-Gulf LoRA.

Combines 4 buckets in fixed ratios:
  - Synthetic medical Gulf  -> 40%  (~60h)
  - Real Gulf rehearsal     -> 30%  (~45h)
  - Real code-switched      -> 20%  (~30h)
  - English medical         -> 10%  (~15h)
                             total ~150h

Each input is a JSONL manifest with at least:
  audio_path: str
  text: str
  duration_s: float

The output is a single shuffled JSONL with all rows tagged by source
bucket. The script reports the actual hours per bucket so we can sanity-
check the ratios.

Usage on the DGX
-----------------
python scripts/build_master_manifest.py \\
    --synthetic     data/training/medical_gulf_v2/manifest.jsonl \\
    --rehearsal     data/training/gulf_rehearsal/manifest.jsonl \\
    --codeswitch    data/raw/masc/manifest.jsonl \\
    --english-med   data/raw/primock57/manifest.jsonl data/raw/cv_medical/manifest.jsonl \\
    --out           data/training/master_v2/manifest.jsonl \\
    --target-hours  150 \\
    --ratios        0.40 0.30 0.20 0.10
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _load(paths: Sequence[str], bucket: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[master] WARN missing manifest: {path}")
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "duration_s" not in row and "duration" in row:
                    row["duration_s"] = float(row["duration"])
                row["bucket"] = bucket
                rows.append(row)
    return rows


def _sample_to_budget(
    rows: List[Dict[str, Any]],
    target_seconds: float,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    """Shuffle, then take rows until total duration >= target_seconds."""
    rng.shuffle(rows)
    chosen: List[Dict[str, Any]] = []
    taken = 0.0
    for r in rows:
        d = float(r.get("duration_s", 0.0))
        if d <= 0:
            continue
        chosen.append(r)
        taken += d
        if taken >= target_seconds:
            break
    return chosen


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--synthetic", nargs="+", required=True,
                   help="One or more synthetic-data manifests.")
    p.add_argument("--rehearsal", nargs="+", required=True,
                   help="One or more rehearsal manifests (sampled from 900h).")
    p.add_argument("--codeswitch", nargs="+", required=True,
                   help="One or more Arabic+English code-switched manifests.")
    p.add_argument("--english-med", nargs="+", required=True,
                   help="One or more English-medical manifests.")
    p.add_argument("--out", required=True,
                   help="Output master JSONL path.")
    p.add_argument("--target-hours", type=float, default=150.0,
                   help="Total target hours across all buckets.")
    p.add_argument("--ratios", nargs=4, type=float,
                   default=[0.40, 0.30, 0.20, 0.10],
                   metavar=("SYN", "REH", "CS", "ENG"),
                   help="Fractions for synthetic / rehearsal / codeswitch / english.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs(sum(args.ratios) - 1.0) > 1e-3:
        raise SystemExit(f"--ratios must sum to 1.0, got {sum(args.ratios)}")

    rng = random.Random(args.seed)

    print("[master] loading buckets ...")
    buckets = {
        "synthetic":  _load(args.synthetic,  "synthetic"),
        "rehearsal":  _load(args.rehearsal,  "rehearsal"),
        "codeswitch": _load(args.codeswitch, "codeswitch"),
        "english":    _load(args.english_med, "english"),
    }
    ratios = dict(zip(["synthetic", "rehearsal", "codeswitch", "english"],
                      args.ratios))

    print()
    print(f"{'bucket':<12} {'available':>12} {'target':>10} {'sampled':>10}")
    print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*10}")

    final_rows: List[Dict[str, Any]] = []
    for name, rows in buckets.items():
        avail_h = sum(float(r.get("duration_s", 0.0)) for r in rows) / 3600
        target_h = args.target_hours * ratios[name]
        if avail_h < target_h:
            print(f"[master] WARN {name}: only {avail_h:.1f}h available "
                  f"but {target_h:.1f}h requested. Using all of it.")
            chosen = list(rows)
        else:
            chosen = _sample_to_budget(rows, target_h * 3600, rng)
        chosen_h = sum(float(r.get("duration_s", 0.0)) for r in chosen) / 3600
        print(f"{name:<12} {avail_h:>10.1f}h  {target_h:>8.1f}h  {chosen_h:>8.1f}h")
        final_rows.extend(chosen)

    total_h = sum(float(r.get("duration_s", 0.0)) for r in final_rows) / 3600
    print(f"{'TOTAL':<12} {'':<12} {args.target_hours:>8.1f}h  {total_h:>8.1f}h")

    rng.shuffle(final_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print()
    print(f"[master] wrote {len(final_rows)} rows -> {out}")


if __name__ == "__main__":
    main()
