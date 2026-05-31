"""Build the master training manifest for the v2 medical-Gulf LoRA.

Combines 4 buckets. The DEFAULT ratio is *real-audio dominant* to avoid the
catastrophic-forgetting / hallucination failure that the synthetic-heavy
40/30/20/10 mix produced (the decoder overfits TTS prosody and starts
*generating* medical text instead of *transcribing* it):

  - Real Gulf rehearsal     -> 50%   (anchors the 5% WER, prevents forgetting)
  - Synthetic medical Gulf  -> 20%   (CAPPED — teaches medical sub-words only)
  - Real code-switched      -> 20%   (Arabic+English, real audio)
  - English medical         -> 10%   (real clinical English)
                             total real audio = 80%, synthetic = 20%

Each input is a JSONL manifest with at least:
  audio_path: str
  text: str
  duration_s: float

The output is a shuffled train JSONL with all rows tagged by source bucket.
With --val-frac > 0 the script ALSO writes a held-out validation manifest
(disjoint from train, sampled per-bucket so every bucket is represented).
That val manifest is what the trainer's early-stopping callback should watch
so we never train blind again.

Usage on the DGX
-----------------
python scripts/build_master_manifest.py \\
    --synthetic     data/training/medical_gulf_v2/manifest.jsonl \\
    --rehearsal     data/training/gulf_rehearsal/manifest.jsonl \\
    --codeswitch    data/raw/masc/manifest.jsonl \\
    --english-med   data/raw/primock57/manifest.jsonl data/raw/cv_medical/manifest.jsonl \\
    --out           data/training/master_v2/manifest.jsonl \\
    --target-hours  150 \\
    --val-frac      0.03
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
                   help="Output master TRAIN JSONL path.")
    p.add_argument("--target-hours", type=float, default=150.0,
                   help="Total target hours across all buckets.")
    p.add_argument("--ratios", nargs=4, type=float,
                   default=[0.20, 0.50, 0.20, 0.10],
                   metavar=("SYN", "REH", "CS", "ENG"),
                   help=("Fractions for synthetic / rehearsal / codeswitch / "
                         "english. Default 0.20/0.50/0.20/0.10 is real-audio "
                         "dominant (synthetic CAPPED at 20%%) to avoid the "
                         "hallucination/forgetting trap."))
    p.add_argument("--val-frac", type=float, default=0.03,
                   help=("Fraction of EACH bucket to hold out as a disjoint "
                         "validation manifest (written next to --out as "
                         "*_val.jsonl). 0 disables. Default 0.03 (~3%%). The "
                         "trainer's early-stopping callback should watch this "
                         "file so training never runs blind."))
    p.add_argument("--val-max-per-bucket", type=int, default=200,
                   help=("Cap the per-bucket validation sample count so eval "
                         "stays fast during training. Default 200."))
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
    print(f"{'bucket':<12} {'available':>12} {'target':>10} {'train':>10} {'val':>8}")
    print(f"{'-'*12} {'-'*12} {'-'*10} {'-'*10} {'-'*8}")

    final_rows: List[Dict[str, Any]] = []
    val_rows: List[Dict[str, Any]] = []
    for name, rows in buckets.items():
        avail_h = sum(float(r.get("duration_s", 0.0)) for r in rows) / 3600
        target_h = args.target_hours * ratios[name]
        if avail_h < target_h:
            print(f"[master] WARN {name}: only {avail_h:.1f}h available "
                  f"but {target_h:.1f}h requested. Using all of it.")
            chosen = list(rows)
        else:
            chosen = _sample_to_budget(rows, target_h * 3600, rng)

        # Hold out a disjoint, per-bucket validation slice BEFORE adding the
        # rest to train. Sampling per-bucket guarantees the val set covers
        # every domain (real Gulf, synthetic medical, code-switch, english),
        # so early-stopping sees a representative signal — not just one bucket.
        bucket_val: List[Dict[str, Any]] = []
        if args.val_frac > 0 and len(chosen) > 1:
            rng.shuffle(chosen)
            n_val = int(round(len(chosen) * args.val_frac))
            n_val = max(1, min(n_val, args.val_max_per_bucket))
            n_val = min(n_val, len(chosen) - 1)  # always leave >=1 for train
            bucket_val = chosen[:n_val]
            chosen = chosen[n_val:]
            val_rows.extend(bucket_val)

        chosen_h = sum(float(r.get("duration_s", 0.0)) for r in chosen) / 3600
        print(f"{name:<12} {avail_h:>10.1f}h  {target_h:>8.1f}h  "
              f"{chosen_h:>8.1f}h  {len(bucket_val):>6d}")
        final_rows.extend(chosen)

    total_h = sum(float(r.get("duration_s", 0.0)) for r in final_rows) / 3600
    print(f"{'TOTAL':<12} {'':<12} {args.target_hours:>8.1f}h  "
          f"{total_h:>8.1f}h  {len(val_rows):>6d}")

    rng.shuffle(final_rows)
    rng.shuffle(val_rows)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for row in final_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print()
    print(f"[master] wrote {len(final_rows)} train rows -> {out}")

    if val_rows:
        val_out = out.with_name(out.stem + "_val" + out.suffix)
        with val_out.open("w", encoding="utf-8") as fh:
            for row in val_rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"[master] wrote {len(val_rows)} val rows   -> {val_out}")
        print(f"[master] -> pass this to training as the FIRST --eval-manifests "
              f"entry so early-stopping watches it.")


if __name__ == "__main__":
    main()
