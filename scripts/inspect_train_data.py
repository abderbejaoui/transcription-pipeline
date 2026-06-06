#!/usr/bin/env python3
"""Inspect the Phase-1 training manifest from the first finetune.

Checks that the real-train manifest + its rejected sidecar exist, then reports
which datasets were used and how many hours each contributed.

Usage:
    python scripts/inspect_train_data.py
    python scripts/inspect_train_data.py --manifest data/dgx_full/preprocessed_audios_full/manifest.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# Schema-tolerant field lookups (different prep runs used different keys).
SOURCE_KEYS = ("source", "dataset", "corpus", "origin")
DURATION_KEYS = ("duration", "duration_sec", "dur", "length", "seconds")
AUDIO_KEYS = ("audio_path", "audio_filepath", "audio", "path", "wav", "file")


def _first(row: dict, keys: tuple[str, ...], default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _human(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:>4}h {m:02d}m {s:02d}s"


def inspect(path: str, label: str, default_clip_sec: float) -> float:
    if not os.path.exists(path):
        print(f"  [MISSING] {label}: {path}")
        return 0.0
    size = os.path.getsize(path)
    by_source_clips = collections.Counter()
    by_source_secs = collections.Counter()
    total_clips = 0
    total_secs = 0.0
    missing_dur = 0
    schema_keys = None
    bad_lines = 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_lines += 1
                continue
            if schema_keys is None:
                schema_keys = list(row.keys())
            src = _first(row, SOURCE_KEYS, "unknown")
            dur = _first(row, DURATION_KEYS)
            if dur is None:
                missing_dur += 1
                dur = default_clip_sec
            try:
                dur = float(dur)
            except (TypeError, ValueError):
                missing_dur += 1
                dur = default_clip_sec
            total_clips += 1
            total_secs += dur
            by_source_clips[src] += 1
            by_source_secs[src] += dur

    print(f"  [OK] {label}: {path}")
    print(f"       size={size/1e6:.1f} MB  clips={total_clips:,}  "
          f"total={_human(total_secs)} ({total_secs/3600:.1f}h)")
    if schema_keys:
        print(f"       schema keys: {schema_keys}")
    if missing_dur:
        print(f"       WARNING: {missing_dur:,} clips had no duration "
              f"-> assumed {default_clip_sec}s each (hours are estimates)")
    if bad_lines:
        print(f"       WARNING: {bad_lines:,} unparseable lines skipped")
    if by_source_clips:
        print(f"       --- per-dataset breakdown ---")
        print(f"       {'clips':>10}  {'hours':>8}  {'%h':>5}  source")
        for src, clips in by_source_clips.most_common():
            secs = by_source_secs[src]
            pct = (secs / total_secs * 100) if total_secs else 0
            print(f"       {clips:>10,}  {secs/3600:>8.1f}  {pct:>4.1f}%  {src}")
    return total_secs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default="data/dgx_full/preprocessed_audios_full/manifest.jsonl",
        help="Real train manifest from the first finetune.",
    )
    ap.add_argument(
        "--rejected",
        default="data/dgx_full/preprocessed_audios_full/rejected.jsonl",
        help="Sidecar of filtered-out clips.",
    )
    ap.add_argument(
        "--default-clip-sec",
        type=float,
        default=8.0,
        help="Assumed clip length when a row has no duration field.",
    )
    args = ap.parse_args()

    print("=" * 70)
    print("TRAINING DATA INSPECTION (Phase-1 / first finetune)")
    print("=" * 70)

    print("\n[1] Real train manifest")
    train_secs = inspect(args.manifest, "train manifest",
                         args.default_clip_sec)

    print("\n[2] Rejected (filtered-out) clips")
    rej_secs = inspect(args.rejected, "rejected manifest",
                       args.default_clip_sec)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Kept (train) : {train_secs/3600:8.1f} h")
    print(f"  Rejected     : {rej_secs/3600:8.1f} h")
    total = train_secs + rej_secs
    if total:
        keep_pct = train_secs / total * 100
        print(f"  Keep rate    : {keep_pct:8.1f}%  "
              f"(of {total/3600:.1f}h decoded)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
