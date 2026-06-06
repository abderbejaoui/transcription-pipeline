#!/usr/bin/env python3
"""Locate synthetic / TTS medical audio corpora anywhere under the repo.

The data/tts_references/ folder is only a 10-clip demo seed. The real
~20h synthetic medical code-switch set lives elsewhere. This walks the
tree, finds directories containing many .wav files (and/or manifests whose
names hint at 'synthetic'/'tts'/'medical'), and reports per-directory wav
counts + total audio hours so you can spot the big one.

Usage:
    python scripts/find_synthetic_data.py
    python scripts/find_synthetic_data.py --root data --min-wavs 5
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import wave

HINT_WORDS = ("synth", "tts", "medical", "cs", "code", "ref")
MANIFEST_EXT = (".jsonl", ".json")


def _wav_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            rate = w.getframerate()
            if rate:
                return w.getnframes() / float(rate)
    except (wave.Error, EOFError, FileNotFoundError, OSError):
        return None
    return None


def _human(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    return f"{h}h {m:02d}m"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data",
                    help="Tree to search.")
    ap.add_argument("--min-wavs", type=int, default=5,
                    help="Only report dirs with at least this many wavs.")
    ap.add_argument("--sample", type=int, default=200,
                    help="Per dir, measure duration of up to N wavs and "
                         "extrapolate (0 = measure all).")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"[MISSING] root: {args.root}")
        return 1

    dir_wavs: dict[str, list[str]] = collections.defaultdict(list)
    hint_dirs: set[str] = set()
    manifests: list[str] = []

    for dirpath, _dirs, files in os.walk(args.root):
        low_dir = dirpath.lower()
        if any(h in low_dir for h in HINT_WORDS):
            hint_dirs.add(dirpath)
        for f in files:
            lf = f.lower()
            if lf.endswith(".wav"):
                dir_wavs[dirpath].append(f)
            elif lf.endswith(MANIFEST_EXT) and any(h in lf for h in HINT_WORDS):
                manifests.append(os.path.join(dirpath, f))

    print("=" * 70)
    print(f"WAV-BEARING DIRECTORIES under {args.root}/ "
          f"(>= {args.min_wavs} wavs)")
    print("=" * 70)
    rows = [(d, w) for d, w in dir_wavs.items() if len(w) >= args.min_wavs]
    rows.sort(key=lambda x: len(x[1]), reverse=True)
    if not rows:
        print("  (none)")
    for d, wavs in rows:
        n = len(wavs)
        sample = wavs if args.sample == 0 else wavs[:args.sample]
        secs = 0.0
        measured = 0
        for w in sample:
            s = _wav_seconds(os.path.join(d, w))
            if s is not None:
                secs += s
                measured += 1
        if measured:
            avg = secs / measured
            est_total = avg * n
            tag = "(measured)" if measured == n else \
                  f"(est from {measured} of {n})"
        else:
            est_total = 0.0
            tag = "(no readable wavs)"
        hint = "  <-- name hints synthetic" \
            if any(h in d.lower() for h in HINT_WORDS) else ""
        print(f"  {n:>7,} wav  ~{_human(est_total):>9} {tag}  {d}{hint}")

    print("\n" + "=" * 70)
    print("MANIFESTS whose name hints synthetic/tts/medical")
    print("=" * 70)
    if not manifests:
        print("  (none)")
    for m in sorted(manifests):
        try:
            n = sum(1 for ln in open(m, encoding="utf-8") if ln.strip())
        except OSError:
            n = -1
        print(f"  {n:>7} lines  {m}")

    print("\n" + "=" * 70)
    print("DIRECTORIES with synthetic-ish names (any size)")
    print("=" * 70)
    for d in sorted(hint_dirs):
        nw = len(dir_wavs.get(d, []))
        print(f"  {nw:>6} wav  {d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
