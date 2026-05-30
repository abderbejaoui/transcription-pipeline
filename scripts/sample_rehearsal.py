"""Sample a fixed-hour subset from the existing 900h Gulf-Arabic training
manifest for use as the rehearsal anchor in a continued fine-tune.

Reads the existing splits/train.jsonl, computes durations from the
audio_path column, then samples manifest lines until we hit the target
hour budget. The sample is stratified by source dataset to keep
diversity (we don't want all 45h from one dataset).

Usage
-----
python scripts/sample_rehearsal.py \
    --manifest data/dgx_full/preprocessed_audios/splits/train.jsonl \
    --out data/training/gulf_rehearsal/manifest.jsonl \
    --target-hours 45 \
    --seed 42

Output is a single .jsonl with the same schema as the input. No audio
files are copied — paths stay as references into the original corpus.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _row_duration_s(row: Dict[str, Any]) -> float:
    """Return the audio duration for a manifest row.

    The various manifests in this repo use slightly different keys:
      - 'duration_s' (newer preprocess output)
      - 'duration' (older)
      - 'end' - 'start' (segment-style)
    """
    for key in ("duration_s", "duration"):
        if key in row and row[key] is not None:
            return float(row[key])
    if "end" in row and "start" in row:
        return float(row["end"]) - float(row["start"])
    return 0.0


def _resolve_audio_abs(audio_path: str, manifest_path: Path) -> str:
    """Resolve a (possibly relative) audio path to an ABSOLUTE path, using the
    same multi-root strategy as the trainer but anchored at the SOURCE manifest.

    The 900h corpus stores paths like ``audio/sada2022_xxx.wav`` relative to the
    preprocess dir (the manifest's parent or grandparent). We must bake the
    absolute path into the sampled manifest, otherwise the trainer — which sits
    in a different directory — can't find the file.
    """
    p = Path(audio_path)
    if p.is_absolute():
        return str(p)
    mdir = manifest_path.resolve().parent
    candidates = [
        mdir / audio_path,                 # splits/audio/foo.wav
        mdir.parent / audio_path,          # preprocessed_audios/audio/foo.wav  <-- expected
        mdir.parent.parent / audio_path,   # one level up
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    # Best guess (grandparent) so a later error is informative.
    return str((mdir.parent / audio_path).resolve())


def _source_of(row: Dict[str, Any]) -> str:
    """Best-effort dataset identifier for stratified sampling.

    Tries explicit fields first, then falls back to the first path
    component of the audio file (worldspeech_kuwait_..., sada_..., etc.).
    """
    for key in ("source", "dataset", "corpus"):
        if key in row and row[key]:
            return str(row[key])
    path = row.get("audio_path") or row.get("audio") or ""
    stem = Path(path).stem.lower()
    m = re.match(r"([a-z_]+?)[_\-]\d", stem)
    if m:
        return m.group(1)
    return "unknown"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True,
                   help="Source train.jsonl from the 900h preprocess.")
    p.add_argument("--out", required=True,
                   help="Output manifest with the sampled rows.")
    p.add_argument("--target-hours", type=float, default=45.0,
                   help="How many hours to sample. Default 45 (~5% of 900h).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-duration-s", type=float, default=1.5,
                   help="Drop clips shorter than this (filler / noise).")
    p.add_argument("--max-duration-s", type=float, default=20.0,
                   help="Drop clips longer than this (rare in spontaneous "
                        "speech and waste batch padding).")
    args = p.parse_args()

    src = Path(args.manifest)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Load and bucket by source dataset.
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_rows = 0
    dropped_short = 0
    dropped_long = 0
    no_dur = 0
    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dur = _row_duration_s(row)
            if dur <= 0:
                no_dur += 1
                continue
            if dur < args.min_duration_s:
                dropped_short += 1
                continue
            if dur > args.max_duration_s:
                dropped_long += 1
                continue
            by_source[_source_of(row)].append(row)
            total_rows += 1

    print(f"[rehearsal] loaded {total_rows} rows from {len(by_source)} sources")
    print(f"[rehearsal] dropped: {dropped_short} short, "
          f"{dropped_long} long, {no_dur} unknown-duration")
    print()
    total_hours = sum(_row_duration_s(r) for rows in by_source.values()
                      for r in rows) / 3600
    print(f"[rehearsal] usable corpus: {total_hours:.1f} hours")
    print("[rehearsal] per-source breakdown:")
    for source, rows in sorted(by_source.items()):
        h = sum(_row_duration_s(r) for r in rows) / 3600
        print(f"  {source:<35} {len(rows):>6} clips   {h:>7.1f} h")
    print()

    # Stratified sample: take roughly the same FRACTION from each source.
    target_seconds = args.target_hours * 3600
    fraction = min(1.0, target_seconds / max(1.0, total_hours * 3600))
    print(f"[rehearsal] target: {args.target_hours:.1f}h "
          f"({fraction*100:.1f}% of corpus)")

    rng = random.Random(args.seed)
    sampled: List[Dict[str, Any]] = []
    for source, rows in by_source.items():
        rng.shuffle(rows)
        # Take fraction of this source.
        budget = (sum(_row_duration_s(r) for r in rows) / 3600) * fraction * 3600
        taken_s = 0.0
        for r in rows:
            d = _row_duration_s(r)
            if taken_s + d > budget * 1.05:  # 5% over is fine
                continue
            sampled.append(r)
            taken_s += d
            if taken_s >= budget:
                break

    sampled_hours = sum(_row_duration_s(r) for r in sampled) / 3600
    print(f"[rehearsal] sampled {len(sampled)} clips, {sampled_hours:.2f} hours")

    rng.shuffle(sampled)  # shuffle across sources before writing
    # Bake absolute audio paths so the trainer (run from a different dir) finds them.
    missing = 0
    with out.open("w", encoding="utf-8") as fh:
        for row in sampled:
            ap = row.get("audio_path") or row.get("audio") or row.get("path")
            if ap:
                abs_ap = _resolve_audio_abs(ap, src)
                if not Path(abs_ap).exists():
                    missing += 1
                row["audio_path"] = abs_ap
                row.pop("audio", None)
                row.pop("path", None)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    if missing:
        print(f"[rehearsal] WARN: {missing} sampled clips have no resolvable audio file")
    print(f"[rehearsal] wrote {out}")


if __name__ == "__main__":
    main()
