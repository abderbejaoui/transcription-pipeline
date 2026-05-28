"""Build a bakeoff-format eval directory from a training split JSONL.

Use case: the original bakeoff test sets (bakeoff_clean, bakeoff_30min)
have lost their audio files on the DGX, but the training data's
held-out test/validation splits still have real audio. This script
samples N clips from such a split, copies the manifest into the schema
bakeoff.py expects, and symlinks the audio files into the right
subdirectory.

Output directory layout (bakeoff-compatible):
    <out_dir>/
        manifest.jsonl                 # 1 record per clip
        audio/<clip_id>.wav            # symlink to real audio

Manifest record schema:
    {
        "id": str,
        "category": "gulf_held_out",
        "language": "ar",
        "audio_path": "audio/<id>.wav",   # relative to <out_dir>
        "duration_s": float,
        "transcript": str,
        "medical_terms": []
    }

Usage on DGX:
    python scripts/build_eval_from_split.py \\
        --split data/dgx_full/preprocessed_audios_full/splits/test.jsonl \\
        --out eval/gulf_held_out \\
        --n 200 \\
        --seed 42

Then:
    python -m scripts.bakeoff --models qwen3 qwen3_gulf \\
        --eval-dir eval/gulf_held_out
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _resolve_audio_abs(rec: Dict, manifest_path: Path) -> Optional[Path]:
    """Find the on-disk audio file for a training-split record.

    The training splits use various keys (audio_path, audio, path) and
    paths may be absolute, manifest-relative, or relative to one of a
    few standard preprocess output roots.
    """
    raw = rec.get("audio_path") or rec.get("audio") or rec.get("path")
    if not raw:
        return None
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p

    manifest_dir = manifest_path.resolve().parent
    candidates = [
        manifest_dir / raw,
        manifest_dir.parent / raw,
        manifest_dir.parent.parent / raw,
    ]
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _derive_id(rec: Dict, fallback_idx: int) -> str:
    for key in ("id", "clip_id", "uid"):
        v = rec.get(key)
        if isinstance(v, str) and v:
            return v
    raw = rec.get("audio_path") or rec.get("audio") or rec.get("path") or ""
    stem = Path(raw).stem
    return stem or f"clip_{fallback_idx:06d}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", type=Path, required=True,
                    help="Source split JSONL (e.g. splits/test.jsonl).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output eval directory (created if missing).")
    ap.add_argument("--n", type=int, default=200,
                    help="Number of clips to sample. Default 200.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for sampling. Default 42.")
    ap.add_argument("--category", default="gulf_held_out",
                    help="Category label written into manifest records.")
    args = ap.parse_args()

    if not args.split.exists():
        print(f"ERROR: split file not found: {args.split}", file=sys.stderr)
        return 2

    records = _read_jsonl(args.split)
    print(f"[build] loaded {len(records)} records from {args.split}")
    if not records:
        print("ERROR: split is empty", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    if args.n > 0 and args.n < len(records):
        sampled = rng.sample(records, args.n)
    else:
        sampled = list(records)
    print(f"[build] sampled {len(sampled)} clips (seed={args.seed})")

    out_dir = args.out.resolve()
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    missing = 0
    written = 0
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for idx, rec in enumerate(sampled):
            src_audio = _resolve_audio_abs(rec, args.split)
            if src_audio is None:
                missing += 1
                continue
            clip_id = _derive_id(rec, idx)
            dst_audio = audio_dir / f"{clip_id}.wav"
            if dst_audio.exists() or dst_audio.is_symlink():
                dst_audio.unlink()
            try:
                # Symlink (saves disk; falls back to copy if symlink fails).
                os.symlink(src_audio, dst_audio)
            except OSError:
                import shutil
                shutil.copy2(src_audio, dst_audio)

            transcript = (
                rec.get("text")
                or rec.get("transcript")
                or rec.get("target")
                or ""
            )
            duration = rec.get("duration") or rec.get("duration_s") or 0.0
            record = {
                "id": clip_id,
                "category": args.category,
                "language": rec.get("language", "ar"),
                "audio_path": f"audio/{clip_id}.wav",
                "duration_s": float(duration),
                "transcript": transcript,
                "medical_terms": [],
                "source": rec.get("source", ""),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    print(f"[build] wrote {written} records to {manifest_path}")
    if missing:
        print(f"[build] skipped {missing} records (audio file not found)")
    print(f"[build] audio symlinks in: {audio_dir}")
    print()
    print("Next step:")
    print(f"  python -m scripts.bakeoff --models qwen3 qwen3_gulf "
          f"--eval-dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
