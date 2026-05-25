"""Download Google FLEURS Arabic test set and build a bakeoff-format eval dir.

WHY FLEURS:
FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
is Google's multilingual benchmark — 102 languages, clean read speech with
human-verified transcripts. Crucially for our purposes:

  * It is NOT in our training data (we trained on SADA22 + MixAT +
    WorldSpeech + Casablanca; FLEURS comes from a different recording
    pipeline entirely — based on the FLoRes-101 text corpus, read by paid
    native speakers in a studio).
  * Its `ar_eg` (Egyptian) split is the closest to dialectal Arabic in
    FLEURS. FLEURS does not have a Gulf-specific subset; the dialectal
    Arabic in FLEURS is mostly MSA-leaning. This makes it a STRICTER test:
    if our Gulf fine-tune still helps here, the improvement is robust;
    if it hurts, we know we overfit to Gulf.
  * Speaker-disjoint from training (paid FLEURS speakers, not the
    SADA22 / MixAT speaker pool).
  * Permissively licensed (CC-BY-SA-4.0).

Output directory layout (matches build_eval_from_split.py):
    <out_dir>/
        manifest.jsonl
        audio/<id>.wav

Usage on DGX:

    # Default: arabic (ar_eg variant), 200 clips
    python scripts/build_eval_from_fleurs.py \\
        --out eval/fleurs_arabic_ood \\
        --n 200

    # Then bake-off (out-of-distribution comparison):
    python -m scripts.bakeoff \\
        --models qwen3 qwen3_ksa qwen3_uae qwen3_gulf whisper \\
        --eval-dir eval/fleurs_arabic_ood

Requirements:
  pip install datasets soundfile
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional


def _load_fleurs(config: str, split: str, hf_token: Optional[str]):
    """Load the FLEURS dataset from Hugging Face Hub.

    Tries the canonical `google/fleurs` repo first; falls back to mirrors
    if the canonical one is unavailable.
    """
    from datasets import load_dataset

    candidates = ["google/fleurs"]
    last_exc = None
    for repo_id in candidates:
        try:
            print(f"[fleurs] loading {repo_id} (config={config}, split={split})")
            ds = load_dataset(
                repo_id,
                config,
                split=split,
                token=hf_token,
                trust_remote_code=True,
            )
            return ds
        except Exception as exc:
            print(f"[fleurs] {repo_id} failed: {exc!r}")
            last_exc = exc
    raise RuntimeError(
        f"Could not load FLEURS {config}/{split} from any source: {last_exc!r}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        default="ar_eg",
        help=("FLEURS config (language code). Default ar_eg (Egyptian "
              "Arabic — closest dialectal variant in FLEURS). Use ar_xx "
              "or other codes for other languages."),
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["train", "validation", "test"],
        help="Which FLEURS split to download. Default: test.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output eval directory (will be created).",
    )
    ap.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of clips to sample. 0 = use all. Default 200.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    ap.add_argument(
        "--hf-token",
        default=None,
        help=("Hugging Face token (optional). Falls back to HF_TOKEN env var "
              "or huggingface-cli login."),
    )
    args = ap.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Run: pip install soundfile",
              file=sys.stderr)
        return 2

    ds = _load_fleurs(args.config, args.split, hf_token)
    print(f"[fleurs] dataset size = {len(ds)}")

    indices = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    if args.n > 0 and args.n < len(indices):
        indices = indices[: args.n]
    print(f"[fleurs] sampling {len(indices)} clips (seed={args.seed})")

    out_dir = args.out.resolve()
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    written = 0
    skipped = 0
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for i, idx in enumerate(indices):
            rec = ds[idx]
            # FLEURS schema: id (int), audio (dict with array+sampling_rate+path),
            # raw_transcription (str), transcription (str), num_samples (int),
            # gender (int), lang_id (int), language (str), lang_group_id (int).
            audio_info = rec.get("audio") or {}
            arr = audio_info.get("array")
            sr = audio_info.get("sampling_rate")
            if arr is None or sr is None:
                skipped += 1
                continue

            clip_id = f"fleurs_{args.config}_{rec.get('id', idx):06d}"
            dst = audio_dir / f"{clip_id}.wav"
            try:
                sf.write(str(dst), arr, sr, subtype="PCM_16")
            except Exception as exc:
                print(f"  [{i+1}/{len(indices)}] write failed for {clip_id}: {exc!r}")
                skipped += 1
                continue

            transcript = (
                rec.get("transcription")
                or rec.get("raw_transcription")
                or ""
            )
            duration = (rec.get("num_samples") or 0) / sr if sr else 0.0
            record = {
                "id": clip_id,
                "category": f"fleurs_{args.config}",
                "language": "ar" if args.config.startswith("ar") else args.config,
                "audio_path": f"audio/{clip_id}.wav",
                "duration_s": float(duration),
                "transcript": transcript,
                "medical_terms": [],
                "source": f"google/fleurs/{args.config}/{args.split}",
                "fleurs_index": int(idx),
                "gender": rec.get("gender", -1),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(indices)}] {clip_id} ({duration:.1f}s)")

    print(f"[fleurs] wrote {written} records to {manifest_path}")
    if skipped:
        print(f"[fleurs] skipped {skipped} records (missing audio or write error)")
    print(f"[fleurs] audio in: {audio_dir}")
    print()
    print("Next step (bake-off):")
    print(f"  python -m scripts.bakeoff \\")
    print(f"      --models qwen3 qwen3_ksa qwen3_uae qwen3_gulf \\")
    print(f"      --eval-dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
