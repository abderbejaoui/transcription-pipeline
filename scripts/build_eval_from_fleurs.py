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

    We cast the `audio` column from the Audio feature back to plain
    bytes, which lets us load FLEURS without requiring `torchcodec`
    (the new mandatory audio backend in `datasets>=4.x`). Each row's
    audio field becomes a dict with the raw file bytes; we decode it
    ourselves with `soundfile` later.
    """
    from datasets import load_dataset, Value

    print(f"[fleurs] loading google/fleurs (config={config}, split={split})")
    ds = load_dataset(
        "google/fleurs",
        config,
        split=split,
        token=hf_token,
    )
    # Replace the Audio feature with a struct that returns raw bytes.
    # This sidesteps the torchcodec-dependent decode path.
    try:
        from datasets import Features, Sequence
        feats = ds.features.copy()
        if "audio" in feats:
            # Audio feature exposes {"bytes": bytes, "path": str}; we want
            # both fields as plain types so __getitem__ never invokes the
            # decoder.
            feats["audio"] = {"bytes": Value("binary"), "path": Value("string")}
            ds = ds.cast(feats)
    except Exception as exc:
        # Some datasets versions don't allow cast on Audio columns.
        # Fall back to disabling the decoder entirely.
        print(f"[fleurs] cast failed ({exc!r}); trying decode-off path")
        try:
            ds = ds.cast_column("audio", ds.features["audio"].__class__(decode=False))
        except Exception as exc2:
            print(f"[fleurs] decode-off failed too: {exc2!r}")
            raise
    return ds


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
    import io
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for i, idx in enumerate(indices):
            rec = ds[idx]
            # FLEURS schema after our cast: id (int), audio = {"bytes":
            # bytes, "path": str}, raw_transcription (str), transcription
            # (str), num_samples (int), gender (int), language (str), ...
            audio_info = rec.get("audio") or {}
            arr = None
            sr = None
            raw_bytes = audio_info.get("bytes") if isinstance(audio_info, dict) else None
            if raw_bytes:
                try:
                    arr, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32",
                                      always_2d=False)
                except Exception as exc:
                    print(f"  [{i+1}/{len(indices)}] decode failed: {exc!r}")
            elif isinstance(audio_info, dict) and audio_info.get("array") is not None:
                # Older datasets versions still expose decoded array.
                arr = audio_info["array"]
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
            # Prefer the actual decoded sample count, fall back to the
            # FLEURS-provided num_samples if available.
            try:
                n_samples = int(len(arr))
            except TypeError:
                n_samples = int(rec.get("num_samples") or 0)
            duration = n_samples / sr if sr else 0.0
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
