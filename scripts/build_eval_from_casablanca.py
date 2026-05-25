"""Download UBC-NLP/Casablanca dialect subset and build a bakeoff-format eval.

ABOUT CASABLANCA:
  - Published Oct 2024 by UBC Deep Learning & NLP Lab (arxiv:2410.04527).
  - 8 Arabic dialects: Algerian, Egyptian, Emirati, Jordanian, Mauritanian,
    Moroccan, Palestinian, Yemeni.
  - Each dialect has a validation + test split (no train, by design — it's
    a held-out benchmark).
  - Used by the Open Universal Arabic ASR Leaderboard.
  - Truly out-of-distribution for our model: published after our training
    data was collected, different recording sources, speaker-disjoint.

For Gulf evaluation we use the Emirati subset (closest to our target
domain). UAE/Saudi/Kuwait don't have their own Casablanca subsets — only
Emirati represents the Gulf cluster.

License: CC-BY-NC-ND-4.0 (research use only — fine for evaluation).

Output layout (matches the other build_eval_* scripts):
    <out_dir>/
        manifest.jsonl
        audio/<id>.wav

Usage on DGX:

    python scripts/build_eval_from_casablanca.py \\
        --dialect Emirati \\
        --split test \\
        --out eval/casablanca_emirati_ood \\
        --n 200

    python -m scripts.bakeoff \\
        --models qwen3 qwen3_ksa qwen3_uae qwen3_gulf \\
        --eval-dir eval/casablanca_emirati_ood
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional


# Casablanca exposes one subset per dialect. The HF subset names are the
# country/dialect labels shown in the dataset viewer. We provide aliases
# so users don't need to remember the exact casing.
_DIALECT_ALIASES = {
    "emirati": ["Emirati", "UAE", "Emirates"],
    "uae":     ["Emirati", "UAE", "Emirates"],
    "egyptian": ["Egyptian", "Egypt"],
    "algerian": ["Algerian", "Algeria"],
    "jordanian": ["Jordanian", "Jordan"],
    "mauritanian": ["Mauritanian", "Mauritania"],
    "moroccan": ["Moroccan", "Morocco"],
    "palestinian": ["Palestinian", "Palestine"],
    "yemeni": ["Yemeni", "Yemen"],
}


def _resolve_subset_candidates(dialect: str) -> list:
    """Return a list of subset names to try (case variants included)."""
    key = dialect.strip().lower()
    if key in _DIALECT_ALIASES:
        candidates = list(_DIALECT_ALIASES[key])
    else:
        candidates = []
    # Also try the literal string the user passed.
    if dialect not in candidates:
        candidates.append(dialect)
    return candidates


def _load_casablanca(dialect: str, split: str, hf_token: Optional[str]):
    """Load a Casablanca dialect subset, bypassing the torchcodec decoder."""
    from datasets import load_dataset, Value

    last_exc = None
    for subset in _resolve_subset_candidates(dialect):
        try:
            print(f"[casablanca] trying subset={subset!r} split={split!r}")
            ds = load_dataset(
                "UBC-NLP/Casablanca",
                subset,
                split=split,
                token=hf_token,
            )
            print(f"[casablanca] loaded subset={subset!r} (n={len(ds)})")
            break
        except Exception as exc:
            print(f"[casablanca] subset={subset!r} failed: {exc!r}")
            last_exc = exc
            ds = None
    if ds is None:
        raise RuntimeError(
            f"Could not load any subset variant of UBC-NLP/Casablanca for "
            f"dialect={dialect!r} split={split!r}. Last error: {last_exc!r}"
        )

    # Bypass torchcodec by casting the Audio column to raw bytes (same
    # trick as build_eval_from_fleurs.py).
    try:
        feats = ds.features.copy()
        if "audio" in feats:
            feats["audio"] = {"bytes": Value("binary"), "path": Value("string")}
            ds = ds.cast(feats)
    except Exception as exc:
        print(f"[casablanca] cast failed ({exc!r}); trying decode-off path")
        try:
            ds = ds.cast_column(
                "audio",
                ds.features["audio"].__class__(decode=False),
            )
        except Exception as exc2:
            print(f"[casablanca] decode-off also failed: {exc2!r}")
            raise
    return ds


def _pick_transcript_field(rec: dict) -> str:
    """Casablanca uses different transcript column names across versions.

    We try the most common candidates in order and return the first
    non-empty string.
    """
    for key in ("transcript", "text", "sentence", "transcription",
                "raw_transcription"):
        v = rec.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dialect",
        default="Emirati",
        help="Casablanca dialect subset. Default: Emirati (Gulf-relevant).",
    )
    ap.add_argument(
        "--split",
        default="test",
        choices=["validation", "test"],
        help="Casablanca only releases validation + test splits.",
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
        help="HF token. Falls back to HF_TOKEN env var or hf auth login.",
    )
    args = ap.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    try:
        import soundfile as sf
    except ImportError:
        print("ERROR: soundfile not installed. Run: pip install soundfile",
              file=sys.stderr)
        return 2

    ds = _load_casablanca(args.dialect, args.split, hf_token)

    indices = list(range(len(ds)))
    rng = random.Random(args.seed)
    rng.shuffle(indices)
    if args.n > 0 and args.n < len(indices):
        indices = indices[: args.n]
    print(f"[casablanca] sampling {len(indices)} clips (seed={args.seed})")

    out_dir = args.out.resolve()
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    written = 0
    skipped = 0
    skipped_no_transcript = 0
    with manifest_path.open("w", encoding="utf-8") as out_f:
        for i, idx in enumerate(indices):
            rec = ds[idx]
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
                arr = audio_info["array"]
                sr = audio_info.get("sampling_rate")
            if arr is None or sr is None:
                skipped += 1
                continue

            transcript = _pick_transcript_field(rec)
            if not transcript:
                skipped_no_transcript += 1
                continue

            clip_id = f"casa_{args.dialect.lower()}_{idx:06d}"
            dst = audio_dir / f"{clip_id}.wav"
            try:
                sf.write(str(dst), arr, sr, subtype="PCM_16")
            except Exception as exc:
                print(f"  [{i+1}/{len(indices)}] write failed: {exc!r}")
                skipped += 1
                continue

            try:
                n_samples = int(len(arr))
            except TypeError:
                n_samples = 0
            duration = n_samples / sr if sr else 0.0

            record = {
                "id": clip_id,
                "category": f"casablanca_{args.dialect.lower()}",
                "language": "ar",
                "audio_path": f"audio/{clip_id}.wav",
                "duration_s": float(duration),
                "transcript": transcript,
                "medical_terms": [],
                "source": f"UBC-NLP/Casablanca/{args.dialect}/{args.split}",
                "casablanca_index": int(idx),
                "gender": rec.get("gender", -1),
                "code_switching": rec.get("code_switching",
                                          rec.get("code_switch", None)),
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if (i + 1) % 20 == 0:
                print(f"  [{i+1}/{len(indices)}] {clip_id} ({duration:.1f}s)")

    print(f"[casablanca] wrote {written} records to {manifest_path}")
    if skipped:
        print(f"[casablanca] skipped {skipped} records (decode/write error)")
    if skipped_no_transcript:
        print(f"[casablanca] skipped {skipped_no_transcript} records "
              f"(no transcript field — try a different --split)")
    print(f"[casablanca] audio in: {audio_dir}")
    print()
    print("Next step (bake-off all 9 models):")
    print(f"  python -m scripts.bakeoff \\")
    print(f"      --models qwen3 qwen3_ksa qwen3_uae \\")
    print(f"              qwen3_gulf_ckpt12k qwen3_gulf_ckpt14k \\")
    print(f"              qwen3_gulf_ckpt16k qwen3_gulf_ckpt18k \\")
    print(f"              qwen3_gulf_ckpt19k qwen3_gulf \\")
    print(f"      --eval-dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
