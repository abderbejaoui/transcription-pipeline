"""Download Casablanca multi-dialect Arabic ASR test set and convert to
bakeoff manifest format.

Casablanca (Talafha et al., 2024, arXiv:2410.04527) is the gold-standard
public benchmark for dialectal Arabic ASR. Eight dialects, all
conversational speech with human-verified transcripts.

  Algeria | Egypt | Jordan | Mauritania | Morocco | Palestine | UAE | Yemen

There is NO Saudi or Kuwait config. For Gulf coverage use `UAE` (closest to
your existing SADA22 / WorldSpeech Gulf clips) and optionally `Yemen` and
`Jordan` for Levantine cross-comparison.

License: **CC-BY-NC-ND-4.0** — research and evaluation only. Do not embed
into a commercial product.

Output:
  eval/casablanca_{dialect}/
    manifest.jsonl
    audio/<id>.wav     (16 kHz mono PCM)
    README.md

Usage:
  pip install datasets soundfile huggingface_hub
  # UAE test split, all 813 clips:
  python -m scripts.build_casablanca_testset --dialect UAE
  # Smoke test with 30 clips:
  python -m scripts.build_casablanca_testset --dialect UAE --max-clips 30
  # Multiple dialects (one folder each):
  python -m scripts.build_casablanca_testset --dialect UAE --dialect Yemen --dialect Jordan

After this, score any backend against it the usual way:
  python -m scripts.bakeoff --eval-dir eval/casablanca_UAE --models qwen3 qwen3_uae
  python3 scripts/eval_standard.py --testset eval/casablanca_UAE
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Per the dataset card: https://huggingface.co/datasets/UBC-NLP/Casablanca
VALID_DIALECTS = {
    "Algeria", "Egypt", "Jordan", "Mauritania",
    "Morocco", "Palestine", "UAE", "Yemen",
}


def build_one(dialect: str, split: str, max_clips: int | None) -> int:
    if dialect not in VALID_DIALECTS:
        print(f"!! unknown dialect '{dialect}'. Valid: {sorted(VALID_DIALECTS)}",
              file=sys.stderr)
        return 1

    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError as exc:
        print(f"!! missing dependency: {exc}. Run: pip install datasets soundfile",
              file=sys.stderr)
        return 1

    out_dir = PROJECT_ROOT / "eval" / f"casablanca_{dialect}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    print(f"\nLoading UBC-NLP/Casablanca config={dialect} split={split} ...")
    ds = load_dataset("UBC-NLP/Casablanca", dialect, split=split)
    print(f"  got {len(ds)} clips")

    if max_clips:
        ds = ds.select(range(min(max_clips, len(ds))))
        print(f"  truncated to {len(ds)} clips")

    manifest = []
    skipped = 0
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"]
        transcript = (ex.get("transcription") or "").strip()

        if not transcript:
            skipped += 1
            continue

        cid = f"casablanca_{dialect.lower()}_{i:04d}"
        wav_path = out_dir / "audio" / f"{cid}.wav"
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")

        manifest.append({
            "id": cid,
            "category": "casablanca",
            "language": "ar",
            "audio_path": f"audio/{cid}.wav",
            "duration_s": round(float(ex.get("duration") or len(arr) / sr), 2),
            "transcript": transcript,
            "source": f"casablanca:{dialect}",
            "tags": ["casablanca", dialect.lower(), split],
            "medical_terms": [],
            "gender": ex.get("gender"),
            "seg_id": ex.get("seg_id"),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{len(ds)}]  {cid}  dur={manifest[-1]['duration_s']:.1f}s")

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
        encoding="utf-8",
    )

    total_dur = sum(r["duration_s"] for r in manifest)
    readme = f"""# Casablanca-{dialect} test set (split={split})

- Clips: {len(manifest)} (skipped {skipped} empty-transcript clips)
- Total duration: {total_dur/60:.1f} min
- Source: [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca), config `{dialect}`
- License: **CC-BY-NC-ND-4.0** — non-commercial, no-derivatives. Research/eval use only.
- Citation: Talafha et al. (2024), *Casablanca: Data and Models for Multidialectal Arabic Speech Recognition*. [arXiv:2410.04527](https://arxiv.org/abs/2410.04527)

## What this is

Casablanca is the current public benchmark for **dialectal Arabic ASR**.
Unlike FLEURS (Modern Standard / Egyptian, read speech), Casablanca is
**conversational dialectal speech** with human-verified transcripts. The
`{dialect}` config covers the {dialect} dialect specifically.

## How to use

```bash
# Run inference on this set (any backend in bakeoff.py):
python -m scripts.bakeoff --eval-dir eval/casablanca_{dialect} --models qwen3 qwen3_uae

# Score with the industry-standard pipeline:
python3 scripts/eval_standard.py --testset eval/casablanca_{dialect}
```

## Reference numbers (from the Casablanca paper)

Best zero-shot WER on the {dialect} test split is in the **30-50%** range
for SOTA models in 2024. Anything below that is competitive.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nwrote manifest with {len(manifest)} clips ({total_dur/60:.1f} min)")
    print(f"  manifest : {out_dir / 'manifest.jsonl'}")
    print(f"  audio    : {out_dir / 'audio'}")
    print(f"  readme   : {out_dir / 'README.md'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", action="append", default=None,
                    help=f"Casablanca config to materialise. Repeat for multiple. "
                         f"Default: UAE. Valid: {sorted(VALID_DIALECTS)}")
    ap.add_argument("--split", default="test", choices=["test", "validation"],
                    help="Which split to materialise (default: test).")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Optional cap per dialect (default: all).")
    args = ap.parse_args()

    dialects = args.dialect or ["UAE"]
    print(f"Dialects to build: {dialects}")
    print(f"Split            : {args.split}")
    if args.max_clips:
        print(f"Max clips        : {args.max_clips}")

    rc = 0
    for d in dialects:
        if build_one(d, args.split, args.max_clips) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
