"""Download FLEURS-Arabic test split and convert to bakeoff manifest format.

FLEURS (Few-shot Learning Evaluation of Universal Representations of Speech)
is Google's standard multilingual ASR benchmark. The Arabic config is `ar_eg`
(read-aloud sentences in Modern Standard / Egyptian Arabic). Gold-quality
reference transcripts — the writer wrote first, the speaker read second.

Output:
  eval/fleurs_ar/
    manifest.jsonl     (one JSON record per clip — same schema as bakeoff)
    audio/<id>.wav     (16 kHz mono PCM)
    README.md          (counts + source citation)

Usage:
  pip install datasets soundfile
  python -m scripts.build_fleurs_testset
  # optional: take only N clips for a quick smoke run
  python -m scripts.build_fleurs_testset --max-clips 50

After this you can score any backend against it:
  python -m scripts.bakeoff --eval-dir eval/fleurs_ar --models qwen3 qwen3_uae
  python3 scripts/eval_standard.py --testset eval/fleurs_ar
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "eval" / "fleurs_ar"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test", choices=["test", "validation", "train"],
                    help="Which FLEURS split to materialise (default: test).")
    ap.add_argument("--config", default="ar_eg",
                    help="FLEURS language config (default: ar_eg).")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Optional cap on number of clips (default: all).")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
        import soundfile as sf
    except ImportError as exc:
        print(f"!! missing dependency: {exc}. Run: pip install datasets soundfile",
              file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "audio").mkdir(parents=True, exist_ok=True)

    print(f"Loading google/fleurs config={args.config} split={args.split} ...")
    ds = load_dataset("google/fleurs", args.config, split=args.split)
    print(f"  got {len(ds)} clips")

    if args.max_clips:
        ds = ds.select(range(min(args.max_clips, len(ds))))
        print(f"  truncated to {len(ds)} clips")

    manifest = []
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        sr = audio["sampling_rate"]
        arr = audio["array"]
        cid = f"fleurs_{args.config}_{i:04d}"
        wav_path = OUT_DIR / "audio" / f"{cid}.wav"
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")

        manifest.append({
            "id": cid,
            "category": "fleurs",
            "language": "ar",
            "audio_path": f"audio/{cid}.wav",
            "duration_s": round(len(arr) / sr, 2),
            "transcript": ex["transcription"],
            "raw_transcript": ex.get("raw_transcription", ex["transcription"]),
            "source": f"fleurs:{args.config}",
            "tags": ["fleurs", args.config, args.split],
            "medical_terms": [],
            "gender": ex.get("gender"),
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1:4d}/{len(ds)}]  {cid}  dur={manifest[-1]['duration_s']:.1f}s")

    (OUT_DIR / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
        encoding="utf-8",
    )

    total_dur = sum(r["duration_s"] for r in manifest)
    readme = f"""# FLEURS-Arabic test set ({args.config}, split={args.split})

- Clips: {len(manifest)}
- Total duration: {total_dur/60:.1f} min
- Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs), config `{args.config}`
- License: CC-BY 4.0
- Citation: Conneau et al., *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*, ICASSP 2023. [arXiv:2205.12446](https://arxiv.org/abs/2205.12446)

## How to use

```bash
# Run inference on this set (any backend in bakeoff.py):
python -m scripts.bakeoff --eval-dir eval/fleurs_ar --models qwen3 qwen3_uae

# Score with the industry-standard pipeline:
python3 scripts/eval_standard.py --testset eval/fleurs_ar
```

## Why FLEURS

Every Arabic ASR paper reports WER on FLEURS. Numbers here are directly
comparable to Whisper, MMS, SeamlessM4T, and the NADI 2025 shared task.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nwrote manifest with {len(manifest)} clips ({total_dur/60:.1f} min)")
    print(f"  manifest : {OUT_DIR / 'manifest.jsonl'}")
    print(f"  audio    : {OUT_DIR / 'audio'} (one wav per clip)")
    print(f"  readme   : {OUT_DIR / 'README.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
