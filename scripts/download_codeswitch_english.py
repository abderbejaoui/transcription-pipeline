"""Download + manifest the Phase-B buckets for the v2 medical fine-tune:

  1. CODE-SWITCH (Arabic<->English):  MASC (Massive Arabic Speech Corpus) — has
     plenty of dialectal Arabic with embedded English, which is exactly the
     "Panadol", "antibiotic", brand-name code-switch we want the model to keep.
     Alternatively a slice of Common Voice Arabic if MASC access is gated.

  2. ENGLISH MEDICAL:  PriMock57 (mock primary-care consultations, en) and/or a
     medical slice of Common Voice English. Keeps the English drug/medical
     vocabulary sharp so the model doesn't over-fit to Arabic-only.

Each bucket is written as a JSONL manifest with the fields the master builder
needs: audio_path, text, duration_s  (+ source, lang for bookkeeping).

Audio is resampled to 16 kHz mono WAV under data/training/<bucket>/wavs/.

Run on the DGX inside the venv. HF datasets that are gated need
`huggingface-cli login` first.

Usage:
  python -m scripts.download_codeswitch_english \
    --codeswitch-out data/training/codeswitch_masc \
    --english-out    data/training/english_medical \
    --max-hours-codeswitch 20 --max-hours-english 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _write_manifest(rows, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    total_h = sum(r["duration_s"] for r in rows) / 3600.0
    print(f"[dl] wrote {len(rows)} clips ({total_h:.2f} h) -> {manifest}")


def _export_hf_split(ds, out_dir: Path, text_key: str, lang: str, source: str,
                     max_seconds: float):
    """Iterate an HF audio dataset, resample to 16k WAV, build manifest rows."""
    import soundfile as sf
    import librosa
    import numpy as np

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    taken = 0.0
    for i, ex in enumerate(ds):
        audio = ex["audio"]
        wav = np.asarray(audio["array"], dtype="float32")
        sr = int(audio["sampling_rate"])
        if sr != 16000:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        dur = len(wav) / float(sr)
        if dur < 1.0 or dur > 25.0:
            continue
        text = (ex.get(text_key) or "").strip()
        if not text:
            continue
        wav_path = wav_dir / f"{source}_{i:06d}.wav"
        sf.write(str(wav_path), wav, sr)
        rows.append({
            "audio_path": str(wav_path),
            "text": text,
            "duration_s": round(dur, 3),
            "source": source,
            "lang": lang,
        })
        taken += dur
        if taken >= max_seconds:
            break
    return rows


def fetch_codeswitch(out_dir: Path, max_hours: float) -> None:
    from datasets import load_dataset
    max_seconds = max_hours * 3600.0
    print(f"[dl] code-switch: trying MASC (streaming), budget {max_hours} h")
    try:
        ds = load_dataset("pain/MASC", split="train", streaming=True,
                          trust_remote_code=True)
        rows = _export_hf_split(ds, out_dir, text_key="transcript",
                                lang="ar-en", source="masc",
                                max_seconds=max_seconds)
    except Exception as e:  # noqa: BLE001
        print(f"[dl] MASC unavailable ({e}); falling back to Common Voice ar")
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "ar",
                          split="train", streaming=True, trust_remote_code=True)
        rows = _export_hf_split(ds, out_dir, text_key="sentence",
                                lang="ar", source="cv_ar",
                                max_seconds=max_seconds)
    _write_manifest(rows, out_dir)


def fetch_english_medical(out_dir: Path, max_hours: float) -> None:
    from datasets import load_dataset
    max_seconds = max_hours * 3600.0
    print(f"[dl] english medical: trying PriMock57, budget {max_hours} h")
    try:
        ds = load_dataset("Hani89/primock57", split="train", streaming=True,
                          trust_remote_code=True)
        rows = _export_hf_split(ds, out_dir, text_key="transcript",
                                lang="en", source="primock57",
                                max_seconds=max_seconds)
    except Exception as e:  # noqa: BLE001
        print(f"[dl] PriMock57 unavailable ({e}); falling back to Common Voice en")
        ds = load_dataset("mozilla-foundation/common_voice_17_0", "en",
                          split="train", streaming=True, trust_remote_code=True)
        rows = _export_hf_split(ds, out_dir, text_key="sentence",
                                lang="en", source="cv_en",
                                max_seconds=max_seconds)
    _write_manifest(rows, out_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codeswitch-out", type=Path,
                    default=Path("data/training/codeswitch_masc"))
    ap.add_argument("--english-out", type=Path,
                    default=Path("data/training/english_medical"))
    ap.add_argument("--max-hours-codeswitch", type=float, default=20.0)
    ap.add_argument("--max-hours-english", type=float, default=12.0)
    ap.add_argument("--skip-codeswitch", action="store_true")
    ap.add_argument("--skip-english", action="store_true")
    args = ap.parse_args()

    if not args.skip_codeswitch:
        fetch_codeswitch(args.codeswitch_out, args.max_hours_codeswitch)
    if not args.skip_english:
        fetch_english_medical(args.english_out, args.max_hours_english)
    return 0


if __name__ == "__main__":
    sys.exit(main())
