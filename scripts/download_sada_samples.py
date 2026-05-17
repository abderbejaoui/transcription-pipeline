"""
Download and sample ~10,000 utterances from SADA (Saudi Arabic speech corpus).

SADA is a 668-hour Saudi Arabic corpus available via HuggingFace.
We stream it and pick 10k random short utterances for mixing with
medical TTS data during fine-tuning.

Usage:
    python scripts/download_sada_samples.py \
        --out data/training/sada_gulf \
        --n 10000

Output:
    data/training/sada_gulf/wavs/sada_00000.wav ...
    data/training/sada_gulf/manifest.jsonl
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/training/sada_gulf")
    parser.add_argument("--n", type=int, default=10000, help="Number of samples to keep")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Max seconds per clip")
    parser.add_argument("--min-duration", type=float, default=1.0, help="Min seconds per clip")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out)
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    random.seed(args.seed)

    try:
        from datasets import load_dataset
    except ImportError:
        print("pip install datasets soundfile")
        sys.exit(1)

    print(f"[sada] Streaming SADA dataset, target {args.n} samples...")

    # SADA is available as speech-da/SADA on HuggingFace
    # It's large (668h), so we stream and reservoir-sample
    ds = load_dataset(
        "speech-da/SADA",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    reservoir = []
    seen = 0
    t0 = time.time()

    for example in ds:
        audio = example.get("audio", {})
        text = example.get("text") or example.get("sentence") or example.get("transcription") or ""
        if not text or not audio:
            continue

        arr = audio.get("array")
        sr = audio.get("sampling_rate", 16000)
        if arr is None:
            continue

        duration = len(arr) / sr
        if duration < args.min_duration or duration > args.max_duration:
            continue

        text = text.strip()
        if len(text) < 3:
            continue

        seen += 1

        # Reservoir sampling
        if len(reservoir) < args.n:
            reservoir.append((arr, sr, text, duration))
        else:
            j = random.randint(0, seen - 1)
            if j < args.n:
                reservoir[j] = (arr, sr, text, duration)

        if seen % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  scanned {seen} clips, reservoir {len(reservoir)}/{args.n} ({elapsed:.0f}s)")

        # Stop after scanning enough to get a good sample
        if seen >= args.n * 5:
            break

    print(f"[sada] Scanned {seen} clips, kept {len(reservoir)}")

    # Write WAVs + manifest
    manifest = []
    for i, (arr, sr, text, dur) in enumerate(reservoir):
        fname = f"sada_{i:05d}.wav"
        wav_path = wav_dir / fname
        sf.write(str(wav_path), arr, sr)
        manifest.append({
            "audio": str(wav_path),
            "text": text,
            "duration_s": round(dur, 2),
            "source": "sada",
        })
        if (i + 1) % 1000 == 0:
            print(f"  wrote {i+1}/{len(reservoir)} WAVs")

    with open(manifest_path, "w") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    total_hours = sum(e["duration_s"] for e in manifest) / 3600
    print(f"\n[sada] Done: {len(manifest)} samples, {total_hours:.1f} hours")
    print(f"[sada] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
