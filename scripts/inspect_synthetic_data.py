#!/usr/bin/env python3
"""Inspect the synthetic TTS reference set used for medical code-switch.

Verifies data/tts_references/ exists with references.jsonl and that each
referenced .wav + .txt is present, then reports counts and total duration.

Usage:
    python scripts/inspect_synthetic_data.py
    python scripts/inspect_synthetic_data.py --root data/tts_references
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import wave

AUDIO_KEYS = ("audio_path", "audio_filepath", "audio", "path", "wav", "file")
TEXT_KEYS = ("text", "transcript", "transcription", "sentence")
TEXTFILE_KEYS = ("text_path", "txt", "txt_path", "transcript_path")


def _first(row: dict, keys, default=None):
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    return default


def _wav_seconds(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            frames = w.getnframes()
            rate = w.getframerate()
            if rate:
                return frames / float(rate)
    except (wave.Error, EOFError, FileNotFoundError):
        return None
    return None


def _resolve(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    cand = os.path.join(root, p)
    if os.path.exists(cand):
        return cand
    # Maybe the path is relative to repo root, not the tts root.
    if os.path.exists(p):
        return p
    return cand


def _human(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m {s:02d}s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/tts_references")
    ap.add_argument("--index", default="references.jsonl",
                    help="Index filename inside --root.")
    args = ap.parse_args()

    root = args.root
    index_path = os.path.join(root, args.index)

    print("=" * 70)
    print("SYNTHETIC TTS DATA INSPECTION")
    print("=" * 70)

    if not os.path.isdir(root):
        print(f"  [MISSING] directory: {root}")
        return 1
    print(f"  [OK] directory: {root}")

    # Inventory raw files on disk.
    wavs = sorted(f for f in os.listdir(root) if f.lower().endswith(".wav"))
    txts = sorted(f for f in os.listdir(root) if f.lower().endswith(".txt"))
    print(f"       on disk: {len(wavs)} .wav  /  {len(txts)} .txt")

    if not os.path.exists(index_path):
        print(f"  [MISSING] index: {index_path}")
        index_rows = []
    else:
        print(f"  [OK] index: {index_path}")
        index_rows = []
        with open(index_path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    index_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        print(f"       index entries: {len(index_rows)}")

    # Validate each index row -> wav + transcript present, measure duration.
    total_secs = 0.0
    measured = 0
    missing_wav = []
    missing_txt = []
    empty_text = 0

    for i, row in enumerate(index_rows):
        ap_val = _first(row, AUDIO_KEYS)
        if ap_val:
            wav_path = _resolve(root, ap_val)
            if os.path.exists(wav_path):
                secs = _wav_seconds(wav_path)
                if secs is not None:
                    total_secs += secs
                    measured += 1
            else:
                missing_wav.append(ap_val)

        # Transcript: inline text field or sidecar .txt path.
        inline = _first(row, TEXT_KEYS)
        tf = _first(row, TEXTFILE_KEYS)
        if inline:
            if not str(inline).strip():
                empty_text += 1
        elif tf:
            tpath = _resolve(root, tf)
            if not os.path.exists(tpath):
                missing_txt.append(tf)
        else:
            # Infer sidecar from wav name (ref_001.wav -> ref_001.txt).
            if ap_val:
                guess = os.path.splitext(_resolve(root, ap_val))[0] + ".txt"
                if not os.path.exists(guess):
                    missing_txt.append(os.path.basename(guess))

    print("\n  --- validation ---")
    if measured:
        print(f"       measured duration on {measured} clips: "
              f"{_human(total_secs)} ({total_secs/3600:.2f}h, "
              f"avg {total_secs/measured:.1f}s/clip)")
    else:
        print("       no .wav durations measured "
              "(files missing or not PCM wav)")
    if missing_wav:
        print(f"       MISSING WAVS ({len(missing_wav)}): "
              f"{missing_wav[:10]}{' ...' if len(missing_wav) > 10 else ''}")
    if missing_txt:
        print(f"       MISSING TRANSCRIPTS ({len(missing_txt)}): "
              f"{missing_txt[:10]}{' ...' if len(missing_txt) > 10 else ''}")
    if empty_text:
        print(f"       EMPTY inline transcripts: {empty_text}")
    if not (missing_wav or missing_txt or empty_text) and index_rows:
        print("       all index entries have a wav + transcript ✔")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  index entries : {len(index_rows)}")
    print(f"  wav files     : {len(wavs)}")
    print(f"  txt files     : {len(txts)}")
    print(f"  total audio   : {total_secs/3600:.2f} h "
          f"({_human(total_secs)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
