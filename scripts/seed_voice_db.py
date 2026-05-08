"""One-time seed of the medical voice index + descriptions.

For every term in `data/medical_lexicon.jsonl`:
  1. LLM call -> short clinical description (cached in descriptions.jsonl)
  2. SpeechT5 TTS -> synthetic 16 kHz waveform
  3. wav2vec2 CTC -> phonetic transcript string
  4. voice_match.register_with_embedding(...)

Run:
    source .venv/bin/activate
    python -m scripts.seed_voice_db [--limit N] [--skip-voice] [--reset]

This is idempotent: it skips any term already present in voice_match by
canonical name, and any term already in descriptions.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

LEXICON_PATH = PROJECT_ROOT / "data" / "medical_lexicon.jsonl"


def _load_lexicon():
    rows = []
    with LEXICON_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="seed only first N terms")
    parser.add_argument("--skip-voice", action="store_true", help="only generate descriptions, no TTS")
    parser.add_argument("--skip-descriptions", action="store_true", help="only seed voices, no descriptions")
    parser.add_argument("--reset", action="store_true", help="wipe existing voice index before seeding")
    args = parser.parse_args()

    from app.services import descriptions, voice_match

    rows = _load_lexicon()
    if args.limit:
        rows = rows[: args.limit]
    print(f"Seeding {len(rows)} terms from {LEXICON_PATH.name}")

    if args.reset:
        print("Resetting voice index...")
        voice_match.reset()

    # Step A: descriptions, parallelised.
    if not args.skip_descriptions:
        print("--- Step A: descriptions ---")
        from concurrent.futures import ThreadPoolExecutor, as_completed

        todo = []
        n_skip = 0
        for i, row in enumerate(rows, 1):
            term = row["term"]
            if descriptions.get(term):
                n_skip += 1
                continue
            todo.append((i, term, row.get("type")))

        n_done = 0
        n_fail = 0
        max_workers = int(__import__("os").environ.get("SEED_PARALLEL", "4"))
        print(f"  ({n_skip} already cached, {len(todo)} to fetch, {max_workers} workers)")

        def _one(item):
            i, term, type_hint = item
            t0 = time.time()
            desc = descriptions.get_or_generate(term, type_hint=type_hint)
            return i, term, desc, time.time() - t0

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_one, item) for item in todo]
            for fut in as_completed(futures):
                i, term, desc, dt = fut.result()
                if desc:
                    n_done += 1
                    print(f"  [{i:>3}/{len(rows)}] +desc {term!r} ({dt:.1f}s): {desc[:80]!r}")
                else:
                    n_fail += 1
                    print(f"  [{i:>3}/{len(rows)}] !desc {term!r} (failed)")

        print(f"descriptions: {n_done} new, {n_skip} cached, {n_fail} failed")

    # Step B: voice fingerprints from TTS.
    if not args.skip_voice:
        print("--- Step B: TTS-derived voice fingerprints ---")
        # Use legacy SoundEmbedder only for TTS synthesis.
        from legacy.pipeline import SoundEmbedder

        emb = SoundEmbedder.load()
        n_done = 0
        n_skip = 0
        n_fail = 0
        for i, row in enumerate(rows, 1):
            term = row["term"]
            if voice_match.has_term(term):
                n_skip += 1
                continue
            t0 = time.time()
            try:
                wav = emb.synthesize(term)
                phon = voice_match.embed(wav)
                if not phon:
                    n_fail += 1
                    print(f"  [{i:>3}/{len(rows)}] !voice {term!r} empty phonetic")
                    continue
                desc = descriptions.get(term)
                voice_match.register_with_embedding(
                    term=term,
                    embedding=phon,
                    duration_s=len(wav) / 16_000,
                    description=desc,
                    source="seed",
                )
                n_done += 1
                print(f"  [{i:>3}/{len(rows)}] +voice {term!r} ({time.time()-t0:.1f}s)")
            except Exception as exc:
                n_fail += 1
                print(f"  [{i:>3}/{len(rows)}] !voice {term!r} failed: {exc!r}")
        print(f"voices: {n_done} new, {n_skip} already present, {n_fail} failed")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
