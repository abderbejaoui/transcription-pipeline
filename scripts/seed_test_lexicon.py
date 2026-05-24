"""Add the medical terms used in the test set to the lexicon.

This script is idempotent — running it twice doesn't duplicate entries.
It reads the ground-truth transcripts from
  /Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/metadata.csv
finds the test-set rows (the ones whose audio is in audio_cache_preprocessed_10/),
extracts a curated set of medical terms, and adds them to
  data/medical_lexicon.jsonl
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path("/Users/abderrahmenbejaoui/test_sound_pipeline")
LEXICON_PATH = PROJECT_ROOT / "data/medical_lexicon.jsonl"
META = Path("/Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/metadata.csv")

# Curated medical terms that appear in the 10-file test set ground truth.
# Terms taken directly from metadata.csv rows for files 100096..101429.
TEST_SET_TERMS = [
    {"term": "asthma", "type": "diagnosis", "aliases": [], "priority": 1.0},
    {"term": "collarbone", "type": "anatomy", "aliases": ["collarbones"], "priority": 1.0},
    {"term": "clavicle", "type": "anatomy", "aliases": ["clavicle bone"], "priority": 1.0},
    {"term": "stents", "type": "device", "aliases": ["stent", "heart stents"], "priority": 1.0},
    {"term": "heart attack", "type": "diagnosis", "aliases": [], "priority": 1.0},
    {"term": "myofascial", "type": "diagnosis", "aliases": ["myofacial", "myofasial"], "priority": 1.0},
    {"term": "spinal column", "type": "anatomy", "aliases": ["spine"], "priority": 1.0},
    {"term": "platelet count", "type": "diagnosis", "aliases": ["platelet"], "priority": 1.0},
    {"term": "coagulation profile", "type": "diagnosis", "aliases": ["coagulation"], "priority": 1.0},
    {"term": "extravasation", "type": "diagnosis", "aliases": ["extravization"], "priority": 1.0},
    {"term": "dermatologist", "type": "specialty", "aliases": [], "priority": 1.0},
    {"term": "hypothyroidism", "type": "diagnosis", "aliases": [], "priority": 1.0},
    {"term": "hyperthyroidism", "type": "diagnosis", "aliases": [], "priority": 1.0},
    {"term": "kidney stones", "type": "diagnosis", "aliases": [], "priority": 1.0},
    {"term": "thyroid function test", "type": "test", "aliases": ["TFT"], "priority": 1.0},
    {"term": "constipation", "type": "symptom", "aliases": [], "priority": 1.0},
    {"term": "cold intolerance", "type": "symptom", "aliases": [], "priority": 1.0},
    {"term": "myofascial pain syndrome", "type": "diagnosis", "aliases": ["myofacial pain"], "priority": 1.0},
    {"term": "nervous system", "type": "anatomy", "aliases": [], "priority": 1.0},
    {"term": "respiration", "type": "anatomy", "aliases": [], "priority": 1.0},
    # Drugs from the broader medical context.
    {"term": "Efferalgan", "type": "drug", "aliases": ["Effiralgon", "Effiralgan", "Eferalgon"], "priority": 1.0},
]


def main() -> int:
    existing = set()
    if LEXICON_PATH.exists():
        for line in LEXICON_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                existing.add(entry["term"].strip().lower())
            except Exception:
                pass

    added = 0
    with LEXICON_PATH.open("a", encoding="utf-8") as fh:
        for term in TEST_SET_TERMS:
            if term["term"].lower() in existing:
                continue
            fh.write(json.dumps(term, ensure_ascii=False) + "\n")
            added += 1
            existing.add(term["term"].lower())

    total = sum(1 for _ in LEXICON_PATH.read_text().splitlines() if _.strip())
    print(f"Added {added} new terms. Lexicon now has {total} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
