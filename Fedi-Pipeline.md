# Fedi Pipeline

This document describes the ASR post-correction pipeline implemented in this repo.

## Purpose

- Input: a raw audio file
- Output: a corrected medical transcript plus audit metadata
- Design goals: conservative auto-apply, doctor review for medical names, and a learning loop

## High-level flow

1) ASR transcription with word-level confidence
2) Error lexicon lookup (doctor-confirmed corrections)
3) Local medical entities KG lookup
4) MedSpeak KG retrieval (semantic + phonetic)
5) LLM fallback (Calme-7B), medical route always reviewed
6) Coherence verification (Calme-7B); revert if low confidence

## Inputs and outputs

Input
- Audio file uploaded to /api/transcribe or /api/transcribe_stream

Output
- raw_text
- corrected_text
- suspicious: list of decisions for each flagged span
- review_queue: items routed to doctor review
- coherence: verifier score and revert flag
- asr: ASR metadata and word list

## Stage 1: Transcription and flagging

- The ASR engine returns text and word-level confidences.
- Any word with confidence below WORD_CONFIDENCE_THRESHOLD is flagged.
- Confident words are never sent to the LLM.
- Adjacent low-confidence words are merged into a span using FLAG_MERGE_GAP_S.

## Stage 2: Error lexicon (doctor-confirmed corrections)

- Backing store: lexicon/corrections.json
- Keyed by lowercased ASR form, human readable and version controlled.
- Lookup order:
  1) Exact match
  2) Double Metaphone match
  3) Fuzzy similarity (RapidFuzz)
- If matched, the correction is applied immediately unless it is a drug name.
- Drug name corrections are queued for doctor review.

## Stage 3: Local medical entities KG

- Backing store: data/medical_entities.json
- Highest priority for UAE-specific brand names and local entities.
- Structure:
  - drugs: list of objects { canonical, aliases }
  - diagnoses: list of strings
  - procedures: list of strings
- Behavior:
  - score >= KG_AUTOFIX_THRESHOLD: auto-apply
  - score >= KG_SUSPECT_THRESHOLD: queue for review
  - otherwise fall through to MedSpeak

## Stage 4: MedSpeak KG retrieval

- Artifacts:
  - vendor/medspeakian/artifacts/kg_semantic.sqlite
  - vendor/medspeakian/artifacts/kg_phonetic.jsonl
- Uses both semantic and phonetic signals from the KG.
- Behavior:
  - score >= MEDSPEAK_AUTO_THRESHOLD: auto-apply
  - score >= MEDSPEAK_MIN_SCORE: queue for review
  - otherwise fall through to LLM

## Stage 5: LLM fallback (Calme-7B)

- Used only when MedSpeak does not produce a usable match.
- General route is auto-applied only if confidence >= LLM_CONFIDENCE_THRESHOLD.
- Medical route is always queued for doctor review, regardless of confidence.

## Stage 6: Coherence verification

- The verifier compares raw_text and corrected_text.
- If coherence confidence < COHERENCE_THRESHOLD, the pipeline reverts to raw_text.

## Doctor review queue

- Backing store: data/doctor_review_queue.jsonl
- Any of the following are queued:
  - drug name corrections
  - low confidence local KG matches
  - MedSpeak matches below MEDSPEAK_AUTO_THRESHOLD
  - any LLM medical correction

## Learning loop

- /api/learn_from_edit writes confirmed corrections to:
  - lexicon/corrections.json (error lexicon)
  - data/medical_entities.json (KG)
- For drugs, the original wrong phrase is stored as an alias.

## MedSpeak integration notes

- The MedSpeak repo is cloned into vendor/medspeakian.
- This pipeline uses the KG artifacts only, not the MedSpeak LLM.
- If KG artifacts are missing, MedSpeak retrieval is skipped.

## Environment variables

Core thresholds
- WORD_CONFIDENCE_THRESHOLD (default 0.70)
- KG_AUTOFIX_THRESHOLD (default 90)
- KG_SUSPECT_THRESHOLD (default 80)
- MEDSPEAK_AUTO_THRESHOLD (default 0.60)
- MEDSPEAK_MIN_SCORE (default 0.60)
- LLM_CONFIDENCE_THRESHOLD (default 0.70)
- COHERENCE_THRESHOLD (default 0.60)

Paths
- KG_ENTITIES_PATH (default data/medical_entities.json)
- MEDSPEAK_KG_SQLITE (default vendor/medspeakian/artifacts/kg_semantic.sqlite)
- MEDSPEAK_KG_PHONETIC (default vendor/medspeakian/artifacts/kg_phonetic.jsonl)

Models
- LLM_MODEL_GENERAL (default MaziyarPanahi/Calme-7B-Instruct-v0.2)
- LLM_MODEL_MEDICAL (default MaziyarPanahi/Calme-7B-Instruct-v0.2)
- LLM_MODEL_VERIFY (default MaziyarPanahi/Calme-7B-Instruct-v0.2)

## Run locally

- Install dependencies: pip install -r requirements.txt
- Start server: ./run.sh
- Health check: curl http://127.0.0.1:8000/api/healthz

## Data files (created or updated)

- data/medical_entities.json
- lexicon/corrections.json
- data/doctor_review_queue.jsonl
- data/sessions/ (uploaded audio)

## Where the code lives

- Pipeline orchestration: app/main.py
- Error lexicon: app/services/error_lexicon.py
- Local KG: app/services/kg_lookup.py
- MedSpeak KG retrieval: app/services/medspeakian.py
- LLM correction: app/services/llm_correct.py
- LLM verification: app/services/llm_verify.py
- LLM runtime warmup: app/services/llm_runtime.py
