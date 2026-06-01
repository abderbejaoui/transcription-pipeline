# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

End-to-end Arabic medical transcription pipeline for Gulf Arabic-speaking clinicians. Transcribes audio via Whisper/Qwen3 ASR, then corrects the output — fixing English medical misspellings, Arabic phonetic errors, and Arabic→English transliterations (e.g. `هستوري` → `history`). Served as a FastAPI web app with a browser UI.

## Setup

Requires **Python 3.11** and **ffmpeg** on `$PATH`.

```bash
python3.11 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# For precise word-level alignment (optional, preferred):
pip install git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
# NOTE: Do NOT install ctc-forced-aligner from PyPI — it's a different project
```

## Running the App

```bash
source .venv/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://localhost:8000>. On first boot, Whisper and wav2vec2 warm up in the background.

## Running Tests

```bash
pytest tests/                              # all tests
pytest tests/test_api_correct.py           # specific file
pytest tests/test_arabic_correction.py -v  # verbose
```

## Key Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `base` | tiny/base/small/medium/large-v3 |
| `WHISPER_DEVICE` | `cpu` | `cuda` on GPU |
| `WHISPER_COMPUTE_TYPE` | `int8` | `float16` on GPU |
| `WHISPER_LANGUAGE` | `en` | Force language (UI dropdown overrides per-request) |
| `OLLAMA_URL` | `http://100.68.87.28:11434/api/chat` | LLM endpoint |
| `OLLAMA_MODEL` | `hf.co/bartowski/calme-3.2-instruct-78b-GGUF:IQ4_XS` | LLM model tag |
| `USE_LLM` | `1` | Set to `0` to skip both LLM calls (rule-based only) |
| `USE_LLM_CORRECTOR` | `1` | Local 4-bit Qwen2.5-1.5B corrector |
| `USE_API_FALLBACK` | `1` | OpenRouter API fallback when local LLM is low-confidence |
| `VECTOR_BACKEND` | `ngram` | `ngram` (fast) or `transformer` (semantic) |

All pipeline feature flags are defined in [app/services/config.py](app/services/config.py) as a `PipelineConfig` dataclass. The singleton is accessed via `get_config()`.

## Architecture

### Request Flow (`/api/transcribe`)

```
Audio upload
  → faster-whisper → raw text + word timestamps + per-word probability
  → LLM DETECT (one call) → suspicious medical-term spans
  → For each span: slice audio → wav2vec2 embedding → top-K voice index hits
  → LLM DECIDE (batched) → pick best candidate or NO_CHANGE
  → Return raw_text, corrected_text, session_id, words, suspicious
```

### Correction Pipeline Stages (in `correction.py`)

The `MedicalCorrector` applies stages in order, stopping when confidence exceeds the threshold:

1. **LLM Corrector** (`llm_corrector.py`) — local 4-bit Qwen2.5-1.5B via transformers/bitsandbytes
2. **OpenRouter API fallback** — `qwen/qwen-2.5-72b-instruct` when local model is low-confidence
3. **Arabic Spelling Correction** (`arabic_spelling.py`) — fixes Gulf ASR errors in Arabic script
4. **English fuzzy match** — rapidfuzz + jellyfish Jaro-Winkler against the medical lexicon
5. **Arabic→English skeleton matching** — consonant skeleton extraction for transliterations like `دايابيتس` → `diabetes`
6. **Multi-word phrase matching** (`phonetic.py`) — e.g. `بلاد شوجر` → `blood sugar`
7. **Vector lexicon** (`vector_lexicon.py`) — n-gram or transformer-based term retrieval via faiss

### Learning Loop (`/api/learn_from_edit`)

When a user corrects a transcript in the UI, word-level diffs are computed against the cached session audio. For each corrected span: slice audio → wav2vec2 embedding → store in `voice_match.py` index under the canonical term. This allows the same spoken sound to be recognized in future sessions regardless of how Whisper transcribes it.

### Key Service Files

| File | Responsibility |
|---|---|
| `app/main.py` | FastAPI app, all endpoints, session management |
| `app/services/correction.py` | `MedicalCorrector` — orchestrates all correction stages |
| `app/services/flag.py` | Phonetic + LLM suspicious-word detection |
| `app/services/arabic_matcher.py` | `HybridMatcher` — skeleton + embedding + LLM open correction |
| `app/services/voice_match.py` | wav2vec2 fingerprints, persistent NPZ index |
| `app/services/llm_detect.py` | LLM DETECT call (find suspicious spans) |
| `app/services/llm_decide.py` | LLM DECIDE call (pick from candidates) |
| `app/services/asr.py` / `asr_dual.py` | Whisper and Qwen3 ASR wrappers |
| `app/services/lexicon.py` | Loads `data/medical_lexicon.jsonl` |
| `app/services/tracing.py` | Per-request NDJSON event stream for UI live updates |
| `app/services/ngram_lm.py` | Kneser-Ney LM for perplexity scoring |

### Data Files

- `data/medical_lexicon.jsonl` — ~200 medical terms with aliases; edit this to add terms
- `data/descriptions.jsonl` — LLM-generated clinical descriptions (cache); safe to delete to regenerate
- `data/voice_index.npz` / `data/voice_index.jsonl` — wav2vec2 fingerprints from user corrections
- `data/user_corrections.jsonl` — HITL feedback log used for fine-tuning

### Scripts

- `scripts/train_lm.py` — Train the Kneser-Ney LM on `data/lm_training_corpus.txt`
- `scripts/generate_lm_training_data.py` — Build the LM training corpus from the lexicon
- `scripts/finetune_llm.py` — Fine-tune local Qwen2.5 on `data/user_corrections.jsonl`
- `scripts/eval_multi_word_phrases.py` — Evaluate phrase-matching accuracy

## transformers Version Pin

`requirements.txt` pins `transformers==4.57.6` and `huggingface-hub==0.36.2`. **Do not upgrade transformers to 5.x** — it removes `AutoModel` and breaks the `Wav2Vec2ForCTC` imports used in `voice_match.py` and the ASR wrappers.
