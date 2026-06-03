# Transcription Pipeline — Summary (v0.4)

> Current date: June 2026
> Branch: `arabic-pipeline-improvement`

---

## Goal

Build an end-to-end Arabic medical transcription pipeline that:

1. **Transcribes** Gulf-accented Arabic medical speech to text (ASR)
2. **Corrects** the transcript — fixing both:
   - **English medical misspellings** from ASR (e.g. `hyperglacymia` → `hyperglycemia`, `wheezeng` → `wheezing`)
   - **Arabic phonetic misspellings** from ASR (e.g. `سداع` → `صداع`, `التهب` → `التهاب`)
   - **Arabic→English transliterations** used in Gulf clinical speech (e.g. `هستوري` → `history`, `دايابيتس` → `diabetes`, `بلاد شوجر` → `blood sugar`)
3. **Flags** suspicious terms for human review (HITL)
4. **Learns** from user corrections via feedback loop + LoRA fine-tuning
5. **Serves** via a FastAPI web UI + API

---

## Pipeline Architecture (v0.4)

```
Input Transcript
       │
       ▼
┌─────────────────────────────────────┐
│ Phase 1: LLM Corrector (local 4-bit)│  → app/services/llm_corrector.py
│ Qwen2.5-1.5B-Instruct (4-bit)       │
│ If confidence >= threshold → use    │
│ Fallback: OpenRouter API (72B)      │
└─────────────────────────────────────┘
       │ (low confidence or failure)
       ▼
┌─────────────────────────────────────┐
│ Phase 2: Rule-Based Pipeline        │  → app/services/correction.py
│   a. Arabic Spelling Correction     │  → app/services/arabic_spelling.py
│   b. Vector Lexicon (n-gram)        │  → app/services/vector_lexicon.py
│   c. Multi-word Phrase Matcher      │  → app/services/phonetic.py
│   d. Standard Lexicon Fuzzy Score   │
│   e. Flagging for HITL              │  → app/services/flag.py
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Stage A: Suspicion Scoring          │  → app/services/flag.py
│ Multiplicative LLM gating:          │
│   Normalcy ×0.30 + Perplexity ×0.35│
│   + Semantic ×0.20 + Feedback ×0.15│
│   × LLM gate (0.20 to 2.0)         │
└─────────────────────────────────────┘
       │
       ▼
  Corrected Transcript + Flags + HITL
```

---

## New Modules (v0.4)

### `app/services/config.py` — Pipeline Configuration

Central configuration dataclass with environment variable overrides:

| Field | Default | Env Var |
|-------|---------|---------|
| `use_llm_corrector` | `True` | `USE_LLM_CORRECTOR` |
| `llm_model_name` | `Qwen/Qwen2.5-1.5B-Instruct` | `LLM_MODEL_NAME` |
| `llm_confidence_threshold` | `0.85` | `LLM_CONFIDENCE_THRESHOLD` |
| `use_api_fallback` | `True` | `USE_API_FALLBACK` |
| `api_model` | `qwen/qwen-2.5-72b-instruct` | — |
| `vector_lexicon_enabled` | `True` | `VECTOR_LEXICON_ENABLED` |
| `vector_backend` | `ngram` | `VECTOR_BACKEND` |
| `vector_similarity_threshold` | `0.15` | — |
| `fallback_to_rules` | `True` | `FALLBACK_TO_RULES` |
| `feedback_data_path` | `data/user_corrections.jsonl` | — |
| `use_perplexity_scorer` | `True` | — |

Singleton pattern via `get_config()`. `load_config()` creates fresh instances.

### `app/services/llm_corrector.py` — Local LLM Corrector

**Architecture:**
1. Lazy-loads 4-bit Qwen2.5-1.5B-Instruct on first call (~1.2GB VRAM)
2. Sends full transcript to LLM with structured JSON prompt
3. Parses returned `{corrected, corrections[], confidence}` JSON
4. If confidence ≥ threshold → use LLM output directly
5. If local model fails → try OpenRouter API (Qwen 2.5 72B)
6. If both fail → fall through to rule-based pipeline

**Key design decisions:**
- `_parse_json_response()` handles 4 formats: plain JSON, markdown-fenced, leading/trailing text, nested braces
- `SYSTEM_PROMPT` includes explicit Arabic transliteration examples (هستوري→history, etc.)
- Double-checked locking on model loading for thread safety

**Known issue:** Requires `transformers>=4.58.0` for `AutoModelForCausalLM`. Currently pinned to `4.57.6` which lacks this export.

### `app/services/vector_lexicon.py` — Multi-View N-Gram Retrieval

Replaces skeleton matching with FAISS-based approximate nearest-neighbour search:

**Multi-view approach:** Each word is converted to 2-4 character-level views:
1. Normalised text (original script)
2. Consonant skeleton (vowels removed)
3. Arabic transliteration (if Arabic text)
4. Transliteration skeleton

**Example:** `هستوري` → `["هستوري", "hstwry", "hstwr"]` — the skeleton `hstwr` shares n-grams with `history`'s skeleton `hstr`.

**Backends:**
- `ngram`: character 3-gram TF-IDF vectors via FAISS (fast, CPU, deterministic)
- `transformer`: DistilBERT multilingual embeddings (semantic, slower)

**Usage:** `vlex = get_vector_lexicon()` → singleton. `vlex.query("هستوري")` returns `[{term, score, term_type}]`.

### `app/services/llm_scorer.py` — Single-API-Call Suspicion Scoring

Scores ALL words in a transcript in a single LLM API call to respect rate limits:

- **Cache:** Results cached by transcript MD5 for 5 minutes
- **Prompt:** LLM asked to return indices of potentially medical words
- **Usage:** `score_words(words)` → `{index: 1.0}` dict or `None` on failure
- **Integration:** flag.py uses this as a high-weight signal in Stage A scoring

### `app/services/ngram_lm.py` — Kneser-Ney N-Gram Language Model

Pure Python (no external deps) n-gram language model for perplexity scoring:

- Modified Kneser-Ney smoothing with absolute discounting
- Supports orders 1-4 (default 4)
- `word_perplexity(word, context)` → returns `-log10(word|context)`
- `sentence_perplexity(tokens)` → standard perplexity
- Save/load via pickle

Used by flag.py's `_context_perplexity()` to detect contextually anomalous words.

### `app/services/flag.py` — Stage A Suspicion Scoring (Enhanced)

**Multiplicative LLM Gating:**
```
algorithmic = normalcy × 0.30 + perplexity × 0.35 + semantic × 0.20 + feedback × 0.15
fused = algorithmic × gate_factor
gate_factor = 0.20 + (2.0 - 0.20) × llm_suspicion
```

When LLM says "not suspicious" (0.0): dampen algorithmic by 5× (×0.20)
When LLM says "suspicious" (1.0): amplify by 2× (×2.0)

**New signals:**
- **Arabic normalcy** — auto-detection via consonant skeleton matching against the full lexicon
- **Feedback loop** — `_record_correction()` stores skeletons of confirmed corrections, boosting future suspicion
- **LM perplexity** — uses `ngram_lm.py` to detect contextually surprising words
- **Semantic coherence** — detects script-mismatch anomalies (Arabic word in Latin context, etc.)

**Arabic filler set:** ~200+ manually curated Gulf clinical words. No longer used as a gate — the auto-normalcy detection (`_is_arabic_normalcy`) handles Arabic classification via skeleton matching against the lexicon.

### `app/services/correction.py` — LLM-First Pipeline Integration

`MedicalCorrector.correct_transcript()` now runs:
1. **Phase 1: LLM** — calls `llm_corrector.correct_transcript()`, uses output if confidence ≥ threshold
2. **Phase 2: Rule-based** — same deterministic pipeline as before (Arabic spelling → vector lexicon → phrase matching → fuzzy scoring → HITL flagging)

New: `_try_llm_correction()` method and `_try_vector_lexicon()` method for vector lexicon fallback.

### `data/user_corrections.jsonl` — Seed Correction Data

12 seed records covering:
- Arabic transliterations (هستوري→history, دايابيتس→diabetes, بلاد شوجر→blood sugar)
- English misspellings (hyperglacymia→hyperglycemia, wheezeng→wheezing, clopidogr→clopidogrel)
- Multi-word Arabic phrases (شورتنس اوف بريث→shortness of breath)

### `scripts/finetune_llm.py` — LoRA Fine-Tuning from Corrections

Trains Qwen2.5-1.5B-Instruct on correction pairs using:

- **4-bit quantization** (same config as llm_corrector.py)
- **LoRA adapters** on attention + FFN linears (r=16, alpha=32)
- **Proper label masking:** system/user tokens masked to -100 so loss only flows through assistant response
- **HuggingFace Trainer** for distributed training support

Usage: `python -m scripts.finetune_llm --output-dir runs/llm_finetune_r1`

---

## Test Suite (v0.4)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `tests/test_config.py` | 17 | Defaults, env var overrides, singleton behavior |
| `tests/test_vector_lexicon.py` | 19 | Build/query, Arabic transliteration, singleton (skipped if FAISS unavailable) |
| `tests/test_llm_corrector.py` | 24 | JSON parsing, prompt templates, result class, config short-circuit |
| `tests/test_stage_a_signals.py` | 19 | LM perplexity, semantic coherence, feedback loop, Arabic normalcy, fused scoring |
| `tests/test_arabic_correction.py` | 28 | Arabic false-positive preservation, spelling correction, transliteration, multi-word phrases |
| `tests/test_api_correct.py` | 43 | Endpoint response format, English corrections, Arabic transliterations, edge cases |

**Total: 151 passing, 14 skipped (FAISS), 4 pre-existing network failures**

### Test results from live server test (port 8000):

| Test Case | Result | Notes |
|-----------|--------|-------|
| `clopidogr` → `clopidogrel` | ✅ | English misspelling correction |
| `amoxicilin` → `amoxicillin` | ✅ | English misspelling correction |
| `عنده هستوري دايابيتس` | ⚠️ | Arabic transliteration works but some over-correction |
| `شورتنس اوف بريث + بلاد شوجر` | ⚠️ | Multi-word phrases partially corrected |
| Clean English (no change) | ❌ | "resting" → "lying" false positive |
| Arabic context words preserved | ⚠️ | Some normal Arabic words being translated |

---

## Known Issues

1. **LLM corrector won't load** — `AutoModelForCausalLM` not in pinned `transformers==4.57.6`. Need upgrade to 4.58+.
2. **"resting" → "lying" false positive** — deterministic scorer over-matches on certain English words.
3. **Arabic over-correction** — Arabic context words are sometimes translated to English instead of being preserved.
4. **FAISS unavailable on Python 3.14** — `faiss-cpu` has no prebuilt wheel for Python 3.14 yet. Vector lexicon tests gracefully skip.
5. **Auto-corrections not logged** — Some corrections are applied to the text but not reported in the `auto_corrections` array.

---

## Possible Improvements

### High Impact
1. Upgrade `transformers` to fix the LLM corrector import
2. Fix "resting" → "lying" false positive (add to COMMON_GLUE)
3. Tighten Arabic span boundary logic to prevent context word translation

### Medium Impact
4. Build and distribute a FAISS wheel for Python 3.14
5. Train the n-gram LM on a larger Gulf Arabic clinical corpus
6. Add CI pipeline to auto-run tests on commit

### Low Impact
7. Expand `user_corrections.jsonl` to 100+ records
8. Add tensorboard logging to finetune script
9. Web UI polish — show confidence scores per correction

---

*Generated for Claude context — June 2026*
