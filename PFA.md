# Pipeline Functional Architecture (PFA)

## Lexical & Semantic Correction Pipeline

This document describes the post-ASR correction pipeline — from raw transcript text to corrected, clinically meaningful text. Every stage, sub-stage, decision branch, threshold, and fallback is documented atomically.

---

## Overview

```
ASR Raw Text
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE A — SUSPICION SCORING                                        │
│  Assign a suspicion score (0–1) to every token based on how likely  │
│  the ASR got it wrong.                                              │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  STAGE B — CORRECTION                                               │
│  For each suspicious span, find the correct medical term and apply  │
│  it. Multiple sub-strategies run in order (first-match wins).       │
└─────────────────────────────────────────────────────────────────────┘
    │
    ▼
Corrected Text
```

---

# STAGE A — Suspicion Scoring

**Entry point:** `app/services/flag.py:flag_suspicious()` → `phonetic_pass()` → `score_suspicion()`

**Purpose:** Assign a suspicion score (0.0–1.0) to every token in the raw transcript, **independently of the medical lexicon**. The score represents how likely the ASR mis-transcribed that word. Scores flow into the UI as `scored_words[]` and guide Stage B.

**Key architectural principle:** Stage A runs BEFORE any phonetic matching or correction. Only words with `suspicion > 0.0` enter Stage B at all. This inverts the previous architecture where the pipeline relied on correction guards (retrofitted inside Stage B) to prevent false positives.

**Files:** `flag.py` (`score_suspicion()`, `_is_arabic_filler()`, `_is_arabic_normalcy()`, `phonetic_pass()`), `main.py` (`_build_scored_words`, `_build_spans`)

---

## Sub-stage A0 — Suspicion Pre-Filter (score_suspicion)

**File:** `flag.py` → `score_suspicion()`, `_context_perplexity()`, `_semantic_coherence()`, `_normalize_ppl()`, `_COMMON_ENGLISH`, `_load_lm()`

**Input:** A single word token + optional left/right context (2 words each side).

**Output:** A suspicion score `float` (0.0–1.0). Words scoring >= `SUSPICION_THRESHOLD` enter Stage B.

**Purpose:** Before any phonetic matching or lexicon lookup, fuse multiple lexicon-independent signals to determine if a word is likely an ASR error. This is the primary architectural change: the old pipeline fused suspicion and correction into the same step. Now suspicion is a separate, lexicon-independent gate with four fused signals.

### A0.1 — Suspicion Score Constants

| Constant | Value | Meaning |
|---|---|---|
| `SUSPICION_SCORE_NONE` | 0.0 | Not suspicious — skip entirely. No correction attempted. |
| `SUSPICION_THRESHOLD` | 0.10 | Words scoring >= this threshold enter Stage B for correction. |
| `WEIGHT_NORMALCY` | 0.30 | Weight for Arabic normalcy detection signal. |
| `WEIGHT_PERPLEXITY` | 0.35 | Weight for n-gram LM context perplexity signal. |
| `WEIGHT_SEMANTIC` | 0.20 | Weight for semantic coherence (script-mismatch) signal. |
| `WEIGHT_FEEDBACK` | 0.15 | Weight for feedback loop (prior corrections) signal. |
| `LM_PPL_SCALE` | 4.0 | Scaling factor mapping LM perplexity → 0–1 via `1 - exp(-ppl / scale)`. |

### A0.2 — Fused Scoring Logic

The suspicion score is a weighted combination of four independent signals:

```
fused = (normalcy × 0.30) + (perplexity × 0.35) + (semantic × 0.20) + (feedback × 0.15)
```

**Signal 1 — Normalcy (30%):**
- **Hard gate:** Pure digits → return 0.0. Words < 3 chars → return 0.0.
- **Latin words in `_COMMON_ENGLISH`** → return 0.0 (safe common English).
- **Arabic words:** If `_is_arabic_normalcy(word)` returns True → 0.0 (normal Arabic). If False → 0.30 (could be a transliteration).
- **Other Latin words** → 0.05 (minimal baseline — might be misspelled drug).

**Signal 2 — LM Perplexity (35%):**
- Uses a trained 4-gram Kneser-Ney language model (`NGramLM` in `ngram_lm.py`).
- Computes `word_perplexity(word, left_context)` — how surprising is this word given its left context?
- Maps raw PPL to 0–1 via `_normalize_ppl()`: `1 - exp(-ppl / 4.0)`.
- PPL=0 → 0.0 (very expected), PPL=10 → ~0.92 (very surprising).
- Returns 0.0 if no LM is available (e.g., first call before model loads).

**Signal 3 — Semantic Coherence (20%):**
- Detects script-mismatch anomalies between a word and its surrounding context.
- If word is Arabic but most context (2 words each side) is Latin → +0.15.
- If word is Latin but most context is Arabic → +0.10.
- If word contains MIXED Arabic+Latin letters → +0.30 (almost always an ASR error).

**Signal 4 — Feedback Loop (15%):**
- Checks if this word's consonant skeleton was seen in previous high-confidence corrections.
- Returns 0.0–0.15 based on correction count: 1→0.05, 2→0.08, 3→0.10, 4+→up to 0.15.
- Uses partial skeleton matching (exact, first 4 chars, first 3 chars).

### A0.3 — Common English Whitelist (`_COMMON_ENGLISH`)

~300 common English words that appear in clinical dictation but are NOT medical terms. Covers:
- Determiners/pronouns (that, this, these, those)
- Common verbs (been, having, take, given, said, need, feel)
- Adverbs/prepositions (about, after, always, around, before)
- Time words (today, yesterday, week, month, year)
- Patient context (patient, doctor, hospital, clinic, family)
- Clinical context words (reason, result, cause, change, test, report)
- Common adjectives (able, clear, good, normal, possible, mild, severe)
- Written-out numbers (one, two, three, single, double)

Latin words NOT in this set get `SUSPICION_SCORE_LOW` (0.05) and enter phonetic matching — they could be misspelled drugs ("panadol" is not in the whitelist, correctly enters Stage B).

---

## Sub-stage A1 — Tokenization & Pre-filtering

**File:** `correction.py` → `MedicalCorrector._tokenize()`, `_generate_spans()`

**Input:** Raw transcript string.

**Output:** List of `Span` objects (text, start/end char positions, token_start/token_end indices).

**Logic:**
1. Tokenize using `TOKEN_RE` (regex):
   - Latin words: `[A-Za-z][A-Za-z'\-]*`
   - Arabic words: Arabic-range characters + hyphens
   - Numbers: `\d+(?:\.\d+)?`
2. Generate spans from every contiguous subsequence of 1 to `max_span_tokens` (default 6).
3. Pre-filter with `_bad_span_boundary()`:
   - Arabic path: If ALL Arabic words in the span pass `_is_arabic_filler()` (i.e., are normal clinical Arabic), the span is skipped.
   - If span contains sentence-ending punctuation (`.!?;:`), skip.
   - If span has no `WORDISH_RE` content tokens, skip.
   - If span exactly matches a `_known_compacts` entry (precomputed from lexicon aliases), keep it.
   - Single-token spans: skip if < 4 chars or in COMMON_GLUE set.
   - Multi-token spans: skip if first or last word is in COMMON_GLUE set.

**Thresholds/Constants:**
- `max_span_tokens = 6`
- `MIN_SPAN_CHARS = 4`
- `COMMON_GLUE`: ~150 English function words (a, an, and, the, of, etc.)

---

## Sub-stage A2 — Arabic Filler Auto-Detection (Skeleton-Based)

**File:** `flag.py` → `_is_arabic_filler()`, `_is_arabic_normalcy()`, `_ensure_lexicon_skeletons()`

**Purpose:** Classify Arabic-script words as either "normal clinical Arabic" (skip → suspicion = 0) or "potential medical transliteration" (keep for further scoring). This is the gatekeeper that prevents flooding the pipeline with false positives from ordinary Arabic words.

### A2.1 — Manual Filler Whitelist (Backward Compatibility Only)

**File:** `flag.py` → `_ARABIC_FILLER` set

**NOTE:** The `_ARABIC_FILLER` set is **no longer used as a fast path** in `_is_arabic_filler()`. The current implementation delegates entirely to `_is_arabic_normalcy()`, which uses consonant skeleton matching against the full medical lexicon. This auto-detection approach handles the full distribution of Arabic clinical vocabulary without manual maintenance.

The `_ARABIC_FILLER` set is preserved only for backward compatibility — `correction.py` imports it as a vocabulary reference for the Arabic spelling corrector.

**Contents:** ~600 entries covering:
- Particles & prepositions (في, من, على, عن, مع, etc.)
- Greetings & polite forms (السلام, عليكم, مرحبا, شكرا, etc.)
- Common Gulf verbs (بدا, يشتكي, قلت, يقول, etc.)
- Common adjectives (بسيط, خفيف, حاد, etc.)
- Pronouns & demonstratives (هذا, هذه, ذلك, اللي, etc.)
- Conjunctions & adverbs (كذلك, حاليا, سابقا, etc.)
- Anatomical/medical context words (يمتد, لليسار, فحص, etc.)
- Body/symptom/anatomy words (صداع, دوخه, قلب, صدر, كبد, etc.)
- Time words (اليوم, ساعه, اسبوع, شهر, etc.)
- Dosage/form words (مرات, حبه, شراب, بخاخ, etc.)
- Numbers (واحد, اثنين, ثلاثه, etc.)
- Honorifics/roles (دكتور, طبيب, مريض, etc.)
- Common Gulf first names (~80 entries)
- Tribal/family-name particles (ابو, ابن, بنت, etc.)
- And many more categories covering the full clinical dictation vocabulary

### A2.2 — Auto-Detection via Lexicon Skeleton Matching (Slow Path)

**File:** `flag.py` → `_is_arabic_normalcy()`

**Call:** When a word has Arabic script and is NOT in the manual `_ARABIC_FILLER` set.

**Logic:**
1. Check if word contains Arabic letters → if not, return True (pass through, it's Latin).
2. Ensure lexicon skeletons are loaded (lazy, cached).
3. If lexicon is empty → return False (may be transliteration, let it through).
4. Transliterate word to Latin using `_translit(word, strip_clitics=True)`.
5. If transliterated length < 3 → return True (too short to be meaningful).
6. Compute Arabic consonant skeleton via `_consonant_skeleton_ar()`.
7. If skeleton length < 3 → return True (too short).
8. For each Latin skeleton in `_LEXICON_SKELETONS`:
   - Length ratio pre-filter: `len(arabic_sk) / len(latin_sk)` must be 0.35–3.0
   - Fuzzy similarity via `rapidfuzz.ratio(arabic_sk, latin_sk)`
   - If similarity >= 40.0 → return False (COULD be a transliteration)
9. If no match above threshold → return True (normal Arabic, skip it).

**Thresholds:**
- Skeleton similarity threshold: **40%** (intentionally low — more permissive than the phonetic_candidates threshold of 45%)
- Length ratio tolerance: **0.35–3.0**
- Minimum skeleton length: **3 chars**
- Short skeleton length: **3 chars** (triggers length ratio tightening)

### A2.3 — Lexicon Skeleton Cache

**File:** `flag.py` → `_LEXICON_SKELETONS`, `_LEXICON_SKELETONS_LOADED`, `_load_lexicon_terms()`, `_clear_lexicon_skeleton_cache()`

**Logic:**
1. On first call to `_ensure_lexicon_skeletons()`, loads all terms + aliases from `data/medical_lexicon.jsonl`.
2. For each term (minimum length 4), computes Latin consonant skeleton via `_consonant_skeleton_latin()`.
3. Deduplicates skeletons, stores in `_LEXICON_SKELETONS` list.
4. Cache invalidated by `_clear_lexicon_skeleton_cache()` — called from `/api/teach` and `/api/learn_from_edit`.

---

## Sub-stage A3 — Phonetic Pass (Arabic→Latin Skeleton Matching)

**File:** `flag.py` → `phonetic_pass()`, `_phonetic_candidates()`

**Purpose:** For every token, find medical lexicon terms that are phonetically similar via consonant skeleton comparison. This is the primary signal for detecting Arabic medical transliterations.

### A3.1 — Transliteration

**File:** `flag.py` → `_translit()`

**Logic:**
1. Normalize Unicode (NFKC).
2. Strip Arabic tashkeel (diacritics).
3. Optionally strip Arabic clitics (الـ, و, ف, ب, ل, etc. — only when remainder ≥ 3 chars).
4. Map each Arabic character to Latin using `_AR2LAT` table (e.g., ا→a, ب→b, ت→t, ث→th, etc.).
5. Non-Arabic alphanumeric characters pass through as-is.

### A3.2 — Consonant Skeletons

**File:** `flag.py` → `_consonant_skeleton_ar()`, `_consonant_skeleton_latin()`

**Arabic skeleton** (`_consonant_skeleton_ar`):
- Strips vowels (a, e, i, o, u, y, w) — Arabic doesn't write short vowels
- Keeps 'h' as a real consonant (represents ه)
- BUT drops 'h' from digraphs: gh→g, sh→s, th→t, kh→k, dh→d
- Examples: `هستوري` → `hstwry` → `hstr`, `دايابيتس` → `dyabts` → `dyabts`

**Latin skeleton** (`_consonant_skeleton_latin`):
- Strips vowels (a, e, i, o, u, y)
- Maps phonetic classes Arabic loses: p→b, v→f, c→k, g→k, q→k, x→ks
- Examples: `paracetamol` → `brktml`, `history` → `hstr`, `efferalgan` → `ffrlkn`

### A3.3 — Candidate Scoring

**File:** `flag.py` → `_phonetic_candidates()`

**Input:** A single word (Arabic or Latin), the medical lexicon list.

**Output:** Up to `k=3` candidates with phonetic_similarity scores.

**Logic for each lexicon term:**
1. Compute Latin skeleton of the term.
2. Compute transliterations of the input word (2 variants: with clitic stripping, without).
3. For each variant:
   - Raw string: Levenshtein similarity of transliterated variant vs term_lat.
   - Consonant skeleton: Levenshtein similarity of Arabic skeleton vs Latin skeleton.
   - Both checks have length-ratio pre-filter (tolerance 0.5, tightened to 0.65 for short strings ≤6 chars).
4. Take the maximum of all similarity scores.
5. If best < threshold (0.45), discard.

**Ranking:**
1. Similarity DESC
2. Drugs before non-drugs at same score (drug hints: suffixes like -in, -ol, -ide, -ine, -ate, etc., and explicit drug terms)
3. Smaller |len(needle) - len(term)| first
4. Longest common substring length DESC

**Tiebreaker (positional character match):**
- `_longest_common_substring_len()` — counts contiguous shared characters between needle skeleton and term. Strong signal even when overall edit distance is mediocre (e.g., `ليفوثيروكسين` shares `thyr` with levothyroxine, but only `f` with ceftriaxone).

**Phonetic Alias Rescue:**
- If the transliterated span matches a known English-mishearing pattern (e.g., `اف اول قن` → "if all gone" → efferalgan), promote that drug to top with similarity 0.95.
- Also matches via consonant skeleton of the alias.
- Defined in `_PHONETIC_ALIAS` dict **(~100 entries covering 25+ drug families:** Efferalgan, Augmentin, Amoxicillin, Paracetamol, Ciprofloxacin, Atorvastatin, Metformin, Nitroglycerin, Omeprazole, Prednisolone, Clopidogrel, Warfarin, Heparin, Insulin, Tramadol, Voltaren, Azithromycin, Vancomycin, Levofloxacin, Metoprolol, Amlodipine, Lorazepam, Alprazolam, Morphine, Codeine, Diazepam, Fluconazole, Pantoprazole, Misoprostol, Aspirin, Ceftriaxone, Flagyl, Ibuprofen).

### A3.4 — Single-Word Pass

**File:** `flag.py` → `phonetic_pass()`

**Logic:**
1. For each word, call `score_suspicion(word, left_context, right_context)` (Stage A0) passing 2 words of context on each side.
   - If `score < SUSPICION_THRESHOLD` (0.10) → skip entirely (word is common English, Arabic normal, short, digit, or contextually coherent).
   - Otherwise (suspicion >= threshold) → run `_phonetic_candidates()` to find matches.
2. For n-gram pass, `_try_ngram()` uses `_is_arabic_filler()` (not `score_suspicion()`) to count filler words in windows — this is appropriate because the n-gram logic needs to know about filler words for window-merging decisions, not for suspicion scoring.
3. For words not consumed by an n-gram:
   - If top candidate similarity < 0.55, skip.
   - If the literal word IS the same as the top candidate term (case-insensitive), skip.
   - **Precision check:** If similarity is 0.55–0.65 AND longest common substring < 3, skip (kills scattered-letter coincidences).
   - Otherwise, create a flag record with reason `phonetic_near_medical`.

### A3.5 — N-Gram Multi-Word Pass

**File:** `flag.py` → `phonetic_pass()` → `_try_ngram()`

**Purpose:** Drug names sometimes split into 2–3 tokens by the ASR (e.g., `برسي تمر` → paracetamol, `اوغ من تين` → augmentin).

**Order:** 3-grams first, then 2-grams.

**Logic:**
1. Try windows of n consecutive unconsumed words.
2. Skip windows with pure-Latin or digit tokens (avoids mixing scripts).
3. Skip windows containing Arabic conjunction `و` (and) — it separates distinct drugs.
4. Count filler words in window:
   - n=2 and both filler → skip
   - n=3 and ≥2 filler → skip
5. Compute phonetic candidates on the joined string.
6. **Two thresholds:**
   - No filler: `threshold` (0.50 for bigrams, 0.55 for trigrams)
   - Has filler: `filler_threshold` (0.70 for bigrams, 0.75 for trigrams)
7. **Precision check:** If similarity < 0.65 AND LCS < 3 → skip.
8. **Single-blocking check:** If a component word has a near-perfect (≥0.80) single-word match of similar length, prefer the single word over the n-gram — UNLESS the joined length is >1.7× the single length (strong signal of a split drug).

**Thresholds:**
- Bigram threshold: **0.50**, filler threshold: **0.70**
- Trigram threshold: **0.55**, filler threshold: **0.75**
- Single-blocking similarity: **0.80**
- Single-blocking length ratio: **0.65**
- Joined > 1.7× single-length override

---

## Sub-stage A4 — LLM Detection (Optional)

**File:** `flag.py` → `llm_pass()`, `llm_detect.py` → `detect()`

**Purpose:** Supplementary flagging using an LLM to catch novel misspellings or rare terms the phonetic pass missed.

**When triggered:** Controlled by `USE_LLM` env var (default: True).

**Logic (`llm_pass`):**
1. Sends the full transcript + token enumeration to an LLM.
2. LLM returns a JSON list of flagged spans with indices, reasons, likely terms, and confidence scores.
3. Results are merged with phonetic flags: if existing phonetic flag exists, LLM info is attached as auxiliary fields (`llm_reason`, `llm_likely_term`, `llm_confidence`).
4. If no phonetic flag exists, a new flag is created with both phonetic candidates AND LLM data.

**Logic (`llm_detect.detect` — used in transcribe pipeline):**
1. Takes word-level tokens with timestamps.
2. Sends to LLM with system prompt requesting span-level mishearing detection.
3. Parses JSON response, validates indices against token array.
4. Merges adjacent/overlapping spans into single acoustic units.
5. Includes retry logic (4 attempts with exponential backoff).

**LLM System Prompt guidance:**
- "Be biased toward flagging — better to over-flag a weird word than miss a real drug."
- Arabic-script spans flagged by LLM are **filtered out** in `/api/correct` to avoid hallucinated English medical terms for normal Arabic words.

---

## Sub-stage A3 — N-Gram Language Model (LM Perplexity Signal)

**File:** `ngram_lm.py` → `NGramLM`, `flag.py` → `_load_lm()`, `_context_perplexity()`, `_normalize_ppl()`

**Purpose:** Provide a lexicon-independent suspicion signal based on n-gram context probability. A word that is surprising given its left context (e.g., "chest elephant" instead of "chest pain") is likely an ASR error.

### A3.1 — Model Architecture

**File:** `ngram_lm.py` → `NGramLM`

- **Order:** 4-gram (looks at up to 3 words of left context).
- **Smoothing:** Modified Kneser-Ney discounting (discount=0.75), the state of the art for n-gram LMs.
- **Pure Python:** Zero external dependencies — uses only built-in `collections.Counter`. Trained on a corpus of clean medical text.
- **Training data sources:** Extracted from `eval/medical_transcript_eval.jsonl` (20 clean English medical transcripts), lexicon entries with template sentences, and synthetic Arabic-English code-switched examples.
- **Trained model saved to:** `app/services/medical_lm.pkl` (~656 KB).

### A3.2 — Word Perplexity

**File:** `ngram_lm.py` → `NGramLM.word_perplexity()`

Computes `-log P(word | context)` where:
- If context matches a known n-gram (order 1–4), uses the discounted probability.
- If context is unseen, backs off to lower-order n-grams (stupid backoff).
- If the word is entirely unknown (OOV), returns a default perplexity of `-log(1e-6)` ≈ 13.8.

### A3.3 — Perplexity → Suspicion Score

**File:** `flag.py` → `_normalize_ppl()`

```python
def _normalize_ppl(ppl: float) -> float:
    return 1.0 - math.exp(-ppl / LM_PPL_SCALE)  # LM_PPL_SCALE = 4.0
```

Maps PPL to 0–1:
- PPL = 0.0 (very expected) → score 0.0
- PPL = 2.0 (mildly surprising) → score ~0.39
- PPL = 5.0 (very surprising) → score ~0.71
- PPL = 10.0 (extremely surprising) → score ~0.92

### A3.4 — Context Window

Called from `phonetic_pass()` with 2 words of left context for each token. If no LM is available (not yet loaded), returns 0.0 gracefully.

---

## Sub-stage A4 — Semantic Coherence Signal

**File:** `flag.py` → `_semantic_coherence()`

**Purpose:** Detect script-mismatch anomalies: an English word surrounded by Arabic (or vice versa) is more suspicious because the ASR may have produced a wrong word that happens to be in the wrong script. Mixed-script tokens (Arabic+Latin letters in one word) are almost always ASR errors.

### A4.1 — Signals

1. **Script mismatch with context (+0.15):** If word is Arabic but most of its context (2 words each side) is Latin → suspicious.
2. **Inverse script mismatch (+0.10):** If word is Latin but most context is Arabic → mildly suspicious.
3. **Mixed-script tokens (+0.30):** If word contains both Arabic and Latin letters → highly suspicious (almost certainly an ASR hallucination).

---

## Sub-stage A5 — Feedback Loop

**File:** `flag.py` → `_CORRECTION_FEEDBACK`, `_record_correction()`, `_get_feedback_boost()`

**Purpose:** Record high-confidence corrections so Stage A learns which Arabic transliteration skeletons are genuinely medical. After Stage B corrects a word with high confidence (phonetic similarity >= 0.85 or LLM confidence >= 0.90 with lexicon validation), the original word's consonant skeleton is recorded. In future transcripts, Stage A gives a small suspicion boost to words with similar skeletons, making the pipeline self-improving over time.

### A5.1 — Recording

**File:** `flag.py` → `_record_correction(original_word, corrected_term)`

Called from `apply_high_confidence_corrections()` after each successful high-confidence correction.

1. Detects script type (Arabic/Latin) from the original word.
2. Computes consonant skeleton:
   - Arabic: `_translit()` → `_consonant_skeleton_ar()`
   - Latin: `_consonant_skeleton_latin()`
3. Stores count in `_CORRECTION_FEEDBACK[(skeleton, tag)]`.
4. Also stores prefix-generalized key `(skeleton[:4], tag)` so similar patterns reinforce each other.

### A5.2 — Retrieval

**File:** `flag.py` → `_get_feedback_boost(word)`

Called from `score_suspicion()` as Signal 4 (weight 15%).

1. Computes consonant skeleton of the input word (same algorithm as recording).
2. Checks exact key match → count.
3. Falls back to first 4 chars prefix match → count.
4. Falls back to first 3 chars prefix match → count.
5. Maps count to boost:
   - 1 correction → 0.05
   - 2 corrections → 0.08
   - 3 corrections → 0.10
   - 4+ corrections → `min(0.15, 0.10 + 0.02 × (count - 3))`

### A5.3 — Persistence

The `_CORRECTION_FEEDBACK` dict is in-memory (module-level global). Corrections accumulate within a server session and are lost on restart. This is acceptable — the feedback loop primarily helps within a single dictation session where the same speaker's patterns repeat.

---

## Sub-stage A7 — Suspicion Score Assembly

**File:** `main.py` → `_build_scored_words()`, `_build_spans()`, `_build_candidates_list()`, `_build_decisions()`

**Purpose:** Convert the raw flags into structured arrays for the UI, computing per-token suspicion scores.

### A7.1 — Scored Words

**File:** `main.py` → `_build_scored_words()`

**Logic:**
1. Tokenize transcript on whitespace.
2. For each token, look up if it has a flag.
3. If flagged:
   - If flag has candidates → `suspicion = min(1.0, candidates[0].phonetic_similarity)` (typically 0.6–1.0)
   - If flag has no candidates → `suspicion = 0.7` (LLM-only flag with no phonetic match)
4. If not flagged → `suspicion = 0.0`
5. `in_lexicon = suspicion < 0.3 AND not flag`

**Output format:**
```json
{"text": "هستوري", "index": 4, "suspicion": 0.83, "in_lexicon": false}
```

### A7.2 — Flags Array

**File:** `main.py` → structured from `result["suspicious_spans"]`

**Each flag contains:**
- `index`: token index (approximate for text-only path)
- `word`: the original word/phrase
- `reason`: issue type (phonetic_near_medical, phonetic_near_medical_2gram, etc.)
- `candidates`: up to 3 candidate corrections with `term`, `phonetic_similarity`, `match_type`
- `start_s`/`end_s`: None for text-only, audio timestamps for debug pipeline

---

## Sub-stage A6 — Legacy: Whisper-Confidence Suspicion (suspect.py)

**File:** `suspect.py` → `detect()`

**Purpose:** Alternative suspicion method used in the audio-pipeline path (not text-only). Based on Whisper per-word log-probabilities.

**Logic:**
1. Check each word against:
   - Common English whitelist (~250 words)
   - Known lexicon terms (exact match)
   - Unknown term set
2. Score similarity to known medical terms via fuzzy ratio + metaphone.
3. Three signals:
   - **Low confidence** (prob < 0.60): flag unconditionally
   - **Mid confidence** (prob 0.60–0.85): flag only if looks medical (fuzzy ≥ 78 OR phonetic ≥ 82)
   - **High confidence** (prob ≥ 0.85): flag only if looks medical
4. Minimum word length: 4 chars.
5. Adjacent suspicious words merged via `merge_adjacent()` (gap ≤ 0.10s).

---

# STAGE B — Correction

**Entry point:** `app/main.py:correct_text_only()` → `app/services/correction.py:MedicalCorrector.correct_transcript()`

**Purpose:** For each suspicious span, determine the correct medical term and apply it. Multiple sub-strategies are tried in order; non-overlapping selections are applied.

**Files:** `correction.py`, `arabic_matcher.py`, `arabic_spelling.py`, `phonetic.py`, `main.py` (hybrid matcher, LLM open correction, LLM reranking)

---

## Sub-stage B0 — Span Generation (MedicalCorrector)

**File:** `correction.py` → `MedicalCorrector._generate_spans()`

**Logic:** Same as A1 — generates all possible spans of 1 to `max_span_tokens` tokens, pre-filtered.

**MedicalCorrector constructor parameters:**
- `max_span_tokens`: 6
- `accept_threshold`: **88.0** (from `_build_corrector()` in main.py, was 80.0)
- `single_word_phonetic_floor`: **92.0**
- `single_word_score_floor`: **80.0**
- Lexicon loaded from `data/medical_lexicon.jsonl`
- Short aliases (compact ≤ 3 chars) are filtered out at build time

---

## Sub-stage B1 — Arabic Spelling Correction

**File:** `arabic_spelling.py` → `correct_arabic_spelling()`

**When triggered:** Inside `MedicalCorrector._best_candidate_for_span()` for Arabic-script spans where the non-filler Arabic word count is exactly 1.

**Purpose:** Fix common Arabic→Arabic misspellings BEFORE the pipeline falls through to English transliteration matching. Prevents false positives like `سداع` coincidentally matching English medical terms.

**Logic:**
1. Check if word (with clitic prefix) is already in the vocabulary (filler set).
2. If yes → return None (already correct).
3. If word < 3 chars → return None.
4. **Explicit misspelling map check** (`_ARABIC_MISSPELLING`):
   - `مرد` → `مريض` (missing ي + د→ض)
   - `هسري` → `هستوري` (missing ت)
   - `تهب` → `تهاب` (aspirated b→hb)
   - Returns corrected word with confidence 0.85.
5. **Single-substitution pass** (via `_generate_single_substitutions()`):
   - For each character position, try alternative letters from the phonetic merger map (`_ARABIC_MERGER`).
   - The merger map defines which letters are commonly substituted in Gulf Arabic:
     - س ↔ ص ↔ ث (sibilants)
     - ط ↔ ت (emphatic/non-emphatic)
     - د ↔ ض ↔ ظ ↔ ذ ↔ ز (dental mergers)
     - ع ↔ أ ↔ إ ↔ ء ↔ ا (gutturals)
     - غ ↔ ق ↔ ك (back consonants)
     - ي ↔ ى (yaa/alif maqsura)
     - ة ↔ ه (ta marbuta/ha)
     - etc.
6. Check each variant against the vocabulary (with clitic handling).
7. Score = 1.0 - (n_changes × 0.5) / max_len.
8. Return best variant if score ≥ 0.60.

**Thresholds:**
- Explicit map confidence: **0.85**
- Single substitution minimum score: **0.60**
- Minimum word length: **3 chars**

---

## Sub-stage B2 — Arabic Multi-Word Phrase Matching

**File:** `correction.py` → in `_best_candidate_for_span()`, `phonetic.py` → `match_multi_word_arabic()`

**When triggered:** For Arabic-only spans (no Latin/number tokens) in `_best_candidate_for_span()`.

**Purpose:** Detect multi-word Arabic transliterations of English medical phrases (e.g., `بلاد شوجر` → blood sugar, `شورتنس اوف بريث` → shortness of breath).

**Logic:**
1. Extract raw Arabic tokens from the span.
2. Build two token lists:
   - **Content tokens:** non-filler Arabic words (transliterated, no clitic stripping), requiring length ≥ 2
   - **All tokens:** every Arabic word (including fillers like اوف → of)
3. **Strategy A — Content-only:** If content tokens ≥ 70% of total Arabic tokens, try matching content tokens.
4. **Strategy B — All tokens:** If Strategy A fails, try matching all tokens (may include filler words).
5. For each strategy, call `match_multi_word_arabic()` with sliding windows of length 2–4.
6. Each window is compared against `_MULTI_WORD_PHRASES` list (~50 entries covering vital signs, symptoms, pain descriptions, therapies, heart-related terms, oxygen, ischemic changes).
7. **Exact match:** If window matches a phrase transliteration exactly → score 100.0.
8. **Fuzzy match:** If window_size matches the phrase word count, compute skeleton similarity. If ≥ 80.0 → accept with that score.
9. Dedup by start position, keep highest score.
10. If top match score ≥ `accept_threshold - 5.0` (83.0), apply as candidate.

**Narrowing:** If the multi-word match covers only a portion of the span's Arabic tokens, `_narrow_span_to_matched_arabic()` creates a sub-span that covers only the matched tokens in the original transcript. This prevents wide conversational spans from being entirely replaced by a sub-phrase match.

**Thresholds:**
- Content ratio threshold: **70%**
- Fuzzy skeleton threshold: **80.0**
- Multi-word acceptance threshold: `accept_threshold - 5.0` = **83.0**
- Multi-word dominate margin: **20.0** (in span selection)

---

## Sub-stage B3 — English Fuzzy + Phonetic Scoring (MedicalCorrector)

**File:** `correction.py` → `_score_pair()`, `_best_candidate_for_span()`

**Purpose:** Score a suspicious span against all lexicon variants using multiple complementary signals.

### B3.1 — Scoring Functions

**File:** `correction.py` → `_score_pair()` (English→English)

**Signals (combined into final score):**

1. **Fuzzy match** — `fuzz.token_set_ratio()` or `fuzz.token_sort_ratio()` on normalized text.
   - If word counts differ: `token_sort_ratio`
   - If word counts match: `token_set_ratio`

2. **Compact match** — `fuzz.ratio()` on compact form (letters only, no separators, no "and").

3. **Partial alignment** — If span's compact is ≥ variant's compact AND variant's compact ≥ 5:
   - `fuzz.partial_ratio()` with length-ratio weighting.
   - Only fires when partial ≥ 90 AND length ratio ≥ 0.55.

4. **Glueless compact** — Same as compact but also drops tiny filler words (a, an, the, of, etc.).
   - Used when compact ≠ glueless (i.e., filler words were in the span).

5. **Phonetic match** — Metaphone of normalized text, compared via `fuzz.ratio()`.
   - Also includes **glueless metaphone** (metaphone of the glueless compact form) as the strongest signal for split-by-filler ASR errors.

**Combined formula:**
```
combined = max(
    0.50 × fuzzy + 0.20 × compact + 0.30 × phonetic,
    0.92 × compact,
    0.85 × phonetic,
)
```

**Length-mismatch penalty:**
- If `len(s_compact) / len(v_compact) < 0.85`:
  - penalty = (0.85 - ratio)² × 200
  - combined = max(0, combined - penalty)

### B3.2 — Arabic→English Scoring

**File:** `correction.py` → `_arabic_score_pair()`

**Signals:**
1. **Skeleton score** — `fuzz.ratio()` on Arabic skeleton (from `_consonant_skeleton_ar()`) vs Latin skeleton (from `_consonant_skeleton_latin()`). Primary signal.
2. **Fuzzy score** — `fuzz.token_sort_ratio()` on normalized texts. Secondary signal, weighted at 0.85×.

**Combined:** `max(skeleton_score, fuzzy_score × 0.85)`

**Short-skeleton guard:** If either skeleton < 4 chars:
- Cap combined at 70.0 (prevents short coincidental matches like `بريث` skeleton `brt` matching `pariet` at skeleton `brt` at 100%)
- Increase fuzzy weight: `combined = max(combined, fuzzy × 0.50)`

**Length-mismatch penalty:**
- If `len_ratio < 0.60`: penalty = (0.60 - ratio)² × 200
- If `len_ratio > 1.50`: penalty = (excess)² × 100

### B3.3 — Selection & Thresholding

**File:** `correction.py` → `_best_candidate_for_span()`

**Logic:**
1. For each variant of each lexicon entry, compute score via `_score_pair()` (English) or `_arabic_score_pair()` (Arabic).
2. Keep the best-scoring candidate.
3. If no candidate → return None.
4. **Arabic path:** Use `accept_threshold` (88.0) directly — no phonetic relaxation.
5. **English path:**
   - Default threshold = `accept_threshold` (88.0)
   - **Relaxation:** If phonetic ≥ 92.0 AND score ≥ 80.0 AND has_char_evidence (fuzzy ≥ 70 OR compact ≥ 70) AND good_coverage (span length ≥ max(5, 0.85 × variant_length)):
     - Lower threshold to `single_word_score_floor` (80.0)
6. If score < threshold → return None.
7. **Arabic guard:** If correction equals the transliterated scoring text → return None (no-op).
8. **English guard:** If correction equals the span text → return None (no-op).
9. **Substring guard:** If correction is a proper substring of span text AND fuzzy < 92 → return None.
10. Assign issue type:
    - Multi-word span → single-word correction: `split_phrase_should_merge`
    - Case-only difference: `capitalization`
    - Phonetic ≥ 90 but fuzzy < 90: `sound_alike`
    - Otherwise: `single_word_misspelling` or `wrong_term`

---

## Sub-stage B4 — Hybrid Matcher (Supplementary)

**File:** `arabic_matcher.py` → `SkeletonMatcher`, `HybridMatcher`

**When triggered:** In `main.py:correct_text_only()` after Stage 1 (MedicalCorrector), for Arabic-script words that weren't corrected.

**Purpose:** Catch Arabic transliterations that the MedicalCorrector missed (e.g., terms in the lexicon but not matched by its scoring pipeline).

### B4.1 — Skeleton Matcher

**File:** `arabic_matcher.py` → `SkeletonMatcher.match()`

**Logic:**
1. Detect Arabic script — if none, return empty.
2. Pre-filter via `_is_arabic_filler()` — skip filler words entirely.
3. Transliterate via `_translit(span_text, strip_clitics=True)`.
4. If transliteration < 3 chars, return empty.
5. Compute Arabic consonant skeleton (`_arabic_skeleton()`).
6. If skeleton < 2 chars, return empty.
7. For each (latin_form, latin_skeleton, term) in the flat variant index:
   - Compute `fuzz.ratio()` on transliteration vs latin_form (raw signal).
   - Compute `fuzz.ratio()` on arabic_skeleton vs latin_skeleton (skeleton signal).
   - Combined = max(raw, skeleton × 1.15).
   - **Length-mismatch penalty:** If `len(arabic_sk) / len(latin_sk) < 0.75`:
     - penalty = (0.75 - ratio)² × 1500
   - If `len(arabic_sk) / len(latin_sk) > 2.0`:
     - penalty = (excess)² × 100
   - **Short-skeleton floor:** If arabic_sk ≤ 3 AND combined ≥ 85.0 AND raw_score < 55.0:
     - Cap combined at 84.9 (prevents 'str' matching unrelated terms like 'statin')
8. Accept threshold: **85.0**
9. Return up to `top_k=5` candidates.

### B4.2 — Embedding Matcher (DISABLED)

**File:** `arabic_matcher.py` → `EmbeddingMatcher`

**Status:** DISABLED in `_get_hybrid_matcher()` (main.py). Reason: LaBSE cross-lingual matching matches Arabic semantic equivalents (قلب → cardiac) rather than transliterations (هستوري → history), which is the wrong signal.

---

## Sub-stage B5 — LLM Open Correction (Optional)

**File:** `arabic_matcher.py` → `LLMOpenCorrector`

**When triggered:** In `main.py:correct_text_only()` for remaining uncorrected English words that weren't flagged by any earlier stage. Arabic-script words are **explicitly excluded** to prevent LLM hallucination of English medical terms for normal Arabic words.

**Logic:**
1. Collect words from the transcript that weren't changed AND weren't already flagged AND have no Arabic script AND length ≥ 4.
2. Send batch to LLM via `correct_batch()`.
3. LLM system prompt instructs constrained JSON output: `{correction: "term" | "UNSURE", confidence: 0-1, reason: "..."}`
4. If LLM returns UNSURE or confidence < 0.60 → skip.
5. If correction score ≥ 80.0 → auto-apply to corrected_text.

**Thresholds:**
- LLM confidence threshold: **0.60**
- Auto-apply score threshold: **80.0**

---

## Sub-stage B6 — LLM Reranking (Optional, Audio Pipeline)

**File:** `llm_rerank.py` → `rerank()`, `llm_decide.py` → `decide()`

**When triggered:** In the audio transcribe pipeline (`/api/transcribe`), NOT in the text-only path.

**Purpose:** For each suspicious span with audio timestamps, choose the best candidate based on clinical context.

**Logic:**
1. For each span, collect candidates from voice matching (user fingerprints + seed fingerprints).
2. Build a batched LLM request with span, candidates (each with similarity + description).
3. LLM returns ONE choice per span or NO_CHANGE.
4. Strong user-voice matches (similarity ≥ 0.85) auto-choose without LLM call.
5. Apply word-level replacements: first word replaced, subsequent words in span dropped.

**Thresholds:**
- Voice auto-fix threshold: **0.85** (short-circuit for strong user match)
- User voice threshold: **0.55**
- Seed voice threshold: **0.45**

---

## Sub-stage B7 — Span Selection (Non-Overlapping)

**File:** `correction.py` → `MedicalCorrector._select_non_overlapping()`

**Purpose:** Given all candidate corrections (across all spans of all lengths), pick the optimal non-overlapping set.

**Logic:**
1. Sort candidates by: `(-adjusted_score, -length, token_start)`.
   - Multi-word phrase matches get +10 score boost for sorting.
2. For each candidate (in sorted order):
   - If its tokens overlap with any already-selected candidate → skip.
   - **Domination check:** If a longer candidate contains this one's span AND scores within `dominate_margin` of it → the longer candidate takes priority.
     - Base margin: **6.0**
     - Phrase match margin: **20.0**
3. Return selected candidates sorted by their original position in the text.

---

## Sub-stage B8 — Correction Application

**File:** `correction.py` → `MedicalCorrector._apply_corrections()`

**Logic:**
1. Iterate over selected candidates sorted by start position.
2. Concatenate: text_before_span + correction + text_after_span.
3. Preserve all text outside the corrected spans.

**For auto-corrections (high-confidence path in main.py):**
- `apply_high_confidence_corrections()` in flag.py:
  - If top phonetic candidate has similarity ≥ 0.85 → auto-apply.
  - If phonetic weak but LLM confidence ≥ 0.90 AND term exists in lexicon → auto-apply.
- Multi-word spans: replace first token, blank subsequent tokens and their whitespace.

**Formatting:**
- Runs of empty tokens collapsed: `re.sub(r"\s+", " ", out).strip()`

---

# Supporting Infrastructure

## Medical Lexicon

**Files:** `data/medical_lexicon.jsonl`, `medical_terms.txt`

**Format:** JSONL with entries:
```json
{"term": "paracetamol", "type": "drug", "aliases": ["acetaminophen", "panadol"], "priority": 1.0}
```

**Management:** `lexicon.py` — `list_terms()`, `add_term()`, `_rewrite()`.

**Cache invalidation:** When `/api/teach` or `/api/learn_from_edit` adds a term:
1. `_TEXT_CORRECTOR = None` (rebuilt on next call).
2. `_HYBRID_MATCHER = None` (rebuilt on next call).
3. `_clear_lexicon_skeleton_cache()` (flag.py skeleton cache cleared).

## API Endpoints Summary

| Endpoint | Method | Purpose | Stages Used |
|---|---|---|---|
| `/api/correct` | POST | Text-only correction | A1–A5, B0–B8 |
| `/api/transcribe` | POST | Audio→corrected text | Full pipeline incl. ASR, voice matching |
| `/api/transcribe_debug` | POST | Debug: ASR + alignment + flags | A1–A5 + forced alignment |
| `/api/transcribe_stream` | POST | NDJSON-streamed pipeline | Same as transcribe with tracing |
| `/api/teach` | POST | Add term to lexicon | Cache invalidation |
| `/api/learn_from_edit` | POST | Learn from user correction | Cache invalidation + voice registration |

## Key Thresholds Reference Table

| Threshold | Value | File | Stage |
|---|---|---|---|
| SUSPICION_THRESHOLD | 0.10 | flag.py | A0 |
| WEIGHT_NORMALCY | 0.30 | flag.py | A0 |
| WEIGHT_PERPLEXITY | 0.35 | flag.py | A0 |
| WEIGHT_SEMANTIC | 0.20 | flag.py | A0 |
| WEIGHT_FEEDBACK | 0.15 | flag.py | A0 |
| LM_PPL_SCALE | 4.0 | flag.py | A3.3 |
| Phonetic candidate minimum | 0.45 | flag.py | A3.3 |
| Single-word flagging | 0.55 | flag.py | A3.4 |
| Bigram threshold | 0.50 | flag.py | A3.5 |
| Bigram with filler | 0.70 | flag.py | A3.5 |
| Trigram threshold | 0.55 | flag.py | A3.5 |
| Trigram with filler | 0.75 | flag.py | A3.5 |
| Phonetic auto-accept | 0.85 | flag.py | B8 |
| Skeleton normalcy threshold | 40% | flag.py | A2.2 |
| Skeleton matcher accept | 85.0 | arabic_matcher.py | B4.1 |
| Accept threshold (main) | 88.0 | main.py | B3.3 |
| Single-word phonetic floor | 92.0 | main.py | B3.3 |
| Single-word score floor | 80.0 | main.py | B3.3 |
| Arabic spelling threshold | 65.0 | correction.py | B1 |
| LLM confidence threshold | 0.60 | arabic_matcher.py | B5 |
| LLM auto-apply score | 80.0 | main.py | B5 |
| Voice auto-fix | 0.85 | main.py | B6 |
| Span selection margin | 6.0 | correction.py | B7 |
| Phrase match margin | 20.0 | correction.py | B7 |
| Arabic spelling min confidence | 0.60 | arabic_spelling.py | B1 |
| Multi-word phrase accept offset | -5.0 | correction.py | B2 |
| Multi-word phrase skeleton threshold | 80.0 | phonetic.py | B2 |
| Feedback boost (1 correction) | 0.05 | flag.py | A5.2 |
| Feedback boost (2 corrections) | 0.08 | flag.py | A5.2 |
| Feedback boost (3 corrections) | 0.10 | flag.py | A5.2 |
| Feedback boost (4+ corrections) | 0.15 | flag.py | A5.2 |
