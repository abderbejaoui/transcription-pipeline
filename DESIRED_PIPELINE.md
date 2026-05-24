# DESIRED_PIPELINE.md
# Medical Transcript Auto-Correction — Full Pipeline Specification

> **Purpose of this file:** Define every stage of the pipeline with
> precision: what model is used, what goes in, what must come out, and
> what the correct answer looks like on the canonical test transcript.
> Copilot must produce output that matches the expected results shown here.
> If it does not, the implementation is wrong — not the expected results.

---

## Canonical Test Transcript

This is the single reference input used throughout this document.
Every stage shows exactly what should happen to this string.

```
"The patient presents with fever and should take dolly prahn twice daily
alongside salbu tamol for the wheeze. Blood pressure was measured
using a sfigmomanometre. The attending physician prescribed
amoxicilin for the secondary infection."
```

### Known errors in this transcript

| Wrong form | Correct term | Type | Why it is wrong |
|---|---|---|---|
| `dolly prahn` | Doliprane | drug brand | Phonetic mishearing of a French drug name |
| `salbu tamol` | Salbutamol | drug generic | Whisper split one word into two |
| `sfigmomanometre` | sphygmomanometer | medical device | Spelling corruption of a long technical word |
| `amoxicilin` | amoxicillin | drug generic | Single-letter spelling error (one `l` missing) |

All other words in the transcript are correct and must NOT be changed.

---

## Stage 0 — Pre-processing

### What it does
Tokenise the input string into a flat list of words with their positions.
Strip punctuation from each token for matching purposes but preserve it
for reconstruction. Record the original position of each word so the
corrected transcript can be rebuilt exactly.

### Model / tool used
No model. Pure Python string operations.
Use `str.split()` for tokenisation.
Preserve punctuation attached to words separately so "fever." becomes
`{text: "fever", punct: "."}` — not `"fever."`.

### Input
```
"The patient presents with fever and should take dolly prahn twice daily
alongside salbu tamol for the wheeze. Blood pressure was measured
using a sfigmomanometre. The attending physician prescribed
amoxicilin for the secondary infection."
```

### Output
A list of token objects. Each token:

```python
@dataclass
class Token:
    index: int       # position in the word list, 0-based
    text: str        # the word, lowercased, no punctuation
    original: str    # exactly as it appeared in the input
    punct: str       # any punctuation that followed (". , ?" etc.) or ""
```

**Expected output for the test transcript (first 12 tokens shown):**

| index | text | original | punct |
|---|---|---|---|
| 0 | the | The | |
| 1 | patient | patient | |
| 2 | presents | presents | |
| 3 | with | with | |
| 4 | fever | fever | |
| 5 | and | and | |
| 6 | should | should | |
| 7 | take | take | |
| 8 | dolly | dolly | |
| 9 | prahn | prahn | |
| 10 | twice | twice | |
| 11 | daily | daily | |
| 12 | alongside | alongside | |
| 13 | salbu | salbu | |
| 14 | tamol | tamol | |
| 15 | for | for | |
| 16 | the | the | |
| 17 | wheeze | wheeze | . |
| 18 | blood | Blood | |
| 19 | pressure | pressure | |
| 20 | was | was | |
| 21 | measured | measured | |
| 22 | using | using | |
| 23 | a | a | |
| 24 | sfigmomanometre | sfigmomanometre | . |
| 25 | the | The | |
| 26 | attending | attending | |
| 27 | physician | physician | |
| 28 | prescribed | prescribed | |
| 29 | amoxicilin | amoxicilin | |
| 30 | for | for | |
| 31 | the | the | |
| 32 | secondary | secondary | |
| 33 | infection | infection | . |

---

## Stage 1 — SCORE

### What it does
Assign a suspicion score (0.0–1.0) to every content word.
A high score means "this word is likely a speech-to-text error."
Stop words (the, and, a, is, for, with, of, to, in, by, an) always
receive score 0.0 and are skipped entirely.

### Model / tool used
**`facebook/bart-large`** loaded from local path `D:/HF_CACHE/models/facebook/bart-large`.

BART is a seq2seq model pre-trained on masked token reconstruction.
Use it in **masked scoring mode**:

1. For each content word at position `i`, build a masked version of
   the sentence where token `i` is replaced with `<mask>`.
2. Ask BART to predict the probability of the original word filling
   that mask given the surrounding context.
3. `suspicion = 1.0 - p(original_word | context)`
   A word BART finds very unlikely in context = high suspicion.

**Fallback if BART is unavailable or slow:**
Use a simple heuristic: if the word is not found in any standard English
dictionary AND not found in `data/medical_lexicon.jsonl`, suspicion = 0.85.
If found in the lexicon, suspicion = 0.10. If found in the English
dictionary, suspicion = 0.05.
The heuristic is labelled `source: "heuristic"` in the output so it
is clear which path was taken.

### Input
The flat token list from Stage 0.

### Output
The same token list, each token now has a `suspicion` score and an
`in_lexicon` boolean.

```python
@dataclass
class ScoredWord:
    index: int
    text: str
    original: str
    punct: str
    suspicion: float    # 0.0 = definitely correct, 1.0 = definitely wrong
    in_lexicon: bool    # True if text (lowercased) is in medical_lexicon.jsonl
```

**Expected output for the test transcript (content words only, stop words omitted):**

| index | text | suspicion | in_lexicon | note |
|---|---|---|---|---|
| 1 | patient | 0.05 | false | common English word, fits context |
| 2 | presents | 0.08 | false | common medical usage |
| 4 | fever | 0.06 | true | in lexicon as symptom |
| 6 | should | 0.03 | false | stop-word-like, skip |
| 7 | take | 0.04 | false | fits context |
| 8 | **dolly** | **0.87** | **false** | not English, not in lexicon |
| 9 | **prahn** | **0.92** | **false** | not English, not in lexicon |
| 10 | twice | 0.04 | false | fits context perfectly |
| 11 | daily | 0.04 | false | fits context perfectly |
| 13 | **salbu** | **0.84** | **false** | not English, not in lexicon |
| 14 | **tamol** | **0.81** | **false** | not English, not in lexicon |
| 17 | wheeze | 0.09 | true | in lexicon as symptom |
| 19 | pressure | 0.05 | false | common English |
| 21 | measured | 0.04 | false | fits context |
| 24 | **sfigmomanometre** | **0.96** | **false** | not English, not in lexicon |
| 26 | attending | 0.04 | false | fits context |
| 27 | physician | 0.06 | false | common medical word |
| 28 | prescribed | 0.05 | false | fits context |
| 29 | **amoxicilin** | **0.71** | **false** | close to English but misspelled |
| 32 | secondary | 0.04 | false | fits context |
| 33 | infection | 0.07 | true | in lexicon |

Words that must NOT be flagged (suspicion below threshold 0.50):
`patient, presents, fever, take, twice, daily, wheeze, pressure,
measured, attending, physician, prescribed, secondary, infection`

---

## Stage 2 — FLAG

### What it does
Two things:

1. Mark individual words with `suspicion >= SUSPICION_THRESHOLD (0.50)`
   AND `in_lexicon == False` as suspicious.
2. Merge adjacent suspicious words into a single span.
   "Adjacent" means: next to each other, or separated by at most one
   stop word. `dolly` and `prahn` are adjacent → one span `"dolly prahn"`.

### Model / tool used
No model. Pure Python logic only.

### Input
The `ScoredWord` list from Stage 1.

### Output
A list of `SuspiciousSpan` objects.

```python
@dataclass
class SuspiciousSpan:
    start: int        # index of first word in the span
    end: int          # index of last word in the span (inclusive)
    text: str         # joined original text of the span
    suspicion: float  # max suspicion score among words in span
    reason: str       # "low_score" | "not_in_lexicon" | "both"
```

**Expected output for the test transcript:**

```python
[
    SuspiciousSpan(start=8,  end=9,  text="dolly prahn",      suspicion=0.92, reason="both"),
    SuspiciousSpan(start=13, end=14, text="salbu tamol",      suspicion=0.84, reason="both"),
    SuspiciousSpan(start=24, end=24, text="sfigmomanometre",  suspicion=0.96, reason="both"),
    SuspiciousSpan(start=29, end=29, text="amoxicilin",       suspicion=0.71, reason="both"),
]
```

**Merge rule applied:** `dolly` (index 8) and `prahn` (index 9) are
adjacent → merged into one span. Same for `salbu` (13) and `tamol` (14).
`sfigmomanometre` (24) is a single word. `amoxicilin` (29) is a single word.

**What must NOT appear in this list:**
Any word that is correct English and fits the context: `fever, wheeze,
infection, physician, prescribed, secondary, measured, pressure`.
If any of these appear as a span, Stage 2 is over-flagging and will
cause false positive corrections downstream.

---

## Stage 3 — RETRIEVE

### What it does
For each suspicious span, find the most phonetically similar terms in
`data/medical_lexicon.jsonl` and return the top K=5 as candidates.

Two lookup paths, in this order:

**Path A — Alias lookup (fast, O(1)):**
Check if the span text (lowercased) exactly matches any `aliases` entry
in the lexicon. If yes, return that term immediately with
`phonetic_score = 1.0` and `match_type = "alias"`. Skip Path B entirely.

**Path B — IPA phonetic search (slower, O(n)):**
1. Convert the span text to IPA using `espeak-ng` via the `phonemizer`
   library. Function: `services/phonetics.text_to_ipa(span.text)`.
2. For every term in the lexicon, compute normalised IPA edit distance
   between the span IPA and the term's stored IPA.
   `phonetic_score = 1.0 - normalised_edit_distance`
3. Return the top 5 terms sorted by `phonetic_score` descending.

### Model / tool used
**`espeak-ng`** (system package, not a Python model) for IPA conversion.
**`python-Levenshtein`** for edit distance computation.
**No neural model in this stage.**

espeak-ng must be installed:
- Windows: `winget install eSpeak.eSpeakNG`
- Ubuntu: `sudo apt-get install espeak-ng`

### Input
The list of `SuspiciousSpan` objects from Stage 2,
plus the full contents of `data/medical_lexicon.jsonl`.

### Output
One `SpanWithCandidates` per suspicious span.

```python
@dataclass
class Candidate:
    term: str
    ipa: str
    term_type: str        # drug_brand | drug_generic | disease | symptom | procedure | anatomy | device
    description: str
    phonetic_score: float # 0.0–1.0, higher = more phonetically similar
    source: str           # "seed" | "user"
    match_type: str       # "alias" | "phonetic"

@dataclass
class SpanWithCandidates:
    span: SuspiciousSpan
    candidates: list[Candidate]   # max 5, sorted best first
```

**Expected output for the test transcript:**

```
Span: "dolly prahn"
  IPA of span: /dɒli prɑːn/
  Candidates:
    1. Doliprane    | drug_brand    | /dɒlipreɪn/  | score: 0.82 | seed
    2. Diprivan     | drug_brand    | /dɪˈpraɪvən/ | score: 0.48 | seed
    3. Paracetamol  | drug_generic  | /ˌpærəˈsiːtəmɒl/ | score: 0.31 | seed

Span: "salbu tamol"
  IPA of span: /sælbuː tæmɒl/
  Candidates:
    1. Salbutamol   | drug_generic  | /sælˈbjuːtəmɒl/ | score: 0.89 | seed
    2. Salmeterol   | drug_generic  | /sælˈmɛtərɒl/   | score: 0.61 | seed
    3. Paracetamol  | drug_generic  | /ˌpærəˈsiːtəmɒl/ | score: 0.29 | seed

Span: "sfigmomanometre"
  IPA of span: /sfɪɡmoʊmænoʊmɛtrə/
  Candidates:
    1. sphygmomanometer | device | /ˌsfɪɡmoʊməˈnɒmɪtər/ | score: 0.74 | seed
    2. dynamometer      | device | /ˌdaɪnəˈmɒmɪtər/      | score: 0.41 | seed

Span: "amoxicilin"
  IPA of span: /əˌmɒksɪˈsɪlɪn/
  Candidates:
    1. amoxicillin  | drug_generic  | /əˌmɒksɪˈsɪlɪn/ | score: 0.94 | seed
    2. ampicillin   | drug_generic  | /ˌæmpɪˈsɪlɪn/   | score: 0.67 | seed
    3. amoxiclavic  | drug_generic  | /əˌmɒksɪˈklævɪk/| score: 0.58 | seed
```

**Critical check:** If `sphygmomanometer` is NOT in the lexicon, the
span `sfigmomanometre` will return no good candidate and escalate to HITL.
The term `sphygmomanometer` MUST be present in `data/medical_lexicon.jsonl`
with a real IPA value for retrieval to work.

Similarly, `Doliprane`, `Salbutamol`, and `amoxicillin` must all be
present in the lexicon with real IPA values (not the raw term text).
Run `scripts/seed_lexicon.py` before testing if these are missing.

---

## Stage 4 — DECIDE

### What it does
For each `SpanWithCandidates`, apply the confidence routing table to
decide the correction path, then either auto-fix, call the LLM, or
escalate to human review.

### Routing table (apply in order, first match wins)

| Condition | Path | Action |
|---|---|---|
| `match_type == "alias"` | `auto_fix` | Apply top candidate, no LLM |
| `source == "user"` AND `phonetic_score >= 0.90` | `auto_fix` | Apply, no LLM |
| `phonetic_score >= 0.60` | `llm` | Send to LLM DECIDE prompt |
| LLM returns valid candidate term | `llm` | Apply that term |
| LLM returns `NO_CHANGE` AND word in lexicon | `no_change` | Leave as-is |
| `phonetic_score < 0.60` | `hitl_escalate` | Ask human |
| LLM returns `NO_CHANGE` AND word NOT in lexicon | `hitl_escalate` | Ask human |
| Human provides term | `human_correction` | Apply + save to dataset |
| Human presses Enter / skips | `no_change` | Leave as-is |

### Model / tool used
**Gemini 2.0 Flash** via `GEMINI_API_KEY` from `.env` for the LLM DECIDE call.
API endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent`

**No other model is used in this stage.**

### LLM DECIDE prompt (exact template)

```
You are a medical transcript correction assistant.
A speech-to-text system made an error in the following sentence.

Full sentence:
"{full_sentence}"

The phrase "{span_text}" is likely a mishearing. Here are the
candidate corrections ranked by phonetic similarity:

{for each candidate:}
{rank}. {term} [{term_type}] — {description} (phonetic score: {score:.2f})

Rules you must follow:
- Pick the candidate that makes the most clinical sense in this sentence.
- Consider the patient's symptoms and the surrounding medical context.
- If the phrase already makes sense as written, return NO_CHANGE.
- Return ONLY the exact term string from the list above, or NO_CHANGE.
- Do not explain. Do not add punctuation. One word or phrase only.
```

### Input
List of `SpanWithCandidates` from Stage 3.

### Output
One `Decision` per span.

```python
@dataclass
class Decision:
    span: SuspiciousSpan
    chosen: str | None    # the corrected term, or None if NO_CHANGE
    confidence: float     # phonetic_score of chosen, or 0.0
    path: str             # "auto_fix" | "llm" | "hitl_escalate" | "no_change"
    candidates: list[Candidate]  # pass through for UI display
```

**Expected output for the test transcript:**

```
Span "dolly prahn":
  Top candidate: Doliprane, score 0.82, source seed
  → score >= 0.60 → path = "llm"
  → LLM prompt includes: fever context + drug candidates
  → LLM returns: "Doliprane"
  Decision: chosen="Doliprane", confidence=0.82, path="llm"

Span "salbu tamol":
  Top candidate: Salbutamol, score 0.89, source seed
  → score >= 0.60 → path = "llm"
  → LLM prompt includes: wheeze context + bronchodilator candidates
  → LLM returns: "Salbutamol"
  Decision: chosen="Salbutamol", confidence=0.89, path="llm"

  NOTE: If Salbutamol has previously been taught by a human
  (source == "user") AND score >= 0.90, path becomes "auto_fix"
  and NO LLM call is made. This is the self-improving behaviour.

Span "sfigmomanometre":
  Top candidate: sphygmomanometer, score 0.74, source seed
  → score >= 0.60 → path = "llm"
  → LLM prompt includes: blood pressure measurement context
  → LLM returns: "sphygmomanometer"
  Decision: chosen="sphygmomanometer", confidence=0.74, path="llm"

  NOTE: If sphygmomanometer is NOT in the lexicon at all,
  score will be below 0.60 → path = "hitl_escalate".
  This is the correct behaviour — escalate rather than guess.

Span "amoxicilin":
  Top candidate: amoxicillin, score 0.94, source seed
  → score >= 0.90 BUT source is "seed" not "user" → NOT auto_fix
  → score >= 0.60 → path = "llm"
  → LLM prompt includes: secondary infection context
  → LLM returns: "amoxicillin"
  Decision: chosen="amoxicillin", confidence=0.94, path="llm"
```

**What the LLM must NOT do:**
- Invent a term not in the candidate list.
- Return a corrected spelling of the span text (e.g. "sphygmomanometer"
  when it was not offered as a candidate).
- Return anything other than an exact candidate term or `NO_CHANGE`.

**Validation after LLM response (in `decider.py`):**
```python
valid_terms = {c.term for c in candidates}
if llm_response.strip() not in valid_terms and llm_response.strip() != "NO_CHANGE":
    # LLM hallucinated — treat as NO_CHANGE
    chosen = None
    path = "no_change"
```

---

## Stage 5 — HUMAN-IN-THE-LOOP (HITL)

### What it does
For spans where path is `"hitl_escalate"`, present the problem to
a human via the testing UI. The human types the correct term.
The system applies the correction and permanently writes the new term
to `data/medical_lexicon.jsonl`.

### Model / tool used
No model. Human judgment only.
The Gemini API is optionally called AFTER the human provides the term,
to generate a one-sentence description if the term is new.

### When it triggers (on the test transcript)
In the ideal case where all four terms are in the lexicon with real IPA:
→ HITL does NOT trigger. All four are resolved by the LLM path.

In the realistic first-run case where `sphygmomanometer` is missing
from the lexicon:
→ HITL triggers for `sfigmomanometre`.

The UI shows:
```
Sentence : "Blood pressure was measured using a sfigmomanometre."
Flagged  : "sfigmomanometre"
Best guess: [none — no candidate scored above 0.60]

Enter the correct term: sphygmomanometer
```

### Input
The `Decision` list from Stage 4, filtered to `path == "hitl_escalate"`.
Also the saved session audio path (not used in text-only mode).

### Output
An updated `Decision` with `path = "human_correction"` and
`chosen = <what the human typed>`.

Plus a write to `data/medical_lexicon.jsonl`:

```json
{
  "term": "sphygmomanometer",
  "canonical_form": "sphygmomanometer",
  "term_type": "device",
  "aliases": ["sfigmomanometre"],
  "ipa": "/ˌsfɪɡmoʊməˈnɒmɪtər/",
  "description": "An instrument for measuring blood pressure, typically consisting of an inflatable cuff and a pressure gauge.",
  "source": "user",
  "added_at": "2026-05-24T14:00:00Z"
}
```

And a write to `data/hitl_log.jsonl`:
```json
{
  "timestamp": "2026-05-24T14:00:00Z",
  "wrong_form": "sfigmomanometre",
  "correct_term": "sphygmomanometer",
  "sentence_context": "Blood pressure was measured using a sfigmomanometre.",
  "path": "human_correction"
}
```

**Next time `sfigmomanometre` appears:**
Stage 3 alias lookup finds it in the lexicon immediately.
`phonetic_score = 1.0`, `match_type = "alias"`.
Stage 4 routing: `match_type == "alias"` → `auto_fix`.
No LLM call, no HITL. Resolved in milliseconds.

---

## Final Output Assembly

### What it does
Reconstruct the corrected transcript by replacing each span's original
text with the `chosen` term from its `Decision`, preserving all
punctuation and capitalisation conventions.

### Rules for reconstruction
1. Replace the span words with the chosen term.
2. Preserve the punctuation of the last word in the span.
3. Capitalise the chosen term if the first word of the span was capitalised.
4. Preserve all non-flagged words exactly as they appeared in the input,
   including their original capitalisation.

### Input
Original token list from Stage 0 + Decision list from Stage 4/5.

### Output
```python
@dataclass
class PipelineResult:
    original: str
    corrected: str
    corrections: list[Decision]
    hitl_required: list[Decision]   # empty if interactive=True
    session_id: str
```

**Expected final output for the test transcript:**

```
Original:
"The patient presents with fever and should take dolly prahn twice daily
alongside salbu tamol for the wheeze. Blood pressure was measured
using a sfigmomanometre. The attending physician prescribed
amoxicilin for the secondary infection."

Corrected:
"The patient presents with fever and should take Doliprane twice daily
alongside Salbutamol for the wheeze. Blood pressure was measured
using a sphygmomanometer. The attending physician prescribed
amoxicillin for the secondary infection."

Corrections applied: 4
  - "dolly prahn"       → Doliprane          (path: llm,            confidence: 0.82)
  - "salbu tamol"       → Salbutamol         (path: llm,            confidence: 0.89)
  - "sfigmomanometre"   → sphygmomanometer   (path: llm or human,   confidence: 0.74)
  - "amoxicilin"        → amoxicillin        (path: llm,            confidence: 0.94)

Words unchanged: 30
False positives: 0
```

---

## Summary Table — Models Used Per Stage

| Stage | Name | Model / Tool | Runs where |
|---|---|---|---|
| 0 | Pre-processing | Python `str.split()` | Local, no model |
| 1 | Score | `facebook/bart-large` | Local, `D:/HF_CACHE/models/facebook/bart-large` |
| 1 | Score (fallback) | Dictionary + lexicon lookup heuristic | Local, no model |
| 2 | Flag | Python logic, no model | Local, no model |
| 3 | Retrieve — IPA | `espeak-ng` system package | Local |
| 3 | Retrieve — distance | `python-Levenshtein` | Local, no model |
| 4 | Decide | Gemini 2.0 Flash | Remote API, `GEMINI_API_KEY` |
| 5 | HITL | Human + optionally Gemini for description | Remote API |
| Final | Reconstruct | Python string operations | Local, no model |

---

## What "Getting It Wrong" Looks Like

Use this as a debugging checklist. If the pipeline output does not
match the expected results above, check these causes in order:

| Wrong output | Likely cause | Fix |
|---|---|---|
| `dolly prahn` not flagged | Stage 1 suspicion < 0.50 | BART model not loaded; heuristic fallback not triggered; check `HF_CACHE_DIR` path |
| `fever` or `infection` flagged | Stage 1 over-scoring common words | Stop word list is incomplete; threshold too low |
| `dolly prahn` flagged as two separate spans | Stage 2 span merge not working | Adjacent suspicious words not being merged |
| Doliprane not in candidates | `data/medical_lexicon.jsonl` missing the entry, or IPA field contains raw text | Run `scripts/seed_lexicon.py` to populate real IPA values |
| Candidate returned but score is 0.01 | IPA field in lexicon is raw text, not real IPA | `seed_lexicon.py` did not run; `text_to_ipa()` failed silently |
| LLM returns wrong term or invents one | Validation in `decider.py` not checking against candidate list | Add the validation block from Stage 4 section above |
| Corrected output changes punctuation | Reconstruction not preserving `punct` field | Check final assembly logic |
| `amoxicilin` not corrected | Score below threshold (0.71 should be above 0.50) | Threshold set too high; check `config.py` SUSPICION_THRESHOLD |
| HITL triggers for all four spans | `data/medical_lexicon.jsonl` has no real IPA; all phonetic scores near 0 | Run `scripts/seed_lexicon.py` first |

---

## Required Entries in `data/medical_lexicon.jsonl`

These four entries MUST exist with real IPA values before the pipeline
can process the test transcript without full HITL escalation.
If any are missing or have IPA equal to the raw term text, add them manually
or run `scripts/seed_lexicon.py`.

```jsonl
{"term": "Doliprane", "canonical_form": "doliprane", "term_type": "drug_brand", "aliases": ["dolly prahn", "dolly brain", "dole preen"], "ipa": "/dɒlipreɪn/", "description": "Paracetamol-based analgesic and antipyretic brand common in France and North Africa.", "source": "seed", "added_at": "2026-05-24T00:00:00Z"}
{"term": "Salbutamol", "canonical_form": "salbutamol", "term_type": "drug_generic", "aliases": ["salbu tamol", "salbutanol", "salbutamul"], "ipa": "/sælˈbjuːtəmɒl/", "description": "Short-acting bronchodilator used to treat asthma, COPD, and wheezing.", "source": "seed", "added_at": "2026-05-24T00:00:00Z"}
{"term": "sphygmomanometer", "canonical_form": "sphygmomanometer", "term_type": "device", "aliases": ["sfigmomanometre", "sphygmomanometre", "sfigmomanometer"], "ipa": "/ˌsfɪɡmoʊməˈnɒmɪtər/", "description": "Medical instrument used to measure arterial blood pressure.", "source": "seed", "added_at": "2026-05-24T00:00:00Z"}
{"term": "amoxicillin", "canonical_form": "amoxicillin", "term_type": "drug_generic", "aliases": ["amoxicilin", "amoxycillin", "amoxicillin"], "ipa": "/əˌmɒksɪˈsɪlɪn/", "description": "Broad-spectrum penicillin antibiotic used to treat bacterial infections.", "source": "seed", "added_at": "2026-05-24T00:00:00Z"}
```