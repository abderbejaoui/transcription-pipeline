# Medical Transcript Auto-Correction Pipeline — Copilot Context

> **Read this file completely before writing any code, suggesting any refactor,
> or answering any question about this project.**
> This is the single source of truth for intent, architecture, and design
> decisions. Every implementation choice should trace back to a section here.

---

## 0. What This Project Is (One Paragraph)

A standalone Python pipeline that takes a **wrong transcript** (the output of
an ASR / speech-to-text model that made mistakes on medical vocabulary) and
produces a **corrected transcript**. Correction is grounded in a local medical
vocabulary dataset. When the pipeline cannot confidently correct a word, a
human is asked to supply the right term — and that term is permanently added
to the dataset so the same mistake never happens again. The pipeline gets
smarter with every human correction.

This is **one team member's implementation** of a shared problem. It will be
merged with teammates' implementations later. Keep it clean, modular, and
testable.

---

## 1. The Problem Being Solved

ASR models (e.g. Whisper) are trained on general speech. When a doctor says
a drug name like **"Doliprane"** (a French painkiller), Whisper may write
**"dolly prahn"** — the closest English-sounding tokens it knows. Every
individual token looks fine; the mistake is invisible unless you know medicine.

This pipeline's job: take the broken transcript, find those mistakes, fix
them using medical knowledge, and learn from corrections.

### The Hard Case

> Doctor says: "Effiralgan"
> Whisper writes: "if it all gone"

No single token resembles the drug name. A naive spell-checker or fuzzy
matcher will not find "Effiralgan" from "if it all gone". The fix must come
from **phonetic similarity** (how it sounds) combined with **contextual
plausibility** (does this make sense in a medical sentence?).

---

## 2. Non-Goals (Do Not Build These)

- No real audio processing — the input is already a text transcript (the
  broken string). Audio is outside this pipeline's scope.
- No fine-tuning of ASR models.
- No real-time / streaming processing.
- No multi-tenant or multi-org scoping — single local dataset only.
- No UI beyond a minimal CLI or simple web endpoint.
- No speaker diarization.

---

## 3. Pipeline Overview (The 5 Stages)

```
INPUT: wrong transcript (string)
         │
         ▼
┌─────────────────────────────┐
│ Stage 1: SCORE              │  Assign a "suspicion score" to every word.
│  How out-of-place is this   │  Score = how likely is this word WRONG given
│  word in this context?      │  its neighbours and the medical domain.
└────────────┬────────────────┘
             │ scored word list
             ▼
┌─────────────────────────────┐
│ Stage 2: FLAG               │  Mark words whose score is below a threshold
│  Which words are suspicious?│  as SUSPICIOUS. Also flag words that are
│                             │  unknown to the medical lexicon AND have
│                             │  low ASR confidence (if available).
└────────────┬────────────────┘
             │ list of suspicious spans
             ▼
┌─────────────────────────────┐
│ Stage 3: RETRIEVE           │  For each suspicious span, fetch the top-K
│  What are the candidates?   │  candidate medical terms from the dataset
│                             │  using PHONETIC similarity (IPA edit distance).
│                             │  Also pull each candidate's type and description.
└────────────┬────────────────┘
             │ suspicious spans + ranked candidates
             ▼
┌─────────────────────────────┐
│ Stage 4: DECIDE             │  An LLM picks the best candidate (or says
│  Which candidate fits best? │  NO_CHANGE) using the full sentence context,
│                             │  candidate types, and descriptions.
│                             │  LLM cannot invent — it can only CHOOSE
│                             │  from the provided candidate list.
└────────────┬────────────────┘
             │ corrections decided
             ▼
┌─────────────────────────────┐
│ Stage 5: HUMAN-IN-THE-LOOP  │  When confidence is low OR no candidate
│  Escalate or auto-apply     │  scored well, present the flagged span to
│                             │  a human. Human provides the correct term.
│                             │  System adds it to the dataset permanently.
└────────────┬────────────────┘
             │
             ▼
OUTPUT: corrected transcript (string) + correction report (JSON)
```

---

## 4. Stage-by-Stage Specification

### 4.1 Stage 1 — SCORE

**Goal:** Give every word a suspicion score: how likely is it wrong?

**How it works:**

Use a masked language model (e.g. `bert-base-uncased` or `roberta-base`) to
compute the per-word log-probability given its left and right context. A word
that the model finds surprising (low log-prob) in its context gets a high
suspicion score.

Alternatively, use an LLM via prompt: ask it to rate each word 0.0–1.0 for
"how likely is this word to be a speech-to-text error in a medical context."
This is slower but more semantically aware.

**Key rule:** Do NOT score every word equally. Skip common stop words
("the", "and", "a", "is") — they cannot be medical errors. Focus compute
on content words.

**Output contract:**

```python
@dataclass
class ScoredWord:
    index: int           # position in word list
    text: str            # the word as written
    suspicion: float     # 0.0 = definitely correct, 1.0 = definitely wrong
    in_lexicon: bool     # is this word in our medical dataset?
```

**Threshold:** Words with `suspicion >= SUSPICION_THRESHOLD` (default 0.5)
and `in_lexicon == False` proceed to Stage 2.

---

### 4.2 Stage 2 — FLAG

**Goal:** Turn individual word scores into contiguous suspicious SPANS.

**Why spans, not individual words:** "dolly prahn" is two words that together
represent one medical term. Correcting only one of them is wrong.

**Span merging rule:** If two suspicious words are adjacent (or separated by
only one stop word), merge them into one span.

**Output contract:**

```python
@dataclass
class SuspiciousSpan:
    start: int           # index of first word in span
    end: int             # index of last word in span (inclusive)
    text: str            # joined text of the span, e.g. "dolly prahn"
    suspicion: float     # max suspicion score among words in span
    reason: str          # "low_score" | "not_in_lexicon" | "both"
```

---

### 4.3 Stage 3 — RETRIEVE

**Goal:** For each suspicious span, find the most phonetically similar terms
in the medical dataset.

**How it works:**

1. Convert the span text to IPA (International Phonetic Alphabet) using
   `phonemizer` + `espeak-ng` backend.
2. For every term in the medical dataset, compute IPA edit distance between
   the span's IPA and the term's IPA.
3. Return the top K=5 terms with the lowest edit distance.

**Why IPA, not string edit distance:**
- "dolly prahn" (string) → "Doliprane" (string): edit distance = 8 (fails)
- "dɒli prɑːn" (IPA) → "dɒlipreɪn" (IPA): edit distance = 3 (works)

**Candidate type matters:** Each returned candidate must include its `term_type`
(drug_brand, drug_generic, disease, symptom, procedure, anatomy). Stage 4
uses this to make type-aware decisions.

**Output contract:**

```python
@dataclass
class Candidate:
    term: str               # canonical spelling, e.g. "Doliprane"
    ipa: str                # IPA of the canonical term
    term_type: str          # drug_brand | drug_generic | disease | symptom | procedure | anatomy
    description: str        # one-line clinical description
    phonetic_score: float   # 0.0–1.0, higher = more similar (1 - normalised edit distance)
    source: str             # "seed" | "user" (user-taught terms get priority)
```

**Output per span:**

```python
@dataclass
class SpanWithCandidates:
    span: SuspiciousSpan
    candidates: list[Candidate]   # ranked best-first, max 5
```

---

### 4.4 Stage 4 — DECIDE

**Goal:** Pick one candidate per span, or decide NO_CHANGE.

**Hard rules:**

1. **The LLM cannot invent.** It receives the candidate list and must pick one
   by its `term` value, or return the string `"NO_CHANGE"`. If it returns
   anything else, treat it as `NO_CHANGE`.
2. **User-sourced candidates win ties.** If a candidate with `source="user"`
   scores phonetically similar to a seed candidate, prefer the user one.
3. **High-confidence auto-fix.** If the top candidate has `phonetic_score >= 0.90`
   AND `source == "user"`, skip the LLM entirely and auto-apply. This is the
   "the system has seen this before" fast path.
4. **Low confidence → human.** If the top candidate has `phonetic_score < 0.60`,
   skip the LLM and escalate directly to Stage 5 (HITL).

**LLM prompt structure (the DECIDE prompt):**

```
You are correcting a medical transcript. A word or phrase may have been 
misheard by the speech-to-text system.

Sentence: "{full_sentence}"
Suspicious phrase: "{span_text}"

Candidates (ranked by phonetic similarity):
1. {term} [{term_type}] — {description} (score: {phonetic_score:.2f})
2. ...

Rules:
- Pick the candidate that makes the most clinical sense in this sentence.
- If the phrase already makes sense or no candidate fits, return NO_CHANGE.
- Return ONLY the exact term string or NO_CHANGE. No explanation.
```

**Output contract:**

```python
@dataclass
class Decision:
    span: SuspiciousSpan
    chosen: str | None    # canonical term, or None if NO_CHANGE
    confidence: float     # phonetic_score of chosen, or 0.0 if NO_CHANGE
    path: str             # "auto_fix" | "llm" | "hitl_escalate"
```

---

### 4.5 Stage 5 — HUMAN-IN-THE-LOOP (HITL)

**Goal:** When the pipeline cannot confidently correct a span, ask a human.
The human's answer becomes a permanent addition to the dataset.

**Trigger conditions (any one of):**
- `path == "hitl_escalate"` (low phonetic confidence)
- LLM returned `NO_CHANGE` and the word is not in the lexicon
- The human explicitly flags an auto-fix as wrong

**What the human sees:**

```
[NEEDS YOUR INPUT]
Sentence  : "The patient should take dolly prahn twice daily."
Flagged   : "dolly prahn"
Best guess: Doliprane (phonetic match 0.71) — paracetamol-based analgesic
            
Enter the correct term (or press Enter to leave unchanged): _
```

**What happens after human input:**

1. The correct term is applied to the transcript.
2. If the term is new to the dataset, a new entry is created:
   - `term`: the correct spelling they entered
   - `aliases`: the wrong form(s) that triggered this
   - `ipa`: computed from the correct term via espeak-ng
   - `term_type`: ask the human (with a short menu) OR infer from context
   - `description`: optionally generated by LLM, or left empty
   - `source`: `"user"`
3. The dataset is saved immediately (write-through, never batch).
4. A log entry is created: `{timestamp, wrong_form, correct_term, sentence_context}`.

**This is the moat**: every correction makes the next run smarter for the
same mistake.

---

## 5. Medical Dataset Format

The local medical dataset is a JSONL file (`data/medical_lexicon.jsonl`).
Each line is one term:

```json
{
  "term": "Doliprane",
  "canonical_form": "doliprane",
  "term_type": "drug_brand",
  "aliases": ["dolly prahn", "dole preen", "dolly brain"],
  "ipa": "/dɒlipreɪn/",
  "description": "Paracetamol-based analgesic and antipyretic, common in France and North Africa.",
  "source": "seed",
  "added_at": "2024-01-01T00:00:00Z"
}
```

**Field rules:**
- `term`: canonical human-readable spelling
- `canonical_form`: lowercased, for dedup checks
- `aliases`: wrong forms the ASR has produced in the past; searched first
  before running phonetic retrieval
- `ipa`: used for phonetic retrieval; computed once at insert time
- `source`: `"seed"` (from initial lexicon) or `"user"` (human-taught);
  user-sourced terms get priority in retrieval and auto-fix
- `added_at`: ISO 8601 timestamp

**Alias shortcut:** Before doing any phonetic search in Stage 3, check if the
span text (lowercased) exactly matches any `alias` in the dataset. If yes,
return that term as the top candidate with `phonetic_score=1.0`. This is
O(1) and catches repeat mistakes instantly.

---

## 6. Confidence Routing Table

This table is the single source of truth for how decisions are routed:

| Condition | Path | Action |
|-----------|------|--------|
| Alias exact match | `auto_fix` | Apply immediately, no LLM |
| `source=user` AND `phonetic_score >= 0.90` | `auto_fix` | Apply immediately, no LLM |
| `phonetic_score >= 0.60` AND `< 0.90` | `llm` | Send to LLM DECIDE |
| LLM returns a valid term | `llm` | Apply with note |
| LLM returns `NO_CHANGE` AND word in lexicon | `no_change` | Leave as-is |
| `phonetic_score < 0.60` | `hitl_escalate` | Ask human |
| LLM returns `NO_CHANGE` AND word NOT in lexicon | `hitl_escalate` | Ask human |
| Human provides term | `human_correction` | Apply + save to dataset |
| Human provides no term (Enter) | `no_change` | Leave as-is |

---

## 7. File Structure

```
project/
├── pipeline/
│   ├── __init__.py
│   ├── models.py          # All dataclasses (ScoredWord, SuspiciousSpan, Candidate, Decision)
│   ├── scorer.py          # Stage 1: word scoring (MLM or LLM-based)
│   ├── flagger.py         # Stage 2: span detection from scored words
│   ├── retriever.py       # Stage 3: phonetic search against medical dataset
│   ├── decider.py         # Stage 4: LLM DECIDE + routing logic
│   ├── hitl.py            # Stage 5: human-in-the-loop prompt + dataset write
│   └── runner.py          # Orchestrator: chains stages 1→5, returns result
│
├── data/
│   ├── medical_lexicon.jsonl      # Medical vocabulary dataset (JSONL)
│   ├── descriptions.jsonl         # Cached LLM-generated descriptions
│   └── hitl_log.jsonl             # Log of every human correction
│
├── services/
│   ├── lexicon.py         # Read/write/search medical_lexicon.jsonl
│   ├── phonetics.py       # IPA conversion (espeak-ng wrapper) + edit distance
│   └── llm.py             # LLM client (Ollama or OpenRouter), one function per task
│
├── tests/
│   ├── test_scorer.py
│   ├── test_flagger.py
│   ├── test_retriever.py
│   ├── test_decider.py
│   └── test_hitl.py
│
├── scripts/
│   └── seed_lexicon.py    # One-time: compute IPA for all seed terms, write to lexicon
│
├── main.py                # CLI entry point: python main.py --transcript "..."
├── requirements.txt
└── COPILOT_CONTEXT.md     # This file
```

---

## 8. Key Interfaces (Public API of Each Module)

### `pipeline/runner.py`
```python
def run_pipeline(transcript: str, interactive: bool = True) -> PipelineResult:
    """
    Main entry point. Takes a wrong transcript, returns corrected text + report.
    If interactive=True, pauses for human input at HITL stage.
    If interactive=False, skips HITL (for batch/test runs).
    """
```

### `services/lexicon.py`
```python
def load_lexicon() -> list[LexiconEntry]: ...
def find_by_alias(text: str) -> LexiconEntry | None: ...
def add_entry(entry: LexiconEntry) -> None: ...   # writes to JSONL immediately
def search_phonetic(ipa: str, top_k: int = 5) -> list[Candidate]: ...
```

### `services/phonetics.py`
```python
def text_to_ipa(text: str, language: str = "en-us") -> str: ...
def ipa_edit_distance(a: str, b: str) -> float: ...   # normalised 0.0–1.0
```

### `services/llm.py`
```python
def llm_decide(sentence: str, span: str, candidates: list[Candidate]) -> str:
    """Returns one of: candidate term string | 'NO_CHANGE'"""
```

---

## 9. Critical Design Constraints

These are non-negotiable. Do not work around them.

1. **LLM cannot invent corrections.** It selects from a candidate list only.
   Any response not matching a candidate term or "NO_CHANGE" is treated as
   "NO_CHANGE". Validate this in `decider.py`, not in the LLM prompt alone.

2. **Alias lookup is always tried first**, before any phonetic computation.
   It is O(1) and catches repeat mistakes instantly.

3. **User-sourced terms have retrieval priority.** When a human has taught a
   term, that knowledge trumps seed data in ties.

4. **Dataset writes are synchronous and immediate.** Never batch HITL saves.
   A crash after a human correction must not lose it.

5. **Span merging must happen before retrieval.** Never retrieve candidates for
   half a multi-word medical term.

6. **Phonetic similarity must use IPA, not raw string edit distance.** Raw edit
   distance fails on the primary use case ("dolly prahn" → "Doliprane").

7. **SUSPICION_THRESHOLD, TOP_K, and similarity cutoffs must be named
   constants**, never magic numbers. Put them in a `config.py` or at the top
   of the relevant module.

---

## 10. The HITL Learning Loop (How the System Gets Smarter)

This is the most important feature. Every human correction closes one gap
permanently.

```
Human types "Doliprane"
         │
         ├─► Is "doliprane" already in lexicon?
         │       YES: add "dolly prahn" to its aliases list, save.
         │       NO:  create full new LexiconEntry:
         │               term="Doliprane"
         │               aliases=["dolly prahn"]
         │               ipa=text_to_ipa("Doliprane")
         │               term_type=<ask human from menu>
         │               source="user"
         │           save to medical_lexicon.jsonl immediately.
         │
         └─► Log to data/hitl_log.jsonl:
               {timestamp, wrong_form, correct_term, sentence_excerpt, session_id}
```

**Next time "dolly prahn" appears:**
- Stage 3 alias lookup hits immediately → `phonetic_score=1.0`
- Confidence router → `auto_fix` path
- No LLM needed, no human needed
- Correction applied in milliseconds

---

## 11. Example End-to-End Run

**Input transcript:**
```
"The patient presents with fever and should take dolly prahn twice daily 
alongside salbu tamol for the wheeze."
```

**Stage 1 scoring:**
- "dolly": suspicion=0.82 (not in lexicon, low MLM prob)
- "prahn": suspicion=0.91 (not in lexicon, very low MLM prob)
- "salbu": suspicion=0.78 (not in lexicon)
- "tamol": suspicion=0.76 (not in lexicon)
- all other words: suspicion<0.3

**Stage 2 flagging:**
- Span A: words 7–8, "dolly prahn", suspicion=0.91
- Span B: words 12–13, "salbu tamol", suspicion=0.78

**Stage 3 retrieval:**
- Span A → IPA: "dɒli prɑːn" → top candidates:
  1. Doliprane (score=0.82, drug_brand)
  2. Diprivan (score=0.51, drug_brand)
- Span B → IPA: "sælbuː tæmɒl" → top candidates:
  1. Salbutamol (score=0.89, drug_generic, source=user ← human taught this last week)
  2. Salmeterol (score=0.61, drug_generic)

**Stage 4 decisions:**
- Span A: top score 0.82, path=`llm` → LLM given sentence + candidates →
  returns "Doliprane" (fever context + drug_brand match)
- Span B: top score 0.89, source=user → path=`auto_fix`, no LLM

**Output:**
```
"The patient presents with fever and should take Doliprane twice daily 
alongside Salbutamol for the wheeze."
```

---

## 12. Testing Strategy

Every stage is independently testable with static inputs. Do not test with
live LLM calls in unit tests — mock `services/llm.py`.

**Key test cases to implement:**
- `test_retriever.py`: "dolly prahn" → Doliprane is in top-3 candidates
- `test_retriever.py`: exact alias match returns score=1.0 immediately
- `test_decider.py`: LLM response not in candidate list → treated as NO_CHANGE
- `test_decider.py`: user-source candidate at 0.90 → auto_fix, no LLM call
- `test_hitl.py`: new term saved to lexicon JSONL; appears in next load
- `test_hitl.py`: existing term → alias appended, no duplicate entry created
- `test_runner.py`: full pipeline on the "dolly prahn / salbu tamol" example

---

## 13. Dependencies (Recommended)

```
# Core
phonemizer          # text → IPA via espeak-ng
python-Levenshtein  # fast edit distance
transformers        # BERT/RoBERTa for Stage 1 scoring (optional, can use LLM instead)
requests            # LLM API calls (Ollama or OpenRouter)

# Dev/test
pytest
pytest-mock
```

**espeak-ng must be installed as a system package** (not pip):
```bash
# Ubuntu/Debian
sudo apt-get install espeak-ng

# macOS
brew install espeak-ng
```

---

## 14. What Copilot Should NOT Do

- Do not add a database (SQLite, Postgres, etc.) — JSONL files are the
  intentional choice for simplicity and portability.
- Do not add a web UI — CLI is sufficient for this implementation phase.
- Do not add streaming ASR — input is always a complete text string.
- Do not add multi-language support beyond English in the initial build —
  the IPA layer will handle other languages later.
- Do not collapse stages into one large function — each stage is a separate
  module for testability and teammate readability.
- Do not let the LLM generate corrections from scratch — it selects from
  the candidate list ONLY.

---

## 15. FastAPI Integration — Wiring the New Pipeline

> **Status as of 2026-05-24:** The pipeline is complete as a CLI tool.
> This section tells Copilot exactly how to wire it into the existing
> FastAPI app without breaking anything already deployed.

### 15.1 New endpoint to add

Add one new endpoint to the existing FastAPI app (`app/main.py`).
Do NOT modify `/api/transcribe` — leave the legacy endpoint untouched.
The new endpoint runs alongside it.

```python
POST /api/v2/correct
```

**Request body:**
```json
{
  "transcript": "The patient should take dolly prahn twice daily.",
  "interactive": false
}
```

**Response body:**
```json
{
  "original": "The patient should take dolly prahn twice daily.",
  "corrected": "The patient should take Doliprane twice daily.",
  "corrections": [
    {
      "span_text": "dolly prahn",
      "start": 5,
      "end": 6,
      "chosen": "Doliprane",
      "confidence": 0.82,
      "path": "llm",
      "candidates": [
        {"term": "Doliprane", "term_type": "drug_brand", "phonetic_score": 0.82},
        {"term": "Diprivan",  "term_type": "drug_brand", "phonetic_score": 0.51}
      ]
    }
  ],
  "hitl_required": [],
  "session_id": "uuid-string"
}
```

When `interactive: false`, any HITL cases are NOT paused for input —
instead they are included in the `hitl_required` list so the UI can
display them for human resolution separately.

**Second endpoint for HITL resolution:**

```python
POST /api/v2/correct/teach
```

**Request body:**
```json
{
  "session_id": "uuid-string",
  "wrong_form":   "dolly prahn",
  "correct_term": "Doliprane",
  "sentence_context": "The patient should take dolly prahn twice daily."
}
```

This calls `hitl.apply_human_correction()` and writes to the lexicon.
Returns `{"status": "saved", "term": "Doliprane", "is_new": true}`.

### 15.2 Where to add the code in main.py

```python
# In app/main.py — add after existing imports
from app.pipeline.runner import run_pipeline
from app.pipeline.hitl import apply_human_correction

# Add after existing route definitions
@app.post("/api/v2/correct")
async def correct_transcript(body: CorrectionRequest):
    result = run_pipeline(body.transcript, interactive=False)
    return result.to_dict()

@app.post("/api/v2/correct/teach")
async def teach_correction(body: TeachRequest):
    apply_human_correction(
        wrong_form=body.wrong_form,
        correct_term=body.correct_term,
        sentence_context=body.sentence_context,
    )
    return {"status": "saved"}
```

### 15.3 Pydantic models to add

Add these to `app/main.py` or a new `app/schemas.py`:

```python
from pydantic import BaseModel

class CorrectionRequest(BaseModel):
    transcript: str
    interactive: bool = False

class TeachRequest(BaseModel):
    session_id: str
    wrong_form: str
    correct_term: str
    sentence_context: str
```

### 15.4 What NOT to touch

- Do not modify `/api/transcribe` or any existing route.
- Do not modify `app/services/asr.py` — the new pipeline receives text, not audio.
- Do not change the existing lexicon file format — the new loader is backward-compatible.
- Do not add authentication — this is a local testing tool.

---

## 16. Testing UI — Full Design Specification

> This section tells Copilot exactly how to build the testing web UI.
> It is a **single-page HTML/JS frontend** served by the existing FastAPI app.
> The UI calls `/api/v2/correct` and `/api/v2/correct/teach`.

### 16.1 Serve the UI from FastAPI

Add one route to `app/main.py`:

```python
from fastapi.responses import FileResponse

@app.get("/test")
async def test_ui():
    return FileResponse("app/static/test_ui.html")
```

Create the file at: `app/static/test_ui.html`

### 16.2 Layout — Two-Column

```
┌─────────────────────────────────────┬──────────────────────┐
│  LEFT COLUMN (main, ~65% width)     │  RIGHT SIDEBAR (~35%) │
│                                     │                       │
│  [Input transcript textarea]        │  [HITL panel]         │
│  [Run pipeline button]              │  (appears only when   │
│                                     │   HITL is needed)     │
│  [Stage 1 — Word scores]            │                       │
│  [Stage 2 — Flagged spans]          │  [Correction log]     │
│  [Stage 3 — Candidates per span]    │  (append-only list of │
│  [Stage 4 — Decisions]              │   all actions taken)  │
│  [Output transcript]                │                       │
└─────────────────────────────────────┴──────────────────────┘
```

The left column shows pipeline stages in order, top to bottom.
Stages are hidden until the pipeline runs (start collapsed, expand
with a smooth CSS transition when data arrives for that stage).

### 16.3 Component Specifications

#### Input Panel
- A `<textarea>` for the wrong transcript. Minimum 3 rows, auto-grows.
- Placeholder: `"Paste the wrong transcript here…"`
- A "Run pipeline" `<button>` below it. Full width. Disabled while running.
- While running: button shows a spinner and "Running…" text.

#### Stage 1 — Word Scores Panel
Title: "Stage 1 — word scores"
Badge: `{n} words scored`

Content: a horizontal bar chart of ONLY the suspicious words
(suspicion > 0.3). Each row shows:
- Word label (left, fixed width)
- A coloured progress bar (full width)
- Score value (right, fixed width)

Bar colour by score:
- `>= 0.75`: red (`#D85A30`)
- `0.50–0.74`: amber (`#EF9F27`)
- `< 0.50`: green (`#639922`)

Stop words and low-suspicion words are NOT shown in this panel — they
add noise. Show a muted footnote: `"{n} stop words and low-score words
not shown."`

#### Stage 2 — Flagged Spans Panel
Title: "Stage 2 — flagged spans"
Badge: `{n} spans`

Content: the full transcript rendered as individual word pills.
- Flagged words: coral pill (`background: #FAECE7; color: #993C1D;
  border: 0.5px solid #F0997B`). If two flagged words form one span,
  render them as a single merged pill showing the full span text.
- Normal words: muted grey pill (`background: var(--color-background-secondary);
  color: var(--color-text-secondary)`).
- Stop words: just plain text, no pill, small and muted.

#### Stage 3 — Candidates Panel
Title: "Stage 3 — candidates"
No badge.

Content: one sub-section per suspicious span. Each sub-section shows:
- Sub-header: the span text in bold, plus a small tag if it was
  found via alias (`alias match`) or phonetic search (`phonetic`).
- A ranked list of up to 5 candidates. Each row:
  - Rank number (muted)
  - Term name (bold)
  - Type badge (e.g. `drug brand`, `drug generic`, `disease`)
  - Phonetic score (right-aligned, muted)
  - Source badge only if `"user"`: green `user-taught` pill

#### Stage 4 — Decisions Panel
Title: "Stage 4 — decisions"
Badge: `{n} auto-fixed`, `{n} via LLM`, `{n} escalated`

Content: one row per span showing:
- Original span text (left, muted)
- Arrow `→`
- Chosen term (bold, green text if correction was made, muted if NO_CHANGE)
- Path badge: `auto-fix` (green), `via LLM` (blue), `HITL` (coral)

If path is `HITL`, the row is highlighted coral and a link reads
"resolve in sidebar →" which scrolls to the HITL panel.

#### Output Transcript Panel
Title: "output transcript"
Badge: `{n} corrections`

Content: the corrected transcript as running text. Corrected words
are highlighted with a subtle green background (`#EAF3DE`) and green
text (`#27500A`). On hover, show a tooltip: `was: "{original_span}"`.

Add a "Copy to clipboard" button (top-right of panel).

#### HITL Sidebar Panel (right column, top)
Only visible when `hitl_required` is non-empty.
Title: "needs your input"
Coral accent border: `border: 0.5px solid #F0997B`
Coral header background: `background: #FAECE7`

For each HITL item, show:
1. The sentence context with the flagged span highlighted in bold coral.
2. The best guess: `"Best guess: {term} ({score:.2f}) — {description}"`
3. A text input: `"Enter the correct term or leave blank to skip…"`
4. Two buttons: `"Save + teach"` (coral filled) and `"Skip"` (ghost).

On "Save + teach":
- POST to `/api/v2/correct/teach` with the entered term.
- Show a green confirmation: `"✓ {term} added to dataset"`.
- Move to the next HITL item (if any).
- Update the Output panel to reflect the correction.

On "Skip":
- The span is left uncorrected in the output.
- Move to the next HITL item.

#### Correction Log Panel (right column, bottom)
Title: "correction log"

An append-only list of every action taken in this session.
Each entry has a coloured dot + text:
- Green dot: `{term} auto-fixed ({source}, {score:.2f})`
- Amber dot: `{term} corrected via LLM ({score:.2f})`
- Coral dot: `"{span}" escalated — {reason}`
- Blue dot: `{term} taught by human and saved to dataset`

Newest entries at top. Maximum 50 entries shown; older ones hidden
with a "show more" link.

### 16.4 Behaviour Rules

- **Stages render progressively.** Do not wait for the full pipeline
  response before showing anything. Use streaming or staged fetch:
  call the API, then as soon as the JSON arrives, render stages
  one-by-one with a 120ms CSS fade-in delay between each.
- **No page reload ever.** All interactions (run, teach, skip) are
  `fetch()` calls. The page is a single HTML file with no framework.
- **Mobile is not a priority.** The two-column layout is fixed.
  Minimum viewport width assumed: 900px.
- **Error handling:** If the API call fails, show a red banner at
  the top: `"Pipeline error: {message}. Check the server console."`
  The banner disappears after 6 seconds or on next run.
- **Empty state:** Before first run, show a muted placeholder in
  each stage panel: `"Results will appear here after you run the pipeline."`

### 16.5 Styling Rules for Copilot

- Use only vanilla HTML, CSS, and JavaScript. No React, no Vue, no Tailwind.
- All CSS in a single `<style>` block at the top of the file.
- Use CSS custom properties for all repeated values (colors, radii, etc.).
- Font: `system-ui, -apple-system, sans-serif` — no web font import.
- Colour palette (all used as CSS variables):
  ```css
  --clr-bg: #ffffff;
  --clr-surface: #f7f6f3;
  --clr-border: rgba(0,0,0,0.10);
  --clr-border-strong: rgba(0,0,0,0.18);
  --clr-text: #1a1a1a;
  --clr-muted: #6b6b6b;
  --clr-hint: #a0a0a0;
  --clr-green-bg: #EAF3DE;
  --clr-green-text: #27500A;
  --clr-amber-bg: #FAEEDA;
  --clr-amber-text: #633806;
  --clr-coral-bg: #FAECE7;
  --clr-coral-text: #993C1D;
  --clr-coral-border: #F0997B;
  --clr-blue-bg: #E6F1FB;
  --clr-blue-text: #185FA5;
  --radius: 10px;
  --radius-sm: 6px;
  ```
- All panels share this base style:
  ```css
  .panel {
    background: var(--clr-bg);
    border: 0.5px solid var(--clr-border);
    border-radius: var(--radius);
    overflow: hidden;
    margin-bottom: 12px;
  }
  .panel-head {
    padding: 10px 16px;
    border-bottom: 0.5px solid var(--clr-border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 13px;
    font-weight: 500;
    color: var(--clr-text);
  }
  .panel-body {
    padding: 12px 16px;
  }
  ```
- No animations beyond `opacity` fade-in on panel reveal (`transition: opacity 0.15s ease`).
- No shadows, no gradients, no blur.

### 16.6 Test Transcript to Include as Default

Pre-fill the textarea with this transcript so the UI is immediately
runnable without the user having to type anything:

```
The patient presents with fever and should take dolly prahn twice daily
alongside salbu tamol for the wheeze. Blood pressure was measured
using a sfigmomanometre. The attending physician prescribed
amoxicilin for the secondary infection.
```

This contains four deliberate errors: `dolly prahn` (Doliprane),
`salbu tamol` (Salbutamol), `sfigmomanometre` (sphygmomanometer),
and `amoxicilin` (amoxicillin). It exercises all three pipeline
paths: auto-fix (if Salbutamol was previously taught), LLM decide,
and potentially HITL.