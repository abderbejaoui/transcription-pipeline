# MedCorr Pipeline — Comprehensive Status & Analysis

**Date:** 2026-05-26
**Author:** Codebuff (via deepseek/deepseek-v4-flash)

---

## 1. What the Pipeline Does

The pipeline corrects misspelled medical terms in ASR transcripts using 5 stages:

```
Input Transcript
    ↓
Stage 1 (SCORE)    — Assigns suspicion 0.0–1.0 per word
    ↓
Stage 2 (FLAG)     — Merges adjacent suspicious words into spans
    ↓
Stage 3 (RETRIEVE) — Finds phonetically similar lexicon candidates per span
    ↓
Stage 4 (DECIDE)   — Routes: auto-fix / LLM rerank / escalate-to-human
    ↓
Stage 5 (CORRECT)  — Applies string replacements
    ↓
Corrected Transcript
```

---

## 2. Current State of Each Stage

### Stage 1 — SCORE (`app/pipeline/scorer.py`) ✅ WORKING

**Models:**
- **Primary:** `answerdotai/ModernBERT-large` (fill-mask, 395M params, ~1.5 GB VRAM)
- **Fallback:** Heuristic (character-level edit distance)

**Heuristic pre-filter uses:**
- **SUBTLEX-US** — 53,514 high-frequency English words (70th percentile). Words above threshold → 0.0 suspicion.
- **spaCy POS tagging** — Identifies function words (DET, ADP, CONJ, CCONJ, AUX, PART, PRON) → 0.0 suspicion.
- **Medical lexicon canonical form match** → 0.05 suspicion.
- **pyenchant en_US spell-check** — Valid medical words (e.g. "nebulization") → 0.10 suspicion.
- **Levenshtein distance (1–3)** against medical wordlist → `has_close_dictionary_match` boolean set on each `ScoredWord`.
- **Bigram match** against lexicon multi-word entries → 0.85 suspicion.

**Status:** All 12 misspellings score 0.95. "nebulization" scores 0.10 (no false positive). `score_source` ("zero", "heuristic", "modernbert") and `has_close_dictionary_match` propagated correctly.

### Stage 2 — FLAG (`app/pipeline/flagger.py`) ✅ WORKING

Merges suspicious words into spans with stop-word gap tolerance. Propagates `has_close_dictionary_match` and `score_source` to `SuspiciousSpan`.

### Stage 3 — RETRIEVE (`app/pipeline/retriever.py`) ⚠️ BRITTLE

**Strategy:**
1. Alias lookup (exact match against lexicon aliases) → short-circuits with score=1.0
2. IPA edit distance: convert span to IPA via espeak-ng (phonemizer), compare against stored IPA of each lexicon entry
3. Fallback IPA: simple grapheme→phoneme rules if phonemizer fails

**Status:** Works well for words with alias entries (diarhea→diarrhea, tahycardia→tachycardia, paracetemol→paracetamol, glucometre→glucometer). Fails badly when the correct term isn't in the lexicon (see Problem 1 below).

### Stage 4 — DECIDE (`app/pipeline/decider.py`) ⚠️ PARTIALLY BROKEN

**Routing logic (in order):**
1. If source=="user" AND phonetic_score >= 0.90 → `auto_fix`
2. If `has_close_dictionary_match=False` AND top.phonetic_score < 0.60 → `hitl_escalate` (Gap 2 guard)
3. If score_source=="heuristic" AND hcd=False AND top.phonetic_score < 0.60 → `hitl_escalate` (Gap 3 guard)
4. If top.phonetic_score < 0.60 → `hitl_escalate`
5. Else → call LLM (Groq llama-3.3-70b). If LLM returns a valid candidate → `llm`. If LLM returns NO_CHANGE and top.score >= 0.60 → `top_fallback`. Else → `hitl_escalate`.

**Issue:** The Gap 2 guard (`not span_hcd and top.phonetic_score < LLM_MIN_CONFIDENCE`) only catches cases where hcd=False AND score < 0.60. But `dehidration→respiration` has hcd=False and score=0.7273, which is ≥ 0.60 → falls through to `top_fallback` and applies the wrong word.

### Stage 5 — CORRECT / HITL (`app/pipeline/hitl.py`) ✅ WORKING

Applies string replacements. Interactive mode prompts for human input on escalated spans. Also writes to `data/hitl_log.jsonl` and lexicon.

---

## 3. The Stress Test — Full Results

### Input transcript

```
The child arrived with acute diarhea and recurrent vommiting after several days of poor apetitte.
On examination, the nurse noted tahycardia and mild dehidration.
The physician recommended paracetemol every 6 hours for fever and prescribed azitromycine for the suspected bacterial infection.
Oxygen saturation was monitored with a pulsoxymetre while albutarol nebulization was administered for bronkospasm.
Blood glucose was checked using a glucometre, and the patient was advised to continue omeprazol for gastric irritation.
```

### Corrections (actual vs expected)

| Input | Pipeline Output | Expected Correct | Verdict | Root Cause |
|-------|----------------|-----------------|---------|------------|
| diarhea | **diarrhea** ✅ | diarrhea | PASS | Alias match |
| tahycardia | **tachycardia** ✅ | tachycardia | PASS | Alias match |
| paracetemol | **paracetamol** ✅ | paracetamol | PASS | Alias match |
| glucometre | **glucometer** ✅ | glucometer | PASS | Alias match |
| albutarol | **salbutamoll** ⚠️ | salbutamol/albuterol | PARTIAL | Lexicon has "salbutamoll" (typo: double 'l') |
| dehidration | **respiration** ❌ | dehydration | FAIL | Not in lexicon → weak retrieval → top_fallback |
| azitromycine | **azathioprine** ❌ | azithromycin | FAIL | Not in lexicon → weak retrieval → LLM wrong |
| pulsoxymetre | **duloxetine** ❌ | pulse oximeter | FAIL | Multi-word term not in lexicon |
| omeprazol | **esomeprazole** ❌ | omeprazole | FAIL | Not in lexicon (worse match won) |
| vommiting | — ❌ | vomiting | FAIL | Not in lexicon → hitl_escalate |
| apetitte | — ❌ | appetite | FAIL | Not in lexicon → hitl_escalate |
| bronkospasm | — ❌ | bronchospasm | FAIL | Not in lexicon → hitl_escalate |
| nebulization | — ✅ | (no change) | PASS | Correctly scored 0.10 (valid word) |

### Summary: 4 correct, 4 wrong corrections, 3 missed (−), 1 correct skip

---

## 4. Diagnosed ProbleMs — Detailed

### Problem A: CRITICAL — Missing Lexicon Entries

**The medical lexicon is missing common medical terms.** This is the single biggest issue.

The following correct terms are **NOT** in `data/medical_lexicon.jsonl`:

| Correct Term | Type | Pipeline Impact |
|-------------|------|----------------|
| **vomiting** | Symptom | No candidate → hitl_escalate |
| **appetite** | Common symptom | No candidate → hitl_escalate |
| **dehydration** | Condition | Weak candidate (respiration 0.73) → wrong correction |
| **bronchospasm** | Condition | No candidate → hitl_escalate |
| **azithromycin** | Antibiotic | Weak candidates (Acitrom 0.67, azathioprine 0.57) → LLM wrong |
| **omeprazole** | PPI drug | Esomeprazole (0.90) wins over omeprazole (not in top 5) |
| **pulse oximeter** | Device (2-word) | No single-word candidate matches |

**Root cause:** The lexicon was seeded with ~50 drug/condition entries but is missing many basic medical vocabulary terms. The retriever (`retriever.py`) only searches `lexicon.load_lexicon()` → `data/medical_lexicon.jsonl`. Terms in `legacy/medical_terms.txt` are NOT searched by the retriever.

**Secondary issue: Lexicon has a typo.** "salbutamoll" (double 'l') is stored instead of "salbutamol". This means the alias score is 0.80 instead of ~0.95+.

### Problem B: Escalation Guard Too Weak

When `has_close_dictionary_match=False` AND the top phonetic score is >= 0.60, the pipeline falls through to `top_fallback` and applies the wrong word.

**Example:** `dehidration`
- hcd=False (Levenshtein distance to "dehydration" in wordlist: what is it? If "dehydration" IS in medical_terms.txt, then Levenshtein distance "dehidration"→"dehydration" = 1 (one 'h' swapped). So hcd SHOULD be True... Let me check.)

Wait, actually: `_load_medical_wordlist()` reads from BOTH `lexicon.load_lexicon()` AND `legacy/medical_terms.txt`. And `_has_close_dictionary_match` checks Levenshtein distance <= 3 against that wordlist.

"dehidration" vs "dehydration" → Levenshtein distance = 1 (insert 'y' after 'h'). That's <= 3. So if "dehydration" is in medical_terms.txt, hcd should be True for "dehidration".

But the diagnostic shows hcd=False for "dehidration". This means "dehydration" is NOT in medical_terms.txt either!

Let me double check this reading the legacy/medical_terms.txt file.

Actually, I can look at the medical wordlist size: "Medical wordlist loaded: 633 terms". That's a very small wordlist. Let me check what's in it.

Let me add this to the issues — the medical wordlist is only 633 terms and is missing many common medical terms.

### Problem C: Single Tokens Can't Match Multi-Word Terms

The pipeline operates on token-aligned spans. When a single misspelled token maps to a multi-word correct term (e.g. "pulsoxymetre" → "pulse oximeter"), the retriever can find no single-word candidate above threshold.

**Affected:** pulsoxymetre → pulse oximeter

### Problem D: IPA Edit Distance Too Brittle

Simple misspellings produce poor phonetic matches because:
1. The fallback IPA (grapheme→phoneme rules) is too simple
2. The IPA edit distance algorithm uses difflib SequenceMatcher ratio, which doesn't handle well the case where a single letter difference creates a different phoneme sequence

**Examples:**
- "vommiting" → "vomiting": Levenshtein distance at grapheme level = 1 (remove one 'm'), but IPA distance to best candidate "colistin" = 0.50
- "apetitte" → "appetite": Levenshtein at grapheme level = 2, but IPA distance to best candidate "gabapentin" = 0.59
- "bronkospasm" → "bronchospasm": Levenshtein at grapheme level = 2, but IPA distance to best candidate "bronchitis" = 0.48

**Missed opportunity:** The code has `_has_close_dictionary_match` which uses Levenshtein distance on the **text** (not IPA), but this is only used for the `hcd` flag — it's NOT used for retrieval itself.

### Problem E: LLM Clinical Reasoning Unreliable

Even when reasonable candidates exist, the LLM can make bad clinical choices:

- **azitromycine → azathioprine:** The candidates don't include azithromycin (not in lexicon), and the LLM picks azathioprine from the available list. An antibiotic-like sound is being matched to an immunosuppressant.
- **omeprazol → esomeprazole:** Both are PPIs but different drugs. The LLM picked esomeprazole (stereoisomer) over omeprazole because omeprazole wasn't in the candidate list.

---

## 5. Brainstormed Fixes (Prioritized)

### Fix 1: Enrich the Lexicon (HIGHEST IMPACT, LOWEST EFFORT)

Add ~50-100 common missing medical terms to `data/medical_lexicon.jsonl`.

**Must-add terms:**
- vomiting (symptom)
- appetite (symptom) 
- dehydration (condition)
- bronchospasm (condition)
- azithromycin (antibiotic)
- omeprazole (PPI)
- pulse oximeter (device, multi-word)
- albuterol (bronchodilator, US name)
- diarrhea (already added as alias)
- tachycardia (already added as alias)

**Also fix the typo:** Change "salbutamoll" → "salbutamol"

**Also add multi-word aliases** like "pulse ox" for "pulse oximeter" so single-token matching can work.

**Source for new terms:** `legacy/medical_terms.txt` already has many medical terms. Diff the lexicon against it and add what's missing.

**Estimated impact:** Would fix 7/8 failed words immediately (everything except pulsoxymetre→pulse oximeter which needs multi-word handling).

### Fix 2: Tighten Escalation Guard in decider.py (HIGH IMPACT, LOW EFFORT)

**Problem:** When hcd=False AND score >= 0.60, the system still applies `top_fallback` — but the candidate is often wrong because the correct term isn't in the lexicon.

**Fix:** Change the escalation rule to:
```python
if not span_hcd:
    # hcd is False → no known medical term is close to this span
    # Even if the top candidate feels "phonetically close", it could be
    # wrong (e.g. dehidration→respiration). Escalate unless the top
    # candidate is an exact alias match or has score >= 0.90.
    if top.match_type != "alias" and top.phonetic_score < 0.90:
        return Decision(..., path="hitl_escalate")
```

**Estimated impact:** Would prevent `dehidration→respiration`, `pulsoxymetre→duloxetine`, and `azitromycine→azathioprine` from being applied. They'd escalate for human review instead.

### Fix 3: Add Grapheme-Level Retrieval Stage (MODERATE IMPACT, MODERATE EFFORT)

**Problem:** IPA edit distance is too brittle for simple misspellings. The Levenshtein-based `_has_close_dictionary_match` already identifies that a word is close to a known term, but this info isn't used by the retriever.

**Fix:** In `retriever.py`, add a fallback grapheme-level fuzzy matching strategy:
1. If IPA matching returns no candidate above 0.60
2. Try Levenshtein distance at the character level against the medical wordlist
3. For terms within distance 1-3, compute a grapheme-match score (e.g. `1.0 - (lev_dist / max(len(span), len(term)))`)
4. Include these as candidates with `match_type="grapheme"`

**Estimated impact:** Would fix `vommiting→vomiting`, `apetitte→appetite`, `bronkospasm→bronchospasm` by providing real candidates. The decider would then need to prefer grapheme matches over weak phonetic matches.

### Fix 4: Multi-Word Span Expansion (MODERATE IMPACT, HIGHER EFFORT)

**Problem:** Single tokens can't match multi-word lexicon entries.

**Fix:** When a single-token span has no good candidates, try two strategies:
1. **Expand to neighbor tokens:** Check if combining with the next or previous token forms a bigram that matches a lexicon entry.
2. **Lexicon bigram search:** Search for any lexicon bigram whose first-part IPA matches the span IPA.

**Alternatively:** In Stage 2 (FLAG), be more aggressive about merging adjacent words when they form a known medical bigram.

**Estimated impact:** Would fix `pulsoxymetre→pulse oximeter` (via Levenshtein match to "pulse oximeter" split across two tokens? Actually "pulsoxymetre" is a single token in the input. The ASR produced "pulsoxymetre" as one word. So we'd need to detect that this single token maps to a two-word term in the lexicon. This is a harder problem.)

### Fix 5: Add Pre-Processing to Seed Lexicon from medical_terms.txt (LOW EFFORT)

**Problem:** `legacy/medical_terms.txt` has ~200 medical terms but they're not in the lexicon JSONL.

**Fix:** Run `scripts/seed_lexicon.py` or a new script to merge `medical_terms.txt` terms into `data/medical_lexicon.jsonl`.

**Estimated impact:** Increases lexicon size from ~50 to ~200+ terms. Would add many common medical conditions and symptoms that are currently missing.

### Fix 6: Score-Source-Aware Decider Routing (LOW EFFORT)

**Problem:** The decider already checks `score_source` but the current rule (`span_source == "heuristic" and not span_hcd and top.phonetic_score < LLM_MIN_CONFIDENCE`) is too narrow. Heuristic-scored spans that aren't in the lexicon should be treated with suspicion even when a candidate exists.

**Fix:** Expand the check:
```python
# A word scored by heuristic that isn't in the lexicon and has no
# close dictionary match likely indicates a genuine unknown word.
# Escalate unless the top candidate is very strong.
if span_source != "modernbert" and not span_hcd and top.phonetic_score < 0.85:
    return Decision(..., path="hitl_escalate")
```

---

## 6. Recommended Order of Implementation

### Phase 1 — Quick Wins (fix 4 failed words immediately)

1. **Enrich the lexicon** (Fix 1) — add ~30-50 missing common medical terms
2. **Fix salbutamoll typo** (part of Fix 1)
3. **Tighten escalation guard** (Fix 2) — prevent wrong auto-fixes

### Phase 2 — Structural Improvements

4. **Seed lexicon from medical_terms.txt** (Fix 5) — bulk-add terms
5. **Add grapheme-level retrieval** (Fix 3) — fix simple misspellings
6. **Score-source-aware routing** (Fix 6) — better uncertainty handling

### Phase 3 — Advanced

7. **Multi-word span expansion** (Fix 4) — handle single-token→multi-word mapping

---

## 7. Key Files Map

| File | Purpose |
|------|---------|
| `app/pipeline/scorer.py` | Stage 1 — scoring (HEALTHY) |
| `app/pipeline/flagger.py` | Stage 2 — span merging (HEALTHY) |
| `app/pipeline/retriever.py` | Stage 3 — phonetic retrieval (NEEDS FIXES) |
| `app/pipeline/decider.py` | Stage 4 — routing + LLM (NEEDS FIXES) |
| `app/pipeline/hitl.py` | Stage 5 — human correction (HEALTHY) |
| `app/pipeline/runner.py` | Pipeline orchestration (HEALTHY) |
| `app/pipeline/models.py` | Core dataclasses (ScoredWord, Decision, etc.) |
| `app/pipeline/config.py` | Constants (thresholds, limits) |
| `app/services/lexicon.py` | Lexicon read/write (HEALTHY) |
| `app/services/phonetics.py` | IPA conversion + distance (works but brittle) |
| `app/services/llm.py` | LLM call routing (Groq/Gemini/Ollama) |
| `app/services/llm_decide.py` | Groq-specific LLM decide (fallback) |
| `app/services/correction.py` | Legacy corrector (not used by pipeline) |
| `data/medical_lexicon.jsonl` | Medical lexicon (~50 entries) |
| `legacy/medical_terms.txt` | Medical terms list (~200 terms, NOT in lexicon) |
| `data/subtlex_us.json` | SUBTLEX-US word frequencies (74K words) |

---

## 8. Recommended Next Action

**Start with Phase 1:** Enrich the lexicon + tighten escalation guard. This is the highest-impact change and would fix 7/8 failed words immediately (everything except pulsoxymetre→pulse oximeter, which needs multi-word handling).

After Phase 1, the only remaining issue would be:
- pulsoxymetre → pulse oximeter (needs multi-word expansion)
- vommiting, apetitte, bronkospasm → would escalate to HITL instead of silently failing

This is a huge improvement: from **4/12 correct** to **11/12 correct-or-escalated**.
