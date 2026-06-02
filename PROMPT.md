# PROMPT.md — Implementation Instructions for the Correction Pipeline Rework

You are a coding agent working in the `transcription-pipeline` repo (branch `arabic-pipeline-improvement`). This document is your spec. Follow it phase by phase. Read `SUMMARY.md` and `PFA.md` for the current state, but understand that **this document supersedes the current architecture's design philosophy** — your job is to migrate from a hand-tuned threshold cascade to a measured, mostly-learned pipeline.

---

## Mission

Post-ASR correction of Gulf-Arabic medical transcripts. Input is raw ASR text (later: ASR N-best + confidences). Output is corrected text plus a HITL queue for low-confidence spans. Three error classes:

1. English medical misspellings (`hyperglacymia` → `hyperglycemia`)
2. Arabic phonetic misspellings (`سداع` → `صداع`)
3. Arabic→English transliterations (`هستوري` → `history`, `بلاد شوجر` → `blood sugar`)

---

## Non-Negotiable Principles

Read these before every phase. They override convenience.

1. **Do no harm > recall.** This is clinical text. Corrupting an already-correct word (especially a drug/disease name) is far worse than missing an error. Every change must be defensible. When in doubt, flag for HITL — do not auto-apply.
2. **Measure before and after every change.** No change to correction logic may be merged without running the eval harness (Phase 0) and reporting the before/after numbers. If a change does not improve the metrics on the held-out set, do not keep it.
3. **Kill magic numbers.** The current pipeline has ~40 hand-tuned thresholds (see `PFA.md` "Key Thresholds Reference Table"). Do not add more. Replace guessed weights/thresholds with either (a) values learned from labeled data, or (b) well-calibrated signals (ASR confidence, model probabilities). Every remaining threshold must have a one-line comment justifying its value from data.
4. **Don't break the existing test suite.** 151 tests currently pass. Keep them green or update them deliberately with a stated reason. Run `pytest tests/` before and after each phase.
5. **Constrain generation to the lexicon for drug/disease slots.** A corrector must never invent a medical term that is not in `data/medical_lexicon.jsonl` (or explicitly added via HITL). Hallucination of plausible-but-wrong drug names is the top risk.
6. **Never let the corrector rewrite normal Arabic.** Arabic-script words that are ordinary clinical language must pass through untouched unless there is strong evidence they are a transliteration or misspelling.

---

## Phase 0 — Evaluation Harness (DO THIS FIRST, blocks everything else)

The single highest-leverage task. Nothing else may be merged until this exists.

**Tasks:**
- Create `eval/correction_eval.jsonl` — a held-out labeled set of `{"raw": "...", "gold": "...", "lang": "ar|en|mixed", "notes": "..."}` records. Seed it from `data/user_corrections.jsonl` and the test cases in `SUMMARY.md`. Target ≥ 200 records; if real data is scarce, generate synthetic ones covering all three error classes AND a large block of **clean inputs that must not change** (this measures "do no harm").
- Create `scripts/eval_correction.py` that runs any pipeline version over the eval set and reports:
  - **WER reduction** — WER(raw, gold) vs WER(corrected, gold), using `jiwer` (already a dependency). Report both Arabic and English subsets separately.
  - **Correction precision** — of the spans the pipeline changed, fraction that match gold.
  - **Correction recall** — of the spans that should change, fraction the pipeline fixed.
  - **Do-no-harm rate** — fraction of clean inputs left unchanged (target: ≈100%).
  - **HITL volume** — how many spans were flagged rather than auto-applied.
- Output a markdown report to `eval/reports/<timestamp>.md` and print a one-line summary.
- Wire it so it can target the current pipeline (`MedicalCorrector.correct_transcript`) as the baseline.

**Acceptance criteria:**
- `python -m scripts.eval_correction` runs end-to-end and produces a report.
- A baseline report exists at `eval/reports/baseline.md` documenting the current pipeline's numbers. **All future phases compare against this.**

---

## Phase 1 — Unblock and Stabilize

Resolve the contradictions in the current build before adding anything.

**Tasks:**
- **Resolve the `transformers` version conflict.** `requirements.txt` pins `4.57.6` (needed by the Wav2Vec2/ASR wrappers per `CLAUDE.md`), but `llm_corrector.py` and `finetune_llm.py` need `AutoModelForCausalLM` from 4.58+. Investigate and pick ONE resolution: confirm whether `AutoModelForCausalLM` actually is missing in 4.57.6 (it likely is not — verify before upgrading), or upgrade and re-verify the ASR imports still work. Document the outcome in `CLAUDE.md`.
- Add a **global do-no-harm guardrail** in the correction entry point: if a proposed change is below the auto-apply confidence, it goes to the HITL queue, never to the output text. Make the auto-apply path explicit and auditable.
- Fix the **"resting" → "lying" false positive** and the **Arabic over-correction** issues (SUMMARY Known Issues #2, #3) — but fix them by tightening the *evidence required to change a word*, not by adding word-specific patches. Confirm the fix via the Phase 0 harness (do-no-harm rate must go up, WER must not drop).
- Fix **auto-corrections not being logged** (Known Issue #5) — every applied change must appear in the response's corrections array. This is required for the HITL flywheel.

**Acceptance criteria:** All known issues #1–#5 resolved or explicitly deferred with reason. Eval report shows do-no-harm rate improved with no WER regression.

---

## Phase 2 — Principled Matching (replace the skeleton heuristics)

Replace hand-rolled consonant-skeleton + digraph rules with grounded, learned, or phonetically-principled components. Build these as new modules; keep the old ones until the eval proves the replacement wins, then retire them in Phase 5.

**2a. Lexicon-constrained retrieval grounding.**
- Build a retrieval layer over `data/medical_lexicon.jsonl` (terms + aliases) that, given a suspicious span, returns ranked candidate canonical terms. Reuse `vector_lexicon.py` if adequate; otherwise build a cleaner version.
- This is the candidate generator for everything downstream. The corrector chooses from these candidates rather than generating free text for drug/disease slots.
- **Retrieval MUST be phonetic/surface-based, NOT semantic.** ASR errors are acoustic, not semantic — the right signal is "sounds/looks like," not "means the same as." Do NOT index the lexicon with semantic sentence embeddings (CamelBERT/LaBSE-style). Semantic retrieval matches translations (`قلب` → `cardiac`) instead of transliterations (`هستوري` → `history`), which is the wrong target. **This repo already built a LaBSE `EmbeddingMatcher` and DISABLED it for exactly this reason (see `PFA.md` §B4.2) — do not reintroduce it.** Use phonetic keys / character n-grams / phoneme-space vectors instead.
- Each lexicon term should carry multiple retrieval keys: canonical English, phonetic/IPA form, and known ASR-misspelling variants. Seed these keys from the existing `_PHONETIC_ALIAS` dict and filler/misspelling maps — those hand-built rules are valuable bootstrap data for the index, not throwaway.

**2b. Phoneme-space transliteration matching.**
- Replace the consonant-skeleton + `_AR2LAT` + digraph rules (`gh→g, sh→s`, etc.) with **IPA/phoneme-space matching**. Map both the Arabic span and the English lexicon terms to a phonetic representation (use `epitran` or `phonemizer`; add to `requirements.txt`) and score similarity in phoneme space.
- This subsumes the `_PHONETIC_ALIAS` dictionary and the skeleton logic with one principled representation. Validate against the transliteration subset of the eval set.

**2c. Arabic spelling correction.**
- Replace the `_ARABIC_MISSPELLING` map + brute-force single-substitution generation with **weighted Damerau-Levenshtein** over the Arabic medical vocabulary, where the substitution cost matrix encodes the Gulf phonetic mergers already in `_ARABIC_MERGER` (cheap س↔ص↔ث, ط↔ت, etc.; expensive unrelated pairs). Use SymSpell-style indexing for speed.
- **Compute the edit distance on the original Arabic script, not on a Latin transliteration.** Group merge-equivalent letters (س/ص/ث → one pseudo-character, د/ض/ظ/ذ/ز → one, etc.) so phonetic neighbors are cheap, but do NOT round-trip through `_AR2LAT` first — transliteration loses information (e.g. it can collapse `وايت` to a bare `t`). Keep the script, fold only the phonetic classes.
- Optional stretch: prototype a character-level seq2seq (ByT5 / AraT5) corrector and compare against the weighted-edit-distance approach on the eval set. Keep whichever wins.

**2d. Normalization via CAMeL Tools.**
- Replace hand-rolled Arabic normalization (alef/ya/ta-marbuta unification, dediacritization, clitic stripping) with `camel-tools` where it covers the same ground. Add to `requirements.txt`. Keep a thin fallback if the dependency is unavailable.

**2e. Script-aware segmentation (runs BEFORE 2a–2c).**
- Fix the wide-span problem (SUMMARY Known Issue #3: spans bleed across words and translate normal Arabic) by **segmenting before correcting, not during.** Detect Arabic-script vs Latin-script blocks and treat script switches as hard candidate boundaries. Also treat unambiguous Arabic function words (في، من، مع، و، بسبب، الخاصة …) as hard boundaries — they are almost never part of a medical term.
- A correction span may never cross a hard boundary. This structurally prevents an English drug name from absorbing the adjacent Arabic word, instead of relying on the current tangle of span-boundary thresholds.

**Acceptance criteria:** Each new component beats its predecessor on the relevant eval subset. No net WER regression. Segmentation provably reduces cross-boundary corrections (add a targeted eval case). Phoneme matcher and weighted-edit corrector are unit-tested.

---

## Phase 3 — LLM as Candidate Selector (judge, not free generator)

Make a learned model the decision-maker, grounded by Phase 2 retrieval. **The default auto-apply path uses the LLM as a constrained selector among retrieved candidates — NOT as a free-form generator that rewrites the transcript.** Rationale: a selector cannot hallucinate a drug name that isn't a candidate (the top clinical risk), it only fires on the ~5–10% of spans that are ambiguous, and its output distribution gives a calibrated confidence for free. A generator rewriting whole transcripts is higher-latency, harder to evaluate, and unsafe for clinical text. Reserve free generation (if used at all) for HITL-flagged spans a human will review.

**Tasks:**
- Define the corrector interface: input = `{context, flagged_span, lexicon_candidates[], optional asr_nbest[], optional word_confidences[]}`, output = `{choice ∈ candidates | "NO_CHANGE", confidence, reason}`.
- **Selection prompt:** present the surrounding context, the span, and the retrieved candidates as a constrained choice (e.g. "A) cand1  B) cand2  C) NO_CHANGE — reply with the letter only"). Derive confidence from the model's probability over the choice tokens, not a self-reported number.
- Build training-pair generation from `data/user_corrections.jsonl` and the eval set's training split (NEVER train on the held-out eval split — enforce the split). Train the model on the *selection* objective, keeping the existing LoRA + 4-bit setup in `scripts/finetune_llm.py`.
- **Calibrated confidence model (replaces hard thresholds):** fit a small logistic regression that maps features → P(correction is right). Features: phonetic/retrieval distance to the best candidate, the *gap* between best and second-best candidate, the LLM selection probability, and whether the span is mixed-script. Choose the auto-apply / HITL / leave-as-is cut points **on the eval set's precision/do-no-harm curve** — this is what kills the guessed 0.85/0.60/0.90/88.0 thresholds, replacing them with a learned operating point.
  - `P > ~0.9` → auto-apply; mid-band → apply but flag for audit; low → leave as-is and flag for human correction. (Exact cuts are chosen from data, not fixed here.)

**Acceptance criteria:** Selector + retrieval beats the rule cascade on overall WER reduction AND correction precision on the eval set, at an equal-or-better do-no-harm rate. The auto-apply path provably cannot emit a non-candidate medical term. Document the chosen confidence operating point and why.

---

## Phase 4 — HITL Data Flywheel

Turn the in-memory feedback hack into the project's backbone.

**Tasks:**
- Persist every human correction (currently `_CORRECTION_FEEDBACK` is an in-memory dict lost on restart) to `data/user_corrections.jsonl` as a structured training pair, deduplicated.
- **Capture rich context, not just `(wrong, right)`** — store the surrounding sentence (±5 words), the audio timestamp (from the ASR alignment, when available), the phonetic form of the wrong input, and the candidate list that was shown. This turns HITL from a review log into an active-learning dataset, and lets you cluster recurring ASR error patterns (e.g. ASR repeatedly writing `التهب` for `التهاب`) and add them as new retrieval-index variant keys with no code change.
- On `/api/learn_from_edit` and `/api/teach`: append the pair, update the lexicon/retrieval index, and invalidate caches (the existing cache-invalidation hooks).
- **Synthetic augmentation:** when a new term enters the lexicon, use the local LLM to generate plausible Gulf-ASR mishearings of it and add them as retrieval variant keys — expands coverage without manual labeling. Validate that augmentation doesn't raise the do-no-harm false-positive rate on the eval set.
- Add `scripts/retrain_from_feedback.py` (or extend `finetune_llm.py`) to periodically fine-tune on accumulated pairs, and a way to evaluate the new adapter against `eval/reports/baseline.md` before promoting it. **Train on human-reviewed pairs, not on the rule pipeline's own auto-corrections** — the latter just teaches the model to imitate the rules (including their errors).
- Produce a "learning curve" report: pipeline metrics as a function of accumulated HITL pairs. This is the strongest demonstrable result for the project.

**Acceptance criteria:** A correction made in the UI survives a server restart and measurably influences future corrections. The retrain script produces an adapter and an eval comparison.

---

## Phase 5 — Retire the Threshold Jungle & ASR Interface

Only after Phases 0–4 prove the new components win.

**Tasks:**
- Delete or quarantine the superseded rule-cascade code paths (skeleton matchers, `_PHONETIC_ALIAS`, brute-force substitution, multiplicative LLM gating with guessed weights). Remove the corresponding magic numbers. Keep only what the eval shows is still pulling weight.
- For any signal still fused in Stage A suspicion scoring, **learn the weights** (e.g. logistic regression on the labeled set) instead of the guessed 0.30/0.35/0.20/0.15. Or drop the explicit suspicion gate entirely if the post-editor's per-span confidence does the job better — test both on the eval set.
- **Define and document the ASR interface** the friends' model should expose: 1-best text, N-best hypotheses (or lattice), and word-level confidences. Make the pipeline ingest N-best and confidences when available and degrade gracefully to 1-best when not. Document this contract in `CLAUDE.md` so the ASR team can build to it. (N-best rescoring is the single biggest accuracy lever — the right word is often in hypothesis #2.)

**Acceptance criteria:** Net lines-of-code and threshold count go DOWN. Eval metrics stay equal or better. The ASR interface contract is documented.

---

## Working Rules for the Agent

- Work one phase at a time. After each phase: run `pytest tests/`, run `scripts/eval_correction.py`, and report the before/after metric delta in your summary.
- Add tests for every new module under `tests/`, matching the existing style.
- Prefer extending existing modules over parallel rewrites; but when replacing a heuristic, build the replacement alongside, prove it on the eval set, then delete the old one (Phase 5) — don't delete-then-build.
- If a dependency won't install (e.g. FAISS on Python 3.14, per Known Issue #4), make the feature degrade gracefully and note it; don't block the pipeline.
- When you hit a real fork in the road (which Arabic spelling approach wins, whether to keep the suspicion gate), let the **eval numbers decide** and report them — don't guess.
- **Do not rebuild the UI.** A working FastAPI + JS interface already exists (`app/main.py`, `app/static/`). Extend it for HITL feedback capture — do not rewrite it in Gradio/Streamlit. That's motion, not progress.
- **Prefer phonetic/surface similarity over semantic similarity** anywhere you retrieve or match. Semantic embeddings are the wrong signal for acoustic errors (see Principle on §2a and the disabled `EmbeddingMatcher`).
- Keep `SUMMARY.md` and `CLAUDE.md` updated as the architecture changes.

## Definition of Done

The pipeline is a measured, retrieval-grounded, learned post-editor with a persistent HITL training flywheel; the threshold jungle is gone; every correction decision is justified by data from the eval harness; and the project can show a learning curve of accuracy improving as HITL data accumulates — all while never corrupting clean clinical text.
