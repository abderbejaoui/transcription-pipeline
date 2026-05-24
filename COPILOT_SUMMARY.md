# Copilot Progress Summary

Date: 2026-05-24

## Current status

I am in the final integration/validation phase of the pipeline, centered on Stage 4 (DECIDE) and the Stage 2 span-boundary cleanup that affects the canonical transcript.

### Stage position
- Current stage focus: Stage 4 validation, with a Stage 2 merge issue still affecting exact canonical spans.
- Rough completion:
  - Code structure and main pipeline pieces: about 90% complete.
  - Spec-verified end-to-end behavior against `DESIRED_PIPELINE.md`: about 70% complete.
  - Final remaining work is mostly integration correctness, not missing architecture.

### Will I finish it?
- Yes. I am still actively working toward a complete finish.
- The only reason the canonical path is not fully green yet is an external Gemini quota block plus a span-merging mismatch that needs one more tightening pass.

## What has been implemented so far

The contract-driven pipeline exists under `app/pipeline/`:

- `app/pipeline/models.py` with the core dataclasses: `ScoredWord`, `SuspiciousSpan`, `Candidate`, `SpanWithCandidates`, `Decision`, and `PipelineResult`.
- `app/pipeline/scorer.py` for Stage 1 scoring.
- `app/pipeline/flagger.py` for Stage 2 span merging.
- `app/pipeline/retriever.py` for Stage 3 phonetic retrieval.
- `app/pipeline/decider.py` for Stage 4 routing and LLM-gated choice validation.
- `app/pipeline/hitl.py` for Stage 5 human correction and write-through lexicon logging.
- `app/pipeline/runner.py` to orchestrate the pipeline end to end.
- `app/pipeline/config.py` for the named constants required by the spec.

Supporting services and tools are also in place:

- `app/services/lexicon.py` now supports the new JSONL lexicon shape while remaining compatible with the legacy format.
- `app/services/phonetics.py` provides IPA conversion and IPA distance helpers, plus a safe fallback-only IPA path for offline/blocked environments.
- `app/services/llm.py` makes constrained Gemini calls for Stage 4 and validates the returned choice against the candidate list.
- `scripts/seed_lexicon.py` exists to backfill missing IPA values in `data/medical_lexicon.jsonl`.
- `main.py` exists as the CLI entry point.

I also added focused helper/test scripts during debugging:

- `scripts/run_quick_pipeline.py` for faster heuristic runs.
- `scripts/run_canonical_pipeline.py` for a deterministic canonical transcript run with injected Stage 1 scores.
- `scripts/run_quick_with_gemini.py` to load `.env` and run the quick pipeline with Gemini settings in-process.
- `scripts/test_gemini_decide.py` to call Stage 4 directly and print the raw Gemini response.
- `scripts/check_gpu.py` to verify CUDA / GPU availability.
- `scripts/generate_canonical_output.py` to print the expected canonical corrected transcript.

## What is verified

Verified in the workspace venv:

- `torch.cuda.is_available()` is `True`.
- CUDA tensor allocation succeeds on `cuda:0`.
- The four required lexicon entries exist in `data/medical_lexicon.jsonl` with real IPA values:
  - Doliprane
  - Salbutamol
  - sphygmomanometer
  - amoxicillin
- The canonical expected corrected transcript matches `DESIRED_PIPELINE.md`.
- The live Gemini request path is reachable, but the current API key/project is quota-limited.

## Current blockers / problems

### 1) Gemini quota block
- The live Gemini test returns HTTP 429 with `RESOURCE_EXHAUSTED`.
- The response body says the project/key has zero free-tier quota for `gemini-2.0-flash`.
- Result: Stage 4 cannot use Gemini successfully in this environment right now.

### 2) Stage 2 span merging still needs tightening
- In deterministic canonical runs, Stage 2 can over-merge suspicious words into spans that are larger than the spec’s expected four spans.
- This is why the canonical pipeline still needs one more cleanup pass before it naturally matches the exact Stage 2 contract.

### 3) Offline/blocked fallback path is present, but not the spec’s live Gemini path
- I added a deterministic fallback so the canonical transcript can still be handled offline.
- That helps keep progress moving, but it does not satisfy the spec’s intended live Gemini behavior for Stage 4.

## What I learned from the live Gemini test

- The prompt structure is not the main issue.
- The API request is failing before a model answer is returned.
- The failure is quota/billing related, not a parser bug.
- The validation logic correctly rejects the upstream failure and does not accept hallucinated output.

## Plan to finish the pipeline

1. Tighten Stage 2 so the canonical transcript produces exactly these spans:
   - `dolly prahn`
   - `salbu tamol`
   - `sfigmomanometre`
   - `amoxicilin`
2. Decide how Stage 4 should behave in this workspace:
   - preferred: use a Gemini key/project with usable quota and keep the live DECIDE path,
   - fallback: keep the deterministic offline path for canonical validation.
3. Re-run the canonical pipeline and verify the corrected transcript matches `DESIRED_PIPELINE.md` exactly.
4. If needed, add the GPU device log in `app/pipeline/scorer.py` and run the main CLI with GPU monitoring to prove the Stage 1 path is actually on CUDA.

## Questions I need answered

1. Do you want me to keep the live Gemini requirement as the final target, or should I prioritize a deterministic offline fallback for this workspace?
2. Can you provide a Gemini project/key with quota or billing enabled if you want Stage 4 to run live?
3. Should I spend the next pass on fixing Stage 2 span merging exactly to the spec, or on wiring a canonical-test-only override so the transcript resolves immediately?
4. Do you want me to add the Stage 1 GPU device log now, or leave that for after the canonical transcript is fixed?

## Action log (append-only)

- 2026-05-24: Verified CUDA works in the venv (`torch.cuda.is_available() == True`) and a test tensor allocates on `cuda:0`.
- 2026-05-24: Added safe fallback IPA helpers and lexicon loading improvements to reduce blocking phonemizer/espeak calls during tests.
- 2026-05-24: Added deterministic canonical-run helpers and a direct Gemini decision test harness.
- 2026-05-24: Confirmed live Gemini requests return HTTP 429 `RESOURCE_EXHAUSTED` for the current `.env` key/project.
- 2026-05-24: The canonical corrected transcript expected by `DESIRED_PIPELINE.md` is known and matches the required output.

If you prefer a different format for this summary, I can switch it to JSONL or a shorter changelog-style note.
