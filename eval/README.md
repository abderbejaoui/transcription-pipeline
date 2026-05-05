# Medical Transcript Eval Set

This folder contains a small gold-labeled evaluation set for testing an LLM that detects and corrects suspicious medical words in noisy transcripts.

## Files

- `medical_transcript_eval.jsonl`: one example per line

## Schema

Each JSON object has:

- `id`: unique example ID
- `split`: `dev` or `test`
- `difficulty`: rough difficulty label
- `contains_error`: whether the transcript actually contains a correction target
- `transcript`: raw transcript text
- `gold_spans`: expected suspicious spans

Each item in `gold_spans` has:

- `original_text`: the wrong text span expected to be flagged
- `possible_correction`: the intended correction
- `issue_type`: one of:
  - `single_word_misspelling`
  - `split_phrase_should_merge`
  - `wrong_medical_term`

## Recommended evaluation

For each example, score the model on:

1. `Detection`: did it find each gold span?
2. `Boundary`: did it capture the full bad span, not just part of it?
3. `Correction`: did it return the exact intended correction?
4. `False positives`: on examples with `gold_spans: []`, did it avoid changing already-correct text?

## Suggested usage

- Use `dev` while tuning prompts.
- Use `test` only after you think the prompt is stable.
- Keep temperature low for extraction tasks.

## Notes

- The set intentionally mixes:
  - single-word misspellings
  - fragmented spans that should merge into one term
  - normalization-style cases
  - hard negative examples with difficult but already-correct medical terms
- This is a starter benchmark, not a final production benchmark. The next upgrade should be real ASR output from your own data.
