# 06 — Evaluation

## 6.1 Test sets

Two fixed evaluation sets are pinned for the project. Once built they
must not be modified — that is the only way the WER numbers stay
comparable across model versions.

### `eval/bakeoff_30min/` — general Gulf, ~30 minutes

Pulled from:
- WorldSpeech ar_bh (Bahrain parliamentary)
- WorldSpeech ar_kw (Kuwait parliamentary)
- WorldSpeech ar_sa (Saudi government archive)
- SADA22 (Saudi TV)

Filters at build time:
- Clip duration 3–20 s
- Drop clips where the WorldSpeech-published CER (reference vs the
  dataset's own ASR alignment) is > 0.25.

Manifest schema (see `scripts/build_bakeoff_testset.py`):
```json
{
  "id": "...",
  "category": "gulf_acoustic" | "saudi_tv",
  "duration_s": 6.41,
  "language": "ar",
  "transcript": "...",
  "transcript_normalized": "...",
  "source": "WorldSpeech_ar_kw" | "SADA22" | ...,
  "tags": ["..."]
}
```

### `eval/casablanca_emirati_full/` — UAE conversational, 813 clips

Sourced from `UBC-NLP/Casablanca`, config `UAE`, test split. This is
real spontaneous Emirati Arabic, human-verified by the Casablanca
annotators. It is the same test set used by the Open Universal Arabic
ASR Leaderboard for the Qwen3-ASR-1.7B and Voxtral published numbers.

Why two test sets?
- `bakeoff_30min` measures **general Gulf** WER (regression check
  against published baselines).
- `casablanca_emirati_full` measures **UAE conversational** WER (our
  primary product target).

A medical-specific test set (`eval/medical_transcript_eval.jsonl`)
exists but at v1 time it was too small to produce reliable WER. It is
in active expansion for v2.

## 6.2 Normalizer

All WER and CER numbers use the **Open Universal Arabic ASR
Leaderboard** normalizer from Wang et al. 2024 (arXiv:2412.13788). Steps,
in order:

1. Strip punctuation (Unicode general category P)
2. Strip Tashkeel (diacritics)
3. Map Persian / Urdu letters to Arabic equivalents
   (ك ↔ ک, ي ↔ ی, etc.)
4. Fold hamza variants (أ إ آ → ا)
5. Fold madda
6. Convert Eastern Arabic digits to Western digits

We deliberately **do not**:
- Fold teh-marbuta (ة vs ه matters for medical terms)
- Apply `num2words` (training data has both digits and words)
- Merge compound pairs

This matches the public leaderboard exactly so our numbers are
comparable to Qwen3-ASR / Whisper / MMS published results.

## 6.3 Scoring library

`jiwer 4.0`, corpus-level WER and CER. We use corpus-level (not mean of
per-clip rates) because per-clip averaging gives undue weight to very
short clips that have all-or-nothing error patterns.

## 6.4 Results

### Baseline (zero-shot, no fine-tune)

50-clip Casablanca-UAE smoke test:

| Rank | Model | WER | CER |
|---:|---|---:|---:|
| 1 | Qwen3-ASR-1.7B (base)            | 67.67% | 22.29% |
| 2 | vadimbelsky/qwen3-asr-arabic-uae | 70.85% | 25.58% |
| 3 | Voxtral-Mini-3B                  | 70.85% | 28.43% |

These match the published Casablanca leaderboard within ~3 points (which
is normal for a 50-clip slice vs the full 813), confirming the pipeline
is correct. See `raw_test_results.md` for the full validation table.

### After v1 LoRA

Same test set, after the 900h Gulf-only LoRA from chapter 5:

| Model | WER | CER |
|---|---:|---:|
| Qwen3-ASR-1.7B + Gulf LoRA r=64 (v1) | ~45% | not reported |

That's a ~22 point WER reduction. Confirms dialect adaptation works.
Still well above the 10% target — the residual error is the medical
vocabulary failure mode that v2 is designed to attack.

## 6.5 Failure mode analysis on the v1 model

Post-training error analysis on a 50-clip mixed medical/general slice:

| Category | Failure pattern | Approx % of errors |
|---|---|---:|
| Medical drug names | Drug transliterated to Arabic ("voltaren" → "فولتارين"); occasional total mangle into unrelated words ("voltaren" → "فواد علي النزار") | ~35% |
| Code-switching boundary | Garbage tokens for 2-3 words around an AR↔EN switch | ~20% |
| Dialect contractions | Unmerged contractions: "ما + هو" produced as two words instead of "مهو" | ~15% |
| Diacritic / hamza minor | Off-by-one hamza or teh-marbuta — survives normalizer but inflates raw WER | ~15% |
| Acoustic noise | Background TV / clinic noise causing whole-segment drop | ~10% |
| Other | Numeric reading style, proper nouns, etc. | ~5% |

The top two categories (medical drugs + code-switching boundaries) are
the focus of v2's data composition: synthetic medical + real code-
switched + English medical. See chapter 10.

## 6.6 What the eval pipeline cannot tell us

Three things WER does not measure that matter for the product:

1. **Whether English drug names came out in English**. Our normalizer
   doesn't penalize Arabic-script vs Latin-script differently if they
   normalize to the "same" word, but for the clinical use case the
   Arabic-script version is unusable. This is measured separately by
   the phonetic flagger (chapter 8).
2. **Whether the model hallucinated a plausible-but-wrong drug.** WER
   counts that as 1 error but the clinical impact is far worse than
   1 misspelling.
3. **Latency / streaming quality.** Our eval is offline batch decoding.
   The product target is near-real-time dictation. Latency is measured
   separately by the FastAPI app's timing telemetry (chapter 7).

## 6.7 Files

| File | Purpose |
|---|---|
| `eval/bakeoff_30min/` | general Gulf test set (30 min) |
| `eval/casablanca_emirati_full/` | UAE conversational test set (813 clips) |
| `eval/medical_transcript_eval.jsonl` | small medical eval set (growing) |
| `scripts/bakeoff.py` | runs every backend on every test set, produces report.md |
| `scripts/eval_arabic.py` | direct WER / CER scorer for a single hyp/ref pair |
| `raw_test_results.md` | raw bake-off table for the 50-clip slice |
| `eval/<set>/predictions/<model>.jsonl` | per-model predictions cached for re-scoring |
