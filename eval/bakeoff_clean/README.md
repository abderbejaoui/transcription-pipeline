# Bake-off clean test set

Total clips: 180  |  Total duration: 27.2 min
**Kept (HIGH + MEDIUM):** 122 clips, 17.7 min

## Tier breakdown

| Tier | Clips | Duration | Description |
|---|---:|---:|---|
| HIGH | 45 | 6.9 min | Both models predict near-identical text, both close to reference. |
| MEDIUM | 77 | 10.8 min | Some disagreement but structurally fine reference. |
| HARD | 26 | 2.2 min | Audio genuinely ambiguous OR reference disagrees with model consensus. |
| REJECT | 32 | 7.3 min | Reference defects (split words, length misalignment, too noisy). |

## Per-source breakdown

| source | HIGH | MEDIUM | HARD | REJECT | kept |
|---|---|---|---|---|---|
| SADA22 | 18 | 30 | 11 | 1 | 48 |
| WorldSpeech-BH | 11 | 14 | 3 | 12 | 25 |
| WorldSpeech-KW | 4 | 21 | 12 | 3 | 25 |
| WorldSpeech-SA | 12 | 12 | 0 | 16 | 24 |

## How to evaluate against this set

```bash
# Score the cached predictions against the cleaned references:
python3 scripts/eval_v2.py --testset eval/bakeoff_clean

# Run a new model on the clean set (DGX):
python -m scripts.bakeoff \
    --eval-dir eval/bakeoff_clean \
    --models qwen3 qwen3_uae
```

## Scoring policy

- Report **CER** (mean over kept clips) as the headline metric. CER
  is robust to dialect spelling variants (هذه / هزه) and minor
  formatting differences (digits vs. spelled-out numbers).
- Report WER as a secondary metric; treat WER differences smaller
  than 3 percentage points as noise.
- Apply `norm_text(s, fold_numbers=True)` to both ref and hyp before
  scoring (see `scripts/eval_v2.py`).
