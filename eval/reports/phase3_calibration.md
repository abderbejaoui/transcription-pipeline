# Confidence Model Calibration — `phase3_calibration`

## Summary

| Metric | Value |
|--------|-------|
| **Training samples** | 90 |
| **Correct corrections** | 30 (33.3%) |
| **Incorrect corrections** | 60 |
| **AUC (train)** | 0.9636 |
| **Features** | 10 |

### Features Used

- `phonetic_score_norm`: coefficient = 2.5723
- `score_gap_norm`: coefficient = 0.0000
- `llm_confidence`: coefficient = 0.0000
- `n_candidates_norm`: coefficient = 0.0000
- `is_mixed_script`: coefficient = -0.0104
- `span_length_norm`: coefficient = -0.4595
- `is_arabic`: coefficient = -0.2833
- `best_retrieval_score`: coefficient = 0.0000
- `n_tokens_norm`: coefficient = -0.6823
- `is_multi_word`: coefficient = -2.5925

### Breakdown by Language

| Language | Samples | Correct | Accuracy |
|----------|---------|---------|----------|
| ar | 36 | 6 | 16.7% |
| en | 44 | 17 | 38.6% |
| mixed | 10 | 7 | 70.0% |

## Thresholds & Metrics

| Threshold Name | Value | Precision | Recall | F1 | Applied/Total |
|----------------|-------|-----------|--------|-----|--------------|
| auto_apply | 0.808 | 0.9630 | 0.8667 | 0.9123 | 27/90 |
| hitl | 0.386 | 0.8438 | 0.9000 | 0.8710 | 32/90 |
| permissive | 0.500 | 0.9630 | 0.8667 | 0.9123 | 27/90 |

## Recommended Operating Point

- **Auto-apply threshold**: 0.808 (P(correct) >= this → apply silently)
- **HITL threshold**: 0.386 (P(correct) >= this → apply but flag for review)
- **Below HITL**: leave span unchanged, flag for human correction

## Feature Coefficients

| Feature | Coefficient | Direction |
|---------|-------------|-----------|
| phonetic_score_norm | 2.5723 | positive |
| score_gap_norm | 0.0000 | negative |
| llm_confidence | 0.0000 | negative |
| n_candidates_norm | 0.0000 | negative |
| is_mixed_script | -0.0104 | negative |
| span_length_norm | -0.4595 | negative |
| is_arabic | -0.2833 | negative |
| best_retrieval_score | 0.0000 | negative |
| n_tokens_norm | -0.6823 | negative |
| is_multi_word | -2.5925 | negative |