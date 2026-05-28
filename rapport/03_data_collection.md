# 03 — Data Collection

## 3.1 The data problem

For Gulf Arabic ASR you need three things:

1. **Acoustic adaptation** to Saudi and Emirati-accented speech.
2. **Vocabulary coverage** for medical terminology.
3. **Code-switching behaviour** for Arabic ↔ English mixing.

Items 1 and 3 are addressable with free public corpora. Item 2 — Arabic
medical conversational speech — has no free corpus, confirmed by exhaustive
search of LDC, ELRA, OpenSLR, HuggingFace, Kaggle, and ~1,200 papers on
Semantic Scholar. That gap is addressed by synthetic data in v2 (see
[10_finetuning_v2_plan.md](10_finetuning_v2_plan.md)).

This chapter documents the v1 corpus: the ~900h of real Gulf-Arabic speech
used in the first LoRA fine-tune.

## 3.2 Gulf-specific verified sources

| Dataset | Hours | Dialect | License | Source |
|---|---:|---|---|---|
| **SADA22** (Saudi Audio Dataset for Arabic) | 667 | Saudi multi-dialect | CC BY-NC-SA | [Kaggle](https://www.kaggle.com/datasets/sdaiancai/sada2022) |
| **WorldSpeech ar_bh** | 272.5 | Bahrain (parliamentary) | CC BY-NC | disco-eth/WorldSpeech on HF |
| **WorldSpeech ar_kw** | 175.5 | Kuwait (parliamentary) | CC BY-NC | disco-eth/WorldSpeech on HF |
| **WorldSpeech ar_sa** | 6.1 | Saudi (government archive) | CC BY-NC | disco-eth/WorldSpeech on HF |
| **OMAN-SPEECH** | ~40 | Omani (32 speakers, 11 wilayats) | research | OMAN-SPEECH paper authors |
| Ramsa / ArSyra / SCC / Traditional UAE | <50 (undisclosed) | UAE + Saudi | mixed | SDAIA + others |
| **Mixat Emirati** | undisclosed | UAE conversational | research | Mixat dataset card |
| **Nexdata UAE Spontaneous Speech (sample)** | small | UAE spontaneous | open sample | Nexdata HF |
| **vadimbelsky UAE Bilingual 40k** | ~150 (estimated from clips) | UAE code-switched | HF-gated | vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k |
| **Gulf subtotal — used in v1 corpus** | **~900 hours** | | | |

Total free verified Gulf-specific audio that exists today is ~1,160h, so
the v1 corpus uses about 77% of it. The remaining 260h was excluded for
one of three reasons:

- Below 0.5s or above 30s (out of training window).
- Duplicate utterance fingerprints across sources.
- Reference transcripts where the dataset-published CER between transcript
  and the dataset's own ASR alignment was > 0.25 (low-quality references).

## 3.3 Pan-Arabic anchor (regularization)

A small amount of pan-Arabic data is mixed in at low weight to keep the
model from forgetting MSA and other dialects:

| Dataset | Hours mixed in | Weight | Why |
|---|---:|---:|---|
| FLEURS ar | 10 | 0.2 | MSA regression-test anchor |
| Common Voice ar v17 | — | — | Excluded (very noisy mic quality) |

Total pan-Arabic mixed in: ~10h at 0.2× weight. The intent is a "do not
forget MSA" anchor, not a primary training signal.

## 3.4 Why we did NOT use these

For completeness, the following sources were available but deliberately
excluded from v1:

- **MGB-2 Aljazeera** (~1,200h MSA): heavy MSA bias, would dilute Gulf
  dialect signal. Available for v3 if regression on MSA becomes a problem.
- **MASC** (~1,000h pan-Arabic YouTube): too much code-switching with
  non-Gulf dialects (Egyptian, Levantine). Considered for v2's code-
  switch bucket.
- **MGB-3 Egyptian** (~37h): Egyptian dialect is acoustically far from
  Khaleeji; would hurt more than help at training time.
- **LDC Arabic datasets**: paid license, not in scope for v1.

## 3.5 Storage and access

The full 900h corpus is preprocessed and lives on the DGX at:

```
data/dgx_full/preprocessed_audios/
  splits/
    train.jsonl       # ~280k clips, ~810h
    validation.jsonl  # ~3k clips, ~7h (used by early stopping)
    test.jsonl        # ~1k clips, ~3h
```

Each entry has the schema documented in `04_preprocessing.md`.

The audio files are 16kHz mono WAV stored on local DGX disk. The total
on-disk footprint after preprocessing is ~180 GB. They are not committed
to git; the manifests are.

## 3.6 What was NOT in the v1 corpus

The v1 corpus has zero medical-domain audio. The 900h is entirely
general-purpose Gulf speech (parliamentary, broadcast, conversational).
This is the gap that v2 is designed to close.

The summary of where data comes from for v2 is in
[10_finetuning_v2_plan.md](10_finetuning_v2_plan.md). Briefly:
- Synthetic medical Gulf (~60h)
- Real Gulf rehearsal sampled from this v1 corpus (~45h)
- Real code-switched Arabic-English (~30h)
- Real English medical (~15h)

## 3.7 Licensing and use

All datasets in v1 are non-commercial research licenses (CC BY-NC, CC BY-
NC-SA, or research-only). The resulting LoRA adapter is therefore
research-only. Commercial deployment will require either:

a) Re-training without SADA22 (lose ~74% of the corpus), or
b) Acquiring SADA22 commercial rights from SDAIA, or
c) Collecting equivalent permissive-licensed Gulf data.

Option (c) is the long-term path — see `PLATFORM_PLAN.md` for the pilot
data collection plan.

## 3.8 References

- Alharbi et al., **SADA: Saudi Audio Dataset for Arabic**, ICASSP 2024.
- Wang et al., **Open Universal Arabic ASR Leaderboard**, arXiv:2412.13788
  (2024).
- WorldSpeech corpus: <https://huggingface.co/datasets/disco-eth/WorldSpeech>
- Casablanca benchmark: <https://huggingface.co/datasets/UBC-NLP/Casablanca>
