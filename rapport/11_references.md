# 11 — References

## Papers cited in this report

### Speech recognition foundations

- Radford et al. **Robust Speech Recognition via Large-Scale Weak
  Supervision** (Whisper). arXiv:2212.04356, 2022.
- Pratap et al. **Scaling Speech Technology to 1,000+ Languages**
  (MMS). arXiv:2305.13516, 2023.
- Yang et al. **Qwen2.5-Audio Technical Report**. arXiv:2407.10759,
  2024.

### Arabic ASR specifically

- Alharbi et al. **SADA: Saudi Audio Dataset for Arabic**. ICASSP
  2024. <https://www.semanticscholar.org/paper/SADA%3A-Saudi-Audio-Dataset-for-Arabic-Alharbi-Alowisheq/de2508f2d48ea42653fe11011f24f9f227d38e71>
- Wang et al. **Open Universal Arabic ASR Leaderboard**.
  arXiv:2412.13788, 2024.
  Public leaderboard hosted at
  <https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard>.
- Diab et al. **Casablanca: Data and Models for Multidialectal
  Arabic Speech Recognition** (UBC-NLP). 2024.
  Dataset: <https://huggingface.co/datasets/UBC-NLP/Casablanca>.

### LoRA and parameter-efficient fine-tuning

- Hu et al. **LoRA: Low-Rank Adaptation of Large Language Models**.
  arXiv:2106.09685, 2021.
- Kalajdzievski et al. **A Rank Stabilization Scaling Factor for
  Fine-Tuning with LoRA** (rsLoRA). arXiv:2312.03732, 2023.
- Mangrulkar et al. **PEFT: Parameter-Efficient Fine-Tuning of
  Billion-Scale Models on Low-Resource Hardware**. HuggingFace
  blog, 2023.
  <https://huggingface.co/blog/peft>

### Optimization and training schedule

- Loshchilov & Hutter. **Decoupled Weight Decay Regularization**
  (AdamW). arXiv:1711.05101, 2019.
- Loshchilov & Hutter. **SGDR: Stochastic Gradient Descent with
  Warm Restarts** (cosine schedule). arXiv:1608.03983, 2017.

### Continual fine-tuning and rehearsal

- Robins. **Catastrophic forgetting, rehearsal and pseudorehearsal**.
  Connection Science 7(2), 1995.
- Scialom et al. **Continual Learning with Foundation Models: An
  Empirical Study of Latent Replay**. NeurIPS Workshop, 2022. The
  20–40% rehearsal fraction rule of thumb originates in this
  literature.

### Forced alignment

- Chen et al. **CTC-based Forced Alignment with MMS** (the
  alignment approach used in `app/services/alignment_v2.py`).
  Library: <https://github.com/MahmoudAshraf97/ctc-forced-aligner>.

### Phonetic similarity

- Jaro. **Advances in record-linkage methodology as applied to
  matching the 1985 census of Tampa, Florida**. JASA 84(406), 1989.
- Winkler. **String Comparator Metrics and Enhanced Decision Rules
  in the Fellegi-Sunter Model of Record Linkage**. Proceedings of
  the Section on Survey Research Methods, 1990.
- Knuth. **The Art of Computer Programming, Vol. 3**, on Soundex
  / Metaphone phonetic indexing.

### Synthetic speech for training data

- Rosenberg et al. **Speech Recognition with Augmented Synthesized
  Speech**. ASRU 2019.
- Du et al. **Synthetic Data for ASR Training: A Survey**.
  arXiv:2310.15264, 2023.

### Code-switching ASR

- Sitaram et al. **A Survey of Code-Switched Speech and Language
  Processing**. arXiv:1904.00784, 2019.

## Datasets

### Used in v1 (~900h Gulf Arabic)

| Dataset | URL |
|---|---|
| SADA22 | <https://www.kaggle.com/datasets/sdaiancai/sada2022> |
| WorldSpeech (ar_bh, ar_kw, ar_sa) | <https://huggingface.co/datasets/disco-eth/WorldSpeech> |
| OMAN-SPEECH | OMAN-SPEECH paper, research distribution |
| Mixat Emirati | research distribution |
| Nexdata UAE Spontaneous Speech (sample) | <https://huggingface.co/datasets/Nexdata/UAE-Arabic-Spontaneous-Speech-Data> |
| vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k | <https://huggingface.co/datasets/vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k> |
| Saudilang Code-Switch Corpus | <https://www.kaggle.com/datasets/sdaiancai/saudilang-code-switch-corpus-scc> |

### Eval sets

| Test set | URL |
|---|---|
| Casablanca multilingual benchmark | <https://huggingface.co/datasets/UBC-NLP/Casablanca> |

### Used in v2 (synthetic + rehearsal + code-switched + English medical)

| Dataset | URL |
|---|---|
| MASC (Massive Arabic Speech Corpus) | <https://huggingface.co/datasets/pain/MASC> |
| PriMock57 (mock primary-care consultations) | <https://github.com/babylonhealth/primock57> |
| Common Voice English | <https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0> |

### Lexicon sources

| Source | URL |
|---|---|
| NLM RxNav (RxNorm ingredients and brand names) | <https://rxnav.nlm.nih.gov/REST/allconcepts.json> |
| NLM Clinical Tables ICD-10-CM | <https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search> |

## Models

| Model | URL |
|---|---|
| Qwen3-ASR-1.7B | <https://huggingface.co/Qwen/Qwen3-ASR-1.7B> |
| Qwen2.5-Audio-7B-Instruct | <https://huggingface.co/Qwen/Qwen2.5-Audio-7B-Instruct> |
| Whisper-large-v3 | <https://huggingface.co/openai/whisper-large-v3> |
| MMS-1B | <https://huggingface.co/facebook/mms-1b-all> |
| Voxtral-Mini-3B-2507 | <https://huggingface.co/mistralai/Voxtral-Mini-3B-2507> |
| VibeVoice-ASR | <https://huggingface.co/microsoft/VibeVoice-ASR> |
| VoxCPM2 (TTS for synthetic data) | <https://huggingface.co/openbmb/VoxCPM2> |
| Calme-3.2-Instruct-78B | <https://huggingface.co/bartowski/calme-3.2-instruct-78b-GGUF> |
| ctc-forced-aligner (MMS-based) | <https://github.com/MahmoudAshraf97/ctc-forced-aligner> |

## Internal repo cross-references

| Document | Path |
|---|---|
| Project plan / progress | [../PROGRESS.md](../PROGRESS.md) |
| Dataset catalog | [../DATASETS.md](../DATASETS.md) |
| DGX data pipeline | [../DGX_DATA_PIPELINE.md](../DGX_DATA_PIPELINE.md) |
| Fine-tune runbook (v1) | [../FINETUNE_RUNBOOK.md](../FINETUNE_RUNBOOK.md) |
| Qwen3 finetune doc (v1) | [../QWEN3_FINETUNING_DOCUMENTATION.md](../QWEN3_FINETUNING_DOCUMENTATION.md) |
| Training data plan | [../TRAINING_DATA.md](../TRAINING_DATA.md) |
| Platform plan | [../PLATFORM_PLAN.md](../PLATFORM_PLAN.md) |
| Raw bake-off results | [../raw_test_results.md](../raw_test_results.md) |
