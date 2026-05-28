# Gulf Arabic Medical ASR — Technical Report

This folder is the chronological, factual report of how the Gulf Arabic medical
ASR system was built. Read the documents in the order listed below — each
chapter assumes the previous ones.

The report mirrors what is actually in this repository at commit time. Every
file path, command, hyperparameter and number cited here is traceable back to
a script, manifest, log or evaluation report in this workspace.

## Reading order

| # | Document | Topic |
|---|---|---|
| 01 | [problem_statement.md](01_problem_statement.md) | What we are trying to solve and why off-the-shelf ASR does not. |
| 02 | [model_selection.md](02_model_selection.md) | Why Qwen3-ASR-1.7B and not Whisper / MMS / Voxtral. |
| 03 | [data_collection.md](03_data_collection.md) | The 900h Gulf-Arabic corpus: sources, licenses, hour counts. |
| 04 | [preprocessing.md](04_preprocessing.md) | Audio normalization, text normalization, manifest schema. |
| 05 | [finetuning_v1.md](05_finetuning_v1.md) | The first LoRA fine-tune (r=6 / r=64, rsLoRA), recipe and run notes. |
| 06 | [evaluation.md](06_evaluation.md) | Bake-off test sets, normalizer, results table, error analysis. |
| 07 | [pipeline_architecture.md](07_pipeline_architecture.md) | End-to-end runtime architecture: FastAPI + ASR + alignment + UI. |
| 08 | [correction_layer.md](08_correction_layer.md) | Phonetic flagger, LCS-3 filter, drug-vs-disease tiebreaker, 50/50 hard suite. |
| 09 | [failure_analysis.md](09_failure_analysis.md) | Real-world failure modes that motivated v2. |
| 10 | [finetuning_v2_plan.md](10_finetuning_v2_plan.md) | Synthetic data, tiered 10k lexicon, 70h budget, training recipe. |
| 11 | [references.md](11_references.md) | Papers and external datasets cited above. |

## Quick facts

- **Base model**: `Qwen/Qwen3-ASR-1.7B`
- **Method**: PEFT LoRA on the LLM decoder, audio encoder frozen
- **Training data v1**: ~900h Gulf Arabic (SADA22 + WorldSpeech ar_bh/ar_kw/ar_sa + OMAN-SPEECH + WorldSpeech UAE/Saudi + Mixat Emirati + Nexdata UAE sample)
- **Training data v2 (in progress)**: ~150h mixed (60h synthetic medical + 45h rehearsal + 30h code-switched + 15h English medical)
- **Hardware**: NVIDIA DGX Spark, GB10 Blackwell, 128GB unified memory
- **Eval set**: UBC-NLP/Casablanca UAE split, 813 clips conversational Emirati
- **Live system**: FastAPI app under `app/` with phonetic flagger + LLM judge + MMS alignment
