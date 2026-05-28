# 02 — Model Selection

## 2.1 The candidates considered

| Model                           | Size  | Decoder type | Why considered |
|---|---|---|---|
| OpenAI Whisper-large-v3         | 1.55B | Acoustic transformer + small text decoder | Industry default; well-known |
| Meta MMS-1B                     | 1.0B  | CTC                                       | Best published number on SADA |
| Qwen2.5-Audio-7B-Instruct       | 7.0B  | LLM decoder                               | Multimodal LLM; strong Arabic |
| **Qwen/Qwen3-ASR-1.7B**         | 1.7B  | LLM decoder                               | Specialized ASR variant of Qwen3 |
| Mistral Voxtral-Mini-3B-2507    | 3.0B  | LLM decoder                               | Latest audio-LLM (2025) |
| Microsoft VibeVoice-ASR-7B      | 7.0B  | LLM decoder                               | Newest 2025 release |
| Meta omniASR-LLM-7B             | 7.0B  | LLM decoder                               | Latest 2025 release |

## 2.2 Bake-off methodology

A "bake-off" test set was assembled from real Khaleeji audio
(`eval/bakeoff_30min/` and `eval/casablanca_emirati_full/`) and every
candidate model was run zero-shot. Scoring used the **Open Universal Arabic
ASR Leaderboard** normalizer from Wang et al. 2024 (arXiv:2412.13788) so
the numbers are comparable to the public leaderboard.

The normalizer does the following before computing WER/CER:
- Strip punctuation
- Strip Tashkeel (Arabic diacritics)
- Map Persian/Urdu letters to Arabic equivalents
- Fold hamza and madda variants
- Convert Eastern Arabic digits to Western digits

It does **not** fold teh-marbuta or do num2words conversion, to stay
faithful to the published leaderboard behaviour.

Scoring library: `jiwer 4.0`, corpus-level WER + CER.

## 2.3 Results (50-clip Casablanca-UAE smoke test)

| Rank | Model | WER | CER | Notes |
|---:|---|---:|---:|---|
| 1 | **Qwen3-ASR-1.7B (base)**     | **67.67%** | **22.29%** | Best baseline |
| 2 | vadimbelsky/qwen3-asr-arabic-uae | 70.85% | 25.58% | Worse than base on conversational |
| 3 | Voxtral-Mini-3B               | 70.85% | 28.43% | Higher CER → spells worse |
| – | VibeVoice-ASR                 | —          | —          | Failed to load on aarch64 |
| – | omniASR-LLM-7B                | —          | —          | No aarch64 wheel for fairseq2n |

Whisper and MMS were excluded at this stage because:

- **Whisper-large-v3** has well-documented ~40% WER on SADA-style audio
  (Alharbi et al. 2024). Even after fine-tuning, getting to 22% takes very
  large amounts of data because the decoder is purely acoustic and has no
  morphological prior.
- **MMS-1B** reports 40.9% WER / 17.6% CER on SADA test-clean post-
  fine-tune (its best published number). That is a CTC ceiling: no LM
  prior, struggles with code-switching.

## 2.4 Why Qwen3-ASR-1.7B was chosen

Three reasons:

### Reason 1 — Best zero-shot Arabic accuracy

22.29% CER on real Emirati conversational audio is the lowest of any model
we tested, and it matches the public leaderboard (which reports 26.23% CER
on the full Casablanca 8-dialect average — our 22% on UAE-only is
consistent because Emirati is closer to MSA than e.g. Algerian).

### Reason 2 — LLM decoder handles Arabic morphology

Whisper's decoder is a small transformer trained on acoustic+character
loss. Qwen3-ASR's decoder is a 1.7B language model that has *already*
seen Arabic morphology, code-switching, and English drug names in its
pre-training corpus. The acoustic encoder hands off features and the
LLM does what LLMs are good at: producing fluent text in the right
language with the right vocabulary.

That is exactly the prior we want for code-switched Khaleeji speech.

### Reason 3 — Small enough to LoRA-fine-tune on a single DGX

At 1.7B parameters, the model + activations fit comfortably in 128GB
DGX Spark unified memory at bf16 with batch size 4 × gradient
accumulation 16 (effective batch 64). LoRA on the decoder modules adds
only ~30M trainable parameters, which trains stably with 1e-4 learning
rate.

7B models (Qwen2.5-Audio, VibeVoice, omniASR) were considered but
rejected:
- Training memory pressure is 4× higher.
- Inference latency on DGX Spark is acceptable but 2-3× slower than
  the 1.7B variant.
- The 7B Qwen2.5-Audio was the "instruct" multimodal variant — better
  for general audio Q&A but the same Arabic ASR specialization is not
  present.

## 2.5 Decision

```
Base:    Qwen/Qwen3-ASR-1.7B   (locked in)
Method:  LoRA on LLM decoder modules, audio encoder FROZEN
Why:     Best public baseline on real Khaleeji + LLM decoder prior
         + small enough to iterate fast on a single DGX node.
```

The encoder is frozen because:
- The pre-trained acoustic features are already strong on Arabic
  (encoder was trained on a large multilingual audio corpus including
  Arabic).
- Unfreezing the encoder typically requires 10× more data to avoid
  destroying the learned representations.
- Our 900h corpus is not large enough to safely re-train an encoder
  from scratch; LoRA on the decoder is the right capacity move.

## 2.6 What stage came next

With the base model chosen, the next decision was whether to:

a) Use vadimbelsky's published Qwen3-ASR-UAE adapter as-is (zero-shot
   it scored *worse* than base — so no).
b) Start a fresh LoRA fine-tune from the base.

We went with (b). Why vadimbelsky's adapter looked worse than base on
our test:

- Their published 9.98% WER is on a different (likely cleaner / closer
  to read-speech) test split.
- Their adapter was trained on a UAE-only mix, so it lost some of the
  Saudi / Khaleeji generalisation.

We trained a **single LoRA on the mixed Gulf corpus with UAE
oversampling**, instead of a UAE-only adapter. Details are in
[05_finetuning_v1.md](05_finetuning_v1.md).
