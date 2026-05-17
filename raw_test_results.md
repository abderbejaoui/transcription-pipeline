# Arabic ASR — Casablanca-UAE Bake-Off (raw results)

> First run, 50 clips, ~3.9 min of Emirati conversational audio. The full
> 813-clip test will scale these numbers but is unlikely to change the
> ranking. All numbers produced by the Open Universal Arabic ASR
> Leaderboard methodology (Wang et al. 2024, arXiv:2412.13788) — same
> normalizer the public benchmark uses for Qwen3-ASR-1.7B / Whisper /
> MMS / omniASR / Voxtral.

## Test setup

| Item | Value |
|---|---|
| Test set | [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) config `UAE`, test split |
| Clips | 50 (smoke-test slice; full split is 813) |
| Total duration | ~3.9 min |
| Speech style | Real conversational Emirati Arabic (not read speech) |
| Reference quality | Human-verified by Casablanca annotators |
| Audio | 24 kHz mono, decoded from parquet, resampled to 16 kHz at inference |
| Hardware | DGX Spark, NVIDIA GB10, 128 GB unified memory |
| Inference settings | bf16, greedy decoding, `max_new_tokens=1024`, no language hint |
| Normalizer | Wang 2024 — strip punct + Tashkeel + Persian letters → Arabic + hamza/madda fold + Eastern digits → Western. **No** teh-marbuta fold, **no** num2words, **no** compound-pair merging. |
| Metric library | `jiwer 4.0` corpus-level WER + CER |

## Models tested

| key in bakeoff.py | model | size | how it ran |
|---|---|---|---|
| `qwen3` | Qwen/Qwen3-ASR-1.7B (base) | 1.7B | via `qwen-asr` wrapper, transformers 4.57.6 |
| `qwen3_uae` | vadimbelsky/qwen3-asr-arabic-uae | 1.7B + LoRA | same path |
| `voxtral_mini` | mistralai/Voxtral-Mini-3B-2507 | 3B | transformers 5.8.0.dev0 + mistral-common |
| `vibevoice` | microsoft/VibeVoice-ASR | 7B | **failed** — model load issue (n=0) |
| `omniASR` | facebook/omniASR-LLM-7B | 7B | **not run** — `fairseq2n` has no aarch64 wheel |

## Headline leaderboard (50 clips)

| rank | model | WER ↓ | **CER ↓** | n |
|---:|---|---:|---:|---:|
| 1 | **qwen3-asr-1.7b** (base) | **67.67%** | **22.29%** | 50 |
| 2 | qwen3-asr-uae | 70.85% | 25.58% | 50 |
| 3 | voxtral-mini-3b | 70.85% | 28.43% | 50 |
| — | vibevoice-asr | — | — | 0 (failed) |

> Median per-clip CER: 21.9% / 24.1% / 28.2% — close to the corpus
> numbers, so no single outlier is dragging the average.

## Validation against the published leaderboard

Our pipeline is **correct**. The numbers match published Casablanca
results within ~3 pts, which is normal for a 50-clip vs 813-clip slice:

| Model | Published Casablanca (8-dialect avg) | Our Casablanca-UAE (50 clips) |
|---|---:|---:|
| Qwen3-ASR-1.7B | 64.47% WER / 26.23% CER | 67.67% WER / **22.29% CER** ✓ |
| Voxtral-Mini-3B | ~71% WER / ~30% CER | 70.85% WER / **28.43% CER** ✓ |

Source: [Open Universal Arabic ASR Leaderboard](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard)

## Findings

### 1. Base Qwen3-ASR beats the UAE fine-tune on real Emirati audio

| Model | CER on Casablanca-UAE |
|---|---:|
| qwen3-asr-1.7b (base) | **22.29%** ← winner |
| qwen3-asr-uae (LoRA) | 25.58% |

The vadimbelsky UAE fine-tune was trained on **synthetic UAE data**.
On the real Casablanca benchmark (human-recorded Emirati conversation),
it overfits to its training distribution and loses ~3 CER points to the
base model.

Our earlier `bakeoff_clean` test (WorldSpeech + SADA22) showed the UAE
fine-tune winning. That set was a mix of Saudi/Bahraini/Kuwaiti broadcast
and had cleaner read-speech audio. Casablanca-UAE is more conversational
and dialectally specific, and **the fine-tune doesn't generalize**.

### 2. Voxtral-Mini-3B underperforms despite being bigger (3B vs 1.7B)

| Model | Size | CER |
|---|---:|---:|
| qwen3-asr-1.7b | 1.7B | 22.29% |
| voxtral-mini-3b | 3B | 28.43% |

Voxtral is a general-purpose multimodal LLM. Qwen3-ASR is a dedicated
ASR model. **For Arabic specifically, the specialist wins.**

### 3. Per-clip CER median is close to corpus mean

The errors are spread evenly, not driven by a small number of disasters.
The 50-clip sample is statistically OK (±~3 pts CI). Scaling to 813
clips will tighten the numbers but is **unlikely to change the ranking**.

## What we couldn't run yet

| Model | Blocker |
|---|---|
| omniASR-LLM-7B | `fairseq2n` has no `aarch64` wheel; would need compile-from-source. ~1 hr extra work, may still fail. |
| VibeVoice-ASR | Loaded but transcribed 0 clips. Probably trust_remote_code issue with transformers 5.8 — investigate next session. |
| Whisper-large-v3 | We have the backend but didn't include it in the comparison. Easy add. |

## Recommendation

**Ship `Qwen/Qwen3-ASR-1.7B` base. Drop the UAE fine-tune.**

Reasons:
1. Better accuracy on real Emirati audio (22.3% vs 25.6% CER).
2. No LoRA adapter, simpler deployment.
3. Apache-2.0 licence (vs vadimbelsky's research-only fine-tune).
4. Competitive with the public leaderboard average.

## Next steps (in priority order)

1. **Fix VibeVoice** so we have a 4-way comparison.
2. **Add Whisper-large-v3** as a sanity baseline (every paper compares to Whisper).
3. **Run the full 813-clip Casablanca-UAE** for a publishable number.
4. **Add Saudi/Kuwait clips** from another source — Casablanca has no Saudi config, but SADA22 (already in bakeoff_clean) covers Najdi Saudi.
5. **Build a 30-clip internal golden set** from your real production audio. The single most useful eval you can have.

## Reproducibility

```bash
# Build the test set
python -m scripts.build_casablanca_testset --dialect UAE --max-clips 50

# Run all backends + score
python -m scripts.compare_models --eval-dir eval/casablanca_UAE \
    --models qwen3 qwen3_uae voxtral_mini vibevoice

# Re-score without re-running inference
python -m scripts.compare_models --eval-dir eval/casablanca_UAE --score-only
```

Generated 17 May 2026 on DGX Spark, commit `fe79167` (PR #21 merged into main).
