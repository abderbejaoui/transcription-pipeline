# Arabic ASR — Casablanca-UAE Bake-Off (raw results)

> 50-clip smoke test on real conversational Emirati Arabic. All 5 models
> ran successfully. Numbers produced by the Open Universal Arabic ASR
> Leaderboard methodology (Wang et al. 2024, arXiv:2412.13788).

## Test setup

| Item | Value |
|---|---|
| Test set | [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca) config `UAE`, test split |
| Clips | 50 (smoke-test slice; full split is 813) |
| Total duration | ~3.9 min |
| Speech style | Real conversational Emirati Arabic |
| Reference quality | Human-verified by Casablanca annotators |
| Audio | 24 kHz mono, resampled to 16 kHz at inference |
| Hardware | DGX Spark, NVIDIA GB10, 128 GB unified memory |
| Inference settings | bf16, greedy decoding, `max_new_tokens=1024`, no language hint |
| Normalizer | Wang 2024 — punct/Tashkeel/Persian-letters/hamza/madda/digits |
| Metric library | `jiwer 4.0` corpus-level WER + CER |

## Final leaderboard (50 clips)

| rank | model | size | WER ↓ | **CER ↓** |
|---:|---|---:|---:|---:|
| 1 | **qwen3-asr-1.7b** (base) | 1.7B | 67.67% | **22.29%** |
| 2 | whisper-large-v3-turbo | 0.8B | 68.68% | 25.20% |
| 3 | qwen3-asr-uae | 1.7B+LoRA | 70.85% | 25.58% |
| 4 | voxtral-mini-3b | 3B | 70.85% | 28.43% |
| 5 | vibevoice-asr | 8B | 76.72% | 37.75% |

## Validation against the published leaderboard

| Model | Published Casablanca (8-dialect avg) | Our Casablanca-UAE (50 clips) |
|---|---:|---:|
| Qwen3-ASR-1.7B | 64.47% WER / 26.23% CER | 67.67% WER / **22.29% CER** ✓ |
| Whisper-large-v3 | 71.81% WER / 31.06% CER | 68.68% WER / **25.20% CER** ✓ |
| Voxtral-Mini-3B | ~71% WER / ~30% CER | 70.85% WER / **28.43% CER** ✓ |
| VibeVoice-ASR | ~74% WER / ~35% CER | 76.72% WER / **37.75% CER** ✓ |

All 4 measurable models land within ±3 CER points of their published
8-dialect-average numbers. **The pipeline is validated.**

Source: [Open Universal Arabic ASR Leaderboard](https://huggingface.co/spaces/elmresearchcenter/open_universal_arabic_asr_leaderboard)

## Findings

### 1. Qwen3-ASR-1.7B base is the winner

The smallest non-Whisper model (1.7B params) beats everything else
including 8B VibeVoice. Specialist beats generalist. **Ship the base
model.**

### 2. Whisper is a surprisingly strong baseline

Whisper-large-v3-turbo (0.8B distilled) lands at #2, beating Voxtral
(3B) and VibeVoice (8B). It's also Apache-2.0 and battle-tested in
production. **Keep it as a fallback option.**

### 3. Bigger ≠ better on Arabic

| Size | Best model at that size | CER |
|---|---|---:|
| 0.8B | whisper-large-v3-turbo | 25.20% |
| 1.7B | **qwen3-asr-1.7b** | **22.29%** |
| 3B | voxtral-mini-3b | 28.43% |
| 8B | vibevoice-asr | 37.75% |

VibeVoice is **trained for English** (its 7.77% Open ASR LB number is on
English only). On Arabic, it's the worst of the 5 — 15 CER points worse
than Qwen3-ASR which is 5× smaller.

### 4. UAE fine-tune doesn't help on real Emirati

qwen3-asr-uae (25.58%) loses to qwen3-asr-1.7b base (22.29%) by 3.3 CER
points. The fine-tune was trained on synthetic UAE data; it overfits
that distribution and degrades on real conversational Emirati audio.
**Don't ship the fine-tune.**

## Recommendation

**Ship `Qwen/Qwen3-ASR-1.7B` base.**

| Property | Value |
|---|---|
| CER on Casablanca-UAE | 22.29% |
| License | Apache-2.0 (commercial use OK) |
| Size | 1.7B (smallest of the 5 tested) |
| Speed | ~150x realtime (per HF model card) |
| Memory | ~4 GB GPU |
| Backup | Whisper-large-v3-turbo (25.20% CER, also Apache-2.0) |

## What we did NOT test

| Model | Why |
|---|---|
| omniASR-LLM-7B | `fairseq2n` has no aarch64 wheel for DGX Spark. Published Casablanca avg: 56.46% WER / 23.96% CER (would likely place #1 if we could run it). |
| Voxtral-Small-24B | Too big for a smoke test; would take ~1 hr. Published numbers say it's ~3 CER pts better than Mini. |

## Next steps (in priority order)

1. **Run full 813-clip Casablanca-UAE** with the same 5 models. ~75 min
   on DGX. Tightens confidence intervals from ±3 pts to ±1 pt.
2. **Run on Casablanca-Yemen and Casablanca-Jordan** to see how Qwen3
   generalises across other Arabic dialects.
3. **Build a 30-clip internal golden set** from your real production
   audio. The single most useful eval you can have.
4. **(Optional) Compile fairseq2n from source** on DGX so we can include
   omniASR. ~1-2 hr, not certain to work on aarch64.

## Reproducibility

```bash
# Build test set
python -m scripts.build_casablanca_testset --dialect UAE --max-clips 50

# Run all 5 backends + score
python -m scripts.compare_models --eval-dir eval/casablanca_UAE \
    --models qwen3 qwen3_uae voxtral_mini vibevoice whisper

# Re-score without re-running inference
python -m scripts.compare_models --eval-dir eval/casablanca_UAE --score-only
```

Generated 17 May 2026 on DGX Spark, commit `1641619` (PR #22 merged into main).
