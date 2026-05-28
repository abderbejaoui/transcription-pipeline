# 05 — Fine-Tuning v1 (LoRA, 900h Gulf corpus)

This chapter documents the first LoRA fine-tune of Qwen3-ASR-1.7B on the
~900h Gulf Arabic corpus described in
[03_data_collection.md](03_data_collection.md). The recipe was driven by
the constraint analysis in [02_model_selection.md](02_model_selection.md):
LLM-decoder LoRA, frozen audio encoder, mixed-corpus oversampling
strategy.

## 5.1 Recipe at a glance

```
Base:        Qwen/Qwen3-ASR-1.7B
Method:      LoRA on LLM decoder modules
LoRA r:      6 (round-5) → 64 (final round)
LoRA alpha:  2 × r       (12 → 128)
rsLoRA:      ON          (rank-stabilized scaling)
Dropout:     0.05
Target mods: q_proj, k_proj, v_proj, o_proj,
             gate_proj, up_proj, down_proj
Encoder:     FROZEN
Precision:   bfloat16
Optimizer:   AdamW   (β1=0.9, β2=0.999, eps=1e-8, wd=0.01)
LR:          1e-4 (round-5), 1e-4 cosine + 1k warmup (final)
Batch:       4 per device × grad-accum 16 = effective 64
Epochs:      2  (early-stopping enabled)
Eval every:  500 steps
Hardware:    DGX Spark — GB10 Blackwell — 128 GB unified memory
```

## 5.2 Why these specific choices

### LoRA rank r=64, alpha=128, rsLoRA

`r=64` is a known sweet spot for ASR LoRA on 1B-class models
(Kalajdzievski et al. 2023, "A Rank Stabilization Scaling Factor for
Fine-Tuning with LoRA"). At r=64 the adapter has enough capacity to
learn dialect-specific token sequences without overfitting on a 900h
corpus.

`alpha = 2 × r` is the rsLoRA recommendation: it keeps the effective
adapter scaling constant as rank changes, so we can vary `r` later
without re-tuning the learning rate.

`use_rslora=True` flips the LoRA scaling from `alpha / r` to
`alpha / sqrt(r)`, which empirically gives smoother loss curves at
r ≥ 32. Without rsLoRA we saw oscillations between steps 4k and 8k
in earlier round-3 experiments.

### Target modules: decoder only

The audio encoder is frozen because:
- Pre-trained acoustic features are already strong on Arabic.
- A 900h fine-tune is too small to safely re-train the encoder.
- Decoder LoRA is where the dialect adaptation actually needs to happen
  — most failures are decoding-side (wrong token chosen given correct
  acoustic features), not perception-side.

The target module list `["q_proj", "k_proj", "v_proj", "o_proj",
"gate_proj", "up_proj", "down_proj"]` covers all the Linear modules in
each decoder transformer block. That's ~196 modules. We inspected the
module graph with `scripts/inspect_qwen3_modules.py` to confirm the
audio_tower has 0 LoRA targets and the language_model has all of them.

### Effective batch size 64

Per-device batch 4 × grad-accum 16. We tried batch 8 × grad-accum 8
first (also effective 64) but ran into intermittent OOM on long audio
clips (>20s) because the encoder activations grow with sequence length.
Smaller per-device batch with larger accumulation is more memory-stable.

### Learning rate 1e-4 + cosine + 1k warmup

`1e-4` is the standard LoRA learning rate for transformers. Warmup
prevents the initial steps from disturbing the frozen base model's
representations. Cosine decay to 1e-6 over the run keeps the model in a
stable region near the end.

### Early stopping on validation WER

Patience 3 evaluations, threshold 0.001 (absolute WER drop). The
intent is to stop before the model starts overfitting on the
oversampled UAE clips, which would degrade general Gulf performance.

In practice we ran into a separate problem: evaluation kept failing
due to a BFloat16 dtype mismatch in the WER computation path. We did
NOT fix it during the run — instead we let the run go to its full
2-epoch budget while monitoring the loss curve, which was clean
descent throughout (3.54 → 1.66 → 1.2 → 0.8). The dtype bug is fixed
in the v2 trainer (see chapter 10).

## 5.3 Training execution

Run command on the DGX inside a tmux session:

```bash
python scripts/finetune_qwen3_lora.py \
    --model-name Qwen/Qwen3-ASR-1.7B \
    --train-manifest data/dgx_full/preprocessed_audios/splits/train.jsonl \
    --eval-manifests \
        data/dgx_full/preprocessed_audios/splits/validation.jsonl \
        eval/casablanca_emirati_full/manifest.jsonl \
    --lora-rank 64 --lora-alpha 128 --use-rslora \
    --lora-dropout 0.05 \
    --num-epochs 2 \
    --per-device-train-batch-size 4 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 \
    --warmup-steps 1000 \
    --lr-scheduler cosine \
    --early-stopping-patience 3 \
    --early-stopping-metric wer \
    --early-stopping-threshold 0.001 \
    --eval-steps 500 \
    --logging-steps 50 \
    --save-steps 500 \
    --output-dir runs/qwen3_lora_r6 \
    --report-to tensorboard \
    2>&1 | tee logs/round6.log
```

Run name (output-dir suffix) `qwen3_lora_r6` is historical — the round
was numbered 6 in the experiment log even though the LoRA rank was 64.

## 5.4 Loss curve

| Step | Loss |
|---:|---:|
| 50    | 3.54 |
| 100   | 2.89 |
| 200   | 2.35 |
| 500   | 1.86 |
| 1,000 | 1.66 |
| 5,000 | 1.20 |
| 10,000| 1.00 |
| 15,000| 0.85 |
| 19,636| 0.78 |

The full curve is in TensorBoard at
`runs/qwen3_lora_r6/runs/*/events.out.tfevents.*`. No NaN, no spikes,
no recovery from a divergence — clean descent over the entire 2-epoch
budget.

Final checkpoint: `runs/qwen3_lora_r6/final_adapter/`.

## 5.5 Wall time

| Phase | Time |
|---|---|
| Per step (avg) | 7–8 s |
| Total steps | 19,636 |
| Wall time | ~43 hours |

This is on a single DGX Spark GB10 node at bf16, no flash-attention 2
(we tried, but had a CUDA mismatch on the aarch64 + GB10 build at the
time — non-blocking, just left ~30% performance on the table).

## 5.6 What we cannot say from v1

The evaluation pipeline broke during training (see above), so the v1
"shipped" WER number comes from a post-training manual evaluation
described in [06_evaluation.md](06_evaluation.md):

- **Test set**: UBC-NLP/Casablanca, UAE split, 813 conversational clips.
- **WER after v1 LoRA**: ~45% (down from 67% base).
- **Medical-term failure rate**: unmeasured at v1 time because the
  medical eval set didn't exist yet.

The 45% number was good enough to know we were on the right path
(LoRA adaptation works, dialect transfer is real) but not good enough
to ship. That motivated v2 — see [09_failure_analysis.md](09_failure_analysis.md)
and [10_finetuning_v2_plan.md](10_finetuning_v2_plan.md).

## 5.7 References

- Hu et al., **LoRA: Low-Rank Adaptation of Large Language Models**,
  arXiv:2106.09685 (2021).
- Kalajdzievski et al., **A Rank Stabilization Scaling Factor for
  Fine-Tuning with LoRA** (rsLoRA), arXiv:2312.03732 (2023).
- Mangrulkar et al., **PEFT: Parameter-Efficient Fine-Tuning of
  Billion-Scale Models on Low-Resource Hardware**, HuggingFace blog
  (2023).
- Loshchilov & Hutter, **Decoupled Weight Decay Regularization**
  (AdamW), arXiv:1711.05101 (2019).
- Loshchilov & Hutter, **SGDR: Stochastic Gradient Descent with Warm
  Restarts** (cosine schedule), arXiv:1608.03983 (2017).
