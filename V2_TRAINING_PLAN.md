# v2 Medical Gulf-Arabic Fine-Tune — Plan & Runbook

## TL;DR
We have a **v1 LoRA** trained on **900h Gulf Arabic** that beats stock Qwen3-ASR on
Gulf dialect. We now have **21h of synthetic medical Gulf audio** (14,920 clips,
English drug names spoken in UAE dialect via voice cloning). Goal: teach the model
medical vocabulary **without losing** the Gulf-dialect skill v1 already has.

**Decision: run two arms and keep the winner.**
- **Arm B (PRIMARY):** bake v1 into the base weights (`merge_and_unload`), then train
  a **fresh** medical LoRA on top. Gulf skill is in the weights → it can't be forgotten.
- **Arm A (CONTROL):** fresh medical LoRA on **stock** base, same data/hparams.

Both train on the **same mixed, shuffled manifest** (anti-forgetting), then we compare
A vs B vs v1 on locked eval sets.

---

## Why both arms? (the research)

You were right to push back on "just start from base." Findings:

| Option | What it does | Forgetting risk | Verdict |
|---|---|---|---|
| **A** Fresh LoRA on stock base | Ignores v1; relies only on rehearsal data to keep Gulf | Medium — depends entirely on rehearsal mix | Control |
| **B** Merge v1 → base, fresh LoRA on top | Bakes 900h Gulf into weights, new adapter learns medical | **Lowest** — Gulf is no longer a removable adapter | **Primary** |
| **C** Keep training the v1 adapter | Continue v1 with medical data | **Highest** + compounds v1's existing regressions | Rejected |

Sources:
- **PEFT docs:** `merge_and_unload()` is the standard *sequential fine-tuning* pattern —
  bake the first skill into the base, then add a new adapter.
- **"LoRA vs Full Fine-tuning: An Illusion of Equivalence"** (arXiv 2410.21228): a LoRA
  kept as a removable adapter introduces **intruder dimensions** and forgets prior
  knowledge when trained further. Baking the *already-learned* skill into the base
  sidesteps this for that skill.

Why still run A as a control: it's cheap, isolates how much v1 actually helps, and if
B underperforms (e.g. merge introduces drift) we have a fallback already trained.

---

## Data mix (anti-forgetting)

One **shuffled** manifest, four buckets:

| Bucket | Source | Ratio | Purpose |
|---|---|---|---|
| Synthetic medical Gulf | `data/training/medical_gulf_v2/manifest.jsonl` (21h, done) | 40% | the new skill |
| Gulf rehearsal | sampled from 900h `splits/train.jsonl` | 30% | keep dialect |
| Code-switch (ar↔en) | MASC / CV-ar | 20% | keep ar+en mixing |
| English medical | PriMock57 / CV-en | 10% | keep english vocab |

- **Phase A (do first, fast):** 2 real buckets only — synthetic 40% + rehearsal ~58%.
  Validates that 21h moves medical WER before spending time downloading the rest.
- **Phase B (full):** all 4 buckets at `0.40 / 0.30 / 0.20 / 0.10`.

**Long-tail drugs (~9,700 names not in the 21h):** don't try to train them all. Use
**contextual biasing at inference** — pass a per-utterance term list via the `prompt=`
field. Fine-tune handles the high-frequency tier-1 drugs; biasing handles the tail.

---

## Hyperparameters (both arms, identical)

| Param | Value | Note |
|---|---|---|
| `--lora-r` | 32 | |
| `--lora-alpha` | 64 | |
| `--use-rslora` | on | stabler scaling at r=32 |
| `--learning-rate` | 5e-5 | between script default 1e-4 and HF canonical 1e-5 |
| `--num-epochs` | 2 | early-stop on WER |
| audio encoder | frozen | default; only LLM decoder linears get LoRA |
| `--eval-every-steps` | 1000 | |
| `--early-stopping-patience` | 3 | metric = WER |

---

## Step-by-step (on the DGX, inside the venv)

Everything is wired in [scripts/run_v2_training.sh](scripts/run_v2_training.sh). The
manual sequence it runs:

1. **Sample rehearsal** (~16h Gulf anchor)
   ```bash
   python3 scripts/sample_rehearsal.py \
     --manifest data/dgx_full/preprocessed_audios/splits/train.jsonl \
     --out data/training/gulf_rehearsal/manifest.jsonl \
     --target-hours 16 --seed 42
   ```
2. **Build master manifest** (Phase A) — [scripts/build_master_manifest.py](scripts/build_master_manifest.py)
3. **Arm B prep — bake v1 into base** — [scripts/merge_v1_into_base.py](scripts/merge_v1_into_base.py)
   ```bash
   python3 scripts/merge_v1_into_base.py \
     --base-model Qwen/Qwen3-ASR-1.7B \
     --adapter runs/qwen3_lora_r6/final_adapter \
     --output runs/qwen3_gulf_merged_base
   ```
4. **Train both arms in DETACHED tmux** (so a reboot can't kill them):
   - Arm B: `--model-path runs/qwen3_gulf_merged_base` → `runs/qwen3_lora_v2_medical_B`
   - Arm A: `--model-path Qwen/Qwen3-ASR-1.7B` → `runs/qwen3_lora_v2_medical_A`
   - One GPU → run B first, then A. Two GPUs → run concurrently.
5. **Phase B (optional, after Phase A looks good):** download extra buckets —
   [scripts/download_codeswitch_english.py](scripts/download_codeswitch_english.py) —
   rebuild master with `--ratios 0.40 0.30 0.20 0.10`, retrain.

> **Always launch training detached:** `tmux new -s train_B -d` + `tmux send-keys ...`.
> The runbook already does this. Monitor with `tmux attach -t train_B` or
> `tail -f logs/train_v2_B.log`.

---

## Evaluation & decision

Eval all candidates on the **same locked sets**:
- **Gulf dialect (forgetting check):** `eval/bakeoff_30min/manifest.jsonl`
- **Medical terms (target metric):** `eval/gulf_medical_v1/manifest.jsonl`

| Candidate | Gulf WER | Medical WER |
|---|---|---|
| stock base | (baseline) | (baseline) |
| v1 (current) | (best Gulf) | (poor medical) |
| **v2 Arm A** | ? | ? |
| **v2 Arm B** | ? | ? |

**Pick the model that lowers Medical WER the most while keeping Gulf WER ≤ v1.**
Expected winner: **Arm B**. Target: Gulf-medical WER materially below v1, Gulf dialect
WER not worse than v1.

---

## Files added for this plan
- [scripts/merge_v1_into_base.py](scripts/merge_v1_into_base.py) — bake v1 LoRA into base (Arm B prep).
- [scripts/download_codeswitch_english.py](scripts/download_codeswitch_english.py) — Phase-B buckets (MASC + PriMock57/CV).
- [scripts/run_v2_training.sh](scripts/run_v2_training.sh) — full detached-tmux runbook for both arms.

Reused as-is: `scripts/sample_rehearsal.py`, `scripts/build_master_manifest.py`,
`scripts/finetune_qwen3_lora.py` (note: `scripts/merge_adapters.py` is for adapter
*averaging* — a different purpose, not used here).
