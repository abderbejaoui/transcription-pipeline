# finetuningv2.md — The agreed Qwen3-ASR V2 strategy (single source of truth)

> Written 2026-06-08. This file reconciles `FINETUNE_RUNBOOK.md` (Part A, the
> authoritative "max data" plan) with the actual driver
> `scripts/run_phase1_finetune.sh`, and records exactly what we are going to do,
> what currently matches, and what still needs wiring. Refer to THIS file going
> forward.

---

## 0. The decision in one paragraph

Train the best open-source Gulf-Arabic medical ASR in **two phases** on
**Qwen3-ASR-1.7B** with **LoRA/DoRA** (audio tower frozen):

- **Phase 1 — acoustic.** Start from the Qwen3 **BASE** model. Train on the
  maximum real Gulf/Arabic acoustic data we have. Carve ~100h *out* of this pool
  and set it aside for Phase 2 (so the same audio is never trained in both
  phases). Output: `runs/phase1/best_adapter`.
- **Phase 2 — medical + code-switch.** **Resume from `runs/phase1/best_adapter`**
  and continue training on a mix of **synthetic medical code-switch + all real
  code-switch sets (mixat) + the carved ~100h Gulf rehearsal**, at a lower LR.
  The 100h rehearsal is the anti-forgetting anchor. Output: `runs/phase2/best_adapter`.

LoRA config (both phases): `r=32, alpha=64, dropout=0.05`, **DoRA on**, rsLoRA on,
cosine schedule, frozen encoder. Phase 1 LR `1e-4`; Phase 2 LR `5e-5`.

**Why two phases (not one mixed run):** training the acoustic skill first and
*then* layering medical/code-switch on top — with a rehearsal anchor — is the
catastrophic-forgetting-safe pattern. The model locks in Gulf acoustic before it
ever sees the small, narrow medical/CS data, and the rehearsal pool keeps the
acoustic skill alive during Phase 2. (PEFT sequential-finetune pattern + the
"LoRA intruder dimensions / forgetting" finding, arXiv:2410.21228.)

---

## 1. Data plan

### Phase 1 — acoustic pool (real only, from BASE)

| Dataset | Slug | ~Hrs | Notes |
|---|---|---|---|
| Existing 804h corpus (**contains SADA**, mixat, worldspeech, nexdata baked in) | — | ~804 | base of Phase 1 |
| WorldSpeech Bahrain | `worldspeech_bh` | 272.5 | gated, `cer≤0.25` |
| WorldSpeech Kuwait | `worldspeech_kw` | 175.5 | gated |
| WorldSpeech Saudi | `worldspeech_sa` | 6.1 | gated |
| WorldSpeech UN (MSA anchor) | `worldspeech_un` | 11.1 | weight 0.3 |
| MASC (clean `type='c'`) | `masc` | ~1000* | weight 0.7, pan-Arabic anchor |

\* gated to clean clips only, down-weighted.

**Carve ~100h out** of this pool → `data/splits/phase2_rehearsal.jsonl`
(tagged `rehearsal:true` / `stage:2`, removed from `phase1.train.jsonl`).

### Phase 2 — medical + code-switch (resume from Phase 1)

| Component | Source | ~Hrs |
|---|---|---|
| Synthetic medical code-switch | `data/preprocessed/synthetic_medical_cs/manifest.jsonl` | ~20 |
| ALL real code-switch | `data/preprocessed/mixat/manifest.jsonl` (+ any mined CS) | ~15–25 |
| Gulf rehearsal (carved from Phase 1) | `data/splits/phase2_rehearsal.jsonl` | ~100 |

Phase 2 manifest = concatenation of those three.

### Eval / held-out (never trained)

- `data/splits/phase1.val.jsonl` — validation, drives early stopping.
- `data/preprocessed/scc22/manifest.jsonl` — held-out Saudi-Eng CS benchmark (`eval_only`).
- casablanca (Emirati) — held-out benchmark when prepared.

All `eval_only` rows are dropped by `split_manifest.py`, so they cannot leak.

---

## 2. Hyperparameters

| Param | Phase 1 | Phase 2 |
|---|---|---|
| init | Qwen3 BASE | resume `runs/phase1/best_adapter` |
| `--lora-r / --lora-alpha / --lora-dropout` | 32 / 64 / 0.05 | 32 / 64 / 0.05 |
| `--use-dora` | on | on |
| `--use-rslora` | on | on |
| `--learning-rate` | 1e-4 | 5e-5 |
| `--lr-scheduler-type` / `--warmup-ratio` | cosine / 0.02 | cosine / 0.03 |
| `--num-epochs` | 3 | 2 |
| batch / grad-accum | 4 / 16 | 4 / 16 |
| `--eval-every-steps` | 2000 | 1000 |
| encoder | frozen | frozen |
| early stop | patience 3, metric wer | patience 3, metric wer |

Escalation if Phase-1 WER plateaus: `--lora-r 64 --lora-alpha 128 --learning-rate 3e-4`
(fresh run, never change rank mid-run). Optional encoder unfreeze only after a
good frozen Phase 1: `--unfreeze-encoder-layers 4 --encoder-lora-lr 1e-5`.

---

## 3. Does `scripts/run_phase1_finetune.sh` match this plan?

**Yes — now it does** (implemented 2026-06-08). The driver runs the full two
phases:

| Item | Agreed plan | Current driver | Status |
|---|---|---|---|
| Phase-1 sources | 804h (contains WorldSpeech + mixat + SADA) + MASC + MSA anchors | 804h + MASC + `saudi_asrv1` + `common_voice_ar` | ✅ |
| Carve 100h for Phase 2 | yes | yes (`--carve-hours 100`) | ✅ |
| DoRA | on | on | ✅ |
| rsLoRA | on | **on** (`--use-dora --use-rslora`) | ✅ |
| **Phase 2 run** | synthetic medical + CS + 100h rehearsal | `phase2` mode | ✅ |
| Code-switch data trained | yes (Phase 2) | via the carved rehearsal (contains real mixat CS) | ✅ |
| Synthetic medical data trained | yes (Phase 2) | `data/training/medical_gulf_v2/manifest.jsonl` | ✅ |
| Phase-2 warm start | resume from Phase 1 | `--init-adapter runs/phase1/best_adapter` | ✅ |
| Resume / pause | needed | `resume` mode | ✅ |
| Shared-GPU cap | needed | `GPU_MEM_FRACTION=0.65` | ✅ |

> **Note on WorldSpeech / mixat:** per `paths.md`, the 804h manifest already
> bakes in WorldSpeech (bh/kw/sa) and mixat (14h Emirati-English code-switch).
> So WorldSpeech is in Phase 1, and the 100h carve pulls real code-switch
> acoustic into Phase 2. There are no standalone WorldSpeech/mixat slugs to add
> — doing so would double-count.
>
> **Note on warm start:** `best_adapter/` is a bare PEFT adapter dir (no
> optimizer/scheduler state), so Phase 2 uses the new `--init-adapter` flag
> (load Phase-1 LoRA weights, fresh optimizer at lr 5e-5) rather than
> `--resume-from-checkpoint` (which needs a full Trainer checkpoint).

---

## 4. What we should do (action list)

1. **Decide the Phase-1 source list.** Either:
   - (a) keep the runbook list (add WorldSpeech bh/kw/sa/un, drop saudi_asrv1 +
     common_voice_ar), **or**
   - (b) keep what the driver has now (804h + MASC + saudi_asrv1 + common_voice_ar)
     if WorldSpeech isn't prepped/worth the gate.
   Confirm which on the DGX with `ls data/preprocessed/`.
2. **Re-add `--use-rslora`** to the Phase-1 run (it was dropped). rsLoRA + DoRA
   are independent and the plan calls for both at r=32.
3. **Verify Phase-2 inputs exist on the DGX:**
   `data/preprocessed/synthetic_medical_cs/manifest.jsonl` and
   `data/preprocessed/mixat/manifest.jsonl`. If `synthetic_medical_cs` is not
   preprocessed yet, that is the blocker for Phase 2 — prep it first.
4. **Wire Phase 2 into the driver** (a `phase2` mode) that:
   - builds `data/splits/phase2_mixed.train.jsonl` =
     `synthetic_medical_cs + mixat + phase2_rehearsal`,
   - launches `finetune_qwen3_lora` with `--resume-from-checkpoint runs/phase1/best_adapter`,
     LR `5e-5`, 2 epochs, output `runs/phase2`.
5. **Run order on the DGX:** `full` (Phase 1) → confirm `runs/phase1/best_adapter`
   improved over baseline (WER 36.14% on `phase1.val`) → `phase2`.
6. **Final test:** `scripts/test_asr.py --adapter runs/phase2/best_adapter` on
   `scc22` (+ casablanca) with `--breakdown` (CS vs non-CS).

---

## 5. Run commands (current driver)

```bash
# on DGX, in tmux qwen3, venv active
cd ~/abder/transcription/transcription-pipeline && source .venv/bin/activate
git pull --no-rebase --no-edit origin feat/v2-max-data

# Phase 1 (caps to 65% GPU mem, leaves 35% free):
bash scripts/run_phase1_finetune.sh full      # split -> smoke -> phase1

# pause anytime: Ctrl-c   ;   resume Phase 1 later from latest checkpoint:
bash scripts/run_phase1_finetune.sh resume

# Phase 2 (run AFTER Phase 1 finishes and runs/phase1/best_adapter exists):
#   builds data/splits/phase2_mixed.train.jsonl = synthetic + rehearsal,
#   warm-starts from runs/phase1/best_adapter at lr 5e-5, writes runs/phase2/.
bash scripts/run_phase1_finetune.sh phase2
```

---

## 6. Locked constraints (do not violate)

- `transformers==4.57.6` (NEVER 5.x), `datasets>=3.x`.
- Audio tower frozen in both phases (LoRA only on LLM decoder linears).
- Phase 2 is **never synthetic-only** — always mixed with rehearsal + real CS.
- Same audio never trained in both phases (the 100h carve guarantees this).
- Smoke ≠ real run. Baseline to beat: WER 36.14% on `phase1.val`.
- Branch `feat/v2-max-data`; never push `main`; stage only edited files.
