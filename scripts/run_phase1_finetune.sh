#!/usr/bin/env bash
# =============================================================================
# Two-phase LoRA finetune driver — Qwen3-ASR-1.7B (DGX Spark)
#
# THE STRATEGY (exact, see finetuningv2.md / FINETUNE_RUNBOOK.md Part A):
#
#   Phase 1 = Gulf/Arabic ACOUSTIC from the Qwen3 BASE model
#             (the 804h pool [contains WorldSpeech + mixat + SADA] + MASC,
#              + saudi_asrv1 + common_voice_ar MSA anchors),
#             carving ~100h OUT of that pool for Phase 2.
#   Phase 2 = RESUME from Phase 1 and train
#             synthetic medical data + code-switched data + the 100h Gulf
#             rehearsal TOGETHER (never synthetic-only — the rehearsal
#             prevents catastrophic forgetting). Lower LR.
#
# Runs the WHOLE path end to end with the REAL on-disk paths
# (verified 2026-06-07, see paths.md):
#
#   A3  build the disjoint Phase-1 train/val split + carve ~100h rehearsal
#   A4  step-0 validation smoke test (proves eval works, ~2 min)
#   A5  the real Phase-1 run (DoRA + rsLoRA, from Qwen3 BASE, frozen encoder)
#   A6  build the Phase-2 mixed manifest + resume-train (synthetic + CS + rehearsal)
#
# Phase-1 ACOUSTIC POOL (what actually exists on disk):
#   - data/dgx_full/preprocessed_audios_full/manifest.jsonl   804.3h
#       (already contains SADA + worldspeech_{bh,kw,sa} + mixat + nexdata)
#   - data/preprocessed/masc/manifest.jsonl                   ~393h  (clean)
#   - data/preprocessed/saudi_asrv1/manifest.jsonl            ~86h
#   - data/preprocessed/common_voice_ar/manifest.jsonl        ~88.6h (MSA)
#
#   NOTE: there are NO worldspeech_* slugs to add — they are baked into the
#   804h manifest. Adding them would double-count. mixat (the Emirati-English
#   code-switch set, 14h) is ALSO inside the 804h, so the carved 100h rehearsal
#   already carries real code-switch acoustic into Phase 2. scc22 / casablanca
#   are EVAL-ONLY.
#
# Phase-2 MEDICAL + CODE-SWITCH POOL (what actually exists on disk):
#   - data/training/medical_gulf_v2/manifest.jsonl   21.01h synthetic medical CS
#   - data/splits/phase2_rehearsal.jsonl             ~100h carved Gulf (incl. mixat CS)
#
# Usage (inside the qwen3 tmux on the DGX, venv active):
#   bash scripts/run_phase1_finetune.sh smoke    # A3 + A4 only (fast check)
#   bash scripts/run_phase1_finetune.sh full      # A3 + A4 + A5 (Phase 1, real run)
#   bash scripts/run_phase1_finetune.sh split     # just rebuild the A3 split
#   bash scripts/run_phase1_finetune.sh resume    # continue Phase 1 from latest ckpt
#   bash scripts/run_phase1_finetune.sh phase2    # A6 (resume Phase 1 -> Phase 2)
#
# Default (no arg) = full. Run `full` first, then `phase2`.
#
# Shared-GPU controls (env vars):
#   GPU_MEM_FRACTION=0.65   cap this process to 65% of GPU memory (default)
#   WORKERS=0               DataLoader workers (0 = safe single-process)
# Stop anytime with Ctrl-c to free the card; later run `... resume` to pick up
# EXACTLY where you left off (weights + optimizer + scheduler + step count).
# =============================================================================
set -euo pipefail

MODE="${1:-full}"

REPO="/home/abder/abder/transcription/transcription-pipeline"
cd "$REPO"

PY="${PY:-python}"   # inside the venv `python` is python3.12; override with PY=.venv/bin/python
MODEL="Qwen/Qwen3-ASR-1.7B"

# DataLoader workers. 0 = single-process loading (NO multiprocessing) — this
# avoids the persistent-worker + WeightedRandomSampler + librosa deadlock that
# froze the run at step 0/6 for 21h. If a >0 value trains fine on your box you
# can bump it: WORKERS=4 bash scripts/run_phase1_finetune.sh full
WORKERS="${WORKERS:-0}"

# --- Shared-GPU memory cap -------------------------------------------------
# The DGX is shared. GPU_MEM_FRACTION caps how much of the card's memory THIS
# process may allocate, leaving the rest free for other users. 0.65 = use up
# to 65%, leave 35% free. Enforced at runtime via
# torch.cuda.set_per_process_memory_fraction (see TORCH_MEM_HOOK below).
# Override: GPU_MEM_FRACTION=0.5 bash scripts/run_phase1_finetune.sh full
GPU_MEM_FRACTION="${GPU_MEM_FRACTION:-0.65}"
export GPU_MEM_FRACTION
# Make the CUDA allocator return freed blocks to the driver promptly so the
# 35% we leave free is actually usable by the other user.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# A tiny -X importtime-free shim that applies the memory fraction the instant
# torch is imported, BEFORE the model loads. Injected via PYTHONSTARTUP-style
# usercustomize is fragile, so we instead pass it through a sitecustomize hook
# created on the fly under runs/ and prepended to PYTHONPATH.
HOOK_DIR="$REPO/runs/_mem_hook"
mkdir -p "$HOOK_DIR"
cat > "$HOOK_DIR/sitecustomize.py" <<'PYHOOK'
import os
_frac = os.environ.get("GPU_MEM_FRACTION")
if _frac:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(_frac), 0)
            print(f"[mem-hook] capped GPU 0 to {float(_frac)*100:.0f}% "
                  f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB total)",
                  flush=True)
    except Exception as _e:  # never block training on the cap
        print(f"[mem-hook] could not set memory fraction: {_e!r}", flush=True)
PYHOOK
export PYTHONPATH="$HOOK_DIR:${PYTHONPATH:-}"

# --- Phase-1 acoustic manifests (real, verified) ---------------------------
M_804H="data/dgx_full/preprocessed_audios_full/manifest.jsonl"
M_MASC="data/preprocessed/masc/manifest.jsonl"
M_SAUDI="data/preprocessed/saudi_asrv1/manifest.jsonl"
M_CV="data/preprocessed/common_voice_ar/manifest.jsonl"

# --- Phase-2 medical + code-switch manifests (real, verified) --------------
# Synthetic medical code-switch (21.01h, schema uses `audio` not `audio_path`;
# the loader in finetune_qwen3_lora.py accepts both, so no conversion needed).
M_SYNTH="data/training/medical_gulf_v2/manifest.jsonl"
# ~100h Gulf rehearsal carved OUT of the Phase-1 pool in A3 (this carries the
# real mixat code-switch acoustic forward, so the same audio is never trained
# in both phases). Built by build_split().
M_REHEARSAL="data/splits/phase2_rehearsal.jsonl"
# The concatenated Phase-2 training manifest (synthetic + rehearsal), built by
# build_phase2().
M_PHASE2="data/splits/phase2_mixed.train.jsonl"

# --- Eval-only held-out benchmark ------------------------------------------
M_SCC22="data/preprocessed/scc22/manifest.jsonl"

mkdir -p data/splits runs logs

echo "=== run_phase1_finetune ($MODE): $(date) ==="
echo "Python: $($PY -c 'import sys;print(sys.executable)')"

# Fail fast if a required manifest is missing.
for m in "$M_804H" "$M_MASC" "$M_SAUDI" "$M_CV"; do
  if [[ ! -f "$m" ]]; then
    echo "FATAL: missing manifest: $m" >&2
    echo "Check paths.md — did prep finish for this slug?" >&2
    exit 1
  fi
done

# ---------------------------------------------------------------------------
# A3 — build disjoint Phase-1 split + carve ~100h Phase-2 rehearsal
# ---------------------------------------------------------------------------
build_split() {
  echo "--- A3: split + carve ---"
  $PY scripts/split_manifest.py \
      --in "$M_804H" "$M_MASC" "$M_SAUDI" "$M_CV" \
      --out-prefix data/splits/phase1 \
      --val-frac 0.02 \
      --stratify-by source \
      --dedup-text \
      --carve-hours 100 \
      --carve-out data/splits/phase2_rehearsal.jsonl
  echo "--- A3 done. Confirm the last line shows leakage=0 ---"
}

# ---------------------------------------------------------------------------
# A4 — step-0 validation smoke test (must show real WER, n>0, NOT nan)
# ---------------------------------------------------------------------------
smoke_test() {
  echo "--- A4: validation smoke test (max-steps 6, eval-at-start) ---"
  $PY -m scripts.finetune_qwen3_lora \
      --model-path "$MODEL" \
      --train-manifest data/splits/phase1.train.jsonl \
      --eval-manifests data/splits/phase1.val.jsonl \
      --output-dir runs/smoke \
      --max-steps 6 \
      --eval-at-start \
      --eval-max-samples 8 \
      --early-stopping-patience 0 \
      --num-workers "$WORKERS" \
      2>&1 | tee logs/smoke.log
  echo "--- A4 done. PASS = step-0 line 'WER=..%  CER=..%  n=8' (NOT nan/n=0) ---"
}

# ---------------------------------------------------------------------------
# A5 — Phase-1 real run (DoRA + rsLoRA, from BASE, encoder frozen)
#   DoRA decomposes the weight update into magnitude + direction for more
#   stable adaptation; rsLoRA (rank-stabilised scaling) keeps the LoRA scale
#   sane at r=32. The plan calls for BOTH (independent knobs).
# ---------------------------------------------------------------------------
phase1_run() {
  echo "--- A5: Phase-1 real run ---"
  $PY -m scripts.finetune_qwen3_lora \
      --model-path "$MODEL" \
      --train-manifest data/splits/phase1.train.jsonl \
      --eval-manifests data/splits/phase1.val.jsonl "$M_SCC22" \
      --output-dir runs/phase1 \
      --num-epochs 3 \
      --learning-rate 1e-4 \
      --lr-scheduler-type cosine --warmup-ratio 0.02 --weight-decay 0.01 \
      --max-grad-norm 1.0 \
      --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
      --use-dora --use-rslora \
      --per-device-train-batch-size 4 \
      --gradient-accumulation-steps 16 \
      --eval-every-steps 2000 \
      --eval-at-start \
      --early-stopping-patience 3 --early-stopping-metric wer \
      --gradient-checkpointing \
      --save-total-limit 5 \
      --num-workers "$WORKERS" \
      ${RESUME_CKPT:+--resume-from-checkpoint "$RESUME_CKPT"} \
      2>&1 | tee -a logs/phase1.log
  echo "--- A5 done. Adapter + checkpoints in runs/phase1/ ---"
}

# ---------------------------------------------------------------------------
# A6a — build the Phase-2 mixed manifest:
#   synthetic medical CS  +  code-switch  +  100h Gulf rehearsal
# The rehearsal (carved from Phase 1) already contains the real mixat
# code-switch acoustic, so it satisfies BOTH the code-switch and the
# anti-forgetting requirements. NEVER train synthetic-only.
# ---------------------------------------------------------------------------
build_phase2() {
  echo "--- A6a: build Phase-2 mixed manifest (synthetic + CS + rehearsal) ---"
  if [[ ! -f "$M_SYNTH" ]]; then
    echo "FATAL: missing synthetic medical manifest: $M_SYNTH" >&2
    echo "See paths.md \u00a72 — it should be 21.01h / 14,920 clips." >&2
    exit 1
  fi
  if [[ ! -f "$M_REHEARSAL" ]]; then
    echo "FATAL: missing carved rehearsal: $M_REHEARSAL" >&2
    echo "Run 'bash $0 split' (or 'full') first to carve the 100h pool." >&2
    exit 1
  fi
  cat "$M_SYNTH" "$M_REHEARSAL" > "$M_PHASE2"
  local n_syn n_reh n_tot
  n_syn=$(wc -l < "$M_SYNTH")
  n_reh=$(wc -l < "$M_REHEARSAL")
  n_tot=$(wc -l < "$M_PHASE2")
  echo "--- A6a: phase2 = synthetic($n_syn) + rehearsal($n_reh) = $n_tot rows -> $M_PHASE2 ---"
}

# ---------------------------------------------------------------------------
# A6b — Phase-2 real run: RESUME from Phase-1 best adapter, lower LR (5e-5),
#   2 epochs, DoRA + rsLoRA kept on, encoder frozen. Eval on phase1.val
#   (early stopping) + scc22 (held-out CS generalisation).
# ---------------------------------------------------------------------------
phase2_run() {
  local p1_adapter="runs/phase1/best_adapter"
  if [[ ! -d "$p1_adapter" ]]; then
    echo "FATAL: $p1_adapter not found — run Phase 1 ('bash $0 full') first." >&2
    exit 1
  fi
  echo "--- A6b: Phase-2 real run (warm-start from $p1_adapter) ---"
  # --init-adapter loads the Phase-1 LoRA weights as the starting point and
  # begins a FRESH run (new optimizer/scheduler/LR schedule). We do NOT use
  # --resume-from-checkpoint here: best_adapter/ is a bare PEFT adapter dir
  # (no optimizer.pt/trainer_state.json), so a Trainer resume would fail.
  $PY -m scripts.finetune_qwen3_lora \
      --model-path "$MODEL" \
      --init-adapter "$p1_adapter" \
      --train-manifest "$M_PHASE2" \
      --eval-manifests data/splits/phase1.val.jsonl "$M_SCC22" \
      --output-dir runs/phase2 \
      --num-epochs 2 \
      --learning-rate 5e-5 \
      --lr-scheduler-type cosine --warmup-ratio 0.03 --weight-decay 0.01 \
      --max-grad-norm 1.0 \
      --lora-r 32 --lora-alpha 64 --lora-dropout 0.05 \
      --use-dora --use-rslora \
      --per-device-train-batch-size 4 \
      --gradient-accumulation-steps 16 \
      --eval-every-steps 1000 \
      --eval-at-start \
      --early-stopping-patience 3 --early-stopping-metric wer \
      --gradient-checkpointing \
      --save-total-limit 5 \
      --num-workers "$WORKERS" \
      2>&1 | tee -a logs/phase2.log
  echo "--- A6b done. Phase-2 adapter + checkpoints in runs/phase2/ ---"
}

# ---------------------------------------------------------------------------
# resume — continue Phase-1 from the latest saved checkpoint (no re-split,
# no smoke). Use this after you stopped the run to free the GPU for someone.
# ---------------------------------------------------------------------------
resume_run() {
  local latest
  latest=$(ls -d runs/phase1/checkpoint-* 2>/dev/null \
           | sort -t- -k2 -n | tail -1 || true)
  if [[ -z "$latest" ]]; then
    echo "[resume] no runs/phase1/checkpoint-* found — nothing to resume." >&2
    echo "[resume] start a fresh run with: bash $0 full" >&2
    exit 1
  fi
  echo "--- resume: continuing from $latest ---"
  RESUME_CKPT="$latest" phase1_run
}

case "$MODE" in
  split)  build_split ;;
  smoke)  build_split; smoke_test ;;
  full)   build_split; smoke_test; phase1_run ;;
  resume) resume_run ;;
  phase2) build_phase2; phase2_run ;;
  *) echo "usage: $0 [split|smoke|full|resume|phase2]" >&2; exit 2 ;;
esac

echo "=== run_phase1_finetune ($MODE) complete: $(date) ==="
