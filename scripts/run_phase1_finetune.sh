#!/usr/bin/env bash
# =============================================================================
# Phase-1 LoRA finetune driver — Qwen3-ASR-1.7B (DGX Spark)
#
# Runs the WHOLE Phase-1 path end to end with the REAL on-disk paths
# (verified 2026-06-07, see paths.md):
#
#   A3  build the disjoint Phase-1 train/val split + carve ~100h rehearsal
#   A4  step-0 validation smoke test (proves eval works, ~2 min)
#   A5  the real Phase-1 run (DoRA + rsLoRA, from Qwen3 BASE, frozen encoder)
#
# Phase-1 ACOUSTIC POOL (what actually exists on disk):
#   - data/dgx_full/preprocessed_audios_full/manifest.jsonl   804.3h
#       (already contains SADA + worldspeech_{bh,kw,sa} + mixat + nexdata)
#   - data/preprocessed/masc/manifest.jsonl                   ~393h  (clean)
#   - data/preprocessed/saudi_asrv1/manifest.jsonl            ~86h
#   - data/preprocessed/common_voice_ar/manifest.jsonl        ~88.6h (MSA)
#
#   NOTE: there are NO worldspeech_* slugs to add — they are baked into the
#   804h manifest. Adding them would double-count. mixat is also inside the
#   804h, so it is NOT listed separately. scc22 / casablanca are EVAL-ONLY.
#
# Usage (inside the qwen3 tmux on the DGX, venv active):
#   bash scripts/run_phase1_finetune.sh smoke    # A3 + A4 only (fast check)
#   bash scripts/run_phase1_finetune.sh full      # A3 + A4 + A5 (the real run)
#   bash scripts/run_phase1_finetune.sh split     # just rebuild the A3 split
#
# Default (no arg) = full.
# =============================================================================
set -euo pipefail

MODE="${1:-full}"

REPO="/home/abder/abder/transcription/transcription-pipeline"
cd "$REPO"

PY="${PY:-python}"   # inside the venv `python` is python3.12; override with PY=.venv/bin/python
MODEL="Qwen/Qwen3-ASR-1.7B"

# --- Phase-1 acoustic manifests (real, verified) ---------------------------
M_804H="data/dgx_full/preprocessed_audios_full/manifest.jsonl"
M_MASC="data/preprocessed/masc/manifest.jsonl"
M_SAUDI="data/preprocessed/saudi_asrv1/manifest.jsonl"
M_CV="data/preprocessed/common_voice_ar/manifest.jsonl"

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
      2>&1 | tee logs/smoke.log
  echo "--- A4 done. PASS = step-0 line 'WER=..%  CER=..%  n=8' (NOT nan/n=0) ---"
}

# ---------------------------------------------------------------------------
# A5 — Phase-1 real run (DoRA + rsLoRA, from BASE, encoder frozen)
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
      2>&1 | tee logs/phase1.log
  echo "--- A5 done. Adapter + checkpoints in runs/phase1/ ---"
}

case "$MODE" in
  split) build_split ;;
  smoke) build_split; smoke_test ;;
  full)  build_split; smoke_test; phase1_run ;;
  *) echo "usage: $0 [split|smoke|full]" >&2; exit 2 ;;
esac

echo "=== run_phase1_finetune ($MODE) complete: $(date) ==="
