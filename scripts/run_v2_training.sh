#!/usr/bin/env bash
# =============================================================================
# v2 MEDICAL FINE-TUNE — full runbook (run ON THE DGX, inside the venv)
# =============================================================================
# Strategy (researched + decided):
#   PRIMARY  = Arm B: bake v1 Gulf LoRA into base, then train a FRESH medical
#              LoRA on top -> Gulf dialect cannot be forgotten.
#   CONTROL  = Arm A: fresh medical LoRA on STOCK base, SAME data/hparams.
#   Then eval A vs B vs v1 on the locked test sets and keep the winner.
#
# Mixed, shuffled manifest (anti-forgetting):
#   synthetic medical 40% / Gulf rehearsal 30% / codeswitch 20% / english-med 10%
#   (Phase A below runs a fast 2-bucket subset; Phase B uses all 4.)
#
# ALWAYS launch training in a DETACHED tmux so a reboot/disconnect can't kill it.
# -----------------------------------------------------------------------------
set -euo pipefail

cd ~/abder/transcription/transcription-pipeline
source .venv/bin/activate
git pull origin docs/finetuning-doc

# ---- EDIT THESE TO MATCH YOUR DGX PATHS -------------------------------------
V1_ADAPTER="runs/qwen3_lora_r6/final_adapter"          # your trained Gulf v1 LoRA
GULF_TRAIN="data/dgx_full/preprocessed_audios/splits/train.jsonl"  # 900h source
SYNTH="data/training/medical_gulf_v2/manifest.jsonl"   # the 21h synthetic (done)
# Eval sets (confirm these exist on the DGX):
EVAL_GULF="eval/bakeoff_30min/manifest.jsonl"          # dialect (forgetting check)
EVAL_MED="eval/gulf_medical_v1/manifest.jsonl"         # medical terms (target metric)
# -----------------------------------------------------------------------------

echo "==================================================================="
echo " STEP 0  sanity checks"
echo "==================================================================="
test -f "$V1_ADAPTER/adapter_config.json" && echo "  v1 adapter OK"  || { echo "  !! v1 adapter missing"; exit 1; }
head -1 "$SYNTH" >/dev/null               && echo "  synthetic manifest OK"
test -f "$EVAL_GULF"                       && echo "  gulf eval OK"   || echo "  !! gulf eval missing"
test -f "$EVAL_MED"                        && echo "  medical eval OK" || echo "  !! medical eval missing"

echo "==================================================================="
echo " STEP 0b  BACK UP the v1 LoRA + write-protect it (never overwrite!)"
echo "==================================================================="
# Immutable, timestamped backup so the v1 weights can NEVER be lost.
V1_BACKUP="runs/_backups/qwen3_lora_r6_v1_$(date +%Y%m%d)"
if [ ! -d "$V1_BACKUP" ]; then
  mkdir -p runs/_backups
  cp -a "$V1_ADAPTER" "$V1_BACKUP"
  echo "  backed up v1 -> $V1_BACKUP"
else
  echo "  v1 backup already exists -> $V1_BACKUP"
fi
# Make the ORIGINAL v1 adapter read-only so nothing in this run can clobber it.
chmod -R a-w "$V1_ADAPTER" || true
chmod -R a-w "$V1_BACKUP"  || true
echo "  v1 adapter is now read-only. New training writes ONLY to runs/qwen3_lora_v2_medical_{A,B}."

echo "==================================================================="
echo " STEP 1  sample ~16h Gulf rehearsal (anti-forgetting anchor)"
echo "==================================================================="
python3 scripts/sample_rehearsal.py \
  --manifest "$GULF_TRAIN" \
  --out      data/training/gulf_rehearsal/manifest.jsonl \
  --target-hours 16 --seed 42

echo "==================================================================="
echo " STEP 2  build the mixed+shuffled master manifest"
echo "==================================================================="
# PHASE A (fast, do this FIRST): 2 real buckets only. Codeswitch/english use the
# synthetic+rehearsal as harmless placeholders with near-zero ratio so the script
# accepts 4 inputs but effectively trains on synth(40%)+rehearsal(58%).
python3 scripts/build_master_manifest.py \
  --synthetic   "$SYNTH" \
  --rehearsal   data/training/gulf_rehearsal/manifest.jsonl \
  --codeswitch  data/training/gulf_rehearsal/manifest.jsonl \
  --english-med "$SYNTH" \
  --out         data/training/master_v2/manifest.jsonl \
  --target-hours 37 --ratios 0.40 0.58 0.01 0.01 --seed 42

# PHASE B (after download_codeswitch_english.py finishes) — real 4-bucket:
#   python3 scripts/build_master_manifest.py \
#     --synthetic   "$SYNTH" \
#     --rehearsal   data/training/gulf_rehearsal/manifest.jsonl \
#     --codeswitch  data/training/codeswitch_masc/manifest.jsonl \
#     --english-med data/training/english_medical/manifest.jsonl \
#     --out         data/training/master_v2/manifest.jsonl \
#     --target-hours 52 --ratios 0.40 0.30 0.20 0.10 --seed 42

echo "==================================================================="
echo " STEP 3  ARM B prep — bake v1 Gulf LoRA into the base weights"
echo "==================================================================="
if [ ! -f runs/qwen3_gulf_merged_base/merge_v1_info.json ]; then
  python3 scripts/merge_v1_into_base.py \
    --base-model Qwen/Qwen3-ASR-1.7B \
    --adapter "$V1_ADAPTER" \
    --output  runs/qwen3_gulf_merged_base
else
  echo "  merged base already exists, skipping"
fi

echo "==================================================================="
echo " STEP 4  launch BOTH arms in DETACHED tmux sessions"
echo "==================================================================="
mkdir -p logs
# Guard: refuse to clobber a finished v2 run. Delete the dir yourself to retrain.
for d in runs/qwen3_lora_v2_medical_B runs/qwen3_lora_v2_medical_A; do
  if [ -f "$d/final_adapter/adapter_config.json" ]; then
    echo "  !! $d already has a finished adapter. Remove it manually to retrain. Aborting."
    exit 1
  fi
done
# Shared hyperparams (researched): r=32 alpha=64, lr=5e-5 (between script default
# 1e-4 and HF canonical 1e-5), 2 epochs, freeze audio encoder (default),
# eval every 1000 steps, early-stop on WER.
COMMON_ARGS="\
  --train-manifest data/training/master_v2/manifest.jsonl \
  --eval-manifests $EVAL_GULF $EVAL_MED \
  --lora-r 32 --lora-alpha 64 --use-rslora \
  --learning-rate 5e-5 --num-epochs 2 \
  --eval-every-steps 1000 --early-stopping-patience 3 --early-stopping-metric wer"

# ---- ARM B (PRIMARY): on the merged Gulf base ----
tmux new -s train_B -d
tmux send-keys -t train_B "cd ~/abder/transcription/transcription-pipeline && source .venv/bin/activate" Enter
tmux send-keys -t train_B "python3 scripts/finetune_qwen3_lora.py \
  --model-path runs/qwen3_gulf_merged_base \
  --output-dir runs/qwen3_lora_v2_medical_B \
  $COMMON_ARGS 2>&1 | tee logs/train_v2_B.log" Enter

# ---- ARM A (CONTROL): on stock base, SAME data/hparams ----
# Run AFTER B finishes if you have one GPU; run concurrently only if you have 2.
tmux new -s train_A -d
tmux send-keys -t train_A "cd ~/abder/transcription/transcription-pipeline && source .venv/bin/activate" Enter
tmux send-keys -t train_A "python3 scripts/finetune_qwen3_lora.py \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --output-dir runs/qwen3_lora_v2_medical_A \
  $COMMON_ARGS 2>&1 | tee logs/train_v2_A.log" Enter

echo "  launched. Monitor with:"
echo "    tmux attach -t train_B     # primary"
echo "    tail -f logs/train_v2_B.log"
echo "==================================================================="
echo " STEP 5 (after training) — eval A vs B vs v1, pick winner"
echo "==================================================================="
echo "  python3 scripts/eval_v2.py --model-path runs/qwen3_gulf_merged_base \\"
echo "      --adapter runs/qwen3_lora_v2_medical_B/final_adapter \\"
echo "      --eval-manifests $EVAL_GULF $EVAL_MED"
echo "  (repeat for arm A on stock base, and for v1 baseline)"
