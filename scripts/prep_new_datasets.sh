#!/usr/bin/env bash
# Overnight prep of the NEW Gulf/Arabic ASR datasets (full hours, no caps).
#
# What this does:
#   * Prepares ONLY datasets that are not already part of the Phase-1 corpus
#     (sada22 / worldspeech_* / mixat / nexdata are REUSED from disk and never
#     re-downloaded here).
#   * Runs each new dataset as its own process, in parallel, with the fast HF
#     downloader, decoding the FULL dataset (no --max-hours / --max-clips).
#   * Is resume-safe: a dataset whose manifest already has clips is SKIPPED, so
#     you can re-run after a crash without redoing finished work.
#   * Logs every dataset to logs/<slug>.log and prints a final summary table.
#
# Usage (run inside tmux so it survives SSH disconnect):
#   tmux new -s prep            # or: tmux attach -t prep
#   bash scripts/prep_new_datasets.sh
#
# Force a fresh re-prep of one dataset:  rm -rf data/preprocessed/<slug>
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1
PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Fast multi-threaded HF downloads.
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
"$PY" -c "import hf_transfer" 2>/dev/null || pip install -q hf_transfer || true

mkdir -p logs

# Common Voice 17 is gated (must accept terms + be logged in). Without a token
# it downloads NOTHING and the prep silently writes 0 clips. Detect that up
# front and drop common_voice_ar from the run with a clear message rather than
# wasting a process on a guaranteed-empty download.
HF_OK=1
if ! "$PY" -c "from huggingface_hub import HfApi; HfApi().whoami()" >/dev/null 2>&1; then
  HF_OK=0
  echo "[warn] Not logged in to Hugging Face."
  echo "       -> common_voice_ar is GATED and will be SKIPPED this run."
  echo "       To include it: run 'huggingface-cli login', accept terms at"
  echo "       hf.co/datasets/mozilla-foundation/common_voice_17_0, then re-run."
fi

# NEW datasets only. Phase-1 (sada22/worldspeech_*/mixat/nexdata) are NOT here
# on purpose — they already exist on disk and must be reused, not re-downloaded.
# Gated/request-only (ramsa/zaebuc/arzen/oman_speech) are excluded because they
# cannot be auto-downloaded; prep them later with --local-dir once obtained.
NEW_DATASETS=(
  masc            # ~1000h multi-dialect Arabic (clean subset)
  saudi_asrv1     # Saudi dialect ASR v1.0
  common_voice_ar # ~157h Arabic (MSA padding)
  scc22           # Saudi-English CS (eval-only)
  casablanca      # UAE Emirati benchmark (eval-only)
)

is_complete() {
  # A dataset counts as DONE only if its summary.json exists (the prep script
  # writes summary.json ONLY after a full successful pass) AND its clip count
  # is at/above the smoke-test floor (200). A leftover 200-clip smoke test is
  # NOT a completed full run, so it must be re-prepped, not skipped.
  local slug="$1"
  local s="data/preprocessed/${slug}/summary.json"
  [ -f "$s" ] || return 1
  local clips
  clips=$("$PY" -c "import json;print(json.load(open('$s')).get('clips',0))" 2>/dev/null || echo 0)
  # > 200 means a real full run, not a smoke test.
  [ "$clips" -gt 200 ] 2>/dev/null
}

echo "=== prep_new_datasets: $(date) ==="
echo "Python: $PY"
df -h . | tail -1
echo

pids=()
slugs=()
for slug in "${NEW_DATASETS[@]}"; do
  if [ "$slug" = "common_voice_ar" ] && [ "$HF_OK" -eq 0 ]; then
    echo "[skip] common_voice_ar: gated and not logged in (see warning above)."
    continue
  fi
  if is_complete "$slug"; then
    n=$("$PY" -c "import json;print(json.load(open('data/preprocessed/${slug}/summary.json')).get('clips',0))" 2>/dev/null || echo "?")
    echo "[skip] ${slug}: already fully prepped (${n} clips). rm -rf data/preprocessed/${slug} to redo."
    continue
  fi
  # Stale partial/smoke folder -> wipe so the prep script starts clean.
  if [ -d "data/preprocessed/${slug}" ]; then
    echo "[redo] ${slug}: clearing stale/partial folder before full prep."
    rm -rf "data/preprocessed/${slug}"
  fi
  echo "[start] ${slug} -> logs/${slug}.log (full, no cap)"
  "$PY" scripts/prepare_datasets.py --dataset "$slug" > "logs/${slug}.log" 2>&1 &
  pids+=("$!")
  slugs+=("$slug")
done

# Wait for every launched job and record its exit code.
fail=0
for i in "${!pids[@]}"; do
  slug="${slugs[$i]}"
  if wait "${pids[$i]}"; then
    # exit 0 is NOT enough: a dataset that downloaded nothing (e.g. gated /
    # not logged in) can exit 0 with 0 clips. Treat 0-clip output as a failure.
    clips=$("$PY" -c "import json;print(json.load(open('data/preprocessed/${slug}/summary.json')).get('clips',0))" 2>/dev/null || echo 0)
    if [ "${clips:-0}" -gt 0 ] 2>/dev/null; then
      echo "[done] ${slug} (${clips} clips)"
    else
      echo "[FAIL] ${slug}: exited cleanly but wrote 0 clips — likely gated / not logged in. See logs/${slug}.log"
      fail=1
    fi
  else
    rc=$?
    echo "[FAIL] ${slug} (exit ${rc}) — see logs/${slug}.log"
    fail=1
  fi
done

echo
echo "=== summary: $(date) ==="
printf "%-18s %10s %8s\n" "dataset" "clips" "hours"
for slug in "${NEW_DATASETS[@]}"; do
  s="data/preprocessed/${slug}/summary.json"
  if [ -f "$s" ]; then
    clips=$("$PY" -c "import json,sys;print(json.load(open('$s')).get('clips',0))" 2>/dev/null || echo "?")
    hours=$("$PY" -c "import json,sys;print(json.load(open('$s')).get('hours',0))" 2>/dev/null || echo "?")
    printf "%-18s %10s %8s\n" "$slug" "$clips" "$hours"
  else
    printf "%-18s %10s %8s\n" "$slug" "MISSING" "-"
  fi
done

echo
if [ "$fail" -ne 0 ]; then
  echo "ONE OR MORE DATASETS FAILED — check the logs above."
  exit 1
fi
echo "All new datasets prepared. Phase-1 corpus on disk was reused (not touched)."
