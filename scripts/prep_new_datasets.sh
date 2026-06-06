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

manifest_has_clips() {
  # returns 0 if data/preprocessed/<slug>/manifest.jsonl exists and is non-empty
  local slug="$1"
  local mf="data/preprocessed/${slug}/manifest.jsonl"
  [ -s "$mf" ]
}

echo "=== prep_new_datasets: $(date) ==="
echo "Python: $PY"
df -h . | tail -1
echo

pids=()
slugs=()
for slug in "${NEW_DATASETS[@]}"; do
  if manifest_has_clips "$slug"; then
    n=$(wc -l < "data/preprocessed/${slug}/manifest.jsonl")
    echo "[skip] ${slug}: already prepped (${n} clips). rm -rf data/preprocessed/${slug} to redo."
    continue
  fi
  echo "[start] ${slug} -> logs/${slug}.log (full, no cap)"
  "$PY" scripts/prepare_datasets.py --dataset "$slug" > "logs/${slug}.log" 2>&1 &
  pids+=("$!")
  slugs+=("$slug")
done

# Wait for every launched job and record its exit code.
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[done] ${slugs[$i]} (exit 0)"
  else
    rc=$?
    echo "[FAIL] ${slugs[$i]} (exit ${rc}) — see logs/${slugs[$i]}.log"
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
