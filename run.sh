#!/usr/bin/env bash
# Run the medical transcription pipeline locally.
#
# Usage:
#   ./run.sh                          # default: large-v3-turbo, English, port 8000
#   PORT=9000 ./run.sh                # custom port
#   USE_LLM=0 ./run.sh                # disable LLM (when Ollama is unreachable)
#   WHISPER_MODEL_SIZE=base ./run.sh  # use a smaller/faster Whisper model

set -e

# Resolve project root (where this script lives)
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# Defaults — overridable from the environment
: "${WHISPER_MODEL_SIZE:=large-v3-turbo}"
: "${WHISPER_LANGUAGE:=en}"
: "${USE_LLM:=1}"
: "${KG_ENTITIES_PATH:=data/medical_entities.json}"
: "${LLM_MODEL_GENERAL:=MaziyarPanahi/Calme-7B-Instruct-v0.2}"
: "${LLM_MODEL_MEDICAL:=MaziyarPanahi/Calme-7B-Instruct-v0.2}"
: "${LLM_MODEL_VERIFY:=MaziyarPanahi/Calme-7B-Instruct-v0.2}"
: "${HOST:=127.0.0.1}"
: "${PORT:=8000}"

# Activate the virtualenv if it exists
if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
else
    echo "[run.sh] No .venv/ found. Create it first:"
    echo "         python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Stop any previous instance on the same port
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

echo "[run.sh] Starting server on http://${HOST}:${PORT}"
echo "[run.sh] Whisper:  ${WHISPER_MODEL_SIZE}  (lang=${WHISPER_LANGUAGE})"
echo "[run.sh] LLM:      ${USE_LLM}  (1=on / 0=off)"
echo "[run.sh] KG:       ${KG_ENTITIES_PATH}"
echo "[run.sh] Models:   ${LLM_MODEL_GENERAL}"
echo

export WHISPER_MODEL_SIZE WHISPER_LANGUAGE USE_LLM KG_ENTITIES_PATH
export LLM_MODEL_GENERAL LLM_MODEL_MEDICAL LLM_MODEL_VERIFY
exec uvicorn app.main:app --host "$HOST" --port "$PORT"
