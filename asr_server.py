"""Standalone ASR-only server (Qwen3-ASR-1.7B + Gulf LoRA).

Exposes ONLY the speech-to-text endpoint — no correction pipeline, no UI app.
Intended to be tunneled via ngrok so a teammate without DGX access can hit the
same ASR model we use in the main app.

Run on the DGX:
    cd /home/abder/abder/transcription/transcription-pipeline
    source .venv/bin/activate
    uvicorn asr_server:app --host 0.0.0.0 --port 8010

Then expose it:
    ngrok http 8010

Endpoints:
    GET  /            -> tiny browser test page (record / upload audio)
    GET  /health      -> {"status": "ok"} once the model is loaded
    POST /asr         -> multipart form: audio=<file>, language=<ar|en|"">
                         returns {"text", "raw_text", "language", "duration"}
"""

from __future__ import annotations

import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.services import asr

DEFAULT_LANGUAGE = os.environ.get("ASR_LANGUAGE", "")  # "" = auto-detect
DEFAULT_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")

app = FastAPI(title="Gulf Arabic Medical ASR — standalone", version="1.0.0")

# Allow the teammate to call the ngrok URL from any browser / origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_UPLOAD_DIR = Path(tempfile.gettempdir()) / "asr_standalone_uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _save_upload(upload: UploadFile) -> Path:
    suffix = Path(upload.filename or "audio").suffix or ".webm"
    dest = _UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    with dest.open("wb") as fh:
        shutil.copyfileobj(upload.file, fh)
    return dest


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/asr")
async def asr_endpoint(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model_size: str = Form(DEFAULT_MODEL_SIZE),
) -> Dict[str, Any]:
    """Transcribe an uploaded audio file with the Gulf LoRA ASR model."""
    effective_lang = language if language is not None else DEFAULT_LANGUAGE
    dest = _save_upload(audio)
    size = dest.stat().st_size
    print(
        f"[asr-standalone] file={audio.filename!r} type={audio.content_type!r} "
        f"size={size}B lang={effective_lang!r}"
    )
    if size < 200:
        dest.unlink(missing_ok=True)
        return JSONResponse(
            status_code=400, content={"error": f"audio file is too small ({size} bytes)"}
        )
    try:
        result = asr.transcribe(
            dest, model_size=model_size, language=effective_lang or None
        )
        return {
            "text": result.get("text", ""),
            "raw_text": result.get("raw_text", result.get("text", "")),
            "language": result.get("language", effective_lang),
            "duration": result.get("duration", 0.0),
        }
    except Exception as exc:  # noqa: BLE001 — surface the error to the caller
        print(f"[asr-standalone] error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        dest.unlink(missing_ok=True)


_TEST_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Gulf Arabic Medical ASR</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 1.3rem; }
  button { font-size: 1rem; padding: 8px 16px; margin: 4px 4px 4px 0; cursor: pointer; }
  select, input[type=file] { font-size: 1rem; padding: 4px; }
  #out { white-space: pre-wrap; background: #f4f4f5; border: 1px solid #ddd;
         border-radius: 8px; padding: 16px; margin-top: 16px; min-height: 48px; }
  .row { margin: 12px 0; }
  .muted { color: #666; font-size: 0.85rem; }
</style>
</head>
<body>
  <h1>Gulf Arabic Medical ASR</h1>
  <p class="muted">Qwen3-ASR-1.7B + Gulf LoRA. Record from the mic or upload a file.</p>

  <div class="row">
    <label>Language:
      <select id="lang">
        <option value="">Auto-detect</option>
        <option value="ar">Arabic</option>
        <option value="en">English</option>
      </select>
    </label>
  </div>

  <div class="row">
    <button id="recBtn">● Record</button>
    <span id="recState" class="muted"></span>
  </div>

  <div class="row">
    <input type="file" id="file" accept="audio/*" />
    <button id="uploadBtn">Transcribe file</button>
  </div>

  <div id="out">Transcript will appear here…</div>

<script>
const out = document.getElementById('out');
const langSel = document.getElementById('lang');

async function send(blob, filename) {
  out.textContent = 'Transcribing…';
  const fd = new FormData();
  fd.append('audio', blob, filename);
  fd.append('language', langSel.value);
  try {
    const r = await fetch('/asr', { method: 'POST', body: fd });
    const j = await r.json();
    if (j.error) { out.textContent = 'Error: ' + j.error; return; }
    out.textContent = j.text || '(empty)';
  } catch (e) {
    out.textContent = 'Request failed: ' + e;
  }
}

// File upload
document.getElementById('uploadBtn').onclick = () => {
  const f = document.getElementById('file').files[0];
  if (!f) { out.textContent = 'Pick a file first.'; return; }
  send(f, f.name);
};

// Mic recording
let mediaRecorder = null, chunks = [];
const recBtn = document.getElementById('recBtn');
const recState = document.getElementById('recState');
recBtn.onclick = async () => {
  if (mediaRecorder && mediaRecorder.state === 'recording') {
    mediaRecorder.stop();
    return;
  }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  mediaRecorder = new MediaRecorder(stream);
  chunks = [];
  mediaRecorder.ondataavailable = e => chunks.push(e.data);
  mediaRecorder.onstop = () => {
    stream.getTracks().forEach(t => t.stop());
    const blob = new Blob(chunks, { type: 'audio/webm' });
    send(blob, 'recording.webm');
    recBtn.textContent = '● Record';
    recState.textContent = '';
  };
  mediaRecorder.start();
  recBtn.textContent = '■ Stop';
  recState.textContent = 'recording…';
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _TEST_PAGE
