"""
VoxCPM2 TTS – FastAPI server for Gulf Arabic synthesis.

Run on DGX:
    pip install voxcpm fastapi uvicorn

    # Optional: for best quality voice cloning
    # pip install voxcpm[denoiser]

    python scripts/tts_server.py --port 7900

Endpoints:
    POST /tts          → synthesize text, return WAV bytes
    POST /tts/file     → synthesize text, save to disk, return path
    GET  /health       → liveness check
"""
import argparse
import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI(title="Gulf Arabic TTS (VoxCPM2)")

# ── globals ──────────────────────────────────────────────────────────────────
model = None
SAMPLE_RATE = 48_000


class TTSRequest(BaseModel):
    text: str
    # Voice design: describe the voice in natural language (optional)
    # e.g. "(Gulf Arabic male doctor, calm professional tone)"
    voice_description: str | None = None
    # Voice cloning: path to reference WAV (optional)
    reference_wav_path: str | None = None
    reference_text: str | None = None
    # Generation params
    cfg_value: float = 2.0
    inference_timesteps: int = 10


class TTSFileRequest(TTSRequest):
    output_path: str = "out.wav"


# ── endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "VoxCPM2", "sample_rate": SAMPLE_RATE}


@app.post("/tts")
def synthesize(req: TTSRequest):
    wav = _generate(req)
    buf = io.BytesIO()
    sf.write(buf, wav, SAMPLE_RATE, format="wav")
    buf.seek(0)
    return Response(content=buf.read(), media_type="audio/wav")


@app.post("/tts/file")
def synthesize_to_file(req: TTSFileRequest):
    wav = _generate(req)
    Path(req.output_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(req.output_path, wav, SAMPLE_RATE)
    dur = len(wav) / SAMPLE_RATE
    return {"path": req.output_path, "samples": len(wav), "duration_s": round(dur, 2)}


def _generate(req: TTSRequest) -> np.ndarray:
    if model is None:
        raise HTTPException(503, "Model not loaded yet")

    t0 = time.time()

    # Build the text: prepend voice description if provided
    text = req.text
    if req.voice_description:
        text = f"({req.voice_description}){text}"

    kwargs = dict(
        text=text,
        cfg_value=req.cfg_value,
        inference_timesteps=req.inference_timesteps,
    )

    # Voice cloning mode
    if req.reference_wav_path:
        kwargs["reference_wav_path"] = req.reference_wav_path
        # Ultimate cloning: provide transcript for max fidelity
        if req.reference_text:
            kwargs["prompt_wav_path"] = req.reference_wav_path
            kwargs["prompt_text"] = req.reference_text

    wav = model.generate(**kwargs)
    dur = len(wav) / SAMPLE_RATE
    print(f"[TTS] {len(req.text)} chars → {dur:.1f}s  ({time.time()-t0:.1f}s)")
    return wav


# ── startup ──────────────────────────────────────────────────────────────────
def load_model():
    global model, SAMPLE_RATE
    from voxcpm import VoxCPM

    print("[TTS] Loading VoxCPM2 …")
    model = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)
    SAMPLE_RATE = model.tts_model.sample_rate
    print(f"[TTS] Ready  (sr={SAMPLE_RATE})")


if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7900)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    load_model()
    uvicorn.run(app, host=args.host, port=args.port)
