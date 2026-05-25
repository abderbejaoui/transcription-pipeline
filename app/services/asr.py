"""Speech-to-text service — Gulf Arabic Qwen3-ASR-1.7B LoRA backend.

Uses the fine-tuned Gulf Arabic LoRA adapter on top of Qwen3-ASR-1.7B.
The adapter path is controlled by the QWEN3_GULF_ADAPTER env var
(default: runs/qwen3_lora_r6/final_adapter relative to the project root).

Set QWEN3_GULF_ADAPTER to point at any other checkpoint directory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_DTYPE = None


def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE, _DTYPE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from peft import PeftModel

    base_repo = os.environ.get("QWEN3_ASR_BASE", "Qwen/Qwen3-ASR-1.7B")
    adapter_path = os.environ.get(
        "QWEN3_GULF_ADAPTER", "runs/qwen3_lora_r6/final_adapter"
    )
    adapter_dir = Path(adapter_path)
    if not adapter_dir.is_absolute():
        adapter_dir = (PROJECT_ROOT / adapter_dir).resolve()

    if torch.cuda.is_available():
        _DEVICE, _DTYPE = "cuda:0", torch.bfloat16
    elif torch.backends.mps.is_available():
        _DEVICE, _DTYPE = "mps", torch.float16
    else:
        _DEVICE, _DTYPE = "cpu", torch.float32

    print(f"[asr] loading {base_repo} on {_DEVICE} ({_DTYPE})")
    _PROCESSOR = AutoProcessor.from_pretrained(base_repo, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        base_repo,
        torch_dtype=_DTYPE,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(_DEVICE).eval()

    if adapter_dir.exists():
        print(f"[asr] attaching LoRA adapter: {adapter_dir}")
        _MODEL = PeftModel.from_pretrained(base, str(adapter_dir)).to(_DEVICE).eval()
    else:
        print(f"[asr] WARNING: adapter not found at {adapter_dir}, using base model")
        _MODEL = base

    return _MODEL, _PROCESSOR


def transcribe(audio_path: str | Path, model_size: str = "large-v3", language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a wav/webm/mp3 file using the Gulf Arabic LoRA model."""
    import torch
    import soundfile as sf

    model, processor = _load_model()

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr if sr > 0 else 0.0
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    lang_label = "Arabic" if (language or "ar") in ("ar", "arabic") else None
    user_msg = (
        f"<|audio_1|>Transcribe the audio in {lang_label}."
        if lang_label else
        "<|audio_1|>Transcribe the audio."
    )

    inputs = processor(
        text=user_msg,
        audios=[audio],
        sampling_rate=sr,
        return_tensors="pt",
    )
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    t0 = time.time()
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    text = processor.batch_decode(
        out_ids[:, input_len:], skip_special_tokens=True
    )[0].strip()

    # Return in the same shape the pipeline expects from faster-whisper.
    # No word-level timestamps (Qwen3-ASR doesn't produce them), so words=[].
    return {
        "text": text,
        "language": language or "ar",
        "language_probability": 1.0,
        "duration": duration,
        "words": [],
    }
