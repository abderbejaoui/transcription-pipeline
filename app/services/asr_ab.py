"""A/B inference service for the v2 medical LoRA arms.

This is a *qualitative* tester for the dashboard: record your own voice and
see what each of the two fine-tuned arms transcribes. It is completely
independent from the production ASR mode (services.asr) — loading these models
does NOT touch or replace the Gulf v1 model used by /api/transcribe.

Arms
----
  A = stock base (Qwen/Qwen3-ASR-1.7B)        + medical LoRA A
  B = Gulf-merged base (runs/qwen3_gulf_merged_base) + medical LoRA B

Both arms are loaded lazily on first request and then cached for the life of
the process. They use the SAME qwen_asr wrapper API as services.asr — i.e.
`wrapper.transcribe(audio=path, language=..., context=...)` — which is the
correct call (the manual processor(...) path in scripts/infer_v2.py is what
caused the "string pattern on a bytes-like object" error).
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Arm definitions. Override via env if the checkpoint dirs ever move.
ARMS: Dict[str, Dict[str, str]] = {
    "A": {
        "base": os.environ.get("V2_ARM_A_BASE", "Qwen/Qwen3-ASR-1.7B"),
        "adapter": os.environ.get(
            "V2_ARM_A_ADAPTER", "runs/qwen3_lora_v2_medical_A/final_adapter"
        ),
        "label": "Arm A · stock base + medical LoRA",
    },
    "B": {
        "base": os.environ.get("V2_ARM_B_BASE", "runs/qwen3_gulf_merged_base"),
        "adapter": os.environ.get(
            "V2_ARM_B_ADAPTER", "runs/qwen3_lora_v2_medical_B/final_adapter"
        ),
        "label": "Arm B · Gulf-merged base + medical LoRA",
    },
}

# Cache: arm -> loaded wrapper.
_MODELS: Dict[str, Any] = {}
_DEVICE: Optional[str] = None
_DTYPE = None


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _device_dtype():
    global _DEVICE, _DTYPE
    if _DEVICE is not None:
        return _DEVICE, _DTYPE
    import torch

    if torch.cuda.is_available():
        _DEVICE, _DTYPE = "cuda:0", torch.bfloat16
    elif torch.backends.mps.is_available():
        _DEVICE, _DTYPE = "mps", torch.float16
    else:
        _DEVICE, _DTYPE = "cpu", torch.float32
    return _DEVICE, _DTYPE


def _load_arm(arm: str):
    """Load (and cache) the base+LoRA wrapper for one arm."""
    if arm in _MODELS:
        return _MODELS[arm]
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r} (expected one of {list(ARMS)})")

    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    cfg = ARMS[arm]
    device, dtype = _device_dtype()
    base = cfg["base"]
    # A base may be a local merged-model dir (Arm B) or an HF repo id (Arm A).
    base_path = _abs(base)
    base_ref = str(base_path) if base_path.exists() else base
    adapter_dir = _abs(cfg["adapter"])

    print(f"[asr_ab] loading arm {arm}: base={base_ref} on {device} ({dtype})")
    wrapper = Qwen3ASRModel.from_pretrained(
        base_ref, dtype=dtype, device_map=device, max_new_tokens=1024
    )

    if adapter_dir.exists():
        print(f"[asr_ab] arm {arm}: attaching LoRA adapter {adapter_dir}")
        inner = getattr(wrapper, "model", None) or wrapper
        peft_model = PeftModel.from_pretrained(inner, str(adapter_dir)).to(device).eval()
        if hasattr(wrapper, "model"):
            wrapper.model = peft_model
        else:
            wrapper = peft_model
    else:
        print(f"[asr_ab] arm {arm}: WARNING adapter not found at {adapter_dir}, using base only")

    _MODELS[arm] = wrapper
    return wrapper


def _lang_label(language: Optional[str]) -> Optional[str]:
    if not language or language in ("", "auto"):
        return None
    low = language.lower()
    if low in ("ar", "arabic"):
        return "Arabic"
    if low in ("en", "english"):
        return "English"
    return None


def transcribe_one(arm: str, audio_path: str | Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a single clip with one arm. Errors are returned, not raised,
    so one failing arm doesn't break the other in the A/B view."""
    cfg = ARMS.get(arm, {})
    try:
        wrapper = _load_arm(arm)
        lang_label = _lang_label(language)
        t0 = time.time()
        results = wrapper.transcribe(audio=str(audio_path), language=lang_label)
        text = getattr(results[0], "text", "").strip() if results else ""
        return {
            "arm": arm,
            "label": cfg.get("label", arm),
            "text": text,
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as exc:  # noqa: BLE001 — surface per-arm failures to the UI
        return {
            "arm": arm,
            "label": cfg.get("label", arm),
            "text": "",
            "error": str(exc),
        }


def transcribe_ab(audio_path: str | Path, language: Optional[str] = None) -> Dict[str, Any]:
    """Run BOTH arms on the same clip. Returns {arm_a, arm_b}."""
    return {
        "arm_a": transcribe_one("A", audio_path, language=language),
        "arm_b": transcribe_one("B", audio_path, language=language),
    }
