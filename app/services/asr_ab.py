"""A/B inference service for the v2 medical LoRA arms.

This is a *qualitative* tester for the dashboard: record your own voice and
see what each of the two fine-tuned arms transcribes. It is completely
independent from the production ASR mode (services.asr) — loading these models
does NOT touch or replace the Gulf v1 model used by /api/transcribe.

Arms
----
  A = stock base (Qwen/Qwen3-ASR-1.7B)              + medical LoRA A
  B = Gulf-merged base (runs/qwen3_gulf_merged_base) + medical LoRA B
  C = Gulf-merged base (runs/qwen3_gulf_merged_base) + v3 medical LoRA C
      (real-dominant mix, LR 1e-5, val-set early-stop)

All arms are loaded lazily on first request and then cached for the life of
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

from .drug_normalize import normalize_drugs

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
    "C": {
        "base": os.environ.get("V3_ARM_C_BASE", "runs/qwen3_gulf_merged_base"),
        # Default to final_adapter, NOT best_adapter: the v3 run's eval was
        # broken (every WER=nan) so best_adapter/ was never written. Only
        # final_adapter exists. Override with V3_ARM_C_ADAPTER once a run
        # with working eval produces a real best_adapter.
        "adapter": os.environ.get(
            "V3_ARM_C_ADAPTER", "runs/qwen3_lora_v3_B/final_adapter"
        ),
        "label": "Arm C · v3 (real-dominant mix, val early-stop)",
    },
}

# Cache: arm -> loaded wrapper.
_MODELS: Dict[str, Any] = {}
_DEVICE: Optional[str] = None
_DTYPE = None


def _abs(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _to_wav(audio_path: Path) -> Path:
    """Convert any audio (webm/opus/mp3/...) to 16kHz mono WAV via ffmpeg.

    Mirrors services.asr._to_wav. The browser records webm/opus, which the
    qwen_asr wrapper decodes unreliably (librosa falls back to audioread and
    can produce garbled samples -> hallucinated transcripts). Feeding a clean
    16k mono WAV fixes this. Returns a temp path the caller must delete; if the
    input is already a .wav it is returned unchanged.
    """
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    import subprocess
    import tempfile

    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp)],
        check=True,
        capture_output=True,
    )
    return tmp


def _load_medical_context() -> str:
    """Comma-separated medical terms for Qwen3-ASR context biasing.

    Same source as services.asr — medical_terms.txt at the project root,
    capped at ~500 terms. This is what biases the model toward drug/brand
    names (doliprane, novadol, ...) instead of inventing plausible Arabic.
    """
    terms: list[str] = []
    medical_file = PROJECT_ROOT / "medical_terms.txt"
    if medical_file.exists():
        for line in medical_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    return ", ".join(terms[:500])


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


def _run_full_pipeline(
    audio_path: str | Path,
    transcript: str,
    *,
    use_llm_flag: bool = True,
) -> Dict[str, Any]:
    """Run the SAME downstream pipeline the Record button uses on one
    transcript: CTC word alignment -> suspicious-word flagging -> stitch
    timestamps -> auto-apply high-confidence corrections.

    Imported lazily so the A/B view stays fast when the pipeline isn't
    requested, and so a missing/broken stage degrades gracefully instead
    of taking down the arm. Returns words/flags/corrected_transcript/
    auto_corrections (each stage independently guarded)."""
    from . import alignment_v2 as _alignment, flag as _flag

    # 1) Word-level CTC forced alignment of the full transcript.
    try:
        words_aligned = _alignment.align_words(str(audio_path), transcript)
    except Exception as exc:  # noqa: BLE001
        print(f"[ab pipeline] alignment failed: {exc!r}")
        words_aligned = []

    # 2) Flag suspicious words (phonetic + optional LLM).
    try:
        flags = _flag.flag_suspicious(transcript, use_llm=use_llm_flag)
    except Exception as exc:  # noqa: BLE001
        print(f"[ab pipeline] flagging failed: {exc!r}")
        flags = []

    # 3) Stitch alignment into each flag so the UI can slice the audio.
    for f in flags:
        idx = f.get("index")
        if isinstance(idx, int) and 0 <= idx < len(words_aligned):
            f["start_s"] = words_aligned[idx].get("start_s")
            f["end_s"] = words_aligned[idx].get("end_s")
            f["alignment_confidence"] = words_aligned[idx].get("confidence", 0.0)
        else:
            f["start_s"] = None
            f["end_s"] = None
            f["alignment_confidence"] = 0.0

    # 4) Auto-apply HIGH-confidence corrections (separate string for compare).
    try:
        corrected = _flag.apply_high_confidence_corrections(transcript, flags)
    except Exception as exc:  # noqa: BLE001
        print(f"[ab pipeline] auto-correct failed: {exc!r}")
        corrected = {
            "corrected_transcript": transcript,
            "applied": [],
            "threshold": 0.90,
        }

    return {
        "words": words_aligned,
        "flags": flags,
        "corrected_transcript": corrected["corrected_transcript"],
        "auto_corrections": corrected["applied"],
        "correction_threshold": corrected["threshold"],
    }


def transcribe_one(
    arm: str,
    audio_path: str | Path,
    language: Optional[str] = None,
    context: Optional[str] = None,
    run_pipeline: bool = False,
) -> Dict[str, Any]:
    """Transcribe a single clip with one arm. Errors are returned, not raised,
    so one failing arm doesn't break the other in the A/B view.

    `audio_path` should already be a clean 16k mono WAV (see transcribe_ab).
    `context` is the medical-term bias string (same as services.asr).
    When `run_pipeline` is True, the full Record-button pipeline (alignment +
    flagging + auto-correction) is run on this arm's normalized transcript."""
    cfg = ARMS.get(arm, {})
    try:
        wrapper = _load_arm(arm)
        lang_label = _lang_label(language)
        kwargs: Dict[str, Any] = {"audio": str(audio_path), "language": lang_label}
        if context:
            kwargs["context"] = context
        t0 = time.time()
        results = wrapper.transcribe(**kwargs)
        raw_text = getattr(results[0], "text", "").strip() if results else ""
        # Map Arabic-script drug names back to canonical Latin (panadol,
        # doliprane, ...). The ASR hears them correctly but transliterates;
        # this is a deterministic, drug-only post-fix (see drug_normalize).
        text, drug_fixes = normalize_drugs(raw_text)
        out: Dict[str, Any] = {
            "arm": arm,
            "label": cfg.get("label", arm),
            "text": text,
            "raw_text": raw_text,
            "drug_corrections": drug_fixes,
            "elapsed_s": round(time.time() - t0, 2),
        }
        if run_pipeline:
            # The pipeline runs on the drug-normalized transcript — the same
            # text shown as "Final transcript" — so flags/corrections line up.
            out["pipeline"] = _run_full_pipeline(audio_path, text)
            out["pipeline_elapsed_s"] = round(time.time() - t0, 2)
        return out
    except Exception as exc:  # noqa: BLE001 — surface per-arm failures to the UI
        return {
            "arm": arm,
            "label": cfg.get("label", arm),
            "text": "",
            "error": str(exc),
        }


def transcribe_ab(
    audio_path: str | Path,
    language: Optional[str] = None,
    run_pipeline: bool = False,
) -> Dict[str, Any]:
    """Run ALL arms on the same clip. Returns {arm_a, arm_b, arm_c}.

    Converts the (usually webm/opus) recording to a clean 16k mono WAV ONCE
    and loads the medical context, then feeds every arm identical input — the
    same path the production ASR uses, which avoids the decode-garbage
    hallucination we saw with raw webm.

    An arm whose adapter is missing (e.g. v3 still training) returns an error
    field but does NOT take down the other arms — that's why each call is
    wrapped in transcribe_one which returns errors instead of raising.

    When `run_pipeline` is True, each arm also runs the full Record-button
    pipeline (alignment + flagging + auto-correction) so the A/B view shows
    everything that happens to each model's output.
    """
    src = Path(audio_path)
    context = _load_medical_context()
    tmp_wav: Optional[Path] = None
    try:
        wav_path = _to_wav(src)
        if wav_path != src:
            tmp_wav = wav_path
        wav_str = str(wav_path)
        return {
            "arm_a": transcribe_one(
                "A", wav_str, language=language, context=context,
                run_pipeline=run_pipeline,
            ),
            "arm_b": transcribe_one(
                "B", wav_str, language=language, context=context,
                run_pipeline=run_pipeline,
            ),
            "arm_c": transcribe_one(
                "C", wav_str, language=language, context=context,
                run_pipeline=run_pipeline,
            ),
        }
    finally:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
