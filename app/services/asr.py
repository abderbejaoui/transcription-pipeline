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

from .drug_normalize import normalize_drugs

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_MODEL = None
_PROCESSOR = None
_DEVICE = None
_DTYPE = None


def _qwen3_asr_in_transformers() -> bool:
    try:
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        return "qwen3_asr" in CONFIG_MAPPING_NAMES
    except Exception:
        return False


def _load_model():
    global _MODEL, _PROCESSOR, _DEVICE, _DTYPE
    if _MODEL is not None:
        return _MODEL, _PROCESSOR

    import torch
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

    if _qwen3_asr_in_transformers():
        from transformers import AutoModelForCausalLM, AutoProcessor
        _PROCESSOR = AutoProcessor.from_pretrained(base_repo, trust_remote_code=True)
        base = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=_DTYPE, trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(_DEVICE).eval()
    else:
        # Fallback: qwen-asr pip wrapper (needed when transformers doesn't register qwen3_asr)
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "qwen3_asr is not registered in transformers and qwen-asr wrapper is not installed. "
                "Run: pip install qwen-asr"
            ) from exc
        print(f"[asr] using qwen-asr wrapper (transformers too old for qwen3_asr)")
        _MODEL = Qwen3ASRModel.from_pretrained(
            base_repo, dtype=_DTYPE, device_map=_DEVICE, max_new_tokens=1024,
        )
        _PROCESSOR = None  # wrapper handles its own processor
        if adapter_dir.exists():
            print(f"[asr] attaching LoRA adapter: {adapter_dir}")
            inner = getattr(_MODEL, "model", None) or _MODEL
            peft_model = PeftModel.from_pretrained(inner, str(adapter_dir)).to(_DEVICE).eval()
            if hasattr(_MODEL, "model"):
                _MODEL.model = peft_model
            else:
                _MODEL = peft_model
        return _MODEL, _PROCESSOR

    if adapter_dir.exists():
        print(f"[asr] attaching LoRA adapter: {adapter_dir}")
        _MODEL = PeftModel.from_pretrained(base, str(adapter_dir)).to(_DEVICE).eval()
    else:
        print(f"[asr] WARNING: adapter not found at {adapter_dir}, using base model")
        _MODEL = base

    return _MODEL, _PROCESSOR


def _to_wav(audio_path: Path) -> Path:
    """Convert any audio format to a 16kHz mono WAV using ffmpeg.
    Returns a temp WAV path (caller must delete it). If already WAV, returns as-is."""
    suffix = audio_path.suffix.lower()
    if suffix in (".wav",):
        return audio_path
    import subprocess, tempfile
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-f", "wav", str(tmp)],
        check=True, capture_output=True,
    )
    return tmp


def _load_medical_context() -> str:
    """Load medical terms as a context string for Qwen3-ASR biasing.

    The model biases toward emitting these terms when they sound similar.
    Source: medical_terms.txt (one term per line) + lexicon if available.
    Returns a comma-separated string, capped at ~500 terms to keep prompt short.
    """
    terms: list[str] = []
    medical_file = PROJECT_ROOT / "medical_terms.txt"
    if medical_file.exists():
        for line in medical_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    # Cap to avoid bloating the prompt (Qwen recommends < 500 terms)
    return ", ".join(terms[:500])


def transcribe(audio_path: str | Path, model_size: str = "large-v3", language: Optional[str] = None, context: Optional[str] = None) -> Dict[str, Any]:
    """Transcribe a wav/webm/mp3 file using the Gulf Arabic LoRA model.

    `context` is a free-text string of domain vocabulary (medical terms,
    brand names, etc.) that Qwen3-ASR biases its decoding toward.
    Defaults to the medical_terms.txt file at the project root.
    """
    import torch
    import soundfile as sf

    model, processor = _load_model()

    # Default to medical terms from medical_terms.txt
    if context is None:
        context = _load_medical_context()

    audio_path = Path(audio_path)
    tmp_wav = None
    try:
        wav_path = _to_wav(audio_path)
        if wav_path != audio_path:
            tmp_wav = wav_path
        audio, sr = sf.read(str(wav_path), dtype="float32")
    except Exception as exc:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
        return {"text": "", "language": language or "ar", "language_probability": 0.0, "duration": 0.0, "words": [], "error": str(exc)}
    # NOTE: do NOT delete tmp_wav here. The qwen-asr wrapper path below
    # re-reads the audio FILE by path; if we delete the clean 16k mono WAV
    # now, the wrapper falls back to decoding the original webm/opus via
    # librosa->audioread ("PySoundFile failed. Trying audioread instead."),
    # which produces garbled samples and worse transcripts. We hand the
    # wrapper the clean wav_path and only delete tmp_wav at the very end.

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr if sr > 0 else 0.0

    # Map UI language codes to Qwen3-ASR language labels.
    # "" / None / "auto" -> None (model auto-detects, handles code-switching)
    # "ar" / "arabic"   -> "Arabic"  (forces Arabic decoding)
    # "en" / "english"  -> "English" (forces English decoding)
    if language is None or language == "" or language == "auto":
        lang_label = None
    elif language.lower() in ("ar", "arabic"):
        lang_label = "Arabic"
    elif language.lower() in ("en", "english"):
        lang_label = "English"
    else:
        lang_label = None

    # qwen_asr wrapper path (older transformers)
    if processor is None:
        t0 = time.time()
        # Hand the wrapper the CLEAN 16k mono WAV (wav_path), not the raw
        # webm/opus. Passing the original triggers the librosa->audioread
        # fallback ("PySoundFile failed") and garbled samples. wav_path is
        # the ffmpeg-converted file produced above and is still on disk.
        kwargs = {"audio": str(wav_path), "language": lang_label}
        if context:
            kwargs["context"] = context
            print(f"[asr] using context bias: {len(context)} chars, "
                  f"{context.count(',') + 1} terms")
        try:
            results = model.transcribe(**kwargs)
            raw_text = getattr(results[0], "text", "").strip() if results else ""
        finally:
            # Clean up the temp WAV now that the wrapper is done reading it.
            if tmp_wav and tmp_wav.exists():
                tmp_wav.unlink(missing_ok=True)
        # Phonetic drug-name canonicalization: map Arabic-script brand names
        # (بنادول، دوليبران …) back to their Latin spelling. Drug-only, deterministic.
        text, drug_fixes = normalize_drugs(raw_text)
        return {
            "text": text,
            "raw_text": raw_text,
            "drug_corrections": drug_fixes,
            "language": language or "ar",
            "language_probability": 1.0,
            "duration": duration,
            "words": [],
        }

    # transformers path: we already have the decoded `audio` array in memory,
    # so the temp WAV is no longer needed. Delete it now.
    if tmp_wav and tmp_wav.exists():
        tmp_wav.unlink(missing_ok=True)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Build prompt with optional context biasing for medical/domain terms.
    # Qwen3-ASR uses the format: "Context: term1, term2, ... <|audio_1|>Transcribe..."
    context_prefix = f"Context: {context}\n" if context else ""
    lang_suffix = f" in {lang_label}" if lang_label else ""
    user_msg = f"{context_prefix}<|audio_1|>Transcribe the audio{lang_suffix}."
    if context:
        print(f"[asr] using context bias: {len(context)} chars, "
              f"{context.count(',') + 1} terms")
    inputs = processor(text=user_msg, audios=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    raw_text = processor.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)[0].strip()
    # Phonetic drug-name canonicalization: map Arabic-script brand names
    # (بنادول، دوليبران …) back to their Latin spelling. Drug-only, deterministic.
    text, drug_fixes = normalize_drugs(raw_text)

    return {
        "text": text,
        "raw_text": raw_text,
        "drug_corrections": drug_fixes,
        "language": language or "ar",
        "language_probability": 1.0,
        "duration": duration,
        "words": [],
    }
