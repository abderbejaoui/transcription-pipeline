"""Dual-ASR transcription with LLM-as-judge merging.

Runs the audio through TWO ASR backends in parallel:
  A) The fine-tuned Gulf Arabic LoRA (best for dialect)
  B) The base Qwen3-ASR-1.7B  (best for code-switched English)

Then asks Calme-3.2-78B (via Ollama) to merge them into a final transcript
that keeps the strengths of both — Gulf Arabic vocabulary/morphology from A,
English brand/medical terms preserved in Latin script from B.

Public API
----------
transcribe_and_merge(audio_path, language=None) -> Dict[str, Any]
    Same return shape as services.asr.transcribe(), plus an `extra` dict
    with both raw transcripts and the LLM's merge reasoning.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Optional

from . import asr as asr_module
from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)

# ---------------------------------------------------------------------------
# Base Qwen3 (no LoRA) — loaded lazily, separate from the Gulf LoRA instance.
# ---------------------------------------------------------------------------

_BASE_MODEL = None
_BASE_PROCESSOR = None
_BASE_DEVICE = None
_BASE_DTYPE = None


def _load_base_model():
    """Load Qwen3-ASR-1.7B WITHOUT the LoRA adapter, in a separate slot."""
    global _BASE_MODEL, _BASE_PROCESSOR, _BASE_DEVICE, _BASE_DTYPE
    if _BASE_MODEL is not None:
        return _BASE_MODEL, _BASE_PROCESSOR

    import torch

    base_repo = os.environ.get("QWEN3_ASR_BASE", "Qwen/Qwen3-ASR-1.7B")

    if torch.cuda.is_available():
        _BASE_DEVICE, _BASE_DTYPE = "cuda:0", torch.bfloat16
    elif torch.backends.mps.is_available():
        _BASE_DEVICE, _BASE_DTYPE = "mps", torch.float16
    else:
        _BASE_DEVICE, _BASE_DTYPE = "cpu", torch.float32

    print(f"[asr_dual] loading base {base_repo} on {_BASE_DEVICE} ({_BASE_DTYPE})")

    if asr_module._qwen3_asr_in_transformers():
        from transformers import AutoModelForCausalLM, AutoProcessor
        _BASE_PROCESSOR = AutoProcessor.from_pretrained(base_repo, trust_remote_code=True)
        _BASE_MODEL = AutoModelForCausalLM.from_pretrained(
            base_repo, torch_dtype=_BASE_DTYPE, trust_remote_code=True, low_cpu_mem_usage=True,
        ).to(_BASE_DEVICE).eval()
    else:
        from qwen_asr import Qwen3ASRModel
        print("[asr_dual] using qwen-asr wrapper (transformers too old for qwen3_asr)")
        _BASE_MODEL = Qwen3ASRModel.from_pretrained(
            base_repo, dtype=_BASE_DTYPE, device_map=_BASE_DEVICE, max_new_tokens=1024,
        )
        _BASE_PROCESSOR = None

    return _BASE_MODEL, _BASE_PROCESSOR


def _transcribe_base(audio_path: Path, lang_label: Optional[str]) -> str:
    """Transcribe with the base Qwen3 (no LoRA). Auto language is best for code-switching."""
    model, processor = _load_base_model()
    if processor is None:
        # qwen_asr wrapper path
        results = model.transcribe(audio=str(audio_path), language=lang_label)
        return getattr(results[0], "text", "").strip() if results else ""

    # transformers path
    import torch
    import soundfile as sf

    wav_path = asr_module._to_wav(audio_path)
    tmp_wav = wav_path if wav_path != audio_path else None
    try:
        audio, sr = sf.read(str(wav_path), dtype="float32")
    finally:
        if tmp_wav and tmp_wav.exists():
            tmp_wav.unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    user_msg = (
        f"<|audio_1|>Transcribe the audio in {lang_label}."
        if lang_label else
        "<|audio_1|>Transcribe the audio."
    )
    inputs = processor(text=user_msg, audios=[audio], sampling_rate=sr, return_tensors="pt")
    inputs = {k: v.to(_BASE_DEVICE) for k, v in inputs.items()}
    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    input_len = inputs["input_ids"].shape[1]
    return processor.batch_decode(out_ids[:, input_len:], skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------------------
# LLM judge — merges the two transcripts via Calme-3.2-78B.
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are merging two automatic speech-recognition transcripts of the SAME "
    "Gulf Arabic audio clip. The two transcripts come from two different ASR "
    "systems:\n"
    "  • Transcript A: a Gulf-Arabic fine-tuned model. Very good at Arabic "
    "morphology and dialect, but tends to transliterate English/medical/brand "
    "words into Arabic script (e.g. 'paracetamol' becomes 'فرنسي تمان').\n"
    "  • Transcript B: a multilingual base model. Worse at Arabic dialect but "
    "preserves English/medical/brand words in their original Latin spelling.\n"
    "\n"
    "Your job: output ONE merged transcript that takes the BEST parts of both.\n"
    "Rules:\n"
    "1. Use Transcript A as the BASE for Arabic words, grammar, and morphology.\n"
    "2. Whenever Transcript B contains an English word, brand name, drug "
    "name, or medical term — substitute it back into A in its Latin form.\n"
    "3. Preserve original word order from Transcript A (it usually has the "
    "right structure for Gulf Arabic).\n"
    "4. Do NOT translate, do NOT add commentary, do NOT correct meaning. "
    "Only fix transliterated English back to Latin where B confirms it.\n"
    "5. Output strict JSON only: {\"merged\": \"<final transcript>\", "
    "\"reason\": \"<one short sentence>\"}. No prose outside JSON.\n"
    "6. If both transcripts agree, just return A unchanged."
)


def _judge_url() -> str:
    return get_llm_url(get_llm_provider())


def _judge_model() -> str:
    return get_llm_model(get_llm_provider())


def _llm_judge(transcript_a: str, transcript_b: str, timeout: float = 120.0) -> Dict[str, str]:
    """Ask the LLM to merge transcripts A and B. Returns {merged, reason}."""
    user_msg = json.dumps(
        {"transcript_A_gulf_lora": transcript_a, "transcript_B_base": transcript_b},
        ensure_ascii=False,
    )
    payload = {
        "model": _judge_model(),
        "stream": False,
        "format": "json",
        "think": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
    }
    last_exc: Optional[BaseException] = None
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                _judge_url(),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(get_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            content = parse_chat_content(data, get_llm_provider())
            # Strip non-JSON wrappers if any.
            text = content.strip()
            if not (text.startswith("{") and text.endswith("}")):
                m = re.search(r"\{.*\}", text, re.S)
                if m:
                    text = m.group(0)
            obj = json.loads(text)
            return {
                "merged": str(obj.get("merged") or transcript_a).strip(),
                "reason": str(obj.get("reason") or "").strip(),
            }
        except Exception as exc:
            last_exc = exc
            wait = 1.0 * (2 ** attempt)
            print(f"[asr_dual] LLM judge failed (attempt {attempt+1}/3): {exc!r}; retrying in {wait:.1f}s")
            time.sleep(wait)
    # Fallback: prefer A (Gulf LoRA) on judge failure.
    print(f"[asr_dual] judge unavailable, returning transcript A: {last_exc!r}")
    return {"merged": transcript_a, "reason": f"judge unavailable: {last_exc!r}"}


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def transcribe_and_merge(
    audio_path: str | Path, language: Optional[str] = None
) -> Dict[str, Any]:
    """Run both ASR backends in parallel, merge with the LLM judge, return.

    Shape matches services.asr.transcribe() so it's a drop-in replacement.
    """
    audio_path = Path(audio_path)
    t0 = time.time()

    # Run both ASRs in parallel. They share the same GPU but it's fine —
    # the LoRA and base are roughly the same size and inference is short.
    #
    # Base model: force "Arabic" so the audio gets routed to the Arabic
    # decoder (otherwise Qwen3 sometimes mis-detects Gulf Arabic as Spanish
    # or Persian and outputs gibberish). Qwen3 still preserves English
    # tokens in Latin script when language=Arabic, which is exactly what
    # we want from transcript B for code-switching content.
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(asr_module.transcribe, audio_path, language=language)
        fut_b = pool.submit(_transcribe_base, audio_path, lang_label="Arabic")
        result_a = fut_a.result()
        text_b = fut_b.result()

    text_a = result_a.get("text", "")
    print(f"[asr_dual] A (gulf-lora): {text_a!r}")
    print(f"[asr_dual] B (base):     {text_b!r}")

    # If either side is empty, skip judge and use the other.
    if not text_a:
        merged, reason = text_b, "A empty; used B"
    elif not text_b:
        merged, reason = text_a, "B empty; used A"
    elif text_a.strip() == text_b.strip():
        merged, reason = text_a, "A and B agreed"
    else:
        verdict = _llm_judge(text_a, text_b)
        merged, reason = verdict["merged"], verdict["reason"]

    print(f"[asr_dual] merged:       {merged!r}  ({reason})")

    return {
        "text": merged,
        "language": result_a.get("language") or "ar",
        "language_probability": 1.0,
        "duration": result_a.get("duration", 0.0),
        "words": [],
        "extra": {
            "transcript_a_gulf_lora": text_a,
            "transcript_b_base": text_b,
            "merge_reason": reason,
            "total_seconds": time.time() - t0,
        },
    }
