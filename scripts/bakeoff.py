"""Zero-shot ASR bake-off on the Gulf-medical eval set.

Pluggable backends — each one transcribes the same set of WAV clips, then
we compute per-category metrics:

  * WER (overall, normalized text)
  * Medical-term recall (any of the canonical drug/disease names from the
    medical_terms field present in the prediction)
  * Real-time factor (audio_duration / inference_time)

Output:
    eval/gulf_medical_v1/bakeoff/
        predictions/<model>/<clip_id>.json     raw prediction per clip
        results.csv                            one row per (model, clip)
        report.md                              ranked summary

Usage
-----
  source .venv/bin/activate
  export HF_TOKEN=...
  python -m scripts.bakeoff --models whisper qwen3 vibevoice
  python -m scripts.bakeoff --models whisper            # subset

Backend availability is checked lazily — if a model can't be loaded we skip
it and log the failure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# These globals are rebound by main() once --eval-dir is parsed. Defaults
# keep backward compatibility with prior invocations.
EVAL_DIR = PROJECT_ROOT / "eval" / "gulf_medical_v1"
MANIFEST_PATH = EVAL_DIR / "manifest.jsonl"
OUT_DIR = EVAL_DIR / "bakeoff"
PREDICTIONS_DIR = OUT_DIR / "predictions"


def _set_eval_dir(path: Path) -> None:
    """Re-point all output paths to a different eval directory."""
    global EVAL_DIR, MANIFEST_PATH, OUT_DIR, PREDICTIONS_DIR
    EVAL_DIR = path
    MANIFEST_PATH = EVAL_DIR / "manifest.jsonl"
    OUT_DIR = EVAL_DIR / "bakeoff"
    PREDICTIONS_DIR = OUT_DIR / "predictions"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)


_set_eval_dir(EVAL_DIR)


# ---------------------------------------------------------------------------
# Text normalization
#
# For Arabic we apply the standard ASR-eval normalisation pipeline:
#   - NFKC unicode normalisation
#   - strip Arabic diacritics (tashkeel) and tatweel
#   - unify alef variants  أ إ آ ٱ -> ا
#   - unify hamza carriers ؤ -> و,  ئ -> ي,  ء -> ""
#   - unify yaa             ى -> ي
#   - unify teh marbuta     ة -> ه
#   - map Arabic-Indic digits ٠-٩ -> 0-9
# Then a generic cleanup: lowercase, strip punctuation, collapse whitespace.
#
# Without these, spelling variants count as substitutions and inflate WER
# by 15-25 absolute points on dialectal Arabic. See:
#   - vadimbelsky/qwen3-asr-arabic-uae model card (`text normalization` section)
#   - SADA22 paper, normalisation procedure
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
# Tashkeel (U+064B..U+065F), superscript alef (U+0670), tatweel (U+0640)
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")

# Translation table for Arabic letter unification + digit folding.
_AR_TRANSLIT = {
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("آ"): "ا",
    ord("ٱ"): "ا",
    ord("ى"): "ي",
    ord("ة"): "ه",
    ord("ؤ"): "و",
    ord("ئ"): "ي",
    ord("ء"): "",
    # Arabic-Indic digits 0..9
    ord("٠"): "0", ord("١"): "1", ord("٢"): "2", ord("٣"): "3", ord("٤"): "4",
    ord("٥"): "5", ord("٦"): "6", ord("٧"): "7", ord("٨"): "8", ord("٩"): "9",
    # Persian/Urdu-style alternates that sometimes appear
    ord("ﻻ"): "لا", ord("ﻷ"): "لا", ord("ﻹ"): "لا", ord("ﻵ"): "لا",
}


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS_RE.sub("", s)
    s = s.translate(_AR_TRANSLIT)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# WER
# ---------------------------------------------------------------------------


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.split()
    hyp = hypothesis.split()
    if not ref:
        return 0.0 if not hyp else 1.0
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[n][m] / n


def medical_term_recall(reference_terms: List[str], hypothesis: str) -> Optional[float]:
    """Fraction of expected medical terms found in hypothesis (case-insensitive,
    diacritic-stripped). Returns None if no expected terms were tagged for this
    clip — which means the metric isn't applicable here.
    """
    if not reference_terms:
        return None
    hyp_norm = normalize_text(hypothesis)
    found = 0
    for term in reference_terms:
        term_norm = normalize_text(term)
        if not term_norm:
            continue
        if term_norm in hyp_norm:
            found += 1
    return found / len(reference_terms)


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


@dataclass
class Prediction:
    text: str
    inference_seconds: float
    extra: Dict[str, Any] = field(default_factory=dict)


class Backend:
    """Override `name` and implement `transcribe()`. `prepare()` is called
    once before any clip is transcribed; raise to signal unavailability."""

    name: str = "abstract"

    def prepare(self) -> None:
        raise NotImplementedError

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Backend: faster-whisper large-v3-turbo
# ---------------------------------------------------------------------------


class WhisperBackend(Backend):
    name = "whisper-large-v3-turbo"

    def __init__(self, model_size: str = "large-v3-turbo"):
        self.model_size = model_size
        self._model = None

    def prepare(self) -> None:
        from faster_whisper import WhisperModel
        device = os.environ.get("WHISPER_DEVICE", "auto")
        compute_type = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
        print(f"[whisper] loading {self.model_size} ({device}, {compute_type})")
        self._model = WhisperModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None
        t0 = time.time()
        segs, info = self._model.transcribe(
            str(wav_path),
            language=language,
            beam_size=5,
            vad_filter=False,
            without_timestamps=True,
        )
        text = "".join(s.text for s in segs).strip()
        return Prediction(
            text=text,
            inference_seconds=time.time() - t0,
            extra={"detected_language": info.language, "lang_prob": info.language_probability},
        )


# ---------------------------------------------------------------------------
# Qwen3-ASR shared loader (uses `transformers` directly, no qwen-asr wrapper)
# ---------------------------------------------------------------------------


class _Qwen3AsrBase(Backend):
    """Loads Qwen3-ASR family models.

    Preferred path: `transformers` (clean, no extra deps).
    Fallback path: `qwen_asr` pip wrapper, used automatically when the
    current `transformers` doesn't register `qwen3_asr` (e.g. pre-5.9 dev).
    """

    repo_id: str = ""

    def __init__(self, repo_id: Optional[str] = None):
        if repo_id:
            self.repo_id = repo_id
        self._model = None
        self._processor = None
        self._device = None
        self._backend: str = ""  # "transformers" | "qwen_asr"

    def _pick_device(self):
        import torch
        if torch.cuda.is_available():
            return "cuda:0", torch.bfloat16
        if torch.backends.mps.is_available():
            return "mps", torch.float16
        return "cpu", torch.float32

    def _qwen3_asr_in_transformers(self) -> bool:
        try:
            from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
            return "qwen3_asr" in CONFIG_MAPPING_NAMES
        except Exception:
            return False

    def prepare(self) -> None:
        self._device, dtype = self._pick_device()

        if self._qwen3_asr_in_transformers():
            from transformers import AutoModelForCausalLM, AutoProcessor
            print(f"[{self.name}] loading {self.repo_id} on {self._device} ({dtype}) via transformers")
            self._processor = AutoProcessor.from_pretrained(
                self.repo_id, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.repo_id,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            ).to(self._device).eval()
            self._backend = "transformers"
            return

        # Fallback: qwen-asr pip wrapper.
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as exc:
            raise RuntimeError(
                "transformers does not yet register `qwen3_asr` and the "
                "qwen-asr pip wrapper is not installed. Run: "
                "`pip install qwen-asr` (or upgrade transformers from main "
                "once the model is merged)."
            ) from exc
        print(f"[{self.name}] loading {self.repo_id} on {self._device} ({dtype}) via qwen-asr wrapper")
        self._model = Qwen3ASRModel.from_pretrained(
            self.repo_id,
            dtype=dtype,
            device_map=self._device,
            max_inference_batch_size=1,
            # 1024 matches the official Qwen3-ASR evaluation (HF model card).
            # 256 truncated clips longer than ~20s.
            max_new_tokens=1024,
        )
        self._backend = "qwen_asr"

    def _qwen_language_label(self, language: Optional[str]) -> Optional[str]:
        # Qwen3-ASR uses verbose language names. None = auto-detect.
        return {"en": "English", "ar": "Arabic", "mixed": None}.get(language or "", None)

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None
        qlang = self._qwen_language_label(language)

        if self._backend == "qwen_asr":
            t0 = time.time()
            try:
                results = self._model.transcribe(
                    audio=str(wav_path), language=qlang
                )
                r = results[0]
                return Prediction(
                    text=getattr(r, "text", "").strip(),
                    inference_seconds=time.time() - t0,
                    extra={"lang_hint": qlang,
                           "detected_language": getattr(r, "language", None)},
                )
            except Exception as exc:
                return Prediction(
                    text="", inference_seconds=time.time() - t0,
                    extra={"error": repr(exc)},
                )

        # transformers backend
        import torch
        import soundfile as sf
        assert self._processor is not None

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception as exc:
                return Prediction(
                    text="", inference_seconds=0.0,
                    extra={"error": f"resample failed: {exc!r}"},
                )

        user_msg = (
            f"<|audio_1|>Transcribe the audio in {qlang}."
            if qlang else
            "<|audio_1|>Transcribe the audio."
        )
        t0 = time.time()
        try:
            inputs = self._processor(
                text=user_msg,
                audios=[audio],
                sampling_rate=sr,
                return_tensors="pt",
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.inference_mode():
                out_ids = self._model.generate(
                    # 1024 matches the official Qwen3-ASR evaluation.
                    **inputs, max_new_tokens=1024, do_sample=False,
                )
            input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
            gen_ids = out_ids[:, input_len:] if input_len else out_ids
            text = self._processor.batch_decode(
                gen_ids, skip_special_tokens=True
            )[0].strip()
            return Prediction(
                text=text,
                inference_seconds=time.time() - t0,
                extra={"lang_hint": qlang},
            )
        except Exception as exc:
            return Prediction(
                text="", inference_seconds=time.time() - t0,
                extra={"error": repr(exc)},
            )


# ---------------------------------------------------------------------------
# Backend: Qwen3-ASR-1.7B (base)
# ---------------------------------------------------------------------------


class QwenAsrBackend(_Qwen3AsrBase):
    name = "qwen3-asr-1.7b"
    repo_id = "Qwen/Qwen3-ASR-1.7B"


# ---------------------------------------------------------------------------
# Backend: Microsoft VibeVoice-ASR
# ---------------------------------------------------------------------------


class VibeVoiceBackend(Backend):
    """Microsoft VibeVoice-ASR (8B params, custom vibevoice_asr architecture).

    Uses the transformers-native release `microsoft/VibeVoice-ASR-HF`
    (registered in transformers >= 5.3.0). The original
    `microsoft/VibeVoice-ASR` repo needs Microsoft's vibevoice Python
    package and won't load via stock transformers.

    The model is multilingual (51 langs incl. Arabic) and auto-detects
    language — no `language` hint. The processor's
    `apply_transcription_request(audio=...)` builds the chat-template
    inputs (same pattern as Voxtral); `processor.decode(...,
    return_format="transcription_only")` strips the speaker JSON and
    timestamp metadata so we get plain text for WER scoring.
    """

    name = "vibevoice-asr"

    def __init__(self, repo_id: str = "microsoft/VibeVoice-ASR-HF"):
        self.repo_id = repo_id
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    def prepare(self) -> None:
        try:
            import torch
            from transformers import (
                AutoProcessor,
                VibeVoiceAsrForConditionalGeneration,
            )
        except ImportError as exc:
            raise RuntimeError(
                "VibeVoice requires transformers >= 5.3.0 "
                f"(install from git main). Underlying error: {exc}"
            )

        if torch.cuda.is_available():
            self._device, dtype = "cuda:0", torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device, dtype = "mps", torch.float16
        else:
            self._device, dtype = "cpu", torch.float32
        self._dtype = dtype

        print(f"[vibevoice] loading {self.repo_id} on {self._device} ({dtype})")
        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        self._model = VibeVoiceAsrForConditionalGeneration.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device).eval()

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None and self._processor is not None
        import torch

        t0 = time.time()
        try:
            # apply_transcription_request handles audio loading, resampling
            # (to 24 kHz internally), and chat-template wrapping.
            inputs = self._processor.apply_transcription_request(
                audio=str(wav_path),
            ).to(self._device, self._dtype)
            with torch.inference_mode():
                output_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
            # return_format="transcription_only" extracts just the plain
            # transcribed text, stripping speaker labels + timestamps.
            text = self._processor.decode(
                generated_ids, return_format="transcription_only",
            )[0]
            return Prediction(
                text=str(text).strip(),
                inference_seconds=time.time() - t0,
            )
        except Exception as exc:
            return Prediction(
                text="", inference_seconds=time.time() - t0,
                extra={"error": repr(exc)},
            )


# ---------------------------------------------------------------------------
# Backend: Meta OmniASR-LLM-7B (1600+ languages, incl Gulf Arabic afb_Arab)
# ---------------------------------------------------------------------------


class OmniAsrBackend(Backend):
    name = "omniASR-LLM-7B"

    def __init__(self, model_card: str = "omniASR_LLM_7B"):
        self.model_card = model_card
        self._pipeline = None

    def prepare(self) -> None:
        try:
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        except ImportError:
            raise RuntimeError(
                "omnilingual-asr not installed. Run: pip install omnilingual-asr"
            )
        print(f"[omniASR] loading {self.model_card}")
        self._pipeline = ASRInferencePipeline(model_card=self.model_card)

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._pipeline is not None
        # Gulf Arabic ISO 639-3 code: afb; general Arabic: arb_Arab
        lang_map = {"ar": "arb_Arab", "mixed": "arb_Arab", "en": "eng_Latn"}
        lang_code = lang_map.get(language or "en", "arb_Arab")
        t0 = time.time()
        try:
            transcriptions = self._pipeline.transcribe(
                [str(wav_path)],
                lang=[lang_code],
                batch_size=1,
            )
            text = transcriptions[0] if transcriptions else ""
        except Exception as exc:
            return Prediction(text="", inference_seconds=time.time() - t0,
                              extra={"error": repr(exc)})
        return Prediction(text=str(text).strip(), inference_seconds=time.time() - t0,
                          extra={"lang_code": lang_code})


# ---------------------------------------------------------------------------
# Backend: otozz/whisper-small-dialect_gulf
# (Whisper-small fine-tuned on MASC Gulf Arabic corpus)
# ---------------------------------------------------------------------------


class WhisperGulfBackend(Backend):
    """Whisper-small fine-tuned on MASC Gulf Arabic.

    Originally this backend used faster-whisper (CTranslate2), but the
    DGX Spark builds of CTranslate2 ship without CUDA support on ARM64,
    so model load raised:
        ValueError("This CTranslate2 package was not compiled with CUDA support")
    We instead use plain HF transformers (WhisperForConditionalGeneration +
    AutoProcessor), which works on any CUDA-capable GPU.
    """

    name = "whisper-small-gulf"

    def __init__(self, repo_id: str = "otozz/whisper-small-dialect_gulf"):
        self.repo_id = repo_id
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    def prepare(self) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            WhisperForConditionalGeneration,
        )

        if torch.cuda.is_available():
            self._device, dtype = "cuda:0", torch.float16
        elif torch.backends.mps.is_available():
            self._device, dtype = "mps", torch.float16
        else:
            self._device, dtype = "cpu", torch.float32
        self._dtype = dtype

        print(f"[whisper-gulf] loading {self.repo_id} on {self._device} ({dtype}) via transformers")
        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        self._model = WhisperForConditionalGeneration.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(self._device).eval()

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None and self._processor is not None
        import torch
        import soundfile as sf

        t0 = time.time()
        try:
            audio, sr = sf.read(str(wav_path), dtype="float32")
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    sr = 16000
                except Exception as exc:
                    return Prediction(
                        text="", inference_seconds=time.time() - t0,
                        extra={"error": f"resample failed: {exc!r}"},
                    )

            inputs = self._processor(
                audio, sampling_rate=sr, return_tensors="pt",
            )
            input_features = inputs.input_features.to(self._device, dtype=self._dtype)

            # Force Arabic for Arabic/mixed clips; English for English.
            lang = "ar" if (language or "en") in ("ar", "mixed") else "en"
            try:
                forced_ids = self._processor.get_decoder_prompt_ids(
                    language=lang, task="transcribe",
                )
            except Exception:
                forced_ids = None

            with torch.inference_mode():
                gen = self._model.generate(
                    input_features,
                    forced_decoder_ids=forced_ids,
                    max_new_tokens=440,
                    num_beams=1,
                    do_sample=False,
                )
            text = self._processor.batch_decode(
                gen, skip_special_tokens=True,
            )[0].strip()
            return Prediction(
                text=text, inference_seconds=time.time() - t0,
                extra={"lang_hint": lang},
            )
        except Exception as exc:
            return Prediction(
                text="", inference_seconds=time.time() - t0,
                extra={"error": repr(exc)},
            )


# ---------------------------------------------------------------------------
# Backend: vadimbelsky/qwen3-asr-arabic-ksa (Saudi Arabic fine-tune)
# ---------------------------------------------------------------------------


class QwenKsaBackend(_Qwen3AsrBase):
    name = "qwen3-asr-ksa"
    repo_id = "vadimbelsky/qwen3-asr-arabic-ksa"

    def _qwen_language_label(self, language: Optional[str]) -> Optional[str]:
        # Saudi-tuned model: always force Arabic except for explicit English.
        return "English" if (language or "") == "en" else "Arabic"


# ---------------------------------------------------------------------------
# Backend: vadimbelsky/qwen3-asr-arabic-uae (UAE/Emirati Arabic fine-tune)
# ---------------------------------------------------------------------------


class QwenUaeBackend(_Qwen3AsrBase):
    name = "qwen3-asr-uae"
    repo_id = "vadimbelsky/qwen3-asr-arabic-uae"

    def _qwen_language_label(self, language: Optional[str]) -> Optional[str]:
        # UAE-tuned model: always force Arabic except for explicit English.
        return "English" if (language or "") == "en" else "Arabic"


# ---------------------------------------------------------------------------
# Backend: our own Gulf-Arabic LoRA fine-tune of Qwen3-ASR-1.7B
# ---------------------------------------------------------------------------


class QwenGulfLoraBackend(_Qwen3AsrBase):
    """Loads Qwen3-ASR-1.7B and stacks a locally-trained LoRA adapter on top.

    The adapter directory is controlled by the QWEN3_GULF_ADAPTER env var
    (default: runs/qwen3_lora_r6/final_adapter). It must contain a PEFT
    `adapter_config.json` and `adapter_model.safetensors` saved via
    model.save_pretrained(...).

    Subclasses can override `name` and `DEFAULT_ADAPTER` to evaluate
    specific training checkpoints alongside the final adapter (so each
    checkpoint gets its own predictions/<name>/ cache directory).
    """

    name = "qwen3-asr-gulf-lora"
    repo_id = "Qwen/Qwen3-ASR-1.7B"  # base model
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/final_adapter"

    def __init__(self, adapter_path: Optional[str] = None):
        super().__init__()
        # Precedence: explicit ctor arg > env var > class default.
        # Only the *base* gulf-lora backend honors the env var; per-checkpoint
        # subclasses set their own DEFAULT_ADAPTER and ignore the env so a
        # single bakeoff invocation can evaluate multiple checkpoints.
        if adapter_path is not None:
            self.adapter_path = adapter_path
        elif type(self).DEFAULT_ADAPTER == QwenGulfLoraBackend.DEFAULT_ADAPTER:
            self.adapter_path = os.environ.get(
                "QWEN3_GULF_ADAPTER", self.DEFAULT_ADAPTER
            )
        else:
            self.adapter_path = type(self).DEFAULT_ADAPTER

    def prepare(self) -> None:
        # Load base model + processor via the parent class. This populates
        # either self._model (transformers PreTrainedModel) when
        # transformers registers qwen3_asr, OR self._model (Qwen3ASRModel
        # wrapper) when the qwen_asr pip wrapper is used.
        super().prepare()

        # Resolve adapter path against the repo root if it's relative.
        adapter_dir = Path(self.adapter_path)
        if not adapter_dir.is_absolute():
            adapter_dir = (PROJECT_ROOT / adapter_dir).resolve()
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"Adapter directory not found: {adapter_dir}. Set "
                f"QWEN3_GULF_ADAPTER or pass --adapter-path."
            )

        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "peft is required for LoRA loading. Run: pip install peft"
            ) from exc

        print(f"[{self.name}] attaching LoRA adapter: {adapter_dir}")

        if self._backend == "transformers":
            # Direct: wrap the PreTrainedModel.
            self._model = PeftModel.from_pretrained(
                self._model, str(adapter_dir),
            ).to(self._device).eval()
            return

        # qwen_asr wrapper path: the wrapper holds the actual nn.Module on
        # `.model`. Replace that attribute with the PEFT-wrapped version so
        # subsequent wrapper.transcribe() calls route through the LoRA
        # layers. This matches how finetune_qwen3_lora.py applied PEFT
        # during training.
        inner = getattr(self._model, "model", None)
        if inner is None:
            raise RuntimeError(
                f"qwen_asr wrapper has no .model attribute; cannot attach "
                f"LoRA. Wrapper type: {type(self._model).__name__}"
            )
        peft_model = PeftModel.from_pretrained(inner, str(adapter_dir))
        # Move to the same device the wrapper is on (cuda:0 typically).
        peft_model = peft_model.to(self._device).eval()
        self._model.model = peft_model

    def _qwen_language_label(self, language: Optional[str]) -> Optional[str]:
        # Gulf-tuned model: always force Arabic except for explicit English.
        return "English" if (language or "") == "en" else "Arabic"


# Per-checkpoint variants — each writes to its own predictions/<name>/ dir
# so a single bakeoff run can compare multiple checkpoints side-by-side.

class QwenGulfLoraCkpt12000Backend(QwenGulfLoraBackend):
    name = "qwen3-asr-gulf-lora-ckpt12000"
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/checkpoint-12000"


class QwenGulfLoraCkpt14000Backend(QwenGulfLoraBackend):
    name = "qwen3-asr-gulf-lora-ckpt14000"
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/checkpoint-14000"


class QwenGulfLoraCkpt16000Backend(QwenGulfLoraBackend):
    name = "qwen3-asr-gulf-lora-ckpt16000"
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/checkpoint-16000"


class QwenGulfLoraCkpt18000Backend(QwenGulfLoraBackend):
    name = "qwen3-asr-gulf-lora-ckpt18000"
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/checkpoint-18000"


class QwenGulfLoraCkpt19636Backend(QwenGulfLoraBackend):
    name = "qwen3-asr-gulf-lora-ckpt19636"
    DEFAULT_ADAPTER = "runs/qwen3_lora_r6/checkpoint-19636"


# ---------------------------------------------------------------------------
# Backend: Mistral Voxtral (Voxtral-Mini-3B / Voxtral-Small-24B)
# ---------------------------------------------------------------------------
# Uses the official Open Universal Arabic ASR Leaderboard inference recipe:
# processor.apply_transcription_request(language="ar", ...) at bf16, with
# max_new_tokens=500 and greedy decoding.
# Source: github.com/Natural-Language-Processing-Elm/open_universal_arabic_asr_leaderboard
#         /blob/main/models/voxtral.py


class VoxtralBackend(Backend):
    name = "voxtral-mini-3b"  # overridden by size variants below

    def __init__(self, repo_id: str = "mistralai/Voxtral-Mini-3B-2507"):
        self.repo_id = repo_id
        self._model = None
        self._processor = None
        self._device = None

    def prepare(self) -> None:
        try:
            import torch
            from transformers import VoxtralForConditionalGeneration, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                f"Voxtral requires a recent transformers (≥4.46): {exc}"
            )
        if torch.cuda.is_available():
            self._device, dtype = "cuda:0", torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device, dtype = "mps", torch.float16
        else:
            self._device, dtype = "cpu", torch.float32
        self._dtype = dtype
        print(f"[{self.name}] loading {self.repo_id} on {self._device} ({dtype})")
        self._processor = AutoProcessor.from_pretrained(self.repo_id)
        
        # Use auto device mapping for large models (24B params) to spread across GPU+CPU
        device_map = "auto" if "24b" in self.repo_id.lower() else self._device
        print(f"[{self.name}] using device_map={device_map}")
        
        self._model = VoxtralForConditionalGeneration.from_pretrained(
            self.repo_id, torch_dtype=dtype, device_map=device_map,
        )

    def _voxtral_lang(self, language: Optional[str]) -> str:
        # Voxtral accepts ISO 639-1 codes via apply_transcription_request.
        return {"en": "en", "ar": "ar", "mixed": "ar"}.get(language or "", "ar")

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None and self._processor is not None
        import torch
        lang = self._voxtral_lang(language)
        t0 = time.time()
        try:
            inputs = self._processor.apply_transcription_request(
                language=lang,
                audio=str(wav_path),
                model_id=self.repo_id,
            )
            inputs = inputs.to(self._device, dtype=self._dtype)
            with torch.inference_mode():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=500, do_sample=False,
                )
            decoded = self._processor.batch_decode(
                outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True,
            )
            text = decoded[0].strip() if decoded else ""
            return Prediction(
                text=text, inference_seconds=time.time() - t0,
                extra={"lang_hint": lang},
            )
        except Exception as exc:
            return Prediction(
                text="", inference_seconds=time.time() - t0,
                extra={"error": repr(exc)},
            )


class VoxtralMiniBackend(VoxtralBackend):
    name = "voxtral-mini-3b"
    def __init__(self):
        super().__init__(repo_id="mistralai/Voxtral-Mini-3B-2507")


class VoxtralSmallBackend(VoxtralBackend):
    name = "voxtral-small-24b"
    def __init__(self):
        super().__init__(repo_id="mistralai/Voxtral-Small-24B-2507")


# ---------------------------------------------------------------------------
# Bake-off runner
# ---------------------------------------------------------------------------


def load_manifest() -> List[Dict[str, Any]]:
    out = []
    with MANIFEST_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def run_backend(
    backend: Backend,
    manifest: List[Dict[str, Any]],
    skip_existing: bool = False,
) -> List[Dict[str, Any]]:
    print(f"\n=== {backend.name} ===")
    pred_dir = PREDICTIONS_DIR / backend.name
    pred_dir.mkdir(parents=True, exist_ok=True)

    # If skip_existing, load already-computed rows without preparing the model
    # — but only when ALL manifest clips are already covered.
    if skip_existing:
        existing = {p.stem for p in pred_dir.glob("*.json")}
        manifest_ids = {clip["id"] for clip in manifest}
        if existing >= manifest_ids:
            print(f"  → found {len(existing)} existing predictions (all clips covered), skipping")
            rows = []
            for p in pred_dir.glob("*.json"):
                try:
                    d = json.loads(p.read_text(encoding="utf-8"))
                    rows.append({
                        "model": backend.name,
                        "clip_id": d["id"],
                        "category": d["category"],
                        "language": d["language"],
                        "duration_s": d["duration_s"],
                        "wer": d["wer"],
                        "medical_term_recall": d["medical_term_recall"],
                        "inference_seconds": d["inference_seconds"],
                        "rtf": d["rtf"],
                    })
                except Exception:
                    pass
            return rows
        elif existing:
            print(f"  → found {len(existing)}/{len(manifest_ids)} existing predictions, resuming from where we left off")

    try:
        backend.prepare()
    except Exception as exc:
        print(f"  ! could not prepare {backend.name}: {exc!r}")
        traceback.print_exc()
        return []

    rows: List[Dict[str, Any]] = []
    for i, clip in enumerate(manifest, 1):
        # Skip clips that already have a prediction file (allows resuming).
        pred_file = pred_dir / f"{clip['id']}.json"
        if pred_file.exists() and skip_existing:
            try:
                d = json.loads(pred_file.read_text(encoding="utf-8"))
                rows.append({
                    "model": backend.name,
                    "clip_id": d["id"],
                    "category": d["category"],
                    "language": d["language"],
                    "duration_s": d["duration_s"],
                    "wer": d["wer"],
                    "medical_term_recall": d["medical_term_recall"],
                    "inference_seconds": d["inference_seconds"],
                    "rtf": d["rtf"],
                })
                print(f"  [{i:>3}/{len(manifest)}] ↩ {clip['id']} (cached)")
                continue
            except Exception:
                pass
        wav_path = EVAL_DIR / clip["audio_path"]
        if not wav_path.exists():
            print(f"  [{i:>3}/{len(manifest)}] missing audio: {wav_path}")
            continue
        try:
            pred = backend.transcribe(wav_path, language=clip.get("language"))
        except Exception as exc:
            print(f"  [{i:>3}/{len(manifest)}] {clip['id']}: ERROR {exc!r}")
            pred = Prediction(text="", inference_seconds=0.0, extra={"error": repr(exc)})

        # Per-clip metrics
        ref = clip.get("transcript_normalized") or normalize_text(clip.get("transcript") or "")
        hyp = normalize_text(pred.text)
        clip_wer = wer(ref, hyp)
        rec = medical_term_recall(clip.get("medical_terms") or [], pred.text)
        rtf = (clip["duration_s"] / pred.inference_seconds) if pred.inference_seconds > 0 else 0.0

        # Save raw prediction
        (pred_dir / f"{clip['id']}.json").write_text(
            json.dumps({
                "id": clip["id"],
                "category": clip["category"],
                "language": clip["language"],
                "duration_s": clip["duration_s"],
                "ref": clip["transcript"],
                "ref_normalized": ref,
                "pred": pred.text,
                "pred_normalized": hyp,
                "wer": clip_wer,
                "medical_term_recall": rec,
                "inference_seconds": pred.inference_seconds,
                "rtf": rtf,
                "extra": pred.extra,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        rows.append({
            "model": backend.name,
            "clip_id": clip["id"],
            "category": clip["category"],
            "language": clip["language"],
            "duration_s": clip["duration_s"],
            "wer": clip_wer,
            "medical_term_recall": rec,
            "inference_seconds": pred.inference_seconds,
            "rtf": rtf,
        })
        marker = "✓" if not pred.extra.get("error") else "✗"
        print(f"  [{i:>3}/{len(manifest)}] {marker} {clip['id']:30s} WER={clip_wer:.3f} t={pred.inference_seconds:.1f}s")

    return rows


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {}
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    out: Dict[str, Any] = {}
    for model, items in by_model.items():
        cats: Dict[str, List[Dict[str, Any]]] = {}
        for it in items:
            cats.setdefault(it["category"], []).append(it)
        cat_summary: Dict[str, Any] = {}
        for cat, group in cats.items():
            wers = [g["wer"] for g in group]
            recalls = [g["medical_term_recall"] for g in group if g["medical_term_recall"] is not None]
            rtfs = [g["rtf"] for g in group if g["rtf"] > 0]
            cat_summary[cat] = {
                "n": len(group),
                "wer_mean": sum(wers) / len(wers) if wers else None,
                "medical_term_recall_mean": (sum(recalls) / len(recalls)) if recalls else None,
                "rtf_mean": (sum(rtfs) / len(rtfs)) if rtfs else None,
            }
        all_wers = [it["wer"] for it in items]
        all_recalls = [it["medical_term_recall"] for it in items if it["medical_term_recall"] is not None]
        all_rtfs = [it["rtf"] for it in items if it["rtf"] > 0]
        out[model] = {
            "overall": {
                "n": len(items),
                "wer_mean": sum(all_wers) / len(all_wers) if all_wers else None,
                "medical_term_recall_mean": (sum(all_recalls) / len(all_recalls)) if all_recalls else None,
                "rtf_mean": (sum(all_rtfs) / len(all_rtfs)) if all_rtfs else None,
            },
            "by_category": cat_summary,
        }
    return out


def write_csv(rows: List[Dict[str, Any]]) -> Path:
    path = OUT_DIR / "results.csv"
    fieldnames = ["model", "clip_id", "category", "language", "duration_s",
                  "wer", "medical_term_recall", "inference_seconds", "rtf"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return path


def write_report(summary: Dict[str, Any]) -> Path:
    path = OUT_DIR / "report.md"
    lines: List[str] = []
    lines.append("# Bake-off report — eval/gulf_medical_v1\n")

    # Overall ranking
    n_total = next(iter(summary.values()))["overall"]["n"] if summary else 0
    lines.append(f"## Overall (across all {n_total} clips)\n")
    lines.append("| model | n | WER ↓ | Medical-term recall ↑ | RTF ↑ |")
    lines.append("|---|---:|---:|---:|---:|")
    rows = sorted(summary.items(), key=lambda kv: (kv[1]["overall"].get("wer_mean") or 1e9))
    for model, stats in rows:
        ov = stats["overall"]
        wer_s = f"{ov['wer_mean']:.3f}" if ov["wer_mean"] is not None else "—"
        rec_s = f"{ov['medical_term_recall_mean']:.3f}" if ov["medical_term_recall_mean"] is not None else "—"
        rtf_s = f"{ov['rtf_mean']:.2f}x" if ov["rtf_mean"] is not None else "—"
        lines.append(f"| {model} | {ov['n']} | {wer_s} | {rec_s} | {rtf_s} |")
    lines.append("")

    # Per-category. Discover categories from the summary so any eval set works.
    discovered_cats: List[str] = []
    for stats in summary.values():
        for c in stats["by_category"].keys():
            if c not in discovered_cats:
                discovered_cats.append(c)
    for cat in discovered_cats:
        lines.append(f"## Category: `{cat}`\n")
        lines.append("| model | n | WER ↓ | Medical-term recall ↑ | RTF ↑ |")
        lines.append("|---|---:|---:|---:|---:|")
        cat_rows = []
        for model, stats in summary.items():
            cat_stats = stats["by_category"].get(cat)
            if not cat_stats:
                continue
            cat_rows.append((model, cat_stats))
        cat_rows.sort(key=lambda kv: (kv[1].get("wer_mean") or 1e9))
        for model, c in cat_rows:
            wer_s = f"{c['wer_mean']:.3f}" if c["wer_mean"] is not None else "—"
            rec_s = f"{c['medical_term_recall_mean']:.3f}" if c["medical_term_recall_mean"] is not None else "—"
            rtf_s = f"{c['rtf_mean']:.2f}x" if c["rtf_mean"] is not None else "—"
            lines.append(f"| {model} | {c['n']} | {wer_s} | {rec_s} | {rtf_s} |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


BACKENDS: Dict[str, Callable[[], Backend]] = {
    "whisper":             WhisperBackend,
    "qwen3":               QwenAsrBackend,
    "vibevoice":           VibeVoiceBackend,
    "omniASR":             OmniAsrBackend,
    "whisper_gulf":        WhisperGulfBackend,
    "qwen3_ksa":           QwenKsaBackend,
    "qwen3_uae":           QwenUaeBackend,
    "qwen3_gulf":          QwenGulfLoraBackend,
    "qwen3_gulf_ckpt12k":  QwenGulfLoraCkpt12000Backend,
    "qwen3_gulf_ckpt14k":  QwenGulfLoraCkpt14000Backend,
    "qwen3_gulf_ckpt16k":  QwenGulfLoraCkpt16000Backend,
    "qwen3_gulf_ckpt18k":  QwenGulfLoraCkpt18000Backend,
    "qwen3_gulf_ckpt19k":  QwenGulfLoraCkpt19636Backend,
    "voxtral_mini":        VoxtralMiniBackend,
    "voxtral_small":       VoxtralSmallBackend,
}

# Map CLI model key -> backend class.name (the directory name used to cache
# predictions). Used by --rescore-only to find the cached predictions.
_MODEL_KEY_TO_DIR: Dict[str, str] = {
    "whisper":             WhisperBackend.name,
    "qwen3":               QwenAsrBackend.name,
    "vibevoice":           VibeVoiceBackend.name,
    "omniASR":             OmniAsrBackend.name,
    "whisper_gulf":        WhisperGulfBackend.name,
    "qwen3_ksa":           QwenKsaBackend.name,
    "qwen3_uae":           QwenUaeBackend.name,
    "qwen3_gulf":          QwenGulfLoraBackend.name,
    "qwen3_gulf_ckpt12k":  QwenGulfLoraCkpt12000Backend.name,
    "qwen3_gulf_ckpt14k":  QwenGulfLoraCkpt14000Backend.name,
    "qwen3_gulf_ckpt16k":  QwenGulfLoraCkpt16000Backend.name,
    "qwen3_gulf_ckpt18k":  QwenGulfLoraCkpt18000Backend.name,
    "qwen3_gulf_ckpt19k":  QwenGulfLoraCkpt19636Backend.name,
    "voxtral_mini":        VoxtralMiniBackend.name,
    "voxtral_small":       VoxtralSmallBackend.name,
}


def _rescore_cached_predictions(model_keys: List[str]) -> List[Dict[str, Any]]:
    """Walk cached prediction JSONs and recompute WER + recall.

    Each prediction JSON is updated in place so subsequent reads see the
    new numbers. The returned row list matches what run_backend() would
    have produced for a fresh run, so write_csv / write_report can be
    reused unchanged.
    """
    rows: List[Dict[str, Any]] = []
    for key in model_keys:
        dir_name = _MODEL_KEY_TO_DIR.get(key)
        if dir_name is None:
            print(f"  ! unknown model key {key!r}; skipping")
            continue
        pred_dir = PREDICTIONS_DIR / dir_name
        if not pred_dir.is_dir():
            print(f"  ! no cached predictions for {dir_name} "
                  f"({pred_dir.relative_to(PROJECT_ROOT)})")
            continue
        files = sorted(pred_dir.glob("*.json"))
        if not files:
            print(f"  ! cache dir for {dir_name} is empty")
            continue
        print(f"  rescoring {dir_name}: {len(files)} clips")
        for p in files:
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"    ✗ skip {p.name}: {exc!r}")
                continue
            ref_norm = normalize_text(d.get("ref") or "")
            hyp_norm = normalize_text(d.get("pred") or "")
            new_wer = wer(ref_norm, hyp_norm)
            new_rec = medical_term_recall(
                d.get("medical_terms") or [],
                d.get("pred") or "",
            )
            # Persist new metrics back into the cache file.
            d["ref_normalized"] = ref_norm
            d["pred_normalized"] = hyp_norm
            d["wer"] = new_wer
            d["medical_term_recall"] = new_rec
            p.write_text(json.dumps(d, ensure_ascii=False, indent=2),
                         encoding="utf-8")
            rows.append({
                "model": dir_name,
                "clip_id": d.get("id", p.stem),
                "category": d.get("category", ""),
                "language": d.get("language", ""),
                "duration_s": d.get("duration_s", 0.0),
                "wer": new_wer,
                "medical_term_recall": new_rec,
                "inference_seconds": d.get("inference_seconds", 0.0),
                "rtf": d.get("rtf", 0.0),
            })
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description="Run zero-shot ASR bake-off.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(BACKENDS.keys()) + ["all"],
        default=list(BACKENDS.keys()),
        help="Which backends to run (default: all). Pass 'all' to run every backend.",
    )
    p.add_argument(
        "--max-clips",
        type=int,
        default=None,
        help="Optional: limit to first N clips (for smoke testing)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        default=False,
        help="Skip clips that already have prediction files (resume interrupted runs)",
    )
    p.add_argument(
        "--eval-dir",
        type=Path,
        default=None,
        help="Eval set directory (must contain manifest.jsonl + audio/). "
             "Defaults to eval/gulf_medical_v1.",
    )
    p.add_argument(
        "--rescore-only",
        action="store_true",
        default=False,
        help="Skip model inference. Re-score every cached prediction in "
             "<eval-dir>/bakeoff/predictions/<model>/*.json using the "
             "current normalize_text() and write a fresh report.",
    )
    args = p.parse_args()

    # Expand "all" shorthand to every registered backend.
    if args.models and "all" in args.models:
        args.models = list(BACKENDS.keys())

    if args.eval_dir is not None:
        eval_dir = args.eval_dir
        if not eval_dir.is_absolute():
            eval_dir = (PROJECT_ROOT / eval_dir).resolve()
        _set_eval_dir(eval_dir)
        print(f"using eval dir: {EVAL_DIR.relative_to(PROJECT_ROOT)}")

    if args.rescore_only:
        all_rows = _rescore_cached_predictions(args.models)
        if not all_rows:
            print("no cached predictions found under "
                  f"{PREDICTIONS_DIR.relative_to(PROJECT_ROOT)}")
            return 1
        csv_path = write_csv(all_rows)
        summary = aggregate(all_rows)
        md_path = write_report(summary)
        print("\n" + "=" * 60)
        print(f"rescored {len(all_rows)} cached predictions")
        print(f"results -> {csv_path.relative_to(PROJECT_ROOT)}")
        print(f"report  -> {md_path.relative_to(PROJECT_ROOT)}")
        print("=" * 60)
        print(md_path.read_text(encoding="utf-8"))
        return 0

    manifest = load_manifest()
    if args.max_clips:
        manifest = manifest[: args.max_clips]
    print(f"loaded {len(manifest)} clips from {MANIFEST_PATH.relative_to(PROJECT_ROOT)}")

    all_rows: List[Dict[str, Any]] = []
    for name in args.models:
        backend = BACKENDS[name]()
        rows = run_backend(backend, manifest, skip_existing=args.skip_existing)
        all_rows.extend(rows)

    if not all_rows:
        print("no results — all backends failed")
        return 1

    csv_path = write_csv(all_rows)
    summary = aggregate(all_rows)
    md_path = write_report(summary)

    print("\n" + "=" * 60)
    print(f"results -> {csv_path.relative_to(PROJECT_ROOT)}")
    print(f"report  -> {md_path.relative_to(PROJECT_ROOT)}")
    print("=" * 60)
    print(md_path.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
