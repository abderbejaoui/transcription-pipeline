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
# Text normalization (must match build_eval_set.py)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS_RE.sub("", s)
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
            max_new_tokens=256,
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
                    **inputs, max_new_tokens=256, do_sample=False,
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
    """Loads VibeVoice via transformers. The model auto-detects language and
    natively handles code-switching (matches our use case)."""

    name = "vibevoice-asr"

    def __init__(self, repo_id: str = "microsoft/VibeVoice-ASR"):
        self.repo_id = repo_id
        self._model = None
        self._processor = None
        self._device = None

    def prepare(self) -> None:
        try:
            import torch
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(f"transformers/torch missing: {exc}")

        if torch.cuda.is_available():
            self._device, dtype = "cuda:0", torch.bfloat16
        elif torch.backends.mps.is_available():
            self._device, dtype = "mps", torch.float16
        else:
            self._device, dtype = "cpu", torch.float32

        print(f"[vibevoice] loading {self.repo_id} on {self._device} ({dtype})")
        self._processor = AutoProcessor.from_pretrained(self.repo_id, trust_remote_code=True)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.repo_id,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(self._device).eval()

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None and self._processor is not None
        import torch
        import soundfile as sf

        audio, sr = sf.read(str(wav_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                sr = 16000
            except Exception as exc:
                return Prediction(text="", inference_seconds=0.0,
                                  extra={"error": f"resample failed: {exc!r}"})

        t0 = time.time()
        inputs = self._processor(audio, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, max_new_tokens=400)
        text = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return Prediction(text=text.strip(), inference_seconds=time.time() - t0)


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
    name = "whisper-small-gulf"

    def __init__(self, repo_id: str = "otozz/whisper-small-dialect_gulf"):
        self.repo_id = repo_id
        self._model = None

    def prepare(self) -> None:
        from faster_whisper import WhisperModel
        device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        print(f"[whisper-gulf] loading {self.repo_id} on {device}")
        self._model = WhisperModel(self.repo_id, device=device, compute_type=compute_type)

    def transcribe(self, wav_path: Path, *, language: Optional[str] = None) -> Prediction:
        assert self._model is not None
        t0 = time.time()
        # Force Arabic for Arabic/mixed clips; English for English
        lang = "ar" if (language or "en") in ("ar", "mixed") else "en"
        segs, info = self._model.transcribe(
            str(wav_path),
            language=lang,
            beam_size=5,
            vad_filter=False,
            without_timestamps=True,
        )
        text = "".join(s.text for s in segs).strip()
        return Prediction(text=text, inference_seconds=time.time() - t0,
                          extra={"detected_language": info.language})


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

    # If skip_existing, load already-computed rows without preparing the model.
    if skip_existing:
        existing = list(pred_dir.glob("*.json"))
        if len(existing) > 0:
            print(f"  → found {len(existing)} existing predictions, loading without re-running")
            rows = []
            for p in existing:
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
    "whisper":      WhisperBackend,
    "qwen3":        QwenAsrBackend,
    "vibevoice":    VibeVoiceBackend,
    "omniASR":      OmniAsrBackend,
    "whisper_gulf": WhisperGulfBackend,
    "qwen3_ksa":    QwenKsaBackend,
    "qwen3_uae":    QwenUaeBackend,
}


def main() -> int:
    p = argparse.ArgumentParser(description="Run zero-shot ASR bake-off.")
    p.add_argument(
        "--models",
        nargs="+",
        choices=list(BACKENDS.keys()),
        default=list(BACKENDS.keys()),
        help="Which backends to run (default: all)",
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
    args = p.parse_args()

    if args.eval_dir is not None:
        eval_dir = args.eval_dir
        if not eval_dir.is_absolute():
            eval_dir = (PROJECT_ROOT / eval_dir).resolve()
        _set_eval_dir(eval_dir)
        print(f"using eval dir: {EVAL_DIR.relative_to(PROJECT_ROOT)}")

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
