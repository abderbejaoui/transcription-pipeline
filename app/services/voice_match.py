"""Voice fingerprint store + retrieval (IPA phoneme, speaker-invariant).

This module wraps `facebook/wav2vec2-lv-60-espeak-cv-ft`: a CTC model whose
vocabulary is **IPA phonemes**, trained on multilingual Common Voice. It
outputs sequences like `ɛ f ə ɹ æ l ɡ æ n` for "Efferalgan" — the same
string regardless of language, accent, or how the word is spelled.

Public API (kept stable for callers in main.py)
----------
warm_up()                      # preload the CTC model in a background thread
load_audio(path)               # ffmpeg -> 16 kHz mono float32
slice_audio(wav, t0, t1)       # safe slice with small padding
embed(wav) -> str              # IPA phoneme transcript (no spaces, lowercase)
posteriors(wav) -> ndarray     # (T, V) softmax matrix — used by audio_verify
vocab() -> dict[str, int]      # IPA token -> id, used by audio_verify
register(...), match(...), list_voices(), has_term(...), reset()
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
INDEX_PATH = DATA_DIR / "voice_index.json"
META_PATH = DATA_DIR / "voice_index.jsonl"
LEGACY_NPZ = DATA_DIR / "voice_index.npz"   # removed on first save

SAMPLE_RATE = 16_000
MIN_SLICE_SECONDS = 0.20

DEFAULT_MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"


# ---------------------------------------------------------------------------
# Lazy CTC model loader
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_processor = None
_model = None
_torch = None
_device = "cpu"
_vocab_token_to_id: Optional[Dict[str, int]] = None
_id_to_vocab_token: Optional[Dict[int, str]] = None


def _load_models():
    global _processor, _model, _torch, _device
    global _vocab_token_to_id, _id_to_vocab_token
    if _model is not None:
        return _processor, _model, _torch, _device
    with _lock:
        if _model is not None:
            return _processor, _model, _torch, _device
        import torch
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        model_id = os.environ.get("VOICE_ENCODER_MODEL", DEFAULT_MODEL_ID)
        print(f"[voice_match] loading IPA-CTC {model_id} on {_device}")
        _processor = Wav2Vec2Processor.from_pretrained(model_id)
        _model = Wav2Vec2ForCTC.from_pretrained(model_id).to(_device).eval()
        _torch = torch
        try:
            v = _processor.tokenizer.get_vocab()
            _vocab_token_to_id = dict(v)
            _id_to_vocab_token = {i: t for t, i in v.items()}
        except Exception:
            _vocab_token_to_id = {}
            _id_to_vocab_token = {}
        return _processor, _model, _torch, _device


def warm_up() -> None:
    threading.Thread(target=_load_models, daemon=True).start()


def vocab() -> Dict[str, int]:
    _load_models()
    return dict(_vocab_token_to_id or {})


def id_to_token() -> Dict[int, str]:
    _load_models()
    return dict(_id_to_vocab_token or {})


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def load_audio(path: Union[str, Path]) -> np.ndarray:
    path = str(path)
    cmd = [
        "ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-i", path,
        "-f", "s16le", "-ac", "1", "-ar", str(SAMPLE_RATE), "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"ffmpeg failed to decode {path}: {exc.stderr.decode('utf-8', 'ignore')}"
        ) from exc
    return np.frombuffer(out, dtype=np.int16).astype(np.float32) / 32768.0


def slice_audio(wav: np.ndarray, start_s: float, end_s: float, pad_s: float = 0.05) -> np.ndarray:
    if wav.size == 0:
        return wav
    start = max(0, int((start_s - pad_s) * SAMPLE_RATE))
    end = min(wav.size, int((end_s + pad_s) * SAMPLE_RATE))
    if end <= start:
        return np.zeros(int(SAMPLE_RATE * MIN_SLICE_SECONDS), dtype=np.float32)
    return wav[start:end].astype(np.float32, copy=False)


def _ensure_min_length(wav: np.ndarray) -> np.ndarray:
    min_samples = int(MIN_SLICE_SECONDS * SAMPLE_RATE)
    if wav.size >= min_samples:
        return wav
    pad = np.zeros(min_samples - wav.size, dtype=np.float32)
    return np.concatenate([wav, pad])


# ---------------------------------------------------------------------------
# IPA decoding + posteriors
# ---------------------------------------------------------------------------


def posteriors(wav: np.ndarray) -> np.ndarray:
    """Return the (T, V) softmax matrix. Used by audio_verify."""
    processor, model, torch, device = _load_models()
    wav = _ensure_min_length(wav)
    with torch.inference_mode():
        inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        logits = model(**inputs).logits.squeeze(0)
        probs = torch.softmax(logits, dim=-1)
    return probs.detach().cpu().numpy().astype(np.float32)


def embed(wav: np.ndarray) -> str:
    processor, model, torch, device = _load_models()
    wav = _ensure_min_length(wav)
    with torch.inference_mode():
        inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        logits = model(**inputs).logits.squeeze(0)
        ids = logits.argmax(dim=-1).cpu().tolist()
    pad_id = model.config.pad_token_id
    tokens: List[int] = []
    prev = None
    for i in ids:
        if i != prev and i != pad_id:
            tokens.append(i)
        prev = i
    text = processor.tokenizer.decode(tokens, skip_special_tokens=True)
    return text.strip().replace(" ", "").lower()


def _lev_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return 1.0 - prev[m] / max(n, m)


# ---------------------------------------------------------------------------
# Persistent index
# ---------------------------------------------------------------------------


@dataclass
class VoiceEntry:
    id: str
    term: str
    language: Optional[str]
    duration_s: float
    created_at: float
    phonetic: str
    description: Optional[str] = None
    source: str = "user"


_index_lock = threading.Lock()
_entries: List[VoiceEntry] = []
_loaded = False


def _load_index() -> None:
    global _entries, _loaded
    with _index_lock:
        if _loaded:
            return
        _entries = []
        if INDEX_PATH.exists() and META_PATH.exists():
            try:
                with INDEX_PATH.open("r", encoding="utf-8") as fh:
                    phon_by_id: Dict[str, str] = json.load(fh)
                meta_by_id: Dict[str, Dict[str, Any]] = {}
                with META_PATH.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        m = json.loads(line)
                        meta_by_id[m["id"]] = m
                for vec_id, phon in phon_by_id.items():
                    m = meta_by_id.get(vec_id)
                    if m is None:
                        continue
                    _entries.append(
                        VoiceEntry(
                            id=vec_id,
                            term=m["term"],
                            language=m.get("language"),
                            duration_s=float(m.get("duration_s", 0.0)),
                            created_at=float(m.get("created_at", 0.0)),
                            phonetic=str(phon or "").lower(),
                            description=m.get("description"),
                            source=m.get("source", "user"),
                        )
                    )
            except Exception as exc:
                print(f"[voice_match] failed to load index: {exc}")
                _entries = []
        _loaded = True


def _save_index() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if LEGACY_NPZ.exists():
        try:
            LEGACY_NPZ.unlink()
        except Exception:
            pass
    if not _entries:
        for p in (INDEX_PATH, META_PATH):
            if p.exists():
                p.unlink()
        return
    phon_by_id = {e.id: e.phonetic for e in _entries}
    INDEX_PATH.write_text(json.dumps(phon_by_id, ensure_ascii=False), encoding="utf-8")
    with META_PATH.open("w", encoding="utf-8") as fh:
        for e in _entries:
            fh.write(json.dumps({
                "id": e.id, "term": e.term, "language": e.language,
                "duration_s": e.duration_s, "created_at": e.created_at,
                "phonetic": e.phonetic, "description": e.description,
                "source": e.source,
            }, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Public ops
# ---------------------------------------------------------------------------


def register(
    term: str,
    audio_path: Union[str, Path],
    start_s: float,
    end_s: float,
    language: Optional[str] = None,
    description: Optional[str] = None,
    source: str = "user",
) -> Dict[str, Any]:
    if not term or not term.strip():
        raise ValueError("term is required")
    _load_index()
    wav = load_audio(audio_path)
    seg = slice_audio(wav, start_s, end_s)
    if seg.size < int(0.05 * SAMPLE_RATE):
        raise ValueError(f"audio segment too short: {seg.size / SAMPLE_RATE:.3f}s")
    phon = embed(seg)
    entry = VoiceEntry(
        id=uuid.uuid4().hex,
        term=term.strip(),
        language=language,
        duration_s=seg.size / SAMPLE_RATE,
        created_at=time.time(),
        phonetic=phon,
        description=description,
        source=source,
    )
    with _index_lock:
        _entries.append(entry)
        _save_index()
    return _serialize_entry(entry)


def register_with_embedding(
    term: str,
    embedding: Union[str, np.ndarray],
    *,
    duration_s: float = 0.0,
    language: Optional[str] = None,
    description: Optional[str] = None,
    source: str = "seed",
) -> Dict[str, Any]:
    if not term or not term.strip():
        raise ValueError("term is required")
    if isinstance(embedding, np.ndarray):
        phon = ""
    else:
        phon = str(embedding or "").strip().lower().replace(" ", "")
    _load_index()
    entry = VoiceEntry(
        id=uuid.uuid4().hex,
        term=term.strip(),
        language=language,
        duration_s=float(duration_s),
        created_at=time.time(),
        phonetic=phon,
        description=description,
        source=source,
    )
    with _index_lock:
        _entries.append(entry)
        _save_index()
    return _serialize_entry(entry)


def _serialize_entry(e: VoiceEntry) -> Dict[str, Any]:
    return {
        "id": e.id, "term": e.term, "language": e.language,
        "duration_s": round(e.duration_s, 3),
        "phonetic": e.phonetic, "description": e.description,
        "source": e.source,
    }


def match(
    wav_or_path: Union[str, Path, np.ndarray],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    threshold: float = 0.55,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    _load_index()
    if not _entries:
        return []
    if isinstance(wav_or_path, (str, Path)):
        wav = load_audio(wav_or_path)
        if start_s is not None and end_s is not None:
            wav = slice_audio(wav, start_s, end_s)
    else:
        wav = wav_or_path
    if wav.size < int(0.05 * SAMPLE_RATE):
        return []
    q = embed(wav)
    if not q:
        return []
    sims = []
    for e in _entries:
        sims.append(_lev_sim(q, e.phonetic) if e.phonetic else 0.0)
    sims_arr = np.asarray(sims, dtype=np.float32)
    order = np.argsort(-sims_arr)
    out: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        if sims_arr[i] >= threshold:
            e = _entries[i]
            out.append({
                "id": e.id, "term": e.term,
                "similarity": round(float(sims_arr[i]), 4),
                "language": e.language,
                "phonetic": e.phonetic,
                "query_phonetic": q,
                "description": e.description,
                "source": e.source,
            })
    return out


def list_voices() -> List[Dict[str, Any]]:
    _load_index()
    return [{**_serialize_entry(e), "created_at": e.created_at} for e in _entries]


def has_term(term: str) -> bool:
    _load_index()
    target = term.strip().lower()
    return any(e.term.strip().lower() == target for e in _entries)


def reset() -> None:
    global _entries, _loaded
    with _index_lock:
        _entries = []
        _loaded = True
        _save_index()
