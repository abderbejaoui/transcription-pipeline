"""Voice fingerprint store + retrieval.

When the user fixes a misheard word, we slice the exact audio for that word
using Whisper's timestamps, embed it with wav2vec2-base, and store the
embedding under the canonical term. Next time the same audio appears, the
embedding is close in cosine and the LLM reranker is given the right term
in its candidate list.

Public API
----------
warm_up()                       # preload models in a background thread
load_audio(path)                # ffmpeg -> 16 kHz mono float32
slice_audio(wav, t0, t1)        # safe slice with small padding
embed(wav)                      # 768-d L2-normalised vector
register(term, audio_path, t0, t1, language=None)  # add to index
match(wav_or_path, t0, t1, top_k, threshold)       # nearest-neighbour search
list_voices()
reset()
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
INDEX_PATH = DATA_DIR / "voice_index.npz"
META_PATH = DATA_DIR / "voice_index.jsonl"

SAMPLE_RATE = 16_000
MIN_SLICE_SECONDS = 0.20  # wav2vec2 needs a minimum input length


# ---------------------------------------------------------------------------
# Lazy model loader
# ---------------------------------------------------------------------------

_lock = threading.Lock()
_processor = None
_model = None
_torch = None
_device = "cpu"


def _load_models():
    global _processor, _model, _torch, _device
    if _model is not None:
        return _processor, _model, _torch, _device
    with _lock:
        if _model is not None:
            return _processor, _model, _torch, _device
        import torch
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        model_id = os.environ.get("VOICE_ENCODER_MODEL", "facebook/wav2vec2-base")
        print(f"[voice_match] loading {model_id} on {_device}")
        _processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        _model = Wav2Vec2Model.from_pretrained(model_id).to(_device).eval()
        _torch = torch
        return _processor, _model, _torch, _device


def warm_up() -> None:
    """Trigger model load in a background thread."""
    threading.Thread(target=_load_models, daemon=True).start()


# ---------------------------------------------------------------------------
# Audio decoding
# ---------------------------------------------------------------------------


def load_audio(path: Union[str, Path]) -> np.ndarray:
    """ffmpeg-decode any audio file -> 16 kHz mono float32 in [-1, 1]."""
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
# Embedding
# ---------------------------------------------------------------------------


def embed(wav: np.ndarray) -> np.ndarray:
    """Mean-pool wav2vec2 last hidden state, L2-normalise."""
    processor, model, torch, device = _load_models()
    wav = _ensure_min_length(wav)
    with torch.inference_mode():
        inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        out = model(**inputs)
        hidden = out.last_hidden_state  # (1, T, D)
        pooled = hidden.mean(dim=1).squeeze(0)
        pooled = torch.nn.functional.normalize(pooled, dim=0)
    return pooled.detach().cpu().numpy().astype(np.float32)


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
    embedding: np.ndarray  # (D,)
    description: Optional[str] = None
    source: str = "user"  # "seed" | "user"


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
                npz = np.load(INDEX_PATH)
                vectors = npz["embeddings"]
                ids = list(npz["ids"])
                meta_by_id: Dict[str, Dict[str, Any]] = {}
                with META_PATH.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        m = json.loads(line)
                        meta_by_id[m["id"]] = m
                for i, vec_id in enumerate(ids):
                    vec_id_s = str(vec_id)
                    m = meta_by_id.get(vec_id_s)
                    if m is None:
                        continue
                    _entries.append(
                        VoiceEntry(
                            id=vec_id_s,
                            term=m["term"],
                            language=m.get("language"),
                            duration_s=float(m.get("duration_s", 0.0)),
                            created_at=float(m.get("created_at", 0.0)),
                            embedding=vectors[i].astype(np.float32),
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
    if not _entries:
        for p in (INDEX_PATH, META_PATH):
            if p.exists():
                p.unlink()
        return
    embeddings = np.stack([e.embedding for e in _entries])
    ids = np.array([e.id for e in _entries])
    np.savez(INDEX_PATH, embeddings=embeddings, ids=ids)
    with META_PATH.open("w", encoding="utf-8") as fh:
        for e in _entries:
            fh.write(
                json.dumps(
                    {
                        "id": e.id,
                        "term": e.term,
                        "language": e.language,
                        "duration_s": e.duration_s,
                        "created_at": e.created_at,
                        "description": e.description,
                        "source": e.source,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


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
    """Save the audio fingerprint of `audio[start_s..end_s]` keyed by `term`."""
    if not term or not term.strip():
        raise ValueError("term is required")
    _load_index()
    wav = load_audio(audio_path)
    seg = slice_audio(wav, start_s, end_s)
    if seg.size < int(0.05 * SAMPLE_RATE):
        raise ValueError(f"audio segment too short: {seg.size / SAMPLE_RATE:.3f}s")
    vec = embed(seg)
    entry = VoiceEntry(
        id=uuid.uuid4().hex,
        term=term.strip(),
        language=language,
        duration_s=seg.size / SAMPLE_RATE,
        created_at=time.time(),
        embedding=vec,
        description=description,
        source=source,
    )
    with _index_lock:
        _entries.append(entry)
        _save_index()
    return _serialize_entry(entry)


def register_with_embedding(
    term: str,
    embedding: np.ndarray,
    *,
    duration_s: float = 0.0,
    language: Optional[str] = None,
    description: Optional[str] = None,
    source: str = "seed",
) -> Dict[str, Any]:
    """Register an already-computed embedding (e.g. from TTS reference audio).
    Skips audio decoding and slicing."""
    if not term or not term.strip():
        raise ValueError("term is required")
    if embedding.ndim != 1 or embedding.dtype != np.float32:
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
    _load_index()
    entry = VoiceEntry(
        id=uuid.uuid4().hex,
        term=term.strip(),
        language=language,
        duration_s=float(duration_s),
        created_at=time.time(),
        embedding=embedding,
        description=description,
        source=source,
    )
    with _index_lock:
        _entries.append(entry)
        _save_index()
    return _serialize_entry(entry)


def _serialize_entry(e: VoiceEntry) -> Dict[str, Any]:
    return {
        "id": e.id,
        "term": e.term,
        "language": e.language,
        "duration_s": round(e.duration_s, 3),
        "description": e.description,
        "source": e.source,
    }


def match(
    wav_or_path: Union[str, Path, np.ndarray],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    threshold: float = 0.65,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return up to `top_k` nearest voice entries above `threshold`.

    `wav_or_path` can be a file path (then start_s/end_s slice it) or a
    numpy waveform already prepared by the caller.
    """
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
    sims = np.array([float(np.dot(q, e.embedding)) for e in _entries])
    order = np.argsort(-sims)
    out: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        if sims[i] >= threshold:
            e = _entries[i]
            out.append(
                {
                    "id": e.id,
                    "term": e.term,
                    "similarity": round(float(sims[i]), 4),
                    "language": e.language,
                    "description": e.description,
                    "source": e.source,
                }
            )
    return out


def list_voices() -> List[Dict[str, Any]]:
    _load_index()
    return [
        {
            **_serialize_entry(e),
            "created_at": e.created_at,
        }
        for e in _entries
    ]


def has_term(term: str) -> bool:
    """True if the index has at least one voice entry for this canonical term."""
    _load_index()
    target = term.strip().lower()
    return any(e.term.strip().lower() == target for e in _entries)


def reset() -> None:
    """Clear in-memory + on-disk index. Useful for tests."""
    global _entries, _loaded
    with _index_lock:
        _entries = []
        _loaded = True
        _save_index()
