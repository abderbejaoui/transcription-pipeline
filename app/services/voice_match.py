"""Voice fingerprint store + retrieval (phonetic, speaker-invariant).

Background
----------
The previous version mean-pooled the *self-supervised* `wav2vec2-base`
hidden states. That representation is dominated by speaker timbre and
recording conditions, so two completely different words spoken by the
same person scored ~0.85 cosine — well above unrelated-word thresholds.

This rewrite replaces the embedding with a CTC phonetic transcript:
1. Greedy-decode the audio with `wav2vec2-base-960h` (CTC, character
   vocabulary).
2. Strip spaces and lowercase. The result behaves like a coarse phoneme
   string, e.g. "doliprane" -> "dolarain", "ifer algon" -> "eyeforalgon".
3. Compare two clips with normalized Levenshtein similarity in [0, 1].

Properties (verified empirically with macOS `say`):
- Same word, different voice  -> high similarity (~0.6-1.0)
- Different words, same voice -> low similarity  (~0.2-0.4)
- Empty / silence             -> empty string, similarity 0

The "embedding" stored on disk is the phonetic string itself. The public
API matches what the previous module exposed so callers keep working.

Public API
----------
warm_up()                       # preload the CTC model in a background thread
load_audio(path)                # ffmpeg -> 16 kHz mono float32
slice_audio(wav, t0, t1)        # safe slice with small padding
embed(wav) -> str               # CTC phonetic transcript
register(term, audio_path, t0, t1, ...)
register_with_embedding(term, embedding, ...)   # accepts str OR ndarray
match(wav_or_path, t0, t1, top_k, threshold)
list_voices()
has_term(term)
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
INDEX_PATH = DATA_DIR / "voice_index.json"   # phonetic strings live here
META_PATH = DATA_DIR / "voice_index.jsonl"

# Legacy file from the old SSL-embedding implementation. Removed on first
# save so the migration is clean.
LEGACY_NPZ = DATA_DIR / "voice_index.npz"

SAMPLE_RATE = 16_000
MIN_SLICE_SECONDS = 0.20  # CTC needs a minimum input length


# ---------------------------------------------------------------------------
# Lazy CTC model loader
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
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        if torch.cuda.is_available():
            _device = "cuda"
        elif torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"

        model_id = os.environ.get(
            "VOICE_ENCODER_MODEL", "facebook/wav2vec2-base-960h"
        )
        print(f"[voice_match] loading CTC {model_id} on {_device}")
        _processor = Wav2Vec2Processor.from_pretrained(model_id)
        _model = Wav2Vec2ForCTC.from_pretrained(model_id).to(_device).eval()
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
# CTC phonetic decode + edit similarity
# ---------------------------------------------------------------------------


def embed(wav: np.ndarray) -> str:
    """Greedy-CTC decode `wav` and return its phonetic string.

    Empty for silence / nonsense. Spaces are stripped so the comparison
    operates on a single token sequence.
    """
    processor, model, torch, device = _load_models()
    wav = _ensure_min_length(wav)
    with torch.inference_mode():
        inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        logits = model(**inputs).logits.squeeze(0)        # (T, V)
        ids = logits.argmax(dim=-1).cpu().tolist()        # greedy
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
    """Normalized Levenshtein similarity in [0, 1]."""
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
    dist = prev[m]
    return 1.0 - dist / max(n, m)


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
    phonetic: str  # CTC transcript, lowercase, no spaces
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
            fh.write(
                json.dumps(
                    {
                        "id": e.id,
                        "term": e.term,
                        "language": e.language,
                        "duration_s": e.duration_s,
                        "created_at": e.created_at,
                        "phonetic": e.phonetic,
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
    """Save the phonetic fingerprint of `audio[start_s..end_s]` keyed by `term`."""
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
    """Register an already-computed phonetic string.

    Accepts a string (preferred) or a numpy array (legacy callers — ignored
    and recorded as empty). Numpy arrays from the previous SSL-embedding
    implementation are not phonetic, so they're not useful at match time.
    """
    if not term or not term.strip():
        raise ValueError("term is required")
    if isinstance(embedding, np.ndarray):
        phon = ""  # legacy dense vector, no longer usable
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
        "id": e.id,
        "term": e.term,
        "language": e.language,
        "duration_s": round(e.duration_s, 3),
        "phonetic": e.phonetic,
        "description": e.description,
        "source": e.source,
    }


def match(
    wav_or_path: Union[str, Path, np.ndarray],
    start_s: Optional[float] = None,
    end_s: Optional[float] = None,
    threshold: float = 0.55,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Return up to `top_k` nearest voice entries above `threshold`.

    Similarity is normalized Levenshtein over CTC phonetic transcripts.
    Entries with empty phonetic strings (legacy seed entries) are skipped.
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
    if not q:
        return []
    sims = []
    for e in _entries:
        if not e.phonetic:
            sims.append(0.0)
        else:
            sims.append(_lev_sim(q, e.phonetic))
    sims_arr = np.asarray(sims, dtype=np.float32)
    order = np.argsort(-sims_arr)
    out: List[Dict[str, Any]] = []
    for i in order[:top_k]:
        if sims_arr[i] >= threshold:
            e = _entries[i]
            out.append(
                {
                    "id": e.id,
                    "term": e.term,
                    "similarity": round(float(sims_arr[i]), 4),
                    "language": e.language,
                    "phonetic": e.phonetic,
                    "query_phonetic": q,
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
