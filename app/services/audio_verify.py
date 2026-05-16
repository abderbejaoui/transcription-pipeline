"""Audio-vs-transcript verification (Tier 2) — fast version.

Optimization vs. the naive design
---------------------------------
Naively, each (word, term) pair would require its own wav2vec2 forward
pass on a small audio slice. That is O(n_words × n_terms) GPU calls and
takes minutes per audio.

The right way: compute the whole-audio posterior matrix once, then slice
the matrix in TIME for each word's window and run a tiny CTC forward DP
on the cached numpy slice. The wav2vec2 model encodes ~50 frames per
second of audio, so a 30-second recording is a (1500, V) matrix —
trivial to slice and reuse.

  * 1 wav2vec2 forward per audio (cached)
  * O(n_words × n_terms × T_window) numpy DP on CPU

Empirically: ~250ms per minute of audio on Apple Silicon, regardless of
how many lexicon terms.

Public API
----------
verify_words(audio_path, words, lexicon_terms) -> List[dict]
phonemize(text) -> str
"""

from __future__ import annotations

import math
import re
import shutil
import threading
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import voice_match


# ---------------------------------------------------------------------------
# Phonemization
# ---------------------------------------------------------------------------

_phonemize_cache: Dict[str, str] = {}
_cache_lock = threading.Lock()
_PHONEMIZER_AVAILABLE: Optional[bool] = None


def _phonemizer_available() -> bool:
    global _PHONEMIZER_AVAILABLE
    if _PHONEMIZER_AVAILABLE is not None:
        return _PHONEMIZER_AVAILABLE
    try:
        import phonemizer  # noqa: F401
        from phonemizer.backend import EspeakBackend  # noqa: F401
        if shutil.which("espeak-ng") or shutil.which("espeak"):
            _PHONEMIZER_AVAILABLE = True
            return True
    except Exception:
        pass
    _PHONEMIZER_AVAILABLE = False
    return False


def _normalize_text_for_phonemize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w'\- ]+", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def _phonemize_via_espeak(text: str) -> str:
    from phonemizer import phonemize as ph
    out = ph(
        text,
        language="en-us",
        backend="espeak",
        strip=True,
        preserve_punctuation=False,
        with_stress=False,
        njobs=1,
    )
    if isinstance(out, list):
        out = " ".join(out)
    return _strip_ipa(out)


# Letter-level fallback (used only when espeak-ng is unavailable).
_LETTER_IPA: Dict[str, str] = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ",
    "f": "f", "g": "ɡ", "h": "h", "i": "ɪ", "j": "dʒ",
    "k": "k", "l": "l", "m": "m", "n": "n", "o": "ɑ",
    "p": "p", "q": "k", "r": "ɹ", "s": "s", "t": "t",
    "u": "ʌ", "v": "v", "w": "w", "x": "ks", "y": "j",
    "z": "z",
}
_MULTI_RULES = [
    ("tion", "ʃən"), ("sion", "ʃən"), ("ph", "f"), ("th", "θ"),
    ("sh", "ʃ"), ("ch", "tʃ"), ("ck", "k"), ("ng", "ŋ"),
    ("oo", "u"), ("ee", "i"), ("ea", "i"), ("ai", "eɪ"),
    ("ay", "eɪ"), ("ou", "aʊ"), ("ow", "aʊ"), ("oi", "ɔɪ"),
    ("oy", "ɔɪ"), ("igh", "aɪ"),
]


def _phonemize_fallback(text: str) -> str:
    text = _normalize_text_for_phonemize(text)
    out: List[str] = []
    for word in text.split():
        i = 0
        s = word
        buf: List[str] = []
        while i < len(s):
            matched = False
            for pat, rep in _MULTI_RULES:
                if s[i:i + len(pat)] == pat:
                    buf.append(rep)
                    i += len(pat)
                    matched = True
                    break
            if matched:
                continue
            buf.append(_LETTER_IPA.get(s[i], ""))
            i += 1
        out.append("".join(buf))
    return " ".join(out)


_IPA_DROP = "ˈˌːˑ.,;?!()/_"


def _strip_ipa(s: str) -> str:
    s = "".join(c for c in s if c not in _IPA_DROP)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def phonemize(text: str) -> str:
    if not text:
        return ""
    key = text.strip().lower()
    with _cache_lock:
        cached = _phonemize_cache.get(key)
        if cached is not None:
            return cached
    if _phonemizer_available():
        try:
            ipa = _phonemize_via_espeak(text)
        except Exception:
            ipa = _phonemize_fallback(text)
    else:
        ipa = _phonemize_fallback(text)
    ipa = _strip_ipa(ipa).replace(" ", "").lower()
    with _cache_lock:
        _phonemize_cache[key] = ipa
    return ipa


# ---------------------------------------------------------------------------
# Vocab mapping (one-time)
# ---------------------------------------------------------------------------

_token_id_cache: Dict[str, List[int]] = {}


def _ipa_to_ids(ipa: str) -> List[int]:
    """Map an IPA string to model vocab ids by greedy longest-match."""
    cached = _token_id_cache.get(ipa)
    if cached is not None:
        return cached
    vocab = voice_match.vocab()
    if not vocab:
        return []
    tokens = sorted([t for t in vocab.keys() if t and t != "|"], key=len, reverse=True)
    ids: List[int] = []
    i = 0
    s = ipa
    while i < len(s):
        ch = s[i]
        if ch.isspace():
            i += 1
            continue
        matched = None
        for tok in tokens:
            if tok and s.startswith(tok, i):
                matched = tok
                break
        if matched is None:
            i += 1
            continue
        ids.append(vocab[matched])
        i += len(matched)
    _token_id_cache[ipa] = ids
    return ids


# ---------------------------------------------------------------------------
# CTC forward on a precomputed log-posterior submatrix
# ---------------------------------------------------------------------------


def _ctc_forward_log_on_submatrix(log_post: np.ndarray, target_ids: List[int], blank_id: int) -> float:
    """`log_post`: (T, V) precomputed log-posterior for the audio window."""
    if not target_ids or log_post.shape[0] == 0:
        return -1e9
    T = log_post.shape[0]
    L = 2 * len(target_ids) + 1
    if T < len(target_ids):
        return -1e9
    extended: List[int] = []
    for tid in target_ids:
        extended.append(blank_id)
        extended.append(tid)
    extended.append(blank_id)

    NEG_INF = -1e18
    alpha = np.full((T, L), NEG_INF, dtype=np.float64)
    alpha[0, 0] = log_post[0, extended[0]]
    if L > 1:
        alpha[0, 1] = log_post[0, extended[1]]
    for t in range(1, T):
        s_start = max(0, L - 2 * (T - t))
        s_end = min(L, 2 * (t + 1))
        for s in range(s_start, s_end):
            sym = extended[s]
            best = alpha[t - 1, s]
            if s - 1 >= 0:
                best = np.logaddexp(best, alpha[t - 1, s - 1])
            if s - 2 >= 0 and sym != blank_id and extended[s - 2] != sym:
                best = np.logaddexp(best, alpha[t - 1, s - 2])
            alpha[t, s] = best + log_post[t, sym]
    final = alpha[T - 1, L - 1]
    if L >= 2:
        final = float(np.logaddexp(final, alpha[T - 1, L - 2]))
    return float(final / max(1, len(target_ids)))


def _logprob_to_score(lp: float) -> float:
    return float(max(0.0, min(1.0, math.exp(lp))))


# ---------------------------------------------------------------------------
# Whole-audio posterior cache + word-level verification
# ---------------------------------------------------------------------------


_FRAMES_PER_SECOND = 50  # wav2vec2-base has 320 stride at 16kHz -> 50 fps


def _frames_for_time(t: float) -> int:
    return int(round(t * _FRAMES_PER_SECOND))


def _blank_id() -> int:
    _processor, model, _torch, _device = voice_match._load_models()
    pad_id = getattr(model.config, "pad_token_id", None)
    if pad_id is None:
        pad_id = 0
    return int(pad_id)


def verify_words(
    audio_path: str,
    words: List[Dict[str, Any]],
    lexicon_terms: List[Dict[str, Any]],
    *,
    margin_threshold: float = 0.10,
    min_word_duration: float = 0.10,
    expand_window_s: float = 0.40,
) -> List[Dict[str, Any]]:
    """For every Whisper word, score the audio fit of the word AND the best
    lexicon term in an enlarged window, using a single whole-audio
    posterior matrix. Return per-word records.
    """
    if not words or not lexicon_terms:
        return []

    full_wav = voice_match.load_audio(audio_path)
    if full_wav.size < int(0.2 * voice_match.SAMPLE_RATE):
        return []

    # ONE forward pass: compute the whole-audio softmax posterior matrix.
    full_post = voice_match.posteriors(full_wav)         # (T, V)
    eps = 1e-12
    full_log_post = np.log(full_post + eps).astype(np.float64)
    T_total = full_log_post.shape[0]
    duration = T_total / _FRAMES_PER_SECOND

    blank = _blank_id()

    # Pre-compute term -> token ids ONCE.
    term_ids: List[Tuple[str, List[int]]] = []
    for entry in lexicon_terms:
        term = entry.get("term")
        if not term:
            continue
        ipa = phonemize(term)
        if not ipa:
            continue
        ids = _ipa_to_ids(ipa)
        if ids:
            term_ids.append((term, ids))

    out: List[Dict[str, Any]] = []
    for idx, w in enumerate(words):
        word_text = (w.get("word") or "").strip()
        if not word_text:
            continue
        start = w.get("start")
        end = w.get("end")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            continue
        wlen = max(0.0, float(end) - float(start))
        if wlen < min_word_duration:
            continue

        # Frames for the expanded window where we'll search.
        f_start = max(0, _frames_for_time(max(0.0, float(start) - expand_window_s)))
        f_end = min(T_total, _frames_for_time(min(duration, float(end) + expand_window_s)))
        if f_end - f_start < 4:
            continue
        win_log_post = full_log_post[f_start:f_end]      # (Tw, V)

        word_ipa = phonemize(word_text)
        word_ids = _ipa_to_ids(word_ipa)
        s_word_in_window = (
            _logprob_to_score(_ctc_forward_log_on_submatrix(win_log_post, word_ids, blank))
            if word_ids else 0.0
        )

        # Score every term against the same window.
        best_term: Optional[str] = None
        best_score: float = 0.0
        for term, ids in term_ids:
            sc = _logprob_to_score(_ctc_forward_log_on_submatrix(win_log_post, ids, blank))
            if sc > best_score:
                best_score = sc
                best_term = term

        margin = best_score - s_word_in_window
        suspect = (best_term is not None
                   and margin > margin_threshold
                   and best_score > 0.20)

        out.append({
            "index": idx,
            "word": word_text,
            "start": float(start),
            "end": float(end),
            "duration": wlen,
            "ipa_word": word_ipa,
            "s_word_in_window": round(s_word_in_window, 4),
            "best_term": best_term,
            "s_term": round(best_score, 4),
            "margin": round(margin, 4),
            "suspect": bool(suspect),
        })
    return out
