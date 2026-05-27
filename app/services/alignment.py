"""Word-level forced alignment for ASR transcripts.

The Qwen3-ASR model emits text but no word timestamps. To know WHERE in
the audio a suspicious word was spoken, we run a lightweight CTC model
over the same audio and use its frame-level posteriors to align each
ASR word back to a (start_s, end_s) interval.

Approach
--------
1. Decode the audio with `facebook/wav2vec2-base-960h` (already loaded by
   `voice_match`) to get per-frame token argmax + the time-per-frame.
2. Build a coarse phonetic transcript by collapsing the CTC blanks/dupes.
3. For each ASR-emitted word, find its character span in the CTC string
   (greedy Latin transliteration of Arabic letters first) and map it back
   to time using the frame index.

This is a poor-man's forced aligner, NOT Montreal Forced Aligner. MFA
needs Kaldi + dictionaries; here we trade some precision for zero setup.
Expected error: +/- 100 ms per word, plenty for slicing the audio to
hand to an audio-aware LLM judge.

Public API
----------
align_words(audio_path, transcript) -> list[{
    "word": str,
    "start_s": float,
    "end_s": float,
    "confidence": float,
}]
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import voice_match


# ---------------------------------------------------------------------------
# Arabic -> Latin best-effort transliteration
#
# wav2vec2-base-960h has a Latin vocabulary; running it on Arabic audio
# still gives a sequence that loosely mimics the phonetics. To match ASR
# words back, we transliterate the Arabic transcript to the same Latin
# space and compare via simple substring search.
# ---------------------------------------------------------------------------

_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ٱ": "a",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h",
    "و": "w", "ي": "y", "ى": "a", "ة": "h", "ء": "",
    "ؤ": "w", "ئ": "y",
}

_TASHKEEL_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")


def _translit(word: str) -> str:
    """Cheap Arabic -> Latin phonetic transliteration."""
    word = unicodedata.normalize("NFKC", word)
    word = _TASHKEEL_RE.sub("", word)
    out = []
    for ch in word:
        if ch in _AR2LAT:
            out.append(_AR2LAT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
        # else: drop punctuation/whitespace
    return "".join(out)


# ---------------------------------------------------------------------------
# Core: get a (token, frame_index) sequence from the CTC model
# ---------------------------------------------------------------------------


def _ctc_frame_tokens(wav) -> Tuple[List[str], float]:
    """Return (per-collapsed-token-char, seconds-per-output-frame).

    Each non-blank, non-duplicate token is one character (since the model
    has a character vocab). Frame index of that char tells us when it
    occurred in the audio.
    """
    processor, model, torch, device = voice_match._load_models()
    wav = voice_match._ensure_min_length(wav)
    with torch.inference_mode():
        inputs = processor(
            wav, sampling_rate=voice_match.SAMPLE_RATE, return_tensors="pt"
        ).to(device)
        logits = model(**inputs).logits.squeeze(0)  # (T, V)
        ids = logits.argmax(dim=-1).cpu().tolist()
    pad_id = model.config.pad_token_id

    # Audio length / number of CTC frames -> seconds per frame.
    seconds_per_frame = wav.shape[-1] / voice_match.SAMPLE_RATE / max(1, len(ids))

    chars: List[Tuple[str, int]] = []
    prev = None
    for frame_idx, token_id in enumerate(ids):
        if token_id == prev or token_id == pad_id:
            prev = token_id
            continue
        prev = token_id
        ch = processor.tokenizer.decode([token_id], skip_special_tokens=True)
        if ch:
            chars.append((ch.lower(), frame_idx))
    # Collapse word separator '|' / space-marker -> ' '
    cleaned: List[Tuple[str, int]] = []
    for ch, idx in chars:
        ch = ch.replace("|", " ")
        if ch:
            cleaned.append((ch, idx))
    return cleaned, seconds_per_frame


def _find_substring_in_frames(
    needle: str, haystack_chars: List[Tuple[str, int]]
) -> Optional[Tuple[int, int]]:
    """Locate `needle` (a transliterated word) inside the CTC char stream.

    Returns (start_frame, end_frame) in the haystack's frame numbering, or
    None if not found. Uses a relaxed substring search: skips spaces in
    the haystack while matching needle chars in order.
    """
    if not needle:
        return None
    n = len(needle)
    chars = [c for c, _ in haystack_chars]
    frames = [f for _, f in haystack_chars]
    flat = "".join(chars).replace(" ", "")
    # Map every char in `flat` back to its frame index.
    flat_to_frame: List[int] = []
    for ch, f in zip(chars, frames):
        for _ in ch.replace(" ", ""):
            flat_to_frame.append(f)
    pos = flat.find(needle)
    if pos < 0:
        # Try a fuzzier fallback: find by first 3 chars
        if len(needle) >= 3:
            pos = flat.find(needle[:3])
        if pos < 0:
            return None
    end_pos = min(pos + n, len(flat_to_frame) - 1)
    return flat_to_frame[pos], flat_to_frame[end_pos]


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def align_words(
    audio_path: str | Path,
    transcript: str,
    *,
    pad_s: float = 0.10,
) -> List[Dict[str, Any]]:
    """Align each word in `transcript` to a (start_s, end_s) in the audio.

    Returns one record per word, in transcript order. If a word cannot
    be located, returns it with start_s=end_s=None and confidence=0.
    """
    if not transcript or not transcript.strip():
        return []

    wav = voice_match.load_audio(audio_path)
    frame_chars, sec_per_frame = _ctc_frame_tokens(wav)
    if not frame_chars:
        return []

    total_dur = wav.shape[-1] / voice_match.SAMPLE_RATE

    # Walk through the haystack only once. We keep a running cursor so a
    # word later in the transcript can only match later in the audio.
    cursor_pos = 0  # in the flat (no-space) haystack
    flat = "".join(c for c, _ in frame_chars).replace(" ", "")
    flat_to_frame: List[int] = []
    for ch, f in frame_chars:
        for _ in ch.replace(" ", ""):
            flat_to_frame.append(f)
    if not flat:
        return []

    out: List[Dict[str, Any]] = []
    raw_words = [w for w in re.split(r"\s+", transcript.strip()) if w]
    for word in raw_words:
        needle = _translit(word)
        match_start = match_end = None
        confidence = 0.0
        if needle:
            local = flat[cursor_pos:].find(needle)
            if local >= 0:
                start_pos = cursor_pos + local
                end_pos = start_pos + len(needle)
                confidence = 1.0
            elif len(needle) >= 3:
                local = flat[cursor_pos:].find(needle[:3])
                if local >= 0:
                    start_pos = cursor_pos + local
                    end_pos = start_pos + min(len(needle), 6)
                    confidence = 0.5
                else:
                    start_pos = end_pos = None
            else:
                start_pos = end_pos = None

            if start_pos is not None and end_pos is not None:
                end_pos = min(end_pos, len(flat_to_frame) - 1)
                f0 = flat_to_frame[start_pos]
                f1 = flat_to_frame[end_pos]
                match_start = max(0.0, f0 * sec_per_frame - pad_s)
                match_end = min(total_dur, f1 * sec_per_frame + pad_s)
                cursor_pos = end_pos

        out.append({
            "word": word,
            "start_s": match_start,
            "end_s": match_end,
            "confidence": confidence,
        })
    return out


def align_span(
    audio_path: str | Path, transcript: str, span_text: str
) -> Optional[Dict[str, Any]]:
    """Convenience: align a whole multi-word span (e.g. a flagged
    suspicious phrase) and return one (start_s, end_s) covering it all."""
    aligned = align_words(audio_path, transcript)
    if not aligned:
        return None
    span_words = [w for w in re.split(r"\s+", span_text.strip()) if w]
    if not span_words:
        return None
    # Find the contiguous run in `aligned` that matches `span_words`.
    n = len(span_words)
    for i in range(0, len(aligned) - n + 1):
        if [aligned[i + k]["word"] for k in range(n)] == span_words:
            usable = [aligned[i + k] for k in range(n)
                      if aligned[i + k]["start_s"] is not None]
            if not usable:
                return None
            return {
                "span": span_text,
                "start_s": min(a["start_s"] for a in usable),
                "end_s": max(a["end_s"] for a in usable),
                "confidence": sum(a["confidence"] for a in usable) / len(usable),
            }
    return None
