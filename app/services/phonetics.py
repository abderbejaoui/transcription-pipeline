"""Text-to-IPA helpers for the medical correction pipeline.

This module attempts to make the `espeak` / `espeak-ng` binary
discoverable on Windows (common PATH problems) before calling
`phonemizer.phonemize`. If phonemizer is not available or fails,
a small grapheme-to-phoneme fallback is used.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
from difflib import SequenceMatcher


_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def _fallback_word_to_ipa(word: str) -> str:
    word = word.lower()
    word = word.replace("ph", "f")
    word = word.replace("ch", "tʃ")
    word = word.replace("sh", "ʃ")
    word = word.replace("th", "θ")
    word = word.replace("ng", "ŋ")
    word = word.replace("qu", "kw")
    word = word.replace("x", "ks")
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _ensure_espeak_on_path() -> str | None:
    """Ensure espeak or espeak-ng is discoverable on PATH.

    Returns the path to the executable if found, otherwise None.
    On Windows the function probes common install locations and prepends
    the executable directory to PATH when found.
    """
    # Check PATH first
    for exe in ("espeak-ng", "espeak"):
        found = shutil.which(exe)
        if found:
            return found

    # If on Windows, try common install locations and add to PATH if found
    if platform.system().lower().startswith("win"):
        candidates = [
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files (x86)\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files\eSpeak NG\espeak-ng.exe",
            r"C:\Program Files\espeak-ng\espeak-ng.exe",
            r"C:\Program Files\espeak\espeak.exe",
        ]
        for path in candidates:
            if os.path.exists(path):
                dirpath = os.path.dirname(path)
                os.environ["PATH"] = dirpath + os.pathsep + os.environ.get("PATH", "")
                return path
    return None


def text_to_ipa(text: str, language: str = "en-us") -> str:
    """Convert `text` to IPA using phonemizer (espeak backend) or fallback.

    The function is conservative: it will try phonemizer but will not raise
    if phonemizer is missing — instead it returns a heuristic phonetic
    approximation.
    """
    normalized = " ".join(_WORD_RE.findall(text.lower()))
    if not normalized:
        return ""

    # Ensure espeak is discoverable (helpful for Windows installations)
    _ensure_espeak_on_path()

    try:
        from phonemizer import phonemize

        ipa = phonemize(
            normalized,
            language=language,
            backend="espeak",
            strip=True,
            preserve_punctuation=False,
            with_stress=False,
            njobs=1,
        )
        ipa = ipa.strip()
        if ipa:
            return ipa
    except Exception:
        # Best-effort: try once more, then fall back
        try:
            from phonemizer import phonemize

            ipa = phonemize(
                normalized,
                language=language,
                backend="espeak",
                strip=True,
                preserve_punctuation=False,
                with_stress=False,
                njobs=1,
            )
            ipa = ipa.strip()
            if ipa:
                return ipa
        except Exception:
            pass

    # Last-resort heuristic fallback
    return " ".join(_fallback_word_to_ipa(word) for word in normalized.split())


def fallback_text_to_ipa(text: str) -> str:
    """Lightweight, safe grapheme-to-phoneme fallback that NEVER imports
    or calls the `phonemizer` package or system espeak binaries. Use this
    when you must avoid any blocking system calls (e.g., offline tests).
    """
    normalized = " ".join(_WORD_RE.findall(text.lower()))
    if not normalized:
        return ""
    return " ".join(_fallback_word_to_ipa(word) for word in normalized.split())


def ipa_edit_distance(a: str, b: str) -> float:
    left = str(a or "").strip().strip("/")
    right = str(b or "").strip().strip("/")
    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    matcher = SequenceMatcher(None, left, right)
    return 1.0 - matcher.ratio()