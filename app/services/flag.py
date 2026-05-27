"""Suspicious-word flagger for plain-text ASR transcripts.

Combines two passes:
  1. Phonetic pass — for every transcript word, transliterate to Latin and
     compare against `medical_terms.txt` via normalized edit distance.
     Any word within `phonetic_threshold` of a known term is flagged.
  2. LLM pass — ask the chat model to flag any other words that LOOK or
     SOUND medical but didn't pass the dictionary check (rare disease
     names, brand names not in the file).

The two outputs are merged and deduplicated. Each flag includes the
word, the index in the whitespace-tokenized transcript, and the
candidate medical terms phonetically closest to it.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDICAL_TERMS_PATH = PROJECT_ROOT / "medical_terms.txt"


# ---------------------------------------------------------------------------
# Cheap Arabic -> Latin transliteration (for phonetic comparison only).
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


def _strip_arabic_clitics(word: str) -> str:
    """Drop common attached morphemes before phonetic matching.

    Arabic glues 'al-' (the), 'wa-' (and), 'bi-' (with), 'li-' (to) and
    'fa-' (so) onto the next word. Without stripping these, words like
    'البرسيتامول' (= 'the paracetamol') score badly against
    'paracetamol' because of the extra 'al' prefix.

    We're conservative: we only strip when the remainder is at least
    4 characters, so we don't decapitate short words.
    """
    PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل",
                "و", "ف", "ب", "ل", "ك", "س")
    for pre in PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 4:
            return word[len(pre):]
    return word


def _translit(word: str) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", word)
    s = _TASHKEEL_RE.sub("", s)
    s = _strip_arabic_clitics(s)
    out: List[str] = []
    for ch in s:
        if ch in _AR2LAT:
            out.append(_AR2LAT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# Medical lexicon
# ---------------------------------------------------------------------------

_lex_cache: Optional[List[str]] = None


def load_medical_lexicon() -> List[str]:
    global _lex_cache
    if _lex_cache is not None:
        return _lex_cache
    if not MEDICAL_TERMS_PATH.exists():
        _lex_cache = []
        return _lex_cache
    terms: List[str] = []
    for line in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    _lex_cache = terms
    return _lex_cache


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
# Phonetic pass
# ---------------------------------------------------------------------------


def _phonetic_candidates(
    word: str, lexicon: List[str], k: int = 3,
    *, threshold: float = 0.45,
) -> List[Dict[str, Any]]:
    """Find up to `k` lexicon entries whose Latin form is phonetically
    similar to `word` (after Arabic translit + clitic stripping).

    Threshold 0.45 is intentionally loose: Arabic translit drops vowels
    ('paracetamol' -> 'brsytamwl', similarity ~0.50), and we'd rather
    have an over-flag the LLM can dismiss than miss a real drug.
    """
    needle = _translit(word)
    if len(needle) < 2:
        return []
    scored = []
    for term in lexicon:
        term_lat = re.sub(r"[^a-z]", "", term.lower())
        if not term_lat:
            continue
        sim = _lev_sim(needle, term_lat)
        if sim >= threshold:
            scored.append({"term": term, "phonetic_similarity": round(sim, 3)})
    scored.sort(key=lambda d: -d["phonetic_similarity"])
    return scored[:k]


def phonetic_pass(transcript: str) -> List[Dict[str, Any]]:
    """For each word in `transcript`, return flag records with phonetic
    candidates from the medical lexicon."""
    lexicon = load_medical_lexicon()
    if not lexicon:
        return []
    flags: List[Dict[str, Any]] = []
    for i, word in enumerate(re.split(r"\s+", transcript.strip())):
        if not word:
            continue
        candidates = _phonetic_candidates(word, lexicon)
        if not candidates:
            continue
        # Strong matches (>=0.85) are very likely already correct; only
        # flag medium matches (0.55-0.85) that look mangled.
        top = candidates[0]
        if top["phonetic_similarity"] >= 0.90:
            continue  # already spelled close to the canonical form
        flags.append({
            "index": i,
            "word": word,
            "reason": "phonetic_near_medical",
            "candidates": candidates,
        })
    return flags


# ---------------------------------------------------------------------------
# LLM pass
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You audit ASR transcripts of Gulf Arabic doctor-patient consultations "
    "with code-switched English. Your job: flag every word that LOOKS or "
    "SOUNDS like a mishearing of a medical / pharmaceutical / brand / "
    "anatomical term. Be biased toward flagging — better to over-flag a "
    "weird word than miss a real drug.\n\n"
    "Strict rules:\n"
    "1. Output STRICT JSON only, no prose.\n"
    "2. Word indices are zero-based, computed by splitting the transcript "
    "on whitespace.\n"
    "3. Each flag entry: {\"index\": <int>, \"word\": <str>, "
    "\"reason\": <short string>, "
    "\"likely_term\": <best guess at the intended medical term, "
    "in correct Latin spelling for drug names / English for procedures / "
    "or empty string if you cannot identify>, "
    "\"confidence\": <0.0 to 1.0 — how certain you are about likely_term>}.\n"
    "4. Schema: {\"flags\": [<flag entry>, ...]}.\n"
    "5. Do NOT flag plain Arabic words that aren't medical (e.g. 'لمدة', "
    "'كل', 'اليوم'), normal English filler ('okay'), or numbers.\n"
    "6. Use confidence >= 0.90 ONLY when the audio context (drug + dose + "
    "frequency / indication) makes the term unambiguous. Use 0.5-0.85 for "
    "plausible guesses. Use 0.0 when unsure."
)


def llm_pass(transcript: str, timeout: float = 60.0) -> List[Dict[str, Any]]:
    user = json.dumps(
        {"transcript": transcript,
         "tokens": list(enumerate(re.split(r"\s+", transcript.strip())))},
        ensure_ascii=False,
    )
    payload = {
        "model": get_llm_model(get_llm_provider()),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    try:
        req = urllib.request.Request(
            get_llm_url(get_llm_provider()),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(get_llm_provider()),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = parse_chat_content(data, get_llm_provider()).strip()
        if not (text.startswith("{") and text.endswith("}")):
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                text = m.group(0)
        obj = json.loads(text)
        return list(obj.get("flags", []))
    except Exception as exc:
        print(f"[flag] LLM pass failed: {exc!r}")
        return []


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def flag_suspicious(
    transcript: str, use_llm: bool = True
) -> List[Dict[str, Any]]:
    """Return one entry per suspicious word, with phonetic candidates and
    optionally an LLM 'likely_term' attached."""
    phon = phonetic_pass(transcript)
    phon_by_idx = {f["index"]: f for f in phon}

    if use_llm:
        for entry in llm_pass(transcript):
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            llm_conf = float(entry.get("confidence", 0.0) or 0.0)
            likely = entry.get("likely_term") or ""
            existing = phon_by_idx.get(idx)
            if existing:
                existing["llm_reason"] = entry.get("reason") or existing["reason"]
                if likely:
                    existing["llm_likely_term"] = likely
                existing["llm_confidence"] = llm_conf
            else:
                word = entry.get("word") or ""
                phon_by_idx[idx] = {
                    "index": idx,
                    "word": word,
                    "reason": entry.get("reason") or "llm_flag",
                    "candidates": _phonetic_candidates(word, load_medical_lexicon()),
                    "llm_reason": entry.get("reason"),
                    "llm_likely_term": likely,
                    "llm_confidence": llm_conf,
                }
    return sorted(phon_by_idx.values(), key=lambda f: f["index"])


# ---------------------------------------------------------------------------
# Auto-correction: build a corrected transcript using HIGH-CONFIDENCE LLM
# suggestions only. The dashboard surfaces this as a separate string so the
# user can compare it to the raw transcript without losing the original.
# ---------------------------------------------------------------------------

def apply_high_confidence_corrections(
    transcript: str,
    flags: List[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.90,
) -> Dict[str, Any]:
    """Rewrite the transcript using only LLM suggestions where
    `llm_confidence >= confidence_threshold` AND `llm_likely_term` is set.
    """
    tokens = re.split(r"(\s+)", transcript)  # keep whitespace tokens
    # Build a mapping word-index -> token-index in the split (only non-space
    # tokens count as words).
    word_to_tok: List[int] = []
    for ti, t in enumerate(tokens):
        if t.strip():
            word_to_tok.append(ti)

    applied: List[Dict[str, Any]] = []
    for f in flags:
        idx = f.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(word_to_tok):
            continue
        conf = float(f.get("llm_confidence", 0.0) or 0.0)
        likely = f.get("llm_likely_term") or ""
        if conf < confidence_threshold or not likely:
            continue
        ti = word_to_tok[idx]
        original = tokens[ti]
        tokens[ti] = likely
        applied.append({
            "index": idx,
            "original": original,
            "corrected": likely,
            "confidence": conf,
        })
    return {
        "corrected_transcript": "".join(tokens),
        "applied": applied,
        "threshold": confidence_threshold,
    }
