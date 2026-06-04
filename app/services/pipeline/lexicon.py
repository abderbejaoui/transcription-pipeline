"""pipeline/lexicon.py — Medical lexicon and HITL alias management.

Owns:
  - medical_terms.txt loading and caching
  - HITL alias map (data/hitl_aliases.json) load/write/apply
  - Cache invalidation so taught terms take effect without a restart
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEDICAL_TERMS_PATH = PROJECT_ROOT / "medical_terms.txt"
HITL_ALIASES_PATH = PROJECT_ROOT / "data" / "hitl_aliases.json"

_TASHKEEL_RE = re.compile(r"[ً-ْٰـ]")

_lex_cache: Optional[List[str]] = None
_alias_cache: Optional[Dict[str, str]] = None


def load_medical_lexicon() -> List[str]:
    """Return the candidate-retrieval lexicon (medical_terms.txt).

    HITL-taught terms are appended to this file by add_retrieval_term(),
    so anything a clinician teaches becomes retrievable on the next call.
    Result is cached; call invalidate_lexicon_cache() after teaching.
    """
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


def add_retrieval_term(term: str) -> bool:
    """Append a canonical term to the candidate-retrieval dataset.

    Returns True if the term was newly added. Only Latin terms are useful
    since the retrieval matcher folds Arabic needles against Latin skeletons.
    """
    term = (term or "").strip()
    if not term or not re.search(r"[A-Za-z]", term):
        return False
    existing = {t.lower() for t in load_medical_lexicon()}
    if term.lower() in existing:
        return False
    with MEDICAL_TERMS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(term + "\n")
    invalidate_lexicon_cache()
    return True


def _norm_alias(s: str) -> str:
    """Normalise an alias for exact HITL matching: NFKC, drop tashkeel and
    whitespace, lowercase."""
    s = unicodedata.normalize("NFKC", s)
    s = _TASHKEEL_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _load_taught_alias_map() -> Dict[str, str]:
    """Map normalised clinician-taught aliases → canonical term.

    Sourced only from data/hitl_aliases.json, written by /api/teach.
    Kept separate from the seed lexicon so only human-confirmed mappings
    are auto-applied.
    """
    global _alias_cache
    if _alias_cache is not None:
        return _alias_cache
    amap: Dict[str, str] = {}
    if HITL_ALIASES_PATH.exists():
        try:
            raw = json.loads(HITL_ALIASES_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, term in raw.items():
                    if isinstance(key, str) and isinstance(term, str) and len(key) >= 4:
                        amap[key] = term
        except Exception:
            pass
    _alias_cache = amap
    return amap


def record_taught_aliases(term: str, aliases: List[str]) -> int:
    """Persist clinician-confirmed alias → term mappings.

    Returns the number of new mappings written.
    """
    term = (term or "").strip()
    if not term:
        return 0
    amap: Dict[str, str] = {}
    if HITL_ALIASES_PATH.exists():
        try:
            loaded = json.loads(HITL_ALIASES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                amap = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            amap = {}
    added = 0
    for a in aliases or []:
        key = _norm_alias(str(a))
        if len(key) >= 4 and key not in amap:
            amap[key] = term
            added += 1
    if added:
        HITL_ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        HITL_ALIASES_PATH.write_text(
            json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        invalidate_lexicon_cache()
    return added


def apply_taught_aliases(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Replace clinician-taught alias occurrences with their canonical terms.

    Deterministic exact-match pass (1-3 token windows) that runs before
    phonetic flagging. Returns (new_text, [{"from": ..., "to": ...}, ...]).
    """
    amap = _load_taught_alias_map()
    if not amap:
        return text, []
    tokens = re.split(r"(\s+)", text)
    word_pos = [i for i, t in enumerate(tokens) if t.strip()]
    replacements: List[Dict[str, str]] = []
    n = len(word_pos)
    i = 0
    while i < n:
        matched = False
        for size in (3, 2, 1):
            if i + size > n:
                continue
            positions = word_pos[i:i + size]
            key = _norm_alias("".join(tokens[p] for p in positions))
            if key in amap:
                canonical = amap[key]
                original = " ".join(tokens[p].strip() for p in positions)
                tokens[positions[0]] = canonical
                for p in positions[1:]:
                    tokens[p] = ""
                    if p - 1 >= 0:
                        tokens[p - 1] = ""
                replacements.append({"from": original, "to": canonical})
                i += size
                matched = True
                break
        if not matched:
            i += 1
    out = re.sub(r"\s+", " ", "".join(tokens)).strip()
    return out, replacements


def invalidate_lexicon_cache() -> None:
    """Drop cached lexicon + alias map so newly-taught terms take effect
    immediately without a server restart."""
    global _lex_cache, _alias_cache
    _lex_cache = None
    _alias_cache = None
