"""MedSpeak KG retrieval (semantic + phonetic) for medical correction."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_SQLITE = Path(
    os.environ.get(
        "MEDSPEAK_KG_SQLITE",
        str(PROJECT_ROOT / "vendor" / "medspeakian" / "artifacts" / "kg_semantic.sqlite"),
    )
)
DEFAULT_KG_PHONETIC = Path(
    os.environ.get(
        "MEDSPEAK_KG_PHONETIC",
        str(PROJECT_ROOT / "vendor" / "medspeakian" / "artifacts" / "kg_phonetic.jsonl"),
    )
)
DEFAULT_QUERY_LIMIT = int(os.environ.get("MEDSPEAK_QUERY_LIMIT", "80"))


@dataclass(frozen=True)
class MedSpeakMatch:
    term: str
    score: float
    phonetic_score: float
    semantic_score: float
    source: str
    variant: Optional[str] = None


_phonetic_cache: List[Dict[str, str]] = []
_phonetic_mtime: Optional[float] = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _load_phonetic(path: Path = DEFAULT_KG_PHONETIC) -> List[Dict[str, str]]:
    global _phonetic_cache, _phonetic_mtime
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        return []
    if _phonetic_mtime == mtime:
        return _phonetic_cache

    items: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                term = str(row.get("term") or "").strip()
                similar = str(row.get("similar") or "").strip()
                if term and similar:
                    items.append({"term": term, "similar": similar})
    except OSError:
        return []

    _phonetic_cache = items
    _phonetic_mtime = mtime
    return items


def _phonetic_best(text_norm: str, *, path: Path) -> Tuple[Optional[str], float, Optional[str]]:
    items = _load_phonetic(path)
    best_term = None
    best_score = 0.0
    best_variant = None
    for item in items:
        term = item["term"]
        similar = item["similar"]
        score = float(fuzz.ratio(text_norm, _normalize(similar)))
        if score > best_score:
            best_score = score
            best_term = term
            best_variant = similar
    return best_term, best_score, best_variant


def _semantic_best(text_norm: str, *, path: Path, limit: int) -> Tuple[Optional[str], float, Optional[str]]:
    if not path.exists():
        return None, 0.0, None
    term_best = None
    best_score = 0.0
    best_variant = None
    like = f"%{text_norm}%"
    try:
        conn = sqlite3.connect(str(path))
        try:
            rows = conn.execute(
                "SELECT term, related_term FROM kg WHERE term LIKE ? OR related_term LIKE ? LIMIT ?",
                (like, like, limit),
            ).fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        return None, 0.0, None

    for term, related in rows:
        term_str = str(term or "").strip()
        related_str = str(related or "").strip()
        if term_str:
            score = float(fuzz.ratio(text_norm, _normalize(term_str)))
            if score > best_score:
                best_score = score
                term_best = term_str
                best_variant = term_str
        if related_str:
            score = float(fuzz.ratio(text_norm, _normalize(related_str)))
            if score > best_score:
                best_score = score
                term_best = related_str
                best_variant = related_str
    return term_best, best_score, best_variant


def available(
    *,
    kg_sqlite: Path = DEFAULT_KG_SQLITE,
    kg_phonetic: Path = DEFAULT_KG_PHONETIC,
) -> bool:
    return kg_sqlite.exists() or kg_phonetic.exists()


def retrieve(
    text: str,
    *,
    kg_sqlite: Path = DEFAULT_KG_SQLITE,
    kg_phonetic: Path = DEFAULT_KG_PHONETIC,
    query_limit: int = DEFAULT_QUERY_LIMIT,
) -> Optional[Dict[str, Any]]:
    text_norm = _normalize(text)
    if not text_norm:
        return None

    if not kg_sqlite.exists() and not kg_phonetic.exists():
        return None

    p_term, p_score, p_variant = _phonetic_best(text_norm, path=kg_phonetic)
    s_term, s_score, s_variant = _semantic_best(text_norm, path=kg_sqlite, limit=query_limit)

    best_term = p_term if p_score >= s_score else s_term
    best_score = max(p_score, s_score)
    if not best_term:
        return None

    return {
        "term": best_term,
        "score": best_score / 100.0,
        "phonetic_score": p_score / 100.0,
        "semantic_score": s_score / 100.0,
        "variant": p_variant if p_score >= s_score else s_variant,
        "source": "medspeak",
    }
