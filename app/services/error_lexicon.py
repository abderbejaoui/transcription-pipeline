"""JSON-backed error lexicon for doctor-confirmed corrections.

This lexicon stores observed ASR mistakes (wrong_text) and their confirmed
corrections (correct_text). Lookups follow a three-level cascade:
  1) exact string match
  2) Double Metaphone key match
  3) fuzzy similarity (RapidFuzz)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import jellyfish
from filelock import FileLock
from rapidfuzz import fuzz


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEXICON_PATH = PROJECT_ROOT / "lexicon" / "corrections.json"

DEFAULT_FUZZY_THRESHOLD = float(os.environ.get("ERROR_LEXICON_FUZZY", "90"))
DEFAULT_PHONETIC_THRESHOLD = float(os.environ.get("ERROR_LEXICON_PHONETIC", "85"))


@dataclass(frozen=True)
class LexiconMatch:
    wrong_text: str
    correct_text: str
    match_type: str
    similarity: float
    entry_id: str


@dataclass(frozen=True)
class StoredEntry:
    wrong_norm: str
    wrong_text: str
    correct_text: str
    dm_primary: str
    dm_secondary: str
    source: str
    created_at: float
    updated_at: float
    hit_count: int


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _double_metaphone(text: str) -> Tuple[str, str]:
    cleaned = _normalize(text).replace(" ", "")
    if not cleaned:
        return "", ""
    if hasattr(jellyfish, "double_metaphone"):
        primary, secondary = jellyfish.double_metaphone(cleaned)
        return primary or "", secondary or ""
    code = jellyfish.metaphone(cleaned)
    return code or "", ""


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _read(path: Path = DEFAULT_LEXICON_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    lock = FileLock(str(_lock_path(path)))
    with lock:
        try:
            with path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}
    return data if isinstance(data, dict) else {}


def _write(path: Path, data: Dict[str, Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_lock_path(path)))
    with lock:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)


def _iter_entries(data: Dict[str, Dict[str, Any]]) -> Iterable[StoredEntry]:
    for wrong_norm, row in data.items():
        if not isinstance(row, dict):
            continue
        wrong_text = str(row.get("wrong_text") or wrong_norm).strip() or wrong_norm
        correct_text = str(row.get("correct_text") or "").strip()
        if not correct_text:
            continue
        dm_primary = str(row.get("dm_primary") or "").strip()
        dm_secondary = str(row.get("dm_secondary") or "").strip()
        if not dm_primary and not dm_secondary:
            dm_primary, dm_secondary = _double_metaphone(wrong_norm)
        source = str(row.get("source") or "doctor").strip() or "doctor"
        created_at = float(row.get("created_at") or 0.0)
        updated_at = float(row.get("updated_at") or created_at or 0.0)
        hit_count = int(row.get("hit_count") or 0)
        yield StoredEntry(
            wrong_norm=wrong_norm,
            wrong_text=wrong_text,
            correct_text=correct_text,
            dm_primary=dm_primary,
            dm_secondary=dm_secondary,
            source=source,
            created_at=created_at,
            updated_at=updated_at,
            hit_count=hit_count,
        )


def _best_fuzzy_match(
    entries: Iterable[StoredEntry],
    target_norm: str,
) -> Optional[Tuple[StoredEntry, float]]:
    best_entry: Optional[StoredEntry] = None
    best_score = 0.0
    for entry in entries:
        score = float(fuzz.ratio(target_norm, entry.wrong_norm))
        if score > best_score:
            best_score = score
            best_entry = entry
    if best_entry is None:
        return None
    return best_entry, best_score


def add_correction(
    wrong_text: str,
    correct_text: str,
    *,
    source: str = "doctor",
    path: Path = DEFAULT_LEXICON_PATH,
) -> Dict[str, Any]:
    wrong_text = wrong_text.strip()
    correct_text = correct_text.strip()
    if not wrong_text or not correct_text:
        raise ValueError("wrong_text and correct_text must be non-empty")

    wrong_norm = _normalize(wrong_text)
    correct_norm = _normalize(correct_text)
    dm_primary, dm_secondary = _double_metaphone(wrong_norm)
    now = time.time()

    data = _read(path)
    existing = data.get(wrong_norm) if isinstance(data, dict) else None
    created_at = float(existing.get("created_at") or now) if isinstance(existing, dict) else now
    hit_count = int(existing.get("hit_count") or 0) if isinstance(existing, dict) else 0

    data[wrong_norm] = {
        "wrong_text": wrong_text,
        "wrong_norm": wrong_norm,
        "correct_text": correct_text,
        "correct_norm": correct_norm,
        "dm_primary": dm_primary,
        "dm_secondary": dm_secondary,
        "source": source,
        "created_at": created_at,
        "updated_at": now,
        "hit_count": hit_count,
    }
    _write(path, data)

    return {
        "id": wrong_norm,
        "wrong_text": wrong_text,
        "correct_text": correct_text,
        "source": source,
    }


def lookup(
    text: str,
    *,
    path: Path = DEFAULT_LEXICON_PATH,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
    phonetic_threshold: float = DEFAULT_PHONETIC_THRESHOLD,
) -> Optional[LexiconMatch]:
    target_norm = _normalize(text)
    if not target_norm:
        return None

    data = _read(path)
    row = data.get(target_norm)
    if isinstance(row, dict):
        row["hit_count"] = int(row.get("hit_count") or 0) + 1
        row["updated_at"] = time.time()
        data[target_norm] = row
        _write(path, data)
        return LexiconMatch(
            wrong_text=str(row.get("wrong_text") or text).strip() or text,
            correct_text=str(row.get("correct_text") or "").strip(),
            match_type="exact",
            similarity=100.0,
            entry_id=target_norm,
        )

    entries = list(_iter_entries(data))

    dm_primary, dm_secondary = _double_metaphone(target_norm)
    if dm_primary or dm_secondary:
        phonetic_entries = [
            e
            for e in entries
            if (dm_primary and (dm_primary == e.dm_primary or dm_primary == e.dm_secondary))
            or (dm_secondary and (dm_secondary == e.dm_primary or dm_secondary == e.dm_secondary))
        ]
        best = _best_fuzzy_match(phonetic_entries, target_norm)
        if best is not None:
            best_entry, score = best
            if score >= phonetic_threshold:
                data[best_entry.wrong_norm]["hit_count"] = best_entry.hit_count + 1
                data[best_entry.wrong_norm]["updated_at"] = time.time()
                _write(path, data)
                return LexiconMatch(
                    wrong_text=best_entry.wrong_text,
                    correct_text=best_entry.correct_text,
                    match_type="phonetic",
                    similarity=score,
                    entry_id=best_entry.wrong_norm,
                )

    best = _best_fuzzy_match(entries, target_norm)
    if best is None:
        return None
    best_entry, score = best
    if score < fuzzy_threshold:
        return None

    data[best_entry.wrong_norm]["hit_count"] = best_entry.hit_count + 1
    data[best_entry.wrong_norm]["updated_at"] = time.time()
    _write(path, data)
    return LexiconMatch(
        wrong_text=best_entry.wrong_text,
        correct_text=best_entry.correct_text,
        match_type="fuzzy",
        similarity=score,
        entry_id=best_entry.wrong_norm,
    )
