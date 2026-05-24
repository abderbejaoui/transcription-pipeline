"""Read/write helpers for the medical lexicon JSONL file.

This module supports both the legacy `{term, type, aliases, priority}`
shape and the new pipeline contract with `{term, canonical_form, term_type,
aliases, ipa, description, source, added_at}`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .correction import DEFAULT_LEXICON_PATH
from .phonetics import ipa_edit_distance, text_to_ipa


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    canonical_form: str
    term_type: str
    aliases: tuple[str, ...]
    ipa: str
    description: str
    source: str
    added_at: str

    @property
    def type(self) -> str:
        return self.term_type


def _canonical_form(term: str) -> str:
    return " ".join(term.strip().lower().split())


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _coerce_entry(row: Dict[str, Any]) -> LexiconEntry:
    term = str(row.get("term") or "").strip()
    canonical_form = str(row.get("canonical_form") or _canonical_form(term))
    term_type = str(row.get("term_type") or row.get("type") or "unknown")
    aliases = tuple(str(a).strip() for a in row.get("aliases", []) or [] if str(a).strip())
    ipa = str(row.get("ipa") or text_to_ipa(term))
    description = str(row.get("description") or "")
    source = str(row.get("source") or "seed")
    added_at = str(row.get("added_at") or _now_iso())
    return LexiconEntry(
        term=term,
        canonical_form=canonical_form,
        term_type=term_type,
        aliases=aliases,
        ipa=ipa,
        description=description,
        source=source,
        added_at=added_at,
    )


def _resolve_path(path: Optional[Path]) -> Path:
    return path or DEFAULT_LEXICON_PATH


def list_terms(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all raw entries currently in the lexicon."""
    return [entry.__dict__.copy() for entry in load_lexicon(path)]


def load_lexicon(path: Optional[Path] = None) -> List[LexiconEntry]:
    path = _resolve_path(path)
    entries: List[LexiconEntry] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(_coerce_entry(json.loads(line)))
            except Exception:
                continue
    return entries


def _iter_variants(entry: LexiconEntry) -> Iterable[str]:
    yield entry.term
    for alias in entry.aliases:
        yield alias


def find_by_alias(text: str, path: Optional[Path] = None) -> Optional[LexiconEntry]:
    """Find a lexicon entry by matching text against the canonical term OR any alias."""
    path = _resolve_path(path)
    needle = _canonical_form(text)
    if not needle:
        return None
    for entry in load_lexicon(path):
        if _canonical_form(entry.term) == needle:
            return entry
        for alias in entry.aliases:
            if _canonical_form(alias) == needle:
                return entry
    return None


def find_by_canonical(text: str, path: Optional[Path] = None) -> Optional[LexiconEntry]:
    """Find a lexicon entry by matching text against the canonical term only (NOT aliases).

    This is used by the scorer to determine if a word is a KNOWN correct term,
    rather than a misspelled variant stored as an alias.
    """
    path = _resolve_path(path)
    needle = _canonical_form(text)
    if not needle:
        return None
    for entry in load_lexicon(path):
        if _canonical_form(entry.term) == needle:
            return entry
    return None


def _ensure_unique_aliases(existing: Sequence[str], aliases: Sequence[str], canonical_lower: str) -> List[str]:
    out: List[str] = list(existing)
    seen = {_canonical_form(a) for a in out}
    for alias in aliases:
        alias_norm = _canonical_form(alias)
        if not alias_norm or alias_norm == canonical_lower or alias_norm in seen:
            continue
        out.append(alias)
        seen.add(alias_norm)
    return out


def _rewrite(path: Path, entries: List[LexiconEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry.__dict__, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _entry_to_raw(entry: LexiconEntry) -> Dict[str, Any]:
    return {
        "term": entry.term,
        "canonical_form": entry.canonical_form,
        "term_type": entry.term_type,
        "type": entry.term_type,
        "aliases": list(entry.aliases),
        "ipa": entry.ipa,
        "description": entry.description,
        "source": entry.source,
        "added_at": entry.added_at,
    }


def add_entry(entry: LexiconEntry, path: Optional[Path] = None) -> LexiconEntry:
    path = _resolve_path(path)
    entries = load_lexicon(path)
    canonical_lower = _canonical_form(entry.term)
    for i, existing in enumerate(entries):
        if _canonical_form(existing.term) == canonical_lower:
            merged_aliases = _ensure_unique_aliases(existing.aliases, entry.aliases, canonical_lower)
            entries[i] = LexiconEntry(
                term=existing.term,
                canonical_form=existing.canonical_form,
                term_type=existing.term_type or entry.term_type,
                aliases=tuple(merged_aliases),
                ipa=existing.ipa or entry.ipa,
                description=existing.description or entry.description,
                source=existing.source or entry.source,
                added_at=existing.added_at,
            )
            _rewrite(path, entries)
            return entries[i]

    entries.append(entry)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_entry_to_raw(entry), ensure_ascii=False) + "\n")
    return entry


def add_term(
    term: str,
    type_: str = "unknown",
    aliases: Optional[List[str]] = None,
    priority: float = 1.0,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    term = term.strip()
    if not term:
        raise ValueError("term must be non-empty")
    explicit_aliases = [a.strip() for a in (aliases or []) if a and a.strip()]
    entry = LexiconEntry(
        term=term,
        canonical_form=_canonical_form(term),
        term_type=type_,
        aliases=tuple(explicit_aliases),
        ipa=text_to_ipa(term),
        description="",
        source="seed",
        added_at=_now_iso(),
    )
    saved = add_entry(entry, path=path)
    raw = _entry_to_raw(saved)
    raw["priority"] = float(priority)
    return raw


def search_phonetic(ipa: str, top_k: int = 5, path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = _resolve_path(path)
    query = str(ipa or "").strip()
    if not query:
        return []
    query_norm = query.strip("/")
    ranked: List[Dict[str, Any]] = []
    for entry in load_lexicon(path):
        best_score = 0.0
        best_variant = entry.term
        # Use stored IPA when available to avoid runtime phonemizer calls.
        entry_ipa = (entry.ipa or "").strip("/")
        if entry_ipa:
            score = 1.0 - ipa_edit_distance(query_norm, entry_ipa)
            best_score = score
            best_variant = entry.term
        else:
            for variant in _iter_variants(entry):
                variant_ipa = text_to_ipa(variant).strip("/")
                score = 1.0 - ipa_edit_distance(query_norm, variant_ipa)
                if score > best_score:
                    best_score = score
                    best_variant = variant
        ranked.append(
            {
                "term": entry.term,
                "ipa": entry.ipa,
                "term_type": entry.term_type,
                "description": entry.description,
                "phonetic_score": round(max(0.0, min(1.0, best_score)), 6),
                "source": entry.source,
                "priority": 1.0,
                "matched_variant": best_variant,
            }
        )
    ranked.sort(key=lambda row: (-row["phonetic_score"], 0 if row["source"] == "user" else 1, row["term"].lower()))
    return ranked[:top_k]
