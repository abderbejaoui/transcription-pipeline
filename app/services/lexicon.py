"""Read/write helpers for the vocabulary JSONL file.

The vocabulary database is a list of `{term, type, aliases, priority}`
entries. The corrector matches new text against any variant (term + aliases)
using runtime fuzzy + phonetic similarity, so we deliberately do NOT
auto-generate phonetic aliases here. Saving the raw phrase the user
corrected is enough — the corrector handles future variants on its own.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .correction import DEFAULT_LEXICON_PATH


def list_terms(path: Path = DEFAULT_LEXICON_PATH) -> List[Dict[str, Any]]:
    """Return all entries currently in the lexicon."""
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def add_term(
    term: str,
    type_: str = "term",
    aliases: Optional[List[str]] = None,
    priority: float = 1.0,
    path: Path = DEFAULT_LEXICON_PATH,
) -> Dict[str, Any]:
    """Append a new term, or merge new aliases into an existing one.

    If the term already exists (case-insensitive match on `term`), we add
    any new aliases the caller provided to the existing record. We never
    auto-generate aliases here.
    """
    term = term.strip()
    if not term:
        raise ValueError("term must be non-empty")

    explicit_aliases = [a.strip() for a in (aliases or []) if a and a.strip()]
    canonical_lower = term.lower()

    # Build set of all forms already known across the file (so we don't add
    # an alias that collides with another term).
    existing = list_terms(path)
    known: Set[str] = set()
    for entry in existing:
        known.add(entry.get("term", "").lower())
        for a in entry.get("aliases", []) or []:
            known.add(str(a).lower())

    # Term already in the file? Merge new aliases into it (rewrites the
    # file). We don't add the canonical's own form as an alias.
    for entry in existing:
        if entry.get("term", "").lower() == canonical_lower:
            current = list(entry.get("aliases", []) or [])
            current_lower = {a.lower() for a in current}
            for a in explicit_aliases:
                al = a.lower()
                if al == canonical_lower or al in current_lower:
                    continue
                if al in known and al not in current_lower:
                    # collides with another entry's variant; skip silently
                    continue
                current.append(a)
                current_lower.add(al)
            if current != entry.get("aliases", []):
                entry["aliases"] = current
                _rewrite(path, existing)
            return entry

    # New term. Filter aliases against collisions.
    final_aliases: List[str] = []
    seen: Set[str] = {canonical_lower}
    for a in explicit_aliases:
        al = a.lower()
        if al in seen or al in known:
            continue
        seen.add(al)
        final_aliases.append(a)

    new_entry: Dict[str, Any] = {
        "term": term,
        "type": type_,
        "aliases": final_aliases,
        "priority": float(priority),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(new_entry, ensure_ascii=False) + "\n")
    return new_entry


def _rewrite(path: Path, entries: List[Dict[str, Any]]) -> None:
    """Atomically rewrite the lexicon file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    tmp.replace(path)
