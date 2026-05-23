"""Fuzzy lookup against the medical entities list (KG)."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from filelock import FileLock
from rapidfuzz import fuzz, process


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_KG_PATH = Path(
    os.environ.get("KG_ENTITIES_PATH", str(PROJECT_ROOT / "data" / "medical_entities.json"))
)


@dataclass(frozen=True)
class KgVariant:
    term: str
    type: str
    variant: str
    norm: str


_variants: List[KgVariant] = []
_variant_norms: List[str] = []
_type_by_norm: Dict[str, str] = {}
_loaded_mtime: Optional[float] = None


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _lock_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".lock")


def _read_raw(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"drugs": [], "diagnoses": [], "procedures": []}
    lock = FileLock(str(_lock_path(path)))
    with lock:
        try:
            with path.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (json.JSONDecodeError, OSError):
            obj = {}
    if not isinstance(obj, dict):
        return {"drugs": [], "diagnoses": [], "procedures": []}
    obj.setdefault("drugs", [])
    obj.setdefault("diagnoses", [])
    obj.setdefault("procedures", [])
    return obj


def _write_raw(path: Path, data: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(_lock_path(path)))
    with lock:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.write("\n")
        tmp.replace(path)


def _load(path: Path = DEFAULT_KG_PATH) -> None:
    global _loaded_mtime, _variants, _variant_norms, _type_by_norm
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = None
    if _loaded_mtime is not None and mtime is not None and _loaded_mtime == mtime:
        return

    variants: List[KgVariant] = []
    type_by_norm: Dict[str, str] = {}

    data = _read_raw(path)

    for item in data.get("drugs", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        if not canonical:
            continue
        term_norm = _normalize(canonical)
        type_by_norm[term_norm] = "drug"
        variants.append(
            KgVariant(term=canonical, type="drug", variant=canonical, norm=term_norm)
        )
        for alias in item.get("aliases") or []:
            alias_str = str(alias).strip()
            if not alias_str:
                continue
            alias_norm = _normalize(alias_str)
            type_by_norm[alias_norm] = "drug"
            variants.append(
                KgVariant(
                    term=canonical,
                    type="drug",
                    variant=alias_str,
                    norm=alias_norm,
                )
            )

    for diag in data.get("diagnoses", []) if isinstance(data, dict) else []:
        term = str(diag).strip()
        if not term:
            continue
        norm = _normalize(term)
        type_by_norm[norm] = "diagnosis"
        variants.append(
            KgVariant(term=term, type="diagnosis", variant=term, norm=norm)
        )

    for proc in data.get("procedures", []) if isinstance(data, dict) else []:
        term = str(proc).strip()
        if not term:
            continue
        norm = _normalize(term)
        type_by_norm[norm] = "procedure"
        variants.append(
            KgVariant(term=term, type="procedure", variant=term, norm=norm)
        )

    _variants = variants
    _variant_norms = [v.norm for v in variants]
    _type_by_norm = type_by_norm
    _loaded_mtime = mtime


def find_best(text: str, *, path: Path = DEFAULT_KG_PATH) -> Optional[Dict[str, object]]:
    _load(path)
    if not _variants:
        return None
    norm = _normalize(text)
    if not norm:
        return None

    result = process.extractOne(norm, _variant_norms, scorer=fuzz.ratio)
    if not result:
        return None
    _, score, idx = result
    match = _variants[idx]
    return {
        "term": match.term,
        "type": match.type,
        "variant": match.variant,
        "score": float(score),
    }


def is_drug(term: str, *, path: Path = DEFAULT_KG_PATH) -> bool:
    _load(path)
    norm = _normalize(term)
    return _type_by_norm.get(norm) == "drug"


def add_entity(
    term: str,
    *,
    entity_type: str,
    alias: Optional[str] = None,
    path: Path = DEFAULT_KG_PATH,
) -> Dict[str, object]:
    entity_type = entity_type.strip().lower()
    if entity_type not in {"drug", "diagnosis", "procedure"}:
        return {"ok": False, "reason": "unsupported_type"}

    term = term.strip()
    alias = alias.strip() if alias else ""
    if not term:
        return {"ok": False, "reason": "empty_term"}

    data = _read_raw(path)

    if entity_type == "drug":
        drugs = data.get("drugs") or []
        if not isinstance(drugs, list):
            drugs = []
        canonical_lower = term.lower()
        entry = None
        for item in drugs:
            if isinstance(item, dict) and str(item.get("canonical") or "").lower() == canonical_lower:
                entry = item
                break
        if entry is None:
            entry = {"canonical": term, "aliases": []}
            drugs.append(entry)
        aliases = entry.get("aliases")
        if not isinstance(aliases, list):
            aliases = []
        if alias:
            alias_lower = alias.lower()
            if alias_lower not in {str(a).lower() for a in aliases}:
                aliases.append(alias)
        entry["aliases"] = aliases
        data["drugs"] = drugs
    elif entity_type == "diagnosis":
        diagnoses = data.get("diagnoses") or []
        if not isinstance(diagnoses, list):
            diagnoses = []
        if term.lower() not in {str(d).lower() for d in diagnoses}:
            diagnoses.append(term)
        data["diagnoses"] = diagnoses
    else:
        procedures = data.get("procedures") or []
        if not isinstance(procedures, list):
            procedures = []
        if term.lower() not in {str(p).lower() for p in procedures}:
            procedures.append(term)
        data["procedures"] = procedures

    _write_raw(path, data)
    _load(path)
    return {"ok": True, "term": term, "type": entity_type, "alias": alias}
