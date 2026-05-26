"""Generate minimal MedSpeak KG artifacts for local testing.

Creates:
  vendor/medspeakian/artifacts/kg_semantic.sqlite
  vendor/medspeakian/artifacts/kg_phonetic.jsonl

The generator prefers data/medical_entities.json if populated, and falls
back to a small ASCII-only sample list for smoke tests.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

try:
    from rapidfuzz import fuzz
except Exception:  # pragma: no cover - fallback when rapidfuzz is missing
    import difflib

    class _Fuzz:
        @staticmethod
        def ratio(a: str, b: str) -> float:
            return difflib.SequenceMatcher(None, a, b).ratio() * 100.0

    fuzz = _Fuzz()


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTITIES = PROJECT_ROOT / "data" / "medical_entities.json"
DEFAULT_OUT_SQLITE = PROJECT_ROOT / "vendor" / "medspeakian" / "artifacts" / "kg_semantic.sqlite"
DEFAULT_OUT_PHON = PROJECT_ROOT / "vendor" / "medspeakian" / "artifacts" / "kg_phonetic.jsonl"

SCHEMA = """
CREATE TABLE IF NOT EXISTS kg (
    term TEXT,
    term_cui TEXT,
    rel TEXT,
    rel_detail TEXT,
    related_term TEXT,
    related_cui TEXT
);
"""

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_term ON kg(term)",
    "CREATE INDEX IF NOT EXISTS idx_related_term ON kg(related_term)",
    "CREATE INDEX IF NOT EXISTS idx_rel ON kg(rel)",
]


SAMPLE_DRUGS: List[Tuple[str, List[str]]] = [
    ("acetaminophen", ["paracetamol", "acetaminophen"]),
    ("ibuprofen", ["ibu profen", "ibuprofin"]),
    ("amoxicillin", ["amoxycillin", "amoxil"]),
    ("azithromycin", ["azithromicin", "azithro"]),
    ("ceftriaxone", ["ceftriaxon", "ceftriaxone"]),
    ("ciprofloxacin", ["ciprofloxin", "cipro floxacin"]),
    ("metformin", ["met formin", "metforin"]),
    ("lisinopril", ["lisinopril", "lisino pril"]),
    ("atorvastatin", ["atorvastatin", "atorva statin"]),
    ("omeprazole", ["omeprazol", "omepra zole"]),
    ("albuterol", ["albuterol", "albu terol"]),
    ("prednisone", ["prednison", "predni zone"]),
]
SAMPLE_DIAGNOSES = [
    "hypertension",
    "diabetes mellitus",
    "asthma",
    "pneumonia",
    "myocardial infarction",
    "heart failure",
    "acute kidney injury",
    "stroke",
]
SAMPLE_PROCEDURES = ["cbc", "xray", "ct scan", "mri", "ecg", "ekg"]


def _normalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def _load_entities(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"drugs": [], "diagnoses": [], "procedures": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"drugs": [], "diagnoses": [], "procedures": []}
    if not isinstance(data, dict):
        return {"drugs": [], "diagnoses": [], "procedures": []}
    data.setdefault("drugs", [])
    data.setdefault("diagnoses", [])
    data.setdefault("procedures", [])
    return data


def _extract_entities(data: Dict[str, object]) -> Tuple[List[Tuple[str, List[str]]], List[str], List[str]]:
    drugs: List[Tuple[str, List[str]]] = []
    diagnoses: List[str] = []
    procedures: List[str] = []

    for item in data.get("drugs", []) if isinstance(data, dict) else []:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical") or "").strip()
        if not canonical:
            continue
        aliases = [str(a).strip() for a in (item.get("aliases") or []) if str(a).strip()]
        drugs.append((canonical, aliases))

    for term in data.get("diagnoses", []) if isinstance(data, dict) else []:
        t = str(term).strip()
        if t:
            diagnoses.append(t)

    for term in data.get("procedures", []) if isinstance(data, dict) else []:
        t = str(term).strip()
        if t:
            procedures.append(t)

    return drugs, diagnoses, procedures


def _ensure_samples(
    drugs: List[Tuple[str, List[str]]],
    diagnoses: List[str],
    procedures: List[str],
) -> Tuple[List[Tuple[str, List[str]]], List[str], List[str]]:
    if drugs or diagnoses or procedures:
        return drugs, diagnoses, procedures
    return SAMPLE_DRUGS, SAMPLE_DIAGNOSES, SAMPLE_PROCEDURES


def _basic_variants(term: str) -> List[str]:
    t = _normalize(term)
    if not t:
        return []

    variants = set()
    subs = [
        ("ph", "f"),
        ("f", "ph"),
        ("ae", "e"),
        ("oe", "e"),
        ("ie", "ei"),
        ("ei", "ie"),
        ("y", "i"),
        ("i", "y"),
        ("c", "k"),
        ("k", "c"),
        ("x", "ks"),
        ("z", "s"),
        ("s", "z"),
        ("tion", "shun"),
        ("sion", "shun"),
        ("ck", "k"),
        ("qu", "k"),
    ]
    for a, b in subs:
        if a in t and a != b:
            variants.add(t.replace(a, b))

    if "-" in t:
        variants.add(t.replace("-", ""))
        variants.add(t.replace("-", " "))
    if " " in t:
        variants.add(t.replace(" ", ""))

    if len(t) > 5:
        for i in range(3, min(8, len(t) - 1)):
            variants.add(t[:i] + " " + t[i:])

    vowels = "aeiou"
    for i, ch in enumerate(t):
        if ch in vowels and 2 < i < len(t) - 2:
            variants.add(t[:i] + t[i + 1 :])
            break

    for i in range(2, len(t) - 2):
        if t[i] == t[i + 1]:
            variants.add(t[:i] + t[i + 1 :])
            break

    for i in range(2, len(t) - 2):
        if t[i].isalpha() and t[i] not in vowels:
            variants.add(t[:i] + t[i] + t[i:])
            break

    if len(t) > 4:
        i = 2
        variants.add(t[:i] + t[i + 1] + t[i] + t[i + 2 :])

    variants.discard(t)
    return [v for v in variants if v.strip()]


def _build_phonetic_pairs(
    terms: Sequence[str],
    aliases: Sequence[Tuple[str, str]],
    target_count: int,
    *,
    min_similarity: int,
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    seen = set()

    for canon, alias in aliases:
        canon_norm = _normalize(canon)
        alias_norm = _normalize(alias)
        if not canon_norm or not alias_norm or canon_norm == alias_norm:
            continue
        key = (canon_norm, alias_norm)
        if key in seen:
            continue
        seen.add(key)
        pairs.append(key)

    if target_count <= 0:
        return pairs

    if len(pairs) >= target_count:
        return pairs[:target_count]

    for term in terms:
        canon_norm = _normalize(term)
        variants = _basic_variants(canon_norm)
        for variant in variants:
            if fuzz.ratio(canon_norm, variant) < min_similarity:
                continue
            key = (canon_norm, variant)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
            if len(pairs) >= target_count:
                return pairs

    return pairs


def _write_phonetic(
    out_path: Path,
    drugs: List[Tuple[str, List[str]]],
    diagnoses: List[str],
    procedures: List[str],
    *,
    target_count: int,
    min_similarity: int,
) -> int:
    terms: List[str] = []
    aliases: List[Tuple[str, str]] = []
    for canonical, alias_list in drugs:
        terms.append(canonical)
        for alias in alias_list:
            aliases.append((canonical, alias))
    terms.extend(diagnoses)
    terms.extend(procedures)
    pairs = _build_phonetic_pairs(terms, aliases, target_count, min_similarity=min_similarity)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for canon_norm, alias_norm in pairs:
            rec = {"term": canon_norm, "similar": alias_norm, "cui": ""}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return len(pairs)


def _insert_rows(conn: sqlite3.Connection, rows: List[Tuple[str, str, str, str, str, str]]) -> None:
    if not rows:
        return
    conn.executemany(
        "INSERT INTO kg(term, term_cui, rel, rel_detail, related_term, related_cui) VALUES (?,?,?,?,?,?)",
        rows,
    )


def _write_semantic(out_path: Path, drugs: List[Tuple[str, List[str]]], diagnoses: List[str], procedures: List[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(out_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(SCHEMA)
        rows: List[Tuple[str, str, str, str, str, str]] = []

        for canonical, aliases in drugs:
            canon = _normalize(canonical)
            rows.append((canon, "", "isa", "category", "drug", ""))
            for alias in aliases:
                alias_norm = _normalize(alias)
                rows.append((canon, "", "alias", "phonetic", alias_norm, ""))

        for term in diagnoses:
            t = _normalize(term)
            rows.append((t, "", "isa", "category", "diagnosis", ""))

        for term in procedures:
            t = _normalize(term)
            rows.append((t, "", "isa", "category", "procedure", ""))

        _insert_rows(conn, rows)
        for idx_sql in INDEXES:
            conn.execute(idx_sql)
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entities", default=str(DEFAULT_ENTITIES))
    ap.add_argument("--out_sqlite", default=str(DEFAULT_OUT_SQLITE))
    ap.add_argument("--out_phonetic", default=str(DEFAULT_OUT_PHON))
    ap.add_argument("--phonetic-count", type=int, default=200)
    ap.add_argument("--min-sim", type=int, default=85)
    args = ap.parse_args()

    data = _load_entities(Path(args.entities))
    drugs, diagnoses, procedures = _extract_entities(data)
    drugs, diagnoses, procedures = _ensure_samples(drugs, diagnoses, procedures)

    count = _write_phonetic(
        Path(args.out_phonetic),
        drugs,
        diagnoses,
        procedures,
        target_count=args.phonetic_count,
        min_similarity=args.min_sim,
    )
    _write_semantic(Path(args.out_sqlite), drugs, diagnoses, procedures)

    print("Minimal MedSpeak KG artifacts created:")
    print(f"  {args.out_sqlite}")
    print(f"  {args.out_phonetic} (pairs={count})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
