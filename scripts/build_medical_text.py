"""Build the Arabic medical text corpus for decoder adaptation.

Reads three curated seeds (UAE drugs, Arabic disease list with ICD-10,
Arabic symptom list) plus a clinical template bank and emits a JSONL
corpus that looks like real Gulf-clinic conversation transcripts.

Why this works for ASR adaptation:
  - Qwen3-ASR's decoder is a standard Qwen3 causal LM. Adapting its
    weights on Arabic medical text shifts the language-model prior so
    "روسوفاستاتين" and "أوميبرازول" stop getting rewritten to MSA
    near-spellings at inference.
  - The audio encoder stays frozen — only the decoder LoRA changes.
  - We mix synthetic templated sentences with whatever real Arabic
    medical text the user can scrape (Wikipedia, MoH PDFs, public
    guidelines).

Output:
  data/medical_text/corpus.jsonl   — one record per sentence:
    {"text": "...", "source": "templates|scrape|seed", "weight": 1.0}
  data/medical_text/SUMMARY.json   — counts and a small sample
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = PROJECT_ROOT / "data" / "medical_text" / "seeds"
OUT_DIR = PROJECT_ROOT / "data" / "medical_text"

DOSE_FALLBACKS = [5, 10, 20, 25, 50, 75, 100, 200, 250, 500, 750, 1000]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _pick(rng: random.Random, lst):
    return rng.choice(lst) if lst else ""


# ---------------------------------------------------------------------------
# Template instantiation
# ---------------------------------------------------------------------------


_SLOT_RE = re.compile(r"\{([a-zA-Z_0-9]+)(?:_2)?\}")


def _fill_template(
    template: str,
    rng: random.Random,
    drugs: List[Dict[str, Any]],
    diseases: List[Dict[str, Any]],
    symptoms: List[Dict[str, Any]],
    slot_fillers: Dict[str, List[str]],
) -> Optional[str]:
    """Substitute named slots. Returns None if a required slot can't be filled."""
    # Pre-roll a drug / disease / symptom so multiple slots referencing the
    # same kind in one sentence stay consistent.
    drug = _pick(rng, drugs)
    drug2 = _pick(rng, drugs)
    disease = _pick(rng, diseases)
    sym = _pick(rng, symptoms)
    sym2 = _pick(rng, [s for s in symptoms if s["ar"] != sym["ar"]] or symptoms)

    dose_unit = drug.get("dose_unit", "ملغ")
    doses = drug.get("common_doses_mg") or DOSE_FALLBACKS
    dose = _pick(rng, doses)
    brand_uae_list = drug.get("brand_uae") or []
    brand_uae = _pick(rng, brand_uae_list) or drug.get("generic_ar", "")

    used = 0  # count distinct symptom_ar occurrences so symptom_ar_2 fires
    out = template

    def _sub(match):
        nonlocal used
        slot = match.group(0).strip("{}").rstrip("_2")
        suffix = match.group(0).endswith("_2}")
        if slot == "drug_ar":
            return drug.get("generic_ar", drug2.get("generic_ar", ""))
        if slot == "brand_uae":
            return brand_uae
        if slot == "dose":
            return str(dose)
        if slot == "dose_unit":
            return dose_unit
        if slot == "disease_ar":
            return disease.get("ar", "")
        if slot == "symptom_ar":
            if suffix:
                return sym2.get("ar", "")
            used += 1
            return sym.get("ar", "")
        if slot in slot_fillers:
            return _pick(rng, slot_fillers[slot])
        return match.group(0)

    out = _SLOT_RE.sub(_sub, out)
    # Reject if we ended up with empty critical slots.
    if "{" in out or "}" in out:
        return None
    return out


def synthesize(
    n_target: int,
    rng: random.Random,
    drugs: List[Dict[str, Any]],
    diseases: List[Dict[str, Any]],
    symptoms: List[Dict[str, Any]],
    templates: Dict[str, Any],
) -> List[str]:
    slot_fillers = templates.get("_slot_fillers", {})
    pools = {k: v for k, v in templates.items() if not k.startswith("_") and isinstance(v, list)}
    flat = []
    for tpl_list in pools.values():
        flat.extend(tpl_list)
    print(f"  {len(flat)} template sentences across {len(pools)} pools")

    out: List[str] = []
    tries = 0
    while len(out) < n_target and tries < n_target * 4:
        tpl = rng.choice(flat)
        filled = _fill_template(tpl, rng, drugs, diseases, symptoms, slot_fillers)
        if filled and 4 < len(filled.split()) < 120:
            out.append(filled)
        tries += 1
    return out


# ---------------------------------------------------------------------------
# Seed -> direct sentences (so the model also sees plain term lists)
# ---------------------------------------------------------------------------


def seed_sentences(
    drugs: List[Dict[str, Any]],
    diseases: List[Dict[str, Any]],
    symptoms: List[Dict[str, Any]],
) -> List[str]:
    out: List[str] = []
    for d in drugs:
        ar = d.get("generic_ar", "")
        if ar:
            out.append(ar)
        for b in d.get("brand_uae", []) or []:
            out.append(b)
            if ar:
                out.append(f"{b} هو {ar}")
        if d.get("indication_ar") and ar:
            out.append(f"{ar} يستخدم {d['indication_ar']}")
    for d in diseases:
        ar = d.get("ar", "")
        if ar:
            out.append(ar)
            if d.get("category"):
                out.append(f"{ar} من أمراض {d['category']}")
    for s in symptoms:
        ar = s.get("ar", "")
        if ar:
            out.append(ar)
            out.append(f"المريض يشكو من {ar}")
    return out


# ---------------------------------------------------------------------------
# Optional: external scrape ingestion (user-supplied text files)
# ---------------------------------------------------------------------------


def load_external(text_dirs: List[Path]) -> List[str]:
    """Read every .txt / .jsonl file under given dirs as additional sentences.
    Plain .txt is split per line; .jsonl is expected to have a `text` field.

    Long paragraphs from sources like Wikipedia are split on sentence
    boundaries (.، ؟ !) so each yielded sentence is 4-220 words. Single-word
    or 2-3-word labels (e.g. a Wikidata term on its own line) are also kept
    because they're useful seed data for the LM.
    """
    sent_split = re.compile(r"[\.\u060C\u061B\u061F!؟]\s+")
    out: List[str] = []
    for d in text_dirs:
        if not d.exists():
            print(f"  ! skip missing dir: {d}", file=sys.stderr)
            continue
        for p in d.rglob("*"):
            if p.is_dir():
                continue
            try:
                lines = p.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if p.suffix == ".jsonl":
                    try:
                        rec = json.loads(line)
                        t = rec.get("text") or rec.get("sentence") or ""
                    except Exception:
                        t = ""
                else:
                    t = line
                if not t:
                    continue
                wc = len(t.split())
                if wc <= 3:
                    # Short label (drug name, disease name) — keep as-is.
                    out.append(t)
                    continue
                if wc < 220:
                    out.append(t)
                    continue
                # Long paragraph — split into sentences.
                for sent in sent_split.split(t):
                    sent = sent.strip()
                    sw = len(sent.split())
                    if 4 < sw < 220:
                        out.append(sent)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-templated", type=int, default=200_000,
                    help="How many synthetic templated sentences to emit.")
    ap.add_argument("--external-dirs", type=Path, nargs="*", default=[],
                    help="Optional dirs with .txt/.jsonl Arabic medical text "
                         "(Wikipedia dumps, MoH PDFs, guidelines).")
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("[seeds] loading")
    drugs = _load_jsonl(SEEDS / "uae_drugs.jsonl")
    diseases = _load_jsonl(SEEDS / "diseases_ar.jsonl")
    symptoms = _load_jsonl(SEEDS / "symptoms_ar.jsonl")
    templates = json.loads((SEEDS / "templates.json").read_text(encoding="utf-8"))
    print(f"  drugs={len(drugs)} diseases={len(diseases)} symptoms={len(symptoms)}")

    print(f"[seed-sentences] flattening lexicon to plain sentences")
    seeds = seed_sentences(drugs, diseases, symptoms)
    print(f"  {len(seeds)} seed sentences")

    print(f"[templated] generating {args.n_templated} synthetic clinical sentences")
    tmpl = synthesize(args.n_templated, rng, drugs, diseases, symptoms, templates)
    print(f"  {len(tmpl)} usable")

    external: List[str] = []
    if args.external_dirs:
        print(f"[external] reading user-supplied dirs: {args.external_dirs}")
        external = load_external(args.external_dirs)
        print(f"  {len(external)} external sentences")

    records: List[Dict[str, Any]] = []
    for s in seeds:
        records.append({"text": s, "source": "seed", "weight": 1.0})
    for s in tmpl:
        records.append({"text": s, "source": "templates", "weight": 1.0})
    for s in external:
        records.append({"text": s, "source": "external", "weight": 1.5})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "corpus.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "n_records": len(records),
        "n_seed": len(seeds),
        "n_templated": len(tmpl),
        "n_external": len(external),
        "sample_seed": seeds[:5],
        "sample_templated": tmpl[:5],
    }
    (args.output_dir / "SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    print(f"\n[done] wrote {len(records)} records -> {out_path}")
    print("First 3 templated examples:")
    for s in tmpl[:3]:
        print(f"  • {s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
