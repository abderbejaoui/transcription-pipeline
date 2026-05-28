"""Expand the medical lexicon by prompting Calme across therapeutic categories
and body systems. Designed to produce thousands of unique Gulf-relevant
drugs and diseases with Arabic-script aliases.

The script is resumable: it appends to the output file after every batch and
skips terms that already exist (case-insensitive) in either the seed lexicon
or the running output.

Usage
-----
    python3 scripts/expand_lexicon.py \\
        --ollama-url http://localhost:11434 \\
        --ollama-model calme-3.2-instruct-78b-GGUF:IQ4_XS \\
        --seed data/medical_lexicon.jsonl data/gulf_drug_brands.jsonl \\
        --out data/gulf_lexicon_expanded.jsonl \\
        --target 5000

Strategy
--------
We sweep ~70 therapeutic / clinical categories. For each category we ask the
LLM for N new terms NOT in a provided "already-have" sample. We rotate that
sample every batch so the model keeps proposing novel items. We accept only
JSON-parseable arrays of objects in the required shape and silently drop
duplicates and obviously bogus entries.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

import requests


# ── categories ────────────────────────────────────────────────────────────────
# Drug therapeutic classes. We aim for ~60-100 unique items per class.
DRUG_CATEGORIES: List[str] = [
    # cardiovascular
    "ACE inhibitors (antihypertensive)",
    "angiotensin receptor blockers (ARBs)",
    "beta blockers",
    "calcium channel blockers",
    "thiazide and loop diuretics",
    "potassium-sparing diuretics",
    "statins and other lipid-lowering drugs",
    "fibrates and PCSK9 inhibitors",
    "antiplatelet drugs",
    "anticoagulants (warfarin, DOACs, heparins)",
    "antiarrhythmics",
    "vasodilators and nitrates",
    "heart failure drugs (digoxin, SGLT2 for HF, sacubitril)",
    # diabetes & endocrine
    "insulin formulations (rapid, basal, mixed)",
    "metformin formulations and combinations",
    "GLP-1 receptor agonists",
    "SGLT2 inhibitors",
    "DPP-4 inhibitors",
    "sulfonylureas and meglitinides",
    "thyroid replacement and antithyroid drugs",
    "osteoporosis drugs (bisphosphonates, denosumab, teriparatide)",
    "corticosteroids systemic",
    "sex hormones and HRT",
    "contraceptives (oral, injectable, implants)",
    "fertility drugs (clomid, letrozole, gonadotropins)",
    # antimicrobials
    "penicillins and beta-lactam/inhibitor combinations",
    "cephalosporins (1st-5th generation)",
    "macrolides and ketolides",
    "fluoroquinolones",
    "tetracyclines and glycylcyclines",
    "carbapenems and monobactams",
    "glycopeptides and oxazolidinones (vancomycin, linezolid)",
    "aminoglycosides",
    "metronidazole, sulfa drugs, nitrofurantoin",
    "azole antifungals (oral and topical)",
    "echinocandins and polyene antifungals",
    "antivirals for herpes, HBV, HCV, HIV, influenza, COVID",
    "antimalarials and antiparasitics",
    "antituberculous drugs",
    # respiratory
    "inhaled short-acting bronchodilators (SABA)",
    "inhaled long-acting bronchodilators (LABA, LAMA)",
    "inhaled corticosteroids and combinations",
    "leukotriene receptor antagonists and biologics for asthma",
    "decongestants, antihistamines (1st and 2nd generation)",
    "cough suppressants and expectorants",
    "nasal sprays (steroid and antihistamine)",
    # gastrointestinal
    "proton pump inhibitors and H2 blockers",
    "antacids and alginates",
    "prokinetics, antiemetics",
    "antidiarrheals and rehydration salts",
    "laxatives and stool softeners",
    "antispasmodics and IBS drugs",
    "biologics for IBD (anti-TNF, anti-integrin, anti-IL)",
    "pancreatic enzyme replacement and bile acids",
    # neurology & psychiatry
    "SSRIs and SNRIs",
    "tricyclic and atypical antidepressants",
    "first and second generation antipsychotics",
    "anxiolytics (benzodiazepines, buspirone)",
    "sleep aids (z-drugs, melatonin agonists)",
    "mood stabilizers (lithium, valproate, lamotrigine)",
    "ADHD medications",
    "antiepileptic drugs",
    "Parkinson disease drugs",
    "Alzheimer / dementia drugs",
    "migraine prevention and abortive drugs (triptans, CGRP)",
    "multiple sclerosis disease-modifying therapies",
    # pain & rheumatology
    "non-opioid analgesics",
    "opioid analgesics (immediate and modified release)",
    "topical analgesics (NSAID gels, capsaicin, patches)",
    "muscle relaxants",
    "DMARDs (methotrexate, leflunomide, hydroxychloroquine)",
    "biologic DMARDs and JAK inhibitors",
    "gout drugs (allopurinol, febuxostat, colchicine, uricosurics)",
    # dermatology / ophthalmology / ENT / urology
    "topical corticosteroids by potency",
    "topical antibiotics and antifungals",
    "topical retinoids and acne treatments",
    "psoriasis biologics and topicals",
    "anti-VEGF and ophthalmic drops (glaucoma, dry eye, antibiotic)",
    "ear drops (antibiotic, anti-inflammatory, wax)",
    "BPH and erectile dysfunction drugs",
    "overactive bladder drugs",
    # oncology
    "common chemotherapy agents",
    "targeted therapies for breast, lung, GI cancers",
    "immunotherapy checkpoint inhibitors",
    "hormonal cancer therapies (tamoxifen, AI, GnRH analogs)",
    # vaccines & misc
    "common adult and pediatric vaccines",
    "iron, calcium, vitamin D, B-complex, folate supplements",
    "pediatric syrups (analgesic, antibiotic, antitussive)",
    "emergency drugs (adrenaline, atropine, naloxone, glucagon)",
    "topical wound care and antiseptics",
]

# Disease categories grouped by body system / specialty.
DISEASE_CATEGORIES: List[str] = [
    "common cardiovascular diseases (HTN, IHD, HF, arrhythmias, valvular)",
    "cerebrovascular and stroke-related diagnoses",
    "respiratory infections (URTI, LRTI, pneumonia variants, TB)",
    "asthma, COPD, bronchiectasis, ILD subtypes",
    "type 1 and type 2 diabetes and complications",
    "thyroid disorders (hyper, hypo, nodules, thyroiditis)",
    "obesity, metabolic syndrome, dyslipidemia",
    "common GI diseases (GERD, PUD, gastritis, IBS, IBD, celiac)",
    "liver diseases (hepatitis types, NAFLD, cirrhosis, ALD)",
    "kidney diseases (CKD stages, AKI, nephritis, stones)",
    "urinary tract infections and BPH-related diagnoses",
    "common dermatologic conditions (acne, eczema, psoriasis, fungal, scabies)",
    "skin infections (cellulitis, erysipelas, abscess, impetigo)",
    "common pediatric infectious diseases (measles, mumps, varicella, hand-foot-mouth)",
    "neonatal common diagnoses (jaundice, sepsis, feeding issues)",
    "common gynecologic diagnoses (PCOS, endometriosis, PID, fibroids)",
    "obstetric conditions (GDM, preeclampsia, hyperemesis, threatened miscarriage)",
    "common psychiatric disorders (MDD, anxiety, OCD, bipolar, psychosis, PTSD, ADHD)",
    "neurologic conditions (migraine, epilepsy, Parkinson, MS, stroke subtypes)",
    "common cancers (breast, lung, colorectal, prostate, leukemias)",
    "rheumatologic disorders (RA, SLE, AS, vasculitis, fibromyalgia, gout)",
    "common ENT diagnoses (otitis, sinusitis, pharyngitis, tonsillitis, vertigo)",
    "ophthalmologic diagnoses (conjunctivitis, glaucoma, cataract, AMD, diabetic retinopathy)",
    "musculoskeletal injuries (sprains, fractures, tendinopathies, low back pain syndromes)",
    "hematologic disorders (anemias by type, thalassemia, sickle cell, ITP, leukemia/lymphoma)",
    "infectious diseases of clinical importance (dengue, malaria, brucellosis, MERS, COVID)",
    "common allergic and immunologic conditions (rhinitis, urticaria, anaphylaxis)",
    "vitamin and mineral deficiencies (D, B12, iron, calcium, folate)",
    "common pediatric chronic conditions (asthma, atopic dermatitis, ADHD, autism)",
    "common geriatric syndromes (delirium, dementia, falls, frailty)",
    "common ER presentations as diagnoses (chest pain, abd pain, headache, syncope)",
    "common surgical diagnoses (appendicitis, cholecystitis, hernia, hemorrhoids, varicose veins)",
]


# ── prompt templates ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a clinical pharmacist and physician working in the
Gulf region (Saudi Arabia, UAE, Kuwait, Bahrain, Oman, Qatar).

Generate JSON arrays of medical terms that are REAL and ACTUALLY USED in
Gulf clinical practice. Drug entries should be real generic names or REAL brand
names sold in Gulf pharmacies. Disease entries should be real ICD-recognized
clinical diagnoses or commonly used clinical labels.

For every entry provide 1-3 Arabic-script aliases representing how Gulf
patients, pharmacists, or doctors transliterate or pronounce the term. These
must look plausible to a Khaleeji Arabic speaker.

Output STRICT JSON only — a single array of objects with exactly these keys:
  term       lowercase English term, e.g. "amlodipine"
  type       "drug" or "disease"
  aliases    array of 1-3 Arabic-script strings
  category   short English subcategory, e.g. "calcium channel blocker"

NEVER invent terms that do not exist. NEVER repeat terms across calls. Do not
include markdown, no code fences, no explanation — JSON array only."""

USER_TEMPLATE = """Generate {n} NEW {type_label} entries in the category:
  {category}

Avoid repeating any of these already-collected terms:
  {seen}

Return ONLY the JSON array."""


# ── helpers ──────────────────────────────────────────────────────────────────
def _load_seed(paths: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    """Return dict of term(lowercase) -> entry, merged across files."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        path = Path(p)
        if not path.exists():
            print(f"[expand] WARN seed missing: {path}")
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                term = row.get("term", "").strip().lower()
                if term:
                    out[term] = row
    return out


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    """Robustly extract a JSON array from the model's reply."""
    # Strip ```json ... ``` fences if the model added them
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    blob = text[start:end + 1]
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    return [x for x in data if isinstance(x, dict)]


def _is_valid_entry(entry: Dict[str, Any], expected_type: str) -> bool:
    term = entry.get("term", "").strip().lower()
    if not term or len(term) < 2 or len(term) > 60:
        return False
    if not re.match(r"^[a-z0-9 \-'/().+]+$", term):
        return False
    if entry.get("type") not in ("drug", "disease"):
        return False
    if entry.get("type") != expected_type:
        return False
    aliases = entry.get("aliases") or []
    if not isinstance(aliases, list) or not aliases:
        return False
    # Must have at least one Arabic-script alias
    arabic_re = re.compile(r"[\u0600-\u06FF]")
    if not any(isinstance(a, str) and arabic_re.search(a) for a in aliases):
        return False
    return True


def _ollama_generate(
    url: str,
    model: str,
    system: str,
    prompt: str,
    timeout: int = 600,
) -> str:
    resp = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "system": system,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.95,
                "top_p": 0.95,
                "num_predict": 6144,
            },
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


# ── main loop ────────────────────────────────────────────────────────────────
def expand(
    *,
    ollama_url: str,
    ollama_model: str,
    seed_paths: List[str],
    out_path: Path,
    target: int,
    per_call: int,
    max_calls_per_category: int,
) -> None:
    seed = _load_seed(seed_paths)
    print(f"[expand] seed lexicon: {len(seed)} terms")

    # Load any pre-existing output (resume).
    collected: Dict[str, Dict[str, Any]] = {}
    if out_path.exists():
        with out_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                t = row.get("term", "").strip().lower()
                if t:
                    collected[t] = row
        print(f"[expand] resuming with {len(collected)} previously collected")

    seen_terms: Set[str] = set(seed.keys()) | set(collected.keys())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = out_path.open("a", encoding="utf-8")

    t_start = time.time()
    n_calls = 0
    n_added = 0

    def _round_robin_categories() -> Iterable[tuple[str, str]]:
        """Yield (type, category) endlessly, alternating drug/disease."""
        i = 0
        # Per-category call counter so we don't burn the same prompt forever.
        per_cat: Dict[str, int] = {}
        while True:
            advanced = False
            for cat_type, cat_list in (
                ("drug", DRUG_CATEGORIES),
                ("disease", DISEASE_CATEGORIES),
            ):
                cat = cat_list[i % len(cat_list)]
                key = f"{cat_type}::{cat}"
                if per_cat.get(key, 0) >= max_calls_per_category:
                    continue
                per_cat[key] = per_cat.get(key, 0) + 1
                yield cat_type, cat
                advanced = True
            i += 1
            if not advanced:
                return

    for cat_type, category in _round_robin_categories():
        if len(collected) >= target:
            break

        # Build a small "already-have" sample so the model knows what to skip.
        # Take a rotating window of recent terms to keep the prompt small.
        recent = list(seen_terms)[-120:]
        recent_sample = ", ".join(recent[-60:]) if recent else "(none)"

        prompt = USER_TEMPLATE.format(
            n=per_call,
            type_label=cat_type + "s",
            category=category,
            seen=recent_sample,
        )

        try:
            text = _ollama_generate(ollama_url, ollama_model, SYSTEM_PROMPT, prompt)
        except Exception as e:
            print(f"[expand]   call failed ({cat_type}/{category[:40]}): {e}")
            continue

        n_calls += 1
        entries = _parse_json_array(text)
        kept = 0
        for entry in entries:
            if not _is_valid_entry(entry, cat_type):
                continue
            term = entry["term"].strip().lower()
            if term in seen_terms:
                continue
            # Normalize the shape.
            row = {
                "term": term,
                "type": entry["type"],
                "aliases": [a for a in entry["aliases"] if isinstance(a, str)][:3],
                "category": str(entry.get("category", category))[:80],
                "priority": 0.8,
                "region": "gulf",
                "source": "calme-expanded",
            }
            collected[term] = row
            seen_terms.add(term)
            out_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            kept += 1
            n_added += 1
        out_fh.flush()

        elapsed = (time.time() - t_start) / 60
        print(f"[expand] +{kept:>3} ({cat_type}/{category[:48]:<48}) "
              f"total={len(collected):>5}/{target}  calls={n_calls}  "
              f"elapsed={elapsed:.1f}min")

    out_fh.close()
    print(f"\n[expand] DONE  collected={len(collected)}  "
          f"calls={n_calls}  added_this_run={n_added}")
    print(f"[expand] file: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ollama-url", default="http://localhost:11434")
    ap.add_argument("--ollama-model",
                    default="calme-3.2-instruct-78b-GGUF:IQ4_XS")
    ap.add_argument("--seed", nargs="+",
                    default=["data/medical_lexicon.jsonl",
                             "data/gulf_drug_brands.jsonl"])
    ap.add_argument("--out", default="data/gulf_lexicon_expanded.jsonl")
    ap.add_argument("--target", type=int, default=5000,
                    help="Stop after collecting this many NEW unique terms.")
    ap.add_argument("--per-call", type=int, default=40,
                    help="How many entries to request per LLM call.")
    ap.add_argument("--max-calls-per-category", type=int, default=8,
                    help="Hard cap on how many times we revisit each category.")
    args = ap.parse_args()

    expand(
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model,
        seed_paths=args.seed,
        out_path=Path(args.out),
        target=args.target,
        per_call=args.per_call,
        max_calls_per_category=args.max_calls_per_category,
    )


if __name__ == "__main__":
    main()
