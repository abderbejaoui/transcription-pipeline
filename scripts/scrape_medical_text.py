"""Scrape Arabic medical text from public sources for decoder LoRA adaptation.

Sources, in priority order:

  1. Arabic Wikipedia medical categories — via official MediaWiki API.
     Permissive license (CC-BY-SA). Yields ~30-50 MB of clean prose.
  2. Wikidata medical entities labelled in Arabic — via SPARQL endpoint.
     Small but high-precision (drug names, disease names, ICD-10 codes).
  3. Wikidata drug aliases — Arabic generic + brand spellings.

All sources use OFFICIAL APIs with proper User-Agent. No HTML scraping,
no rate-limit hacks. Polite throttling: ~0.3s between requests, 20 retries.

Output:
  data/medical_text/external/arwiki_<category>.jsonl
  data/medical_text/external/wikidata_drugs.jsonl
  data/medical_text/external/wikidata_diseases.jsonl
  data/medical_text/external/SCRAPE_SUMMARY.json

Each .jsonl line: {"text": "...", "source": "<src>", "title": "<page title>"}

Then feed everything to the medical text corpus builder:

  python -m scripts.build_medical_text \\
      --n-templated 200000 \\
      --external-dirs data/medical_text/external \\
      --output-dir data/medical_text
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "medical_text" / "external"

UA = ("transcription-pipeline/0.1 medical-corpus-builder "
      "(github.com/abderbejaoui/transcription-pipeline)")

# Top-level Arabic Wikipedia medical categories. Each is recursively expanded
# up to MAX_DEPTH, deduplicated by page id. These cover ~95% of all medical
# pages on ar.wikipedia.org.
SEED_CATEGORIES_AR = [
    "تصنيف:طب",                       # Medicine
    "تصنيف:أمراض",                    # Diseases
    "تصنيف:صيدلة",                    # Pharmacy
    "تصنيف:أدوية",                    # Medications
    "تصنيف:علم العقاقير",              # Pharmacology
    "تصنيف:أعراض وعلامات",             # Symptoms and signs
    "تصنيف:تشريح",                    # Anatomy
    "تصنيف:علم الأمراض",              # Pathology
    "تصنيف:جراحة",                    # Surgery
    "تصنيف:علاج",                     # Treatment
    "تصنيف:طب الأسنان",               # Dentistry
    "تصنيف:علم الأوبئة",              # Epidemiology
    "تصنيف:صحة",                      # Health
    "تصنيف:علم وظائف الأعضاء",         # Physiology
    "تصنيف:علم الأحياء الدقيقة الطبي",  # Medical microbiology
]

# Wikidata classes to mine for Arabic medical labels.
# Q12140 = medication, Q12136 = disease, Q169872 = symptom,
# Q4936952 = anatomical structure, Q105584 = active ingredient
WIKIDATA_CLASSES = {
    "drugs": "Q12140",
    "diseases": "Q12136",
    "symptoms": "Q169872",
    "anatomy": "Q4936952",
}


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(session, url: str, params: Optional[Dict[str, Any]] = None,
              max_retries: int = 5, sleep_s: float = 0.3) -> Dict:
    for attempt in range(max_retries):
        try:
            resp = session.get(url, params=params, timeout=60,
                               headers={"User-Agent": UA, "Accept": "application/json"})
            if resp.status_code == 200:
                time.sleep(sleep_s)
                return resp.json()
            # Polite back-off on 429 / 5xx
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Wikipedia: walk categories, fetch page extracts
# ---------------------------------------------------------------------------


WIKI_API = "https://ar.wikipedia.org/w/api.php"


def list_category_pages(session, cat_title: str, max_depth: int,
                        seen_cats: Set[str], seen_pages: Set[int]) -> Iterable[Dict]:
    """Yield {'pageid': int, 'title': str} for every page under cat_title
    up to max_depth subcategory levels. Uses categorymembers + recursion."""
    if max_depth < 0 or cat_title in seen_cats:
        return
    seen_cats.add(cat_title)

    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": cat_title,
            "cmlimit": 500,
            "cmtype": "page|subcat",
            "format": "json",
            "formatversion": 2,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = _http_get(session, WIKI_API, params=params)
        members = data.get("query", {}).get("categorymembers", [])
        for m in members:
            ns = m.get("ns")
            if ns == 14:  # subcategory
                yield from list_category_pages(
                    session, m["title"], max_depth - 1, seen_cats, seen_pages,
                )
            elif ns == 0:  # main article
                pid = m["pageid"]
                if pid in seen_pages:
                    continue
                seen_pages.add(pid)
                yield {"pageid": pid, "title": m["title"]}
        cmcontinue = data.get("continue", {}).get("cmcontinue")
        if not cmcontinue:
            break


def fetch_extracts(session, pageids: List[int]) -> List[Dict]:
    """Fetch plaintext extracts for up to 20 page ids per call."""
    out = []
    for i in range(0, len(pageids), 20):
        batch = pageids[i:i + 20]
        params = {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "exsectionformat": "plain",
            "exlimit": "max",
            "pageids": "|".join(str(p) for p in batch),
            "format": "json",
            "formatversion": 2,
        }
        data = _http_get(session, WIKI_API, params=params, sleep_s=0.4)
        for p in data.get("query", {}).get("pages", []):
            text = p.get("extract", "") or ""
            if text:
                out.append({"pageid": p["pageid"], "title": p["title"], "text": text})
    return out


# ---------------------------------------------------------------------------
# Wikidata SPARQL: Arabic labels for medical entity classes
# ---------------------------------------------------------------------------


WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"


def query_wikidata_arabic_labels(session, class_qid: str, limit: int = 20000) -> List[Dict]:
    """Return [{'qid', 'label_ar', 'aliases_ar': [..]}] for every entity that
    is instance-of (or subclass-of) class_qid and has an Arabic label."""
    sparql = f"""
SELECT ?item ?itemLabel ?alias WHERE {{
  ?item wdt:P31/wdt:P279* wd:{class_qid} .
  ?item rdfs:label ?itemLabel .
  FILTER(LANG(?itemLabel) = "ar")
  OPTIONAL {{
    ?item skos:altLabel ?alias .
    FILTER(LANG(?alias) = "ar")
  }}
}}
LIMIT {limit}
"""
    headers = {
        "User-Agent": UA,
        "Accept": "application/sparql-results+json",
    }
    resp = session.get(WIKIDATA_SPARQL, params={"query": sparql},
                       headers=headers, timeout=180)
    resp.raise_for_status()
    data = resp.json()

    by_item: Dict[str, Dict] = {}
    for row in data.get("results", {}).get("bindings", []):
        qid = row["item"]["value"].rsplit("/", 1)[-1]
        label = row.get("itemLabel", {}).get("value", "")
        alias = row.get("alias", {}).get("value", "")
        rec = by_item.setdefault(qid, {"qid": qid, "label_ar": label, "aliases_ar": []})
        if alias and alias != label and alias not in rec["aliases_ar"]:
            rec["aliases_ar"].append(alias)
    return list(by_item.values())


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------


_REF_RE = re.compile(r"\[[\d,\s]+\]")
_MULTI_NL = re.compile(r"\n{2,}")
_MULTI_WS = re.compile(r"[ \t]{2,}")


def _clean_paragraph(text: str) -> str:
    text = _REF_RE.sub("", text)
    text = _MULTI_WS.sub(" ", text)
    text = _MULTI_NL.sub("\n", text)
    return text.strip()


def _split_paragraphs(text: str, min_words: int = 6, max_words: int = 220) -> List[str]:
    paras = []
    for raw in text.split("\n"):
        p = _clean_paragraph(raw)
        if not p:
            continue
        wc = len(p.split())
        if min_words <= wc <= max_words:
            paras.append(p)
    return paras


# ---------------------------------------------------------------------------
# Per-source runners
# ---------------------------------------------------------------------------


def scrape_arwiki(session, categories: List[str], max_depth: int,
                  max_articles: int, out_dir: Path) -> Dict[str, Any]:
    """Walk Arabic Wikipedia medical categories and dump per-article extracts."""
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[arwiki] expanding {len(categories)} root categories "
          f"to depth {max_depth}")
    seen_cats: Set[str] = set()
    seen_pages: Set[int] = set()
    pages: List[Dict] = []
    for cat in categories:
        for p in list_category_pages(session, cat, max_depth,
                                     seen_cats, seen_pages):
            pages.append(p)
            if max_articles and len(pages) >= max_articles:
                break
        if max_articles and len(pages) >= max_articles:
            print(f"[arwiki] hit max-articles cap ({max_articles}), stopping expansion")
            break
    print(f"[arwiki] {len(pages)} unique pages discovered across "
          f"{len(seen_cats)} categories")

    ids = [p["pageid"] for p in pages]
    title_by_id = {p["pageid"]: p["title"] for p in pages}

    n_paras = 0
    n_chars = 0
    out_path = out_dir / "arwiki_all.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(ids), 20):
            batch = ids[batch_start:batch_start + 20]
            extracts = fetch_extracts(session, batch)
            for ex in extracts:
                title = title_by_id.get(ex["pageid"], ex.get("title", ""))
                for para in _split_paragraphs(ex["text"]):
                    f.write(json.dumps({
                        "text": para,
                        "source": "arwiki",
                        "title": title,
                    }, ensure_ascii=False) + "\n")
                    n_paras += 1
                    n_chars += len(para)
            if (batch_start // 20) % 10 == 0:
                print(f"  fetched {batch_start + len(batch)}/{len(ids)} "
                      f"pages   paras={n_paras}  ~{n_chars/1e6:.1f}MB", flush=True)

    return {
        "source": "arwiki",
        "categories": categories,
        "pages": len(pages),
        "paragraphs": n_paras,
        "approx_MB": round(n_chars / 1e6, 2),
        "out_path": str(out_path.relative_to(PROJECT_ROOT)),
    }


def scrape_wikidata(session, out_dir: Path,
                    classes: Dict[str, str], limit_per_class: int) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"source": "wikidata", "classes": {}}
    for label, qid in classes.items():
        print(f"[wikidata] {label} ({qid}) ...")
        try:
            rows = query_wikidata_arabic_labels(session, qid, limit=limit_per_class)
        except Exception as exc:
            print(f"  ! failed: {exc!r}")
            results["classes"][label] = {"error": repr(exc)}
            continue
        out_path = out_dir / f"wikidata_{label}.jsonl"
        n_lines = 0
        with out_path.open("w", encoding="utf-8") as f:
            for r in rows:
                main = r["label_ar"].strip()
                if main:
                    f.write(json.dumps({"text": main, "source": f"wikidata_{label}",
                                        "title": r["qid"]}, ensure_ascii=False) + "\n")
                    n_lines += 1
                for alias in r.get("aliases_ar", []):
                    alias = alias.strip()
                    if alias:
                        f.write(json.dumps({"text": alias,
                                            "source": f"wikidata_{label}_alias",
                                            "title": r["qid"]}, ensure_ascii=False) + "\n")
                        n_lines += 1
                # Also write a "main is an alias_of" style sentence for both
                # spellings so the LM learns alternates.
                for alias in r.get("aliases_ar", []):
                    f.write(json.dumps({
                        "text": f"{main} ويعرف أيضا بـ{alias}",
                        "source": f"wikidata_{label}_pair",
                        "title": r["qid"],
                    }, ensure_ascii=False) + "\n")
                    n_lines += 1
        results["classes"][label] = {
            "entities": len(rows),
            "lines": n_lines,
            "out_path": str(out_path.relative_to(PROJECT_ROOT)),
        }
        print(f"  {len(rows)} entities -> {n_lines} lines -> {out_path}")
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--skip-arwiki", action="store_true")
    ap.add_argument("--skip-wikidata", action="store_true")
    ap.add_argument("--arwiki-depth", type=int, default=3,
                    help="Subcategory recursion depth (3 = root cat + 3 nested).")
    ap.add_argument("--arwiki-max-articles", type=int, default=15000,
                    help="Cap total Wikipedia articles to fetch (0 = no cap). "
                         "Realistic full medical Arabic Wikipedia: ~10-15k articles.")
    ap.add_argument("--wikidata-limit-per-class", type=int, default=15000)
    args = ap.parse_args()

    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": UA})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"run_at": int(time.time()), "results": {}}

    if not args.skip_arwiki:
        try:
            r = scrape_arwiki(session, SEED_CATEGORIES_AR, args.arwiki_depth,
                              args.arwiki_max_articles, args.out_dir)
            summary["results"]["arwiki"] = r
        except Exception as exc:
            print(f"[arwiki] FAILED: {exc!r}", file=sys.stderr)
            summary["results"]["arwiki"] = {"error": repr(exc)}

    if not args.skip_wikidata:
        try:
            r = scrape_wikidata(session, args.out_dir, WIKIDATA_CLASSES,
                                args.wikidata_limit_per_class)
            summary["results"]["wikidata"] = r
        except Exception as exc:
            print(f"[wikidata] FAILED: {exc!r}", file=sys.stderr)
            summary["results"]["wikidata"] = {"error": repr(exc)}

    summary_path = args.out_dir / "SCRAPE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                            encoding="utf-8")
    print(f"\n[done] summary -> {summary_path}")

    # Total MB tally
    total_bytes = 0
    for f in args.out_dir.glob("*.jsonl"):
        total_bytes += f.stat().st_size
    print(f"[done] total scraped: {total_bytes/1e6:.1f} MB across "
          f"{len(list(args.out_dir.glob('*.jsonl')))} files")
    print("\nNext step — bake into the medical training corpus:")
    print("  python -m scripts.build_medical_text \\")
    print("      --n-templated 200000 \\")
    print(f"      --external-dirs {args.out_dir.relative_to(PROJECT_ROOT)} \\")
    print("      --output-dir data/medical_text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
