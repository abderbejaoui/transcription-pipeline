"""Measure whether a scraped medical text corpus actually covers our
target Arabic medical vocabulary.

Answers the only question that matters before deciding to train:
"Does Wikipedia (or whatever we scraped) contain enough of the drug,
disease, and symptom terms we want the decoder to learn?"

Reports, for each term in our curated seeds:
  - is it present at all (0/1)
  - how many times it appears
  - histogram: terms with 0 / 1-5 / 6-20 / 21-100 / 100+ occurrences

Rule of thumb for decoder LoRA on text:
  ~30+ occurrences  → strong learning signal
  ~5-30 occurrences → weak but useful prior shift
  0-5 occurrences   → effectively unseen, no learning

So if 80%+ of our vocabulary lands in the 30+ bucket, the corpus is
useful. If most terms are at 0-5, we need a different source.

Usage:
  python -m scripts.eval_medical_corpus \\
      --corpus-dir data/medical_text/external \\
      --seeds-dir data/medical_text/seeds
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def collect_seed_terms(seeds_dir: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Returns {category: [(term, type), ...]} for everything in our seeds.

    `type` is "ar" for the Arabic name, "brand" for a UAE brand name.
    """
    terms: Dict[str, List[Tuple[str, str]]] = defaultdict(list)

    for r in _load_jsonl(seeds_dir / "uae_drugs.jsonl"):
        ar = (r.get("generic_ar") or "").strip()
        if ar:
            terms["drug_generic"].append((ar, "generic"))
        for b in r.get("brand_uae", []) or []:
            b = (b or "").strip()
            if b:
                terms["drug_brand"].append((b, "brand"))

    for r in _load_jsonl(seeds_dir / "diseases_ar.jsonl"):
        ar = (r.get("ar") or "").strip()
        if ar:
            terms["disease"].append((ar, "disease"))

    for r in _load_jsonl(seeds_dir / "symptoms_ar.jsonl"):
        ar = (r.get("ar") or "").strip()
        if ar:
            terms["symptom"].append((ar, "symptom"))

    return terms


def load_corpus_text(corpus_dir: Path) -> str:
    """Concatenate all .jsonl files in corpus_dir into one big string for
    fast substring counting. Returns total text."""
    chunks: List[str] = []
    total_files = 0
    for p in sorted(corpus_dir.glob("*.jsonl")):
        total_files += 1
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            t = rec.get("text") or ""
            if t:
                chunks.append(t)
    if total_files == 0:
        raise FileNotFoundError(f"No *.jsonl files under {corpus_dir}")
    return "\n".join(chunks)


def count_occurrences(text: str, terms: List[Tuple[str, str]]) -> List[Tuple[str, str, int]]:
    """For each term, count substring occurrences (case-sensitive — Arabic).
    Long terms (multi-word) use plain str.count. Single-word terms use a
    regex with word boundaries so we don't double-count substrings inside
    longer words.
    """
    results: List[Tuple[str, str, int]] = []
    for term, kind in terms:
        if not term:
            continue
        if len(term.split()) == 1 and len(term) >= 5:
            # Word-boundary count for single Arabic words >= 5 chars.
            # \b doesn't always work cleanly in Arabic, so use a lookahead
            # that ensures the next character is not an Arabic letter.
            pat = re.compile(rf"(?<![\u0600-\u06FF])"
                             rf"{re.escape(term)}"
                             rf"(?![\u0600-\u06FF])")
            n = len(pat.findall(text))
        else:
            n = text.count(term)
        results.append((term, kind, n))
    return results


def bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 5:
        return "1-5"
    if n <= 20:
        return "6-20"
    if n <= 100:
        return "21-100"
    return "100+"


BUCKET_ORDER = ["0", "1-5", "6-20", "21-100", "100+"]


def print_category(name: str, results: List[Tuple[str, str, int]], top: int = 15):
    total = len(results)
    if total == 0:
        print(f"\n=== {name}: 0 terms ===")
        return
    buckets: Counter = Counter()
    sum_occ = 0
    for term, kind, n in results:
        buckets[bucket(n)] += 1
        sum_occ += n
    covered = sum(1 for _, _, n in results if n > 0)
    strong = sum(1 for _, _, n in results if n >= 30)
    print(f"\n=== {name}: {total} terms, "
          f"{covered} present ({100*covered/total:.0f}%), "
          f"{strong} with >=30 occurrences ({100*strong/total:.0f}%) ===")
    print(f"  total occurrences across corpus: {sum_occ:,}")
    print("  distribution:")
    for b in BUCKET_ORDER:
        cnt = buckets.get(b, 0)
        bar = "#" * int(40 * cnt / max(total, 1))
        print(f"    {b:>6s} occ: {cnt:>4d}  {bar}")

    # Best- and worst-covered terms.
    by_occ = sorted(results, key=lambda r: -r[2])
    print(f"  top {top} most-covered:")
    for t, k, n in by_occ[:top]:
        print(f"    {n:>6d}  {t}")
    print(f"  bottom {top} (effectively unseen):")
    for t, k, n in by_occ[-top:]:
        print(f"    {n:>6d}  {t}")


def verdict(category_stats: Dict[str, Tuple[int, int, int]]) -> str:
    """category_stats[name] = (total, covered, strong)"""
    lines: List[str] = []
    lines.append("\n" + "=" * 64)
    lines.append("VERDICT")
    lines.append("=" * 64)
    overall_total = sum(c[0] for c in category_stats.values())
    overall_strong = sum(c[2] for c in category_stats.values())
    overall_covered = sum(c[1] for c in category_stats.values())
    strong_pct = 100 * overall_strong / max(overall_total, 1)
    covered_pct = 100 * overall_covered / max(overall_total, 1)

    lines.append(
        f"Overall: {overall_covered}/{overall_total} terms present "
        f"({covered_pct:.0f}%), {overall_strong} have strong support "
        f"(>=30 occ, {strong_pct:.0f}%)."
    )

    if strong_pct >= 60:
        lines.append("\n  STATUS: GOOD — corpus is usable for decoder LoRA.")
        lines.append("    Training will materially shift the LM prior toward")
        lines.append("    these terms. Expect ~8-15% relative WER improvement")
        lines.append("    on medical content (after merging with Phase 1).")
    elif strong_pct >= 30:
        lines.append("\n  STATUS: PARTIAL — corpus is useful but incomplete.")
        lines.append("    Common drugs/diseases will be learned, but UAE-specific")
        lines.append("    brand names and rarer terms remain unseen. Train on")
        lines.append("    what we have AND add Hindawi books or MoH formulary")
        lines.append("    to close the gap.")
    else:
        lines.append("\n  STATUS: WEAK — corpus alone won't move medical WER much.")
        lines.append("    Too many seed terms have 0-5 occurrences. Either:")
        lines.append("      (a) curated seeds + templates do the heavy lifting,")
        lines.append("          and Wikipedia is just LM-style padding;")
        lines.append("      (b) we need bigger sources (Hindawi, Common Crawl).")

    # Highlight worst category
    worst = min(category_stats.items(), key=lambda kv: kv[1][2] / max(kv[1][0], 1))
    lines.append(f"\n  Weakest category: {worst[0]}  "
                 f"({worst[1][2]}/{worst[1][0]} strong = "
                 f"{100*worst[1][2]/max(worst[1][0],1):.0f}%)")
    best = max(category_stats.items(), key=lambda kv: kv[1][2] / max(kv[1][0], 1))
    lines.append(f"  Strongest category: {best[0]}  "
                 f"({best[1][2]}/{best[1][0]} strong = "
                 f"{100*best[1][2]/max(best[1][0],1):.0f}%)")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "medical_text" / "external")
    ap.add_argument("--seeds-dir", type=Path,
                    default=PROJECT_ROOT / "data" / "medical_text" / "seeds")
    ap.add_argument("--top", type=int, default=10,
                    help="Show top/bottom N terms per category.")
    args = ap.parse_args()

    print(f"[load] corpus from {args.corpus_dir}")
    text = load_corpus_text(args.corpus_dir)
    n_chars = len(text)
    n_words = len(text.split())
    print(f"[load] {n_chars:,} chars / {n_words:,} words "
          f"(~{n_words/1e6:.1f}M tokens estimated)")

    print(f"[load] seed terms from {args.seeds_dir}")
    by_cat = collect_seed_terms(args.seeds_dir)
    for cat, terms in by_cat.items():
        print(f"  {cat:20s} {len(terms)} terms")

    category_stats: Dict[str, Tuple[int, int, int]] = {}
    for cat, terms in by_cat.items():
        results = count_occurrences(text, terms)
        total = len(results)
        covered = sum(1 for _, _, n in results if n > 0)
        strong = sum(1 for _, _, n in results if n >= 30)
        category_stats[cat] = (total, covered, strong)
        print_category(cat, results, top=args.top)

    print(verdict(category_stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
