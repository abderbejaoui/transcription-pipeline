"""Arabic ASR evaluation that exactly matches the published methodology.

This script reproduces the normalization and scoring used by the
**Open Universal Arabic ASR Leaderboard** (Wang et al. 2024, arXiv:2412.13788),
which is THE public benchmark for Arabic ASR. Numbers produced by this
script can be directly compared to:

  * Qwen3-ASR-1.7B published WER/CER
  * Whisper-large-v3 published WER/CER
  * MMS, SeamlessM4T, omniASR-LLM, etc.

Why not eval_standard.py?
-------------------------
eval_standard.py used several normalization steps that DON'T match the
public leaderboard:
  - teh marbuta folding ة→ه: not in the standard normalizer
  - num2words digit folding: refs already use digits, this hurts
  - normalize_compound_pairs: HF English-leaderboard utility, not used
    for Arabic in published work

Those steps gave us misleadingly good numbers. This script uses the EXACT
normalizer from the leaderboard's eval.py (verbatim copy, attributed below).

Two metrics, both computed corpus-level via jiwer:
  - WER (primary in most Arabic ASR papers)
  - CER (more robust to dialect spelling variation; NADI 2025 convention)

Usage:
  python3 scripts/eval_arabic.py
  python3 scripts/eval_arabic.py --testset eval/casablanca_UAE
  python3 scripts/eval_arabic.py --testset eval/bakeoff_30min --model qwen3-asr-1.7b

Reference attribution
---------------------
The `normalize_arabic_text` function below is taken verbatim from:
  Wang, Alhmoud, Alqurishi (2024). Open Universal Arabic ASR Leaderboard.
  arXiv:2412.13788. https://github.com/Natural-Language-Processing-Elm/
  open_universal_arabic_asr_leaderboard/blob/main/eval.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTSET = PROJECT_ROOT / "eval" / "bakeoff_clean"
FALLBACK_PRED_ROOT = PROJECT_ROOT / "eval" / "bakeoff_30min" / "bakeoff" / "predictions"


# ============================================================================
# The official Arabic ASR normalizer
# ============================================================================
# Verbatim from Wang et al. 2024, modulo a small `text = ""` guard. This is
# THE function used to rank Qwen3-ASR-1.7B at WER=33.36% / CER=12.33%.

def normalize_arabic_text(text: str) -> str:
    """Arabic text normalization (Open Universal Arabic ASR Leaderboard).

    1. Remove punctuation (Latin + Arabic ، ؛ ؟)
    2. Remove diacritics (Fatha, Damma, ...)
    3. Map Persian-style letters پ→ب, ڤ→ف
    4. Hamza/madda variants → bare letter
       (آ, أ, إ → ا   ؤ → و   ئ → ي   ء dropped)
    5. Eastern Arabic numerals → Western (٠-٩ → 0-9)
    """
    if text is None:
        return ""
    # 1. Remove punctuation
    punctuation = r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~،؛؟]'
    text = re.sub(punctuation, "", text)

    # 2. Remove diacritics (Arabic diacritical marks)
    diacritics = r"[\u064B-\u0652]"
    text = re.sub(diacritics, "", text)

    # 3. Persian-style letters
    text = re.sub("پ", "ب", text)
    text = re.sub("ڤ", "ف", text)

    # 4. Hamza/madda variants
    text = re.sub(r"[آ]", "ا", text)
    text = re.sub(r"[أإ]", "ا", text)
    text = re.sub(r"[ؤ]", "و", text)
    text = re.sub(r"[ئ]", "ي", text)
    text = re.sub(r"[ء]", "", text)

    # 5. Eastern Arabic to Western Arabic numerals
    eastern_to_western = {
        "٠": "0", "١": "1", "٢": "2", "٣": "3", "٤": "4",
        "٥": "5", "٦": "6", "٧": "7", "٨": "8", "٩": "9",
    }
    for east, west in eastern_to_western.items():
        text = text.replace(east, west)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ============================================================================
# Scoring
# ============================================================================

def score_corpus(refs: List[str], preds: List[str]) -> Tuple[float, float]:
    """Corpus-level WER and CER via jiwer. Empty refs are dropped (jiwer
    would raise otherwise); empty preds against non-empty refs count as
    full deletion."""
    safe_refs, safe_preds = [], []
    for r, p in zip(refs, preds):
        if not r.strip():
            continue
        safe_refs.append(r)
        safe_preds.append(p if p.strip() else " ")
    if not safe_refs:
        return 0.0, 0.0
    w = jiwer.wer(safe_refs, safe_preds)
    c = jiwer.cer(safe_refs, safe_preds)
    return w, c


def score_per_clip(ref: str, hyp: str) -> Tuple[float, float]:
    """Per-clip WER/CER. Empty ref ⇒ 1.0 if hyp non-empty, else 0.0."""
    if not ref.strip():
        return (1.0 if hyp.strip() else 0.0,
                1.0 if hyp.strip() else 0.0)
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


# ============================================================================
# Loading
# ============================================================================

def source_of(clip_id: str) -> str:
    if clip_id.startswith("sada_"):
        return "SADA22"
    if "_ar_sa_" in clip_id:
        return "WorldSpeech-SA"
    if "_ar_kw_" in clip_id:
        return "WorldSpeech-KW"
    if "_ar_bh_" in clip_id:
        return "WorldSpeech-BH"
    if clip_id.startswith("casablanca_"):
        parts = clip_id.split("_")
        if len(parts) >= 2:
            return f"Casablanca-{parts[1].upper()}"
        return "Casablanca"
    if clip_id.startswith("fleurs_"):
        return "FLEURS"
    return "other"


def load_manifest(testset_dir: Path) -> List[Dict]:
    path = testset_dir / "manifest.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pred_root(testset_dir: Path) -> Path:
    own = testset_dir / "bakeoff" / "predictions"
    if own.is_dir():
        return own
    return FALLBACK_PRED_ROOT


def load_predictions(model: str, pred_root: Path) -> Dict[str, str]:
    d = pred_root / model
    out: Dict[str, str] = {}
    if not d.is_dir():
        return out
    for p in d.glob("*.json"):
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
            out[rec["id"]] = rec.get("pred", "")
        except Exception:
            pass
    return out


def list_models(pred_root: Path) -> List[str]:
    if not pred_root.is_dir():
        return []
    return sorted(d.name for d in pred_root.iterdir() if d.is_dir())


# ============================================================================
# Reporting
# ============================================================================

def _fmt_pct(v: Optional[float]) -> str:
    return f"{v*100:.2f}%" if v is not None else "—"


def evaluate_model(model: str, manifest: List[Dict], pred_root: Path) -> Dict:
    pred_by_id = load_predictions(model, pred_root)
    rows: List[Dict] = []
    refs_n, preds_n = [], []

    for clip in manifest:
        cid = clip["id"]
        if cid not in pred_by_id:
            continue
        ref_raw = clip.get("transcript", "") or ""
        pred_raw = pred_by_id[cid]
        ref_n = normalize_arabic_text(ref_raw)
        pred_n = normalize_arabic_text(pred_raw)
        refs_n.append(ref_n)
        preds_n.append(pred_n)
        w, c = score_per_clip(ref_n, pred_n)
        rows.append({
            "id": cid,
            "source": source_of(cid),
            "tier": clip.get("tier", "?"),
            "ref_raw": ref_raw,
            "pred_raw": pred_raw,
            "ref_n": ref_n,
            "pred_n": pred_n,
            "wer": w,
            "cer": c,
        })

    corpus_wer, corpus_cer = score_corpus(refs_n, preds_n)
    return {
        "model": model,
        "n": len(rows),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "rows": rows,
    }


def report_slice(name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    refs = [r["ref_n"] for r in rows]
    preds = [r["pred_n"] for r in rows]
    w, c = score_corpus(refs, preds)
    per_clip_cers = [r["cer"] for r in rows]
    print(f"  {name:24s} n={len(rows):4d}  "
          f"corpus WER={_fmt_pct(w)}  CER={_fmt_pct(c)}  "
          f"median-clip CER={_fmt_pct(statistics.median(per_clip_cers))}")


def report_model(result: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"  {result['model']}  (n={result['n']})")
    print("=" * 72)
    print(f"\nCORPUS-LEVEL (sum of edits / sum of ref tokens — leaderboard convention):")
    print(f"  WER = {_fmt_pct(result['corpus_wer'])}")
    print(f"  CER = {_fmt_pct(result['corpus_cer'])}")

    rows = result["rows"]
    if rows:
        print(f"\nPer source:")
        for src in sorted(set(r["source"] for r in rows)):
            report_slice(src, [r for r in rows if r["source"] == src])

        tiers = sorted(set(r["tier"] for r in rows) - {"?"})
        if tiers:
            print(f"\nPer tier:")
            for tier in tiers:
                report_slice(f"tier={tier}", [r for r in rows if r["tier"] == tier])


def print_samples(result: Dict, n: int = 3) -> None:
    rows = result["rows"]
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["cer"])
    print("\n" + "-" * 72)
    print(f"  TOP {n} clips (lowest CER)")
    print("-" * 72)
    for r in rows_sorted[:n]:
        print(f"\n[{r['id']}]  CER={_fmt_pct(r['cer'])}  WER={_fmt_pct(r['wer'])}")
        print(f"  ref : {r['ref_raw']}")
        print(f"  pred: {r['pred_raw']}")

    print("\n" + "-" * 72)
    print(f"  BOTTOM {n} clips (highest CER)")
    print("-" * 72)
    for r in rows_sorted[-n:]:
        print(f"\n[{r['id']}]  CER={_fmt_pct(r['cer'])}  WER={_fmt_pct(r['wer'])}")
        print(f"  ref : {r['ref_raw']}")
        print(f"  pred: {r['pred_raw']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET,
                    help=f"Test set directory. Default: {DEFAULT_TESTSET.relative_to(PROJECT_ROOT)}")
    ap.add_argument("--model", action="append", help="Model(s) to score. Default: all.")
    ap.add_argument("--samples", action="store_true",
                    help="Show 3 best + 3 worst clips per model.")
    args = ap.parse_args()

    testset = args.testset if args.testset.is_absolute() else (PROJECT_ROOT / args.testset).resolve()
    pred_root = _pred_root(testset)
    print(f"Testset      : {testset.relative_to(PROJECT_ROOT)}")
    print(f"Predictions  : {pred_root.relative_to(PROJECT_ROOT)}")
    manifest = load_manifest(testset)
    print(f"Manifest     : {len(manifest)} clips")

    models = args.model or list_models(pred_root)
    if not models:
        print(f"\n!! no predictions found under {pred_root}", file=sys.stderr)
        return
    print(f"Models       : {', '.join(models)}")
    print(f"\nNormalization: Open Universal Arabic ASR Leaderboard (Wang et al. 2024)")
    print(f"  1. Strip punctuation (Latin + Arabic)")
    print(f"  2. Strip Tashkeel diacritics")
    print(f"  3. Persian-style letters → Arabic (پ→ب, ڤ→ف)")
    print(f"  4. Hamza/madda variants → bare letter (آأإ→ا, ؤ→و, ئ→ي, ء→drop)")
    print(f"  5. Eastern Arabic numerals → Western")
    print(f"\nScoring      : jiwer corpus-level WER + CER")

    summary: Dict[str, Dict] = {}
    for m in models:
        result = evaluate_model(m, manifest, pred_root)
        report_model(result)
        if args.samples:
            print_samples(result)
        summary[m] = {"corpus_wer": result["corpus_wer"],
                      "corpus_cer": result["corpus_cer"],
                      "n": result["n"]}

    if len(summary) > 1:
        print("\n" + "=" * 72)
        print("  LEADERBOARD (sorted by corpus CER)")
        print("=" * 72)
        print(f"{'model':28s} {'n':>5s}  {'WER':>10s}  {'CER':>10s}")
        print("-" * 72)
        for m, s in sorted(summary.items(), key=lambda kv: kv[1]["corpus_cer"]):
            print(f"{m:28s} {s['n']:5d}  {_fmt_pct(s['corpus_wer']):>10s}  {_fmt_pct(s['corpus_cer']):>10s}")


if __name__ == "__main__":
    main()
