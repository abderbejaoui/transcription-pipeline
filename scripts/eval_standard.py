"""Industry-standard ASR evaluation, adapted for Arabic Gulf dialect.

This replaces `eval_v2.py` with a pipeline that matches what Whisper,
HuggingFace's Open ASR Leaderboard, and the NADI 2025 Arabic Shared Task
actually use. Everything below comes from real research practice — nothing
invented for this repo.

The four building blocks:

  1. **`BasicMultilingualTextNormalizer`** from openai/whisper. Lowercase,
     NFKC, strip diacritics (Tashkeel), drop punctuation. The de-facto
     standard for non-English ASR evaluation.

  2. **Arabic letter folding** (alef variants → bare alef, yaa → ya, teh
     marbuta → haa, hamza on waw/yaa → bare letter). Standard CAMeL Tools
     practice for dialectal Arabic, where MSA and dialect transcribers
     interchange these letters.

  3. **`num2words(... lang='ar')`** to convert digits in the reference
     into Arabic spelled-out form. Mirrors what the model actually outputs
     and fixes the largest single source of fake WER errors.

  4. **`normalize_compound_pairs`** from HuggingFace Open ASR Leaderboard —
     uses `difflib.SequenceMatcher` to align ref/pred and merges word
     boundaries when content matches up to whitespace. Fixes WorldSpeech's
     split-word references (`ال خليفه` ↔ `الخليفة`).

  5. **`jiwer`** for WER/CER computation (RapidFuzz under the hood — fast,
     well-tested, the standard library).

Output: WER and CER per model, per source, per tier. Headline metric is
CER, as in NADI 2025.

Usage:
    python3 scripts/eval_standard.py
    python3 scripts/eval_standard.py --testset eval/bakeoff_clean
    python3 scripts/eval_standard.py --testset eval/bakeoff_30min --model qwen3-asr-1.7b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jiwer
import num2words

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTSET = PROJECT_ROOT / "eval" / "bakeoff_clean"
# Where cached predictions live. By default we look first inside the test set
# directory ({testset}/bakeoff/predictions/), and fall back to the original
# bakeoff_30min predictions when scoring a curated re-slice (e.g. bakeoff_clean
# which doesn't have its own predictions/).
FALLBACK_PRED_ROOT = PROJECT_ROOT / "eval" / "bakeoff_30min" / "bakeoff" / "predictions"


# ============================================================================
# 1. Whisper BasicMultilingualTextNormalizer
# ============================================================================
# Source: openai/whisper/normalizers/basic.py
# Reproduced verbatim so we don't have to depend on the whisper package.

ADDITIONAL_DIACRITICS = {
    "œ": "oe", "Œ": "OE", "ø": "o", "Ø": "O",
    "æ": "ae", "Æ": "AE", "ß": "ss", "ẞ": "SS",
    "đ": "d", "Đ": "D", "ð": "d", "Ð": "D",
    "þ": "th", "Þ": "th", "ł": "l", "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """Replace all marks/symbols/punctuation with a space; drop diacritics."""
    def replace(c: str) -> str:
        if c in keep:
            return c
        if c in ADDITIONAL_DIACRITICS:
            return ADDITIONAL_DIACRITICS[c]
        cat = unicodedata.category(c)
        if cat == "Mn":           # nonspacing mark (most diacritics)
            return ""
        if cat[0] in "MSP":       # marks, symbols, punctuation
            return " "
        return c
    return "".join(replace(c) for c in unicodedata.normalize("NFKD", s))


def whisper_basic_normalize(s: str) -> str:
    """Equivalent of whisper.normalizers.BasicTextNormalizer(remove_diacritics=True)."""
    if not s:
        return ""
    s = s.lower()
    s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove [tags] and <tags>
    s = re.sub(r"\(([^)]+?)\)", "", s)        # remove (asides)
    s = remove_symbols_and_diacritics(s).lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


# ============================================================================
# 2. Arabic letter folding
# ============================================================================
# Standard CAMeL Tools / Buckwalter folding for evaluation: collapse spelling
# variants that are interchangeable in dialect/MSA writing.

_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def arabic_fold(s: str) -> str:
    """Letter-level folding for dialectal Arabic eval.

    - Tatweel removed
    - Arabic-Indic digits → ASCII
    - أ إ آ ٱ → ا  (alef variants)
    - ى → ي         (alef maksura → yaa)
    - ة → ه         (teh marbuta → haa)
    - ؤ → و, ئ → ي  (hamza-on-waw/yaa → bare letter)
    - ء dropped     (bare hamza)
    """
    if not s:
        return ""
    s = s.replace("\u0640", "")          # tatweel
    s = s.translate(_ARABIC_DIGITS)
    s = re.sub(r"[\u0623\u0625\u0622\u0671]", "\u0627", s)
    s = s.replace("\u0649", "\u064a")
    s = s.replace("\u0629", "\u0647")
    s = s.replace("\u0624", "\u0648").replace("\u0626", "\u064a").replace("\u0621", "")
    return s


# ============================================================================
# 3. Arabic number folding via num2words
# ============================================================================
# Converts digits in the reference to their Arabic spelled-out form, matching
# what models like Qwen3-ASR actually produce. Uses the standard `num2words`
# library so we don't reinvent a buggy converter.

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def fold_arabic_numbers(s: str) -> str:
    """Replace digit sequences with their `num2words` Arabic form."""
    def _replace(m: re.Match) -> str:
        try:
            val = m.group()
            if "." in val:
                return num2words.num2words(float(val), lang="ar")
            return num2words.num2words(int(val), lang="ar")
        except Exception:
            return m.group()
    return _NUMBER_RE.sub(_replace, s)


# ============================================================================
# 4. Compound-word boundary alignment
# ============================================================================
# Source: HuggingFace open_asr_leaderboard / normalizer / eval_utils.py
# When a region of disagreement contains identical characters once whitespace
# is removed, merge both sides to the joined form. This fixes the split-word
# problem in WorldSpeech ("ال خليفه" ↔ "الخليفة") without affecting real errors.

def normalize_compound_pairs(refs: List[str], preds: List[str]) -> Tuple[List[str], List[str]]:
    new_refs, new_preds = [], []
    for ref_text, pred_text in zip(refs, preds):
        ref_words = ref_text.split()
        pred_words = pred_text.split()
        sm = SequenceMatcher(None, ref_words, pred_words)
        nr, np_ = [], []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                nr.extend(ref_words[i1:i2])
                np_.extend(pred_words[j1:j2])
            else:
                rc = "".join(ref_words[i1:i2])
                pc = "".join(pred_words[j1:j2])
                if rc == pc and rc:
                    nr.append(rc)
                    np_.append(pc)
                else:
                    nr.extend(ref_words[i1:i2])
                    np_.extend(pred_words[j1:j2])
        new_refs.append(" ".join(nr))
        new_preds.append(" ".join(np_))
    return new_refs, new_preds


# ============================================================================
# Pipeline: full normalize() and pair-scorer
# ============================================================================


def normalize_for_eval(s: str, fold_numbers: bool = True) -> str:
    """The recommended end-to-end normalizer for dialectal Arabic ASR eval.

    Steps (matches Open ASR Leaderboard + CAMeL):
      1. Whisper Basic (lowercase, NFKD, strip diacritics + punct)
      2. Arabic letter folding (alef/yaa/teh-marbuta/hamza unification)
      3. Arabic number folding (digits → spelled-out via num2words)
      4. Collapse whitespace
    """
    s = whisper_basic_normalize(s)
    s = arabic_fold(s)
    if fold_numbers:
        s = fold_arabic_numbers(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def score_pair(ref: str, hyp: str) -> Tuple[float, float]:
    """Returns (WER, CER) for a single pair using jiwer."""
    if not ref.strip():
        # jiwer raises on empty ref; treat as 1.0 if pred non-empty.
        return (1.0 if hyp.strip() else 0.0,
                1.0 if hyp.strip() else 0.0)
    return jiwer.wer(ref, hyp), jiwer.cer(ref, hyp)


def score_corpus(refs: List[str], preds: List[str]) -> Tuple[float, float]:
    """Corpus-level WER/CER (the way ALL leaderboards report it).

    Corpus WER aggregates edit operations across the whole corpus first,
    then divides — NOT mean-of-per-clip-WER. Tiny clips don't dominate.
    """
    refs = [r if r.strip() else "EMPTY" for r in refs]  # avoid jiwer empties
    preds = [p if p.strip() else "" for p in preds]
    return jiwer.wer(refs, preds), jiwer.cer(refs, preds)


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
    return "other"


def load_manifest(testset_dir: Path) -> List[Dict]:
    path = testset_dir / "manifest.jsonl"
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _pred_root(testset_dir: Path) -> Path:
    """Where to look for cached predictions for `testset_dir`.

    1. If {testset}/bakeoff/predictions/ exists, use it (the normal case for
       a freshly-run test set like fleurs_ar).
    2. Otherwise fall back to eval/bakeoff_30min/bakeoff/predictions — this
       lets us re-score a curated slice (bakeoff_clean) against the original
       cached predictions without copying files.
    """
    own = testset_dir / "bakeoff" / "predictions"
    if own.is_dir():
        return own
    return FALLBACK_PRED_ROOT


def load_predictions(model: str, pred_root: Path) -> Dict[str, str]:
    """Return {clip_id: raw_pred_text} from cached predictions."""
    d = pred_root / model
    out: Dict[str, str] = {}
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
    return f"{v*100:.1f}%" if v is not None else "—"


def evaluate_model(model: str, manifest: List[Dict], pred_root: Path) -> Dict:
    """Apply normalization + compound alignment + corpus scoring."""
    pred_by_id = load_predictions(model, pred_root)

    rows: List[Dict] = []
    refs_n, preds_n = [], []

    for clip in manifest:
        cid = clip["id"]
        if cid not in pred_by_id:
            continue

        ref_raw = clip.get("transcript", "") or ""
        pred_raw = pred_by_id[cid]

        ref_n = normalize_for_eval(ref_raw, fold_numbers=True)
        pred_n = normalize_for_eval(pred_raw, fold_numbers=True)
        refs_n.append(ref_n)
        preds_n.append(pred_n)

        rows.append({
            "id": cid,
            "source": source_of(cid),
            "tier": clip.get("tier", "?"),
            "ref_raw": ref_raw,
            "pred_raw": pred_raw,
            "ref_n": ref_n,
            "pred_n": pred_n,
        })

    # Apply compound-pair alignment to all kept rows
    aligned_refs, aligned_preds = normalize_compound_pairs(refs_n, preds_n)
    for r, ar, ap in zip(rows, aligned_refs, aligned_preds):
        r["ref_aligned"] = ar
        r["pred_aligned"] = ap
        r["wer"], r["cer"] = score_pair(ar, ap)

    # Corpus-level metrics (the standard for leaderboards)
    corpus_wer, corpus_cer = score_corpus(aligned_refs, aligned_preds)

    return {
        "model": model,
        "n": len(rows),
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "rows": rows,
    }


def slice_by(rows: List[Dict], key: str) -> Dict[str, List[Dict]]:
    out: Dict[str, List[Dict]] = {}
    for r in rows:
        out.setdefault(str(r.get(key, "?")), []).append(r)
    return out


def report_slice(name: str, rows: List[Dict]) -> None:
    if not rows:
        return
    refs = [r["ref_aligned"] for r in rows]
    preds = [r["pred_aligned"] for r in rows]
    w, c = score_corpus(refs, preds)
    per_clip_wers = [r["wer"] for r in rows]
    per_clip_cers = [r["cer"] for r in rows]
    print(f"  {name:24s} n={len(rows):3d}  "
          f"corpus WER={_fmt_pct(w)}  CER={_fmt_pct(c)}  "
          f"median CER={_fmt_pct(statistics.median(per_clip_cers))}")


def print_report(result: Dict) -> None:
    print("\n" + "=" * 72)
    print(f"  {result['model']}  (n={result['n']})")
    print("=" * 72)
    print(f"\nCORPUS-LEVEL (industry standard — sum of edits / sum of ref tokens):")
    print(f"  WER = {_fmt_pct(result['corpus_wer'])}")
    print(f"  CER = {_fmt_pct(result['corpus_cer'])}    ← headline metric for Arabic")

    rows = result["rows"]
    print(f"\nPer source:")
    for src in sorted(set(r["source"] for r in rows)):
        report_slice(src, [r for r in rows if r["source"] == src])

    tiers = sorted(set(r["tier"] for r in rows))
    if tiers and tiers != ["?"]:
        print(f"\nPer tier:")
        for tier in tiers:
            report_slice(f"tier={tier}", [r for r in rows if r["tier"] == tier])


def print_sample(result: Dict, n: int = 3) -> None:
    """Show how normalization actually changes the strings."""
    rows = result["rows"]
    if not rows:
        return
    rows_sorted = sorted(rows, key=lambda r: r["cer"])
    print("\n" + "-" * 72)
    print(f"  3 best CER samples (showing all 4 stages of normalization)")
    print("-" * 72)
    for r in rows_sorted[:3]:
        print(f"\n[{r['id']}] WER={_fmt_pct(r['wer'])} CER={_fmt_pct(r['cer'])} src={r['source']}")
        print(f"  raw ref     : {r['ref_raw']}")
        print(f"  raw pred    : {r['pred_raw']}")
        print(f"  normed ref  : {r['ref_n']}")
        print(f"  normed pred : {r['pred_n']}")
        print(f"  aligned ref : {r['ref_aligned']}")
        print(f"  aligned pred: {r['pred_aligned']}")

    print("\n" + "-" * 72)
    print(f"  3 worst CER samples")
    print("-" * 72)
    for r in rows_sorted[-3:]:
        print(f"\n[{r['id']}] WER={_fmt_pct(r['wer'])} CER={_fmt_pct(r['cer'])} src={r['source']}")
        print(f"  raw ref     : {r['ref_raw']}")
        print(f"  raw pred    : {r['pred_raw']}")
        print(f"  aligned ref : {r['ref_aligned']}")
        print(f"  aligned pred: {r['pred_aligned']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", type=Path, default=DEFAULT_TESTSET,
                    help="Directory containing manifest.jsonl. "
                         f"Default: {DEFAULT_TESTSET.relative_to(PROJECT_ROOT)}")
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
    print(f"\nNormalization pipeline:")
    print(f"  1. Whisper BasicMultilingualTextNormalizer")
    print(f"  2. Arabic letter folding (alef/yaa/teh-marbuta/hamza)")
    print(f"  3. Arabic number folding (digits → words via num2words)")
    print(f"  4. Compound-pair alignment (fixes split-word artefacts)")
    print(f"  5. WER/CER via jiwer (corpus-level + per-clip)")

    summary: Dict[str, Dict] = {}
    for m in models:
        result = evaluate_model(m, manifest, pred_root)
        print_report(result)
        if args.samples:
            print_sample(result)
        summary[m] = {"corpus_wer": result["corpus_wer"],
                      "corpus_cer": result["corpus_cer"],
                      "n": result["n"]}

    # Leaderboard-style summary
    print("\n" + "=" * 72)
    print("  HEADLINE — corpus-level CER (lower is better, NADI 2025 convention)")
    print("=" * 72)
    print(f"{'model':28s} {'n':>5s}  {'WER':>8s}  {'CER':>8s}")
    print("-" * 72)
    for m, s in sorted(summary.items(), key=lambda kv: kv[1]["corpus_cer"]):
        print(f"{m:28s} {s['n']:5d}  {_fmt_pct(s['corpus_wer']):>8s}  {_fmt_pct(s['corpus_cer']):>8s}")


if __name__ == "__main__":
    main()
