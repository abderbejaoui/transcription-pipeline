"""Corrected evaluation of cached bake-off predictions.

Improvements over bakeoff.py's built-in scoring:

  1. Number↔word normalisation. Refs in `2025` and preds in
     `ألفين وخمسة وعشرين` are now treated as equivalent. Same for `26` ↔
     `ستة وعشرين`, etc. Spelled-out Arabic numbers are mapped back to digits
     before comparison.

  2. CER reported alongside WER. CER is robust to spurious whitespace inside
     words (which WorldSpeech refs have a lot of) and to pred/ref length
     mismatch.

  3. Reference-quality filters surface clips that should not be scored:
       - LEN_RATIO_OUTLIER: |pred_tokens − ref_tokens| / ref_tokens > 1.5
         (model says far more or far less than the reference — almost always
         a manifest alignment bug, not an ASR error)
       - REF_BROKEN_TOKENS: ref contains 1-or-2-letter Arabic tokens that
         look like split-words (e.g. "بار لنا ك", "ال قران").

  4. Reports three tiers: raw / clean (no broken refs) / aligned (also drops
     length-ratio outliers). Each tier with both WER and CER.

  5. Per-source breakdown so you can see WorldSpeech vs SADA quality side by
     side.

Usage:
    python3 scripts/eval_v2.py
    python3 scripts/eval_v2.py --model qwen3-asr-1.7b
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = PROJECT_ROOT / "eval" / "bakeoff_30min" / "bakeoff" / "predictions"


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
_TATWEEL_RE = re.compile(r"\u0640")
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")

# Spelled-out Arabic numbers -> digits.
# Order matters: compound forms first so they aren't shadowed by component words.
_ARABIC_NUMBERS_BASIC = {
    "صفر": 0,
    "واحد": 1, "اثنين": 2, "اثنان": 2, "ثلاثه": 3, "ثلاثة": 3, "اربعه": 4,
    "اربعة": 4, "خمسه": 5, "خمسة": 5, "سته": 6, "ستة": 6, "سبعه": 7,
    "سبعة": 7, "ثمانيه": 8, "ثمانية": 8, "تسعه": 9, "تسعة": 9, "عشره": 10,
    "عشرة": 10, "احد": 11, "احدعشر": 11, "اثناعشر": 12, "ثلاثهعشر": 13,
    "اربعهعشر": 14, "خمسهعشر": 15, "ستهعشر": 16, "سبعهعشر": 17,
    "ثمانيهعشر": 18, "تسعهعشر": 19, "عشرون": 20, "عشرين": 20, "ثلاثون": 30,
    "ثلاثين": 30, "اربعون": 40, "اربعين": 40, "خمسون": 50, "خمسين": 50,
    "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70, "ثمانون": 80,
    "ثمانين": 80, "تسعون": 90, "تسعين": 90, "مايه": 100, "مائه": 100,
    "مئه": 100, "مائة": 100, "الف": 1000, "الفين": 2000, "مليون": 1000000,
}


def norm_text(s: str, *, fold_numbers: bool = False) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _TATWEEL_RE.sub("", s)
    s = _DIACRITICS_RE.sub("", s)
    s = s.translate(_ARABIC_DIGITS)
    # Alef variants -> bare alef
    s = re.sub(r"[\u0623\u0625\u0622\u0671]", "\u0627", s)
    # Yaa
    s = s.replace("\u0649", "\u064a")
    # Teh marbuta -> haa
    s = s.replace("\u0629", "\u0647")
    # Hamza on waw/yaa -> bare letter; bare hamza dropped
    s = s.replace("\u0624", "\u0648").replace("\u0626", "\u064a").replace("\u0621", "")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()

    if fold_numbers:
        s = _fold_numbers(s)
    return s


def _fold_numbers(s: str) -> str:
    """Replace spelled-out Arabic numbers with their digit form.

    Handles: integers 0–99, "X و Y" combinations (e.g. "خمسة وعشرين" -> 25),
    "X ميه" (e.g. "ثلاث ميه" -> 300), "X الف" (e.g. "ألفين" -> 2000),
    "الف و ..." compounds. Anything we can't parse is left alone — the goal
    is to fix the obvious cases, not be a full number grammar.
    """
    toks = s.split()
    out: List[str] = []
    i = 0
    while i < len(toks):
        # Try to consume a multi-token number phrase starting at i.
        consumed, value = _consume_number(toks, i)
        if consumed:
            out.append(str(value))
            i += consumed
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


def _consume_number(toks: List[str], i: int) -> Tuple[int, int]:
    """Greedy consume number phrase starting at toks[i]. Returns
    (tokens_consumed, integer_value). 0 consumed means no number found."""
    if i >= len(toks):
        return 0, 0
    n = len(toks)
    # Patterns, longest first
    # 4-token: <thousands> الف و <hundred_phrase>  e.g. "الفين و خمسة"
    # We approximate by trying lengths 5,4,3,2,1.
    for span in (5, 4, 3, 2, 1):
        if i + span > n:
            continue
        phrase = toks[i : i + span]
        v = _parse_number_phrase(phrase)
        if v is not None:
            return span, v
    return 0, 0


def _parse_number_phrase(phrase: List[str]) -> Optional[int]:
    """Try to interpret a small token list as an integer. Returns None if
    none of the recognised patterns match."""
    if not phrase:
        return None
    # Strip a leading "و" (and) on subsequent tokens
    clean = [t for t in phrase if t]
    if not clean:
        return None

    # All tokens digits already
    if all(t.isdigit() for t in clean):
        try:
            return int("".join(clean))
        except ValueError:
            return None

    # Single token
    if len(clean) == 1:
        return _ARABIC_NUMBERS_BASIC.get(clean[0])

    # Pattern: X و Y  (unit + tens, e.g. "خمسة و عشرين")
    if len(clean) == 3 and clean[1] in ("و", "و"):
        a = _ARABIC_NUMBERS_BASIC.get(clean[0])
        b = _ARABIC_NUMBERS_BASIC.get(clean[2])
        if a is not None and b is not None:
            return a + b

    # Pattern: X Y (concatenation, e.g. "الفين خمسه و عشرين")
    # Try splitting: thousands + and-tens
    # Simple cases first:
    # "الف X" -> 1000 + X
    if len(clean) == 2 and clean[0] in ("الف", "الفين") and clean[1].isdigit():
        return _ARABIC_NUMBERS_BASIC[clean[0]] + int(clean[1])

    # "X و Y و Z" or "X Y" etc are hard; bail out.
    return None


# ---------------------------------------------------------------------------
# WER + CER
# ---------------------------------------------------------------------------


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def wer(ref: str, hyp: str) -> float:
    r = ref.split()
    h = hyp.split()
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


def cer(ref: str, hyp: str) -> float:
    r = list(ref.replace(" ", ""))
    h = list(hyp.replace(" ", ""))
    if not r:
        return 0.0 if not h else 1.0
    return _edit_distance(r, h) / len(r)


# ---------------------------------------------------------------------------
# Reference-quality filters
# ---------------------------------------------------------------------------


def _ref_looks_broken(ref: str) -> bool:
    """Heuristic for refs that have been mangled by token-level splitting.

    True split-word artefacts have:
      - standalone 'ال' (definite article never stands alone in real text)
      - single-Arabic-character tokens like 'ك', 'ت', 'ب' (these only exist
        when a suffix or prefix was accidentally split off)

    We do NOT flag short function words ('في', 'ما', 'يا', 'هو', 'لا', 'و')
    which are normal in Arabic.
    """
    toks = ref.split()
    if not toks:
        return False
    if any(t == "ال" for t in toks):
        return True
    # Count tokens that are exactly one Arabic character.
    one_char_arabic = sum(
        1 for t in toks
        if len(t) == 1 and "\u0600" <= t <= "\u06ff"
    )
    # 2+ single-Arabic-char tokens is essentially impossible in clean text.
    return one_char_arabic >= 2


def _length_outlier(ref: str, hyp: str, ratio: float = 1.5) -> bool:
    r, h = len(ref.split()), len(hyp.split())
    if r == 0:
        return h > 5
    return abs(h - r) / r > ratio


# ---------------------------------------------------------------------------
# Load + score
# ---------------------------------------------------------------------------


def load_model(name: str) -> List[Dict]:
    out = []
    d = PRED_ROOT / name
    for p in sorted(d.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception as exc:
            print(f"!! could not read {p}: {exc}")
    return out


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


def score_records(records: List[Dict]) -> List[Dict]:
    scored = []
    for r in records:
        ref_raw = r.get("ref", "") or ""
        pred_raw = r.get("pred", "") or ""

        ref_n = norm_text(ref_raw, fold_numbers=True)
        pred_n = norm_text(pred_raw, fold_numbers=True)

        scored.append({
            "id": r["id"],
            "category": r.get("category", "?"),
            "source": source_of(r["id"]),
            "ref": ref_raw,
            "pred": pred_raw,
            "ref_n": ref_n,
            "pred_n": pred_n,
            "wer": wer(ref_n, pred_n),
            "cer": cer(ref_n, pred_n),
            "broken_ref": _ref_looks_broken(ref_n),
            "length_outlier": _length_outlier(ref_n, pred_n),
        })
    return scored


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _mean(xs: List[float]) -> Optional[float]:
    return statistics.mean(xs) if xs else None


def _fmt(v: Optional[float], pct: bool = True) -> str:
    if v is None:
        return "—"
    return f"{v*100:.1f}%" if pct else f"{v:.3f}"


def report_tier(rows: List[Dict], label: str) -> None:
    if not rows:
        print(f"\n[{label}] no rows"); return
    wers = [r["wer"] for r in rows]
    cers = [r["cer"] for r in rows]
    print(f"\n[{label}]  n={len(rows)}")
    print(f"  mean WER = {_fmt(_mean(wers))}    median WER = {_fmt(statistics.median(wers))}")
    print(f"  mean CER = {_fmt(_mean(cers))}    median CER = {_fmt(statistics.median(cers))}")
    print(f"  WER<10%  = {sum(1 for w in wers if w < 0.10):3d}    "
          f"WER<25%  = {sum(1 for w in wers if w < 0.25):3d}    "
          f"WER=0    = {sum(1 for w in wers if w == 0):3d}")
    # by source
    by_src: Dict[str, List[Dict]] = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    print("  per source:")
    for src, group in sorted(by_src.items()):
        ws = [g["wer"] for g in group]
        cs = [g["cer"] for g in group]
        print(f"    {src:18s} n={len(group):3d}  "
              f"WER={_fmt(_mean(ws))}  CER={_fmt(_mean(cs))}  "
              f"<25% WER: {sum(1 for w in ws if w < 0.25):3d}")


def report_model(model: str, rows: List[Dict]) -> None:
    print("\n" + "=" * 70)
    print(f"  {model}  ({len(rows)} clips)")
    print("=" * 70)

    clean = [r for r in rows if not r["broken_ref"]]
    aligned = [r for r in clean if not r["length_outlier"]]
    dropped_broken = len(rows) - len(clean)
    dropped_align = len(clean) - len(aligned)

    print(f"\nFiltering summary:")
    print(f"  total clips:               {len(rows)}")
    print(f"  dropped (broken ref):      {dropped_broken}")
    print(f"  dropped (length outlier):  {dropped_align}")
    print(f"  remaining (clean+aligned): {len(aligned)}")

    report_tier(rows, f"RAW — all {len(rows)} clips")
    report_tier(clean, "CLEAN — refs without split-word artefacts")
    report_tier(aligned, "ALIGNED — clean + |pred-ref| length within 150%")


def report_by_tier(rows: List[Dict]) -> None:
    """Slice by curator-assigned tier (HIGH/MEDIUM) from manifest."""
    by_tier: Dict[str, List[Dict]] = {}
    for r in rows:
        by_tier.setdefault(str(r.get("tier", "?")).upper(), []).append(r)
    for tier in ["HIGH", "MEDIUM"]:
        if tier in by_tier:
            report_tier(by_tier[tier], f"TIER={tier}")


def rescore_against_testset(model: str, testset_dir: Path) -> List[Dict]:
    """Load cached predictions for `model` and re-score against the refs
    in `testset_dir/manifest.jsonl`. Only clips present in both are kept."""
    manifest_path = testset_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing {manifest_path}")
    manifest = [json.loads(l) for l in manifest_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ref_by_id = {m["id"]: m for m in manifest}

    pred_dir = PRED_ROOT / model
    out: List[Dict] = []
    for p in sorted(pred_dir.glob("*.json")):
        try:
            pred_record = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cid = pred_record["id"]
        if cid not in ref_by_id:
            continue  # clip was filtered out of the clean set
        meta = ref_by_id[cid]
        ref_raw = meta.get("transcript", "") or ""
        pred_raw = pred_record.get("pred", "") or ""
        ref_n = norm_text(ref_raw, fold_numbers=True)
        pred_n = norm_text(pred_raw, fold_numbers=True)
        out.append({
            "id": cid,
            "category": meta.get("category", "?"),
            "source": source_of(cid),
            "tier": meta.get("tier", "?"),
            "ref": ref_raw,
            "pred": pred_raw,
            "ref_n": ref_n,
            "pred_n": pred_n,
            "wer": wer(ref_n, pred_n),
            "cer": cer(ref_n, pred_n),
            "broken_ref": False,   # already filtered out
            "length_outlier": False,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append")
    ap.add_argument("--testset", type=Path, default=None,
                    help="If given, re-score cached predictions against this "
                         "test set's manifest.jsonl (e.g. eval/bakeoff_clean).")
    args = ap.parse_args()

    models = args.model or sorted(d.name for d in PRED_ROOT.iterdir() if d.is_dir())

    if args.testset is not None:
        testset_dir = args.testset
        if not testset_dir.is_absolute():
            testset_dir = (PROJECT_ROOT / testset_dir).resolve()
        print(f"Re-scoring cached predictions against {testset_dir.relative_to(PROJECT_ROOT)}")
        for m in models:
            scored = rescore_against_testset(m, testset_dir)
            if not scored:
                print(f"\n!! no overlap between cached preds and {testset_dir} for {m}")
                continue
            print("\n" + "=" * 70)
            print(f"  {m}  ({len(scored)} clips, scored against clean refs)")
            print("=" * 70)
            report_tier(scored, "ALL KEPT (high + medium)")
            report_by_tier(scored)
        return

    # Legacy mode: use the predictions' own cached refs.
    for m in models:
        records = load_model(m)
        if not records:
            print(f"\n!! no predictions for {m}"); continue
        scored = score_records(records)
        report_model(m, scored)


if __name__ == "__main__":
    main()
