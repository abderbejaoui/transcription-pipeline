"""Audit cached bake-off predictions to find scoring problems.

Reads JSON predictions from eval/bakeoff_30min/bakeoff/predictions/<model>/,
recomputes WER under several normalisation schemes, prints histograms and
worst-offenders, and surfaces clips whose `pred` and `ref` look identical
to a human but still score >0.

Usage:
    python -m scripts.audit_predictions
    python -m scripts.audit_predictions --model qwen3-asr-1.7b
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_ROOT = PROJECT_ROOT / "eval" / "bakeoff_30min" / "bakeoff" / "predictions"


# ---------------------------------------------------------------------------
# Normalisers (graduated strictness)
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")
_TATWEEL_RE = re.compile(r"\u0640")
# Arabic-Indic digits -> ASCII
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def norm_basic(s: str) -> str:
    """What bakeoff.py currently applies (NFKC + diacritics + punct)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_arabic_aware(s: str) -> str:
    """norm_basic + alef/yaa/teh marbuta/hamza unification + digit map + tatweel."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _TATWEEL_RE.sub("", s)
    s = _DIACRITICS_RE.sub("", s)
    s = s.translate(_ARABIC_DIGITS)
    # Alef variants -> bare alef
    s = re.sub(r"[\u0623\u0625\u0622\u0671]", "\u0627", s)  # أ إ آ ٱ -> ا
    # Yaa
    s = s.replace("\u0649", "\u064a")  # ى -> ي
    # Teh marbuta
    s = s.replace("\u0629", "\u0647")  # ة -> ه
    # Hamza on waw / yaa -> bare letter; bare hamza dropped
    s = s.replace("\u0624", "\u0648").replace("\u0626", "\u064a").replace("\u0621", "")
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


def norm_aggressive(s: str) -> str:
    """norm_arabic_aware + remove definite article 'ال' + collapse repeated chars."""
    s = norm_arabic_aware(s)
    if not s:
        return ""
    # Strip leading 'ال' (definite article) from every token
    s = " ".join(re.sub(r"^ال", "", tok) if len(tok) > 3 else tok for tok in s.split())
    return s


# ---------------------------------------------------------------------------
# WER + edit ops (so we can see substitutions / insertions / deletions)
# ---------------------------------------------------------------------------


def wer_ops(ref: str, hyp: str) -> Tuple[float, int, int, int, int]:
    """Returns (wer, substitutions, deletions, insertions, ref_len)."""
    r = ref.split()
    h = hyp.split()
    n, m = len(r), len(h)
    if n == 0:
        return (0.0 if m == 0 else 1.0, 0, 0, m, 0)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    op = [[""] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i; op[i][0] = "D"
    for j in range(m + 1):
        dp[0][j] = j; op[0][j] = "I"
    op[0][0] = ""
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if r[i - 1] == h[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]; op[i][j] = "M"
            else:
                sub = dp[i - 1][j - 1] + 1
                dele = dp[i - 1][j] + 1
                ins = dp[i][j - 1] + 1
                best = min(sub, dele, ins)
                dp[i][j] = best
                op[i][j] = "S" if best == sub else ("D" if best == dele else "I")
    # backtrack
    i, j = n, m
    subs = dels = ins = 0
    while i > 0 or j > 0:
        o = op[i][j]
        if o == "M":
            i -= 1; j -= 1
        elif o == "S":
            subs += 1; i -= 1; j -= 1
        elif o == "D":
            dels += 1; i -= 1
        elif o == "I":
            ins += 1; j -= 1
        else:
            break
    return (dp[n][m] / n, subs, dels, ins, n)


# ---------------------------------------------------------------------------
# Load & score
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


def histogram(values: List[float], buckets: List[float]) -> List[Tuple[str, int]]:
    counts = [0] * (len(buckets) + 1)
    for v in values:
        placed = False
        for i, b in enumerate(buckets):
            if v <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1
    labels = []
    prev = 0.0
    for i, b in enumerate(buckets):
        labels.append((f"{prev:.2f}–{b:.2f}", counts[i]))
        prev = b
    labels.append((f">{buckets[-1]:.2f}", counts[-1]))
    return labels


def score_with(records: List[Dict], normalizer) -> List[Dict]:
    out = []
    for r in records:
        ref_n = normalizer(r.get("ref", "") or "")
        hyp_n = normalizer(r.get("pred", "") or "")
        w, subs, dels, ins, n = wer_ops(ref_n, hyp_n)
        out.append({**r, "_ref_n": ref_n, "_hyp_n": hyp_n,
                    "_wer": w, "_subs": subs, "_dels": dels,
                    "_ins": ins, "_ref_len": n})
    return out


def summarise(rows: List[Dict], label: str) -> None:
    wers = [r["_wer"] for r in rows]
    by_cat: Dict[str, List[float]] = {}
    for r in rows:
        by_cat.setdefault(r.get("category", "?"), []).append(r["_wer"])
    print(f"\n--- {label} ---")
    print(f"n={len(rows)}  mean WER={statistics.mean(wers):.4f}  "
          f"median={statistics.median(wers):.4f}  "
          f"perfect(WER=0)={sum(1 for w in wers if w == 0)}  "
          f"sub-10%={sum(1 for w in wers if w < 0.10)}  "
          f"sub-25%={sum(1 for w in wers if w < 0.25)}")
    for cat, vs in sorted(by_cat.items()):
        print(f"  {cat:20s} n={len(vs):3d}  mean={statistics.mean(vs):.4f}  "
              f"median={statistics.median(vs):.4f}  "
              f"perfect={sum(1 for w in vs if w == 0):2d}  "
              f"sub-10%={sum(1 for w in vs if w < 0.10):2d}")
    print("  histogram (mean WER buckets):")
    for lbl, ct in histogram(wers, [0.0, 0.05, 0.10, 0.20, 0.40, 0.70, 1.0]):
        bar = "█" * ct
        print(f"    {lbl:>12s} | {ct:3d} {bar}")


# ---------------------------------------------------------------------------
# Pathology detectors
# ---------------------------------------------------------------------------


def find_pathologies(rows: List[Dict], n: int = 10) -> None:
    print("\n--- pathology scan (top 10 of each) ---")

    # 1) Cases where pred == ref under aggressive norm but WER > 0 under basic
    bad_norm = []
    for r in rows:
        agg = norm_aggressive(r.get("ref", "") or "")
        agg_h = norm_aggressive(r.get("pred", "") or "")
        if agg == agg_h and r["_wer"] > 0:
            bad_norm.append(r)
    print(f"\n[1] WER>0 but agg-norm equal: {len(bad_norm)} clips  (these are pure normalisation artefacts)")
    for r in bad_norm[:n]:
        print(f"  {r['id']:14s} WER={r['_wer']:.3f}")
        print(f"    ref : {r['ref']}")
        print(f"    pred: {r['pred']}")

    # 2) Insertion-heavy (pred much longer than ref) — model rambles
    inserts = sorted(rows, key=lambda r: -(r["_ins"] - r["_dels"]))[:n]
    print(f"\n[2] Most insertion-heavy (model adds tokens):")
    for r in inserts[:n]:
        delta = r["_ins"] - r["_dels"]
        if delta <= 0:
            continue
        print(f"  {r['id']:14s} WER={r['_wer']:.3f}  ins={r['_ins']} del={r['_dels']} sub={r['_subs']} ref_len={r['_ref_len']}")
        print(f"    ref : {r['ref'][:120]}")
        print(f"    pred: {r['pred'][:120]}")

    # 3) Deletion-heavy (pred much shorter than ref) — model cuts off
    deletes = sorted(rows, key=lambda r: -(r["_dels"] - r["_ins"]))[:n]
    print(f"\n[3] Most deletion-heavy (model truncates):")
    for r in deletes[:n]:
        delta = r["_dels"] - r["_ins"]
        if delta <= 0:
            continue
        print(f"  {r['id']:14s} WER={r['_wer']:.3f}  ins={r['_ins']} del={r['_dels']} sub={r['_subs']} ref_len={r['_ref_len']}")
        print(f"    ref : {r['ref'][:120]}")
        print(f"    pred: {r['pred'][:120]}")

    # 4) Empty predictions
    empties = [r for r in rows if not (r.get("pred") or "").strip()]
    print(f"\n[4] Empty predictions: {len(empties)}")
    for r in empties[:n]:
        print(f"  {r['id']:14s}  ref: {r['ref'][:80]}  extra={r.get('extra')}")

    # 5) Very short refs (1–2 words) — WER is extremely noisy on these
    short = [r for r in rows if r["_ref_len"] <= 2]
    print(f"\n[5] Refs with ≤2 words: {len(short)}  (single-word errors → WER 1.0)")
    if short:
        mean_short_wer = statistics.mean(r["_wer"] for r in short)
        print(f"    mean WER on these: {mean_short_wer:.3f}")
        for r in short[:n]:
            print(f"  {r['id']:14s} ref_len={r['_ref_len']} WER={r['_wer']:.3f}  ref={r['ref'][:60]} | pred={r['pred'][:60]}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", default=None,
                   help="Model name(s) under predictions/. Default: all subdirs.")
    args = p.parse_args()

    if not args.model:
        args.model = [d.name for d in sorted(PRED_ROOT.iterdir()) if d.is_dir()]

    for m in args.model:
        records = load_model(m)
        if not records:
            print(f"no predictions for {m}"); continue
        print(f"\n\n========= MODEL: {m}  ({len(records)} clips) =========")

        for label, fn in [("basic-norm (bakeoff.py current)", norm_basic),
                          ("arabic-aware", norm_arabic_aware),
                          ("aggressive (also strips 'ال')", norm_aggressive)]:
            scored = score_with(records, fn)
            summarise(scored, label)

        # Use arabic-aware scored rows for pathology scan
        scored = score_with(records, norm_arabic_aware)
        find_pathologies(scored, n=8)


if __name__ == "__main__":
    main()
