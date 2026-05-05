"""Evaluate the medical correction pipeline on a JSONL gold set."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from medical_corrector import MedicalCorrector, normalize_text


DEFAULT_EVAL_PATH = Path(__file__).parent / "eval" / "medical_transcript_eval.jsonl"


def norm_correction(text: str) -> str:
    text = normalize_text(text)
    text = text.replace("_", " ")
    text = re.sub(r"\s*-\s*", "-", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def evaluate(path: Path, use_llm: bool, threshold: float, split: str | None) -> Dict[str, Any]:
    rows = load_rows(path)
    if split:
        rows = [row for row in rows if row["split"] == split]

    corrector = MedicalCorrector(use_llm=use_llm, auto_threshold=threshold)

    details = []
    for row in rows:
        result = corrector.correct_transcript(row["transcript"])
        pred_spans = result["suspicious_spans"]
        gold_spans = row["gold_spans"]

        gold_by_span = {normalize_text(g["original_text"]): g for g in gold_spans}
        pred_by_span = {normalize_text(p["original_text"]): p for p in pred_spans}

        detection_hits = 0
        correction_hits = 0
        missed = []
        correction_misses = []

        for gold_key, gold in gold_by_span.items():
            pred = pred_by_span.get(gold_key)
            if pred is None:
                detection_match = _find_near_boundary_match(gold, pred_spans)
                if detection_match is None:
                    missed.append(gold)
                    continue
                pred = detection_match
            detection_hits += 1
            if norm_correction(pred["possible_correction"]) == norm_correction(gold["possible_correction"]):
                correction_hits += 1
            else:
                correction_misses.append({"gold": gold, "pred": pred})

        extra_preds = [
            pred
            for pred_key, pred in pred_by_span.items()
            if pred_key not in gold_by_span
            and _find_gold_overlap(pred, gold_spans) is None
        ]

        details.append(
            {
                "id": row["id"],
                "split": row["split"],
                "difficulty": row["difficulty"],
                "contains_error": row["contains_error"],
                "gold_count": len(gold_spans),
                "pred_count": len(pred_spans),
                "detection_hits": detection_hits,
                "correction_hits": correction_hits,
                "missed": missed,
                "correction_misses": correction_misses,
                "extra_preds": extra_preds,
                "pred": pred_spans,
                "gold": gold_spans,
            }
        )

    return {"summary": summarize(details), "details": details}


def _find_near_boundary_match(gold: Dict[str, Any], pred_spans: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    gold_text = normalize_text(gold["original_text"])
    for pred in pred_spans:
        pred_text = normalize_text(pred["original_text"])
        pred_corr = norm_correction(pred["possible_correction"])
        gold_corr = norm_correction(gold["possible_correction"])
        # Count close boundary variants as detected if the correction is exact
        # and one text span contains the other.
        if pred_corr == gold_corr and (pred_text in gold_text or gold_text in pred_text):
            return pred
    return None


def _find_gold_overlap(pred: Dict[str, Any], gold_spans: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    pred_text = normalize_text(pred["original_text"])
    pred_corr = norm_correction(pred["possible_correction"])
    for gold in gold_spans:
        gold_text = normalize_text(gold["original_text"])
        gold_corr = norm_correction(gold["possible_correction"])
        if pred_corr == gold_corr and (pred_text in gold_text or gold_text in pred_text):
            return gold
    return None


def summarize(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "examples": 0,
            "gold": 0,
            "det": 0,
            "corr": 0,
            "neg_examples": 0,
            "neg_clean": 0,
            "fp_preds_on_negatives": 0,
            "extra_preds": 0,
        }
    )

    for row in details:
        for key in ("overall", row["split"]):
            a = agg[key]
            a["examples"] += 1
            a["gold"] += row["gold_count"]
            a["det"] += row["detection_hits"]
            a["corr"] += row["correction_hits"]
            a["extra_preds"] += len(row["extra_preds"])
            if not row["contains_error"]:
                a["neg_examples"] += 1
                if row["pred_count"] == 0:
                    a["neg_clean"] += 1
                a["fp_preds_on_negatives"] += row["pred_count"]

    out: Dict[str, Any] = {}
    for key, a in agg.items():
        gold = a["gold"]
        neg = a["neg_examples"]
        out[key] = {
            **a,
            "detection_recall": (a["det"] / gold) if gold else None,
            "correction_recall": (a["corr"] / gold) if gold else None,
            "negative_clean_rate": (a["neg_clean"] / neg) if neg else None,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL_PATH)
    parser.add_argument("--split", choices=["dev", "test"], default=None)
    parser.add_argument("--threshold", type=float, default=86.0)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--show-errors", action="store_true")
    args = parser.parse_args()

    report = evaluate(args.eval, args.use_llm, args.threshold, args.split)
    print(json.dumps(report["summary"], indent=2))

    if args.show_errors:
        print("\nERRORS / EXTRAS")
        for row in report["details"]:
            bad = row["missed"] or row["correction_misses"] or row["extra_preds"]
            if not bad:
                continue
            print(f"\n{row['id']} ({row['split']}, {row['difficulty']})")
            print("gold:", json.dumps(row["gold"], ensure_ascii=False))
            print("pred:", json.dumps(row["pred"], ensure_ascii=False))
            if row["missed"]:
                print("missed:", json.dumps(row["missed"], ensure_ascii=False))
            if row["correction_misses"]:
                print("correction_misses:", json.dumps(row["correction_misses"], ensure_ascii=False))
            if row["extra_preds"]:
                print("extra_preds:", json.dumps(row["extra_preds"], ensure_ascii=False))


if __name__ == "__main__":
    main()
