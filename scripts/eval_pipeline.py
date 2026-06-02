#!/usr/bin/env python3
"""\
eval_pipeline.py
================
Competition evaluation script for the Gulf Arabic medical ASR correction pipeline.

Usage
-----
    python eval_pipeline.py --endpoint http://localhost:8000 --test-set test_set.json

Both competitors point at their own server but use the identical test_set.json,
so scores are directly comparable.

Output
------
Prints a per-stage breakdown and an overall leaderboard table.
Writes a detailed JSON report to  eval_results_<timestamp>.json.

Stages evaluated
----------------
  Stage 1 — scoring_and_flagging   : Did the pipeline flag the right spans?
  Stage 2 — phonetic_retrieval     : Did the right candidate appear in the top-K list?
  Stage 3 — correction_decision    : Was the final chosen term correct?
  Stage 4 — false_positive_guard   : Were clean transcripts left untouched?
  Stage 5 — end_to_end             : Full corrected string matches gold?

Each stage produces its own precision / recall / F1 (or exact-match for stages 3 & 5).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

# Fix stdout for Unicode on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test-set loader — supports both JSON and JSONL
# ---------------------------------------------------------------------------


def _load_test_set(path: Path) -> list[dict]:
    """Load cases from either a .json or .jsonl file.

    * .json  — new format: { "version": "...", "cases": [...] }
    * .jsonl — old format: one JSON object per line, each with
      { "id": N, "transcript": "...", "expected": { "flags": [...], ... } }
      which is auto-converted to the new 5-stage expected format.
    """
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    # Default: .json
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    cases: list[dict] = data.get("cases", data if isinstance(data, list) else [])
    return cases


def _convert_old_case(old: dict) -> dict:
    """Convert an old-format JSONL case to the new 5-stage format."""
    cid = old["id"]
    transcript = old["transcript"]
    expected = old.get("expected", {})
    old_flags = expected.get("flags", [])
    old_drug_fixes = expected.get("drug_corrections", [])
    corrected = expected.get("corrected", transcript)

    # Determine dialect / category
    has_arabic = bool(re.search(r"[\u0600-\u06FF]", transcript))

    cat = "drug_normalization"
    if len(transcript) > 150:
        cat = "long_form"
    elif not old_flags:
        cat = "clean"
    elif any("if all gone" in (f.get("word", "") or "").lower() for f in old_flags):
        cat = "phonetic_mishearing"
    elif any(" " in (f.get("word", "") or "") for f in old_flags):
        cat = "split_drug"
    elif all(
        f.get("top_candidate", "").lower() == f.get("word", "").lower()
        for f in old_flags
    ) and not has_arabic:
        cat = "already_correct"
    elif len(old_flags) >= 2:
        cat = "multi_drug"

    dialect = "gulf_arabic" if has_arabic else "english"

    new_id = f"TC-{cid:03d}"

    # scoring_and_flagging
    flagged_spans = []
    for f in old_flags:
        span = {"text": f.get("word", ""), "index": f.get("index", 0)}
        if "span_indices" in f:
            span["span_indices"] = f["span_indices"]
        flagged_spans.append(span)

    # phonetic_retrieval
    spans_list = [
        {"span": f.get("word", ""), "top_candidate": f.get("top_candidate", "")}
        for f in old_flags
    ]
    if len(spans_list) == 1:
        phonetic_retrieval = {"top_candidate": spans_list[0]["top_candidate"]}
    elif spans_list:
        phonetic_retrieval = {"spans": spans_list}
    else:
        phonetic_retrieval = {}

    # correction_decision
    decisions = []
    for f in old_flags:
        decisions.append({
            "span": f.get("word", ""),
            "chosen": f.get("top_candidate", ""),
            "path": "auto_fix",
        })
    for dc in old_drug_fixes:
        if not any(d["span"].strip() == dc.get("from", "").strip() for d in decisions):
            decisions.append({
                "span": dc.get("from", ""),
                "chosen": dc.get("to", ""),
                "path": "auto_fix",
            })
    if len(decisions) == 0:
        correction_decision = {}
    elif len(decisions) == 1:
        correction_decision = decisions[0]
    else:
        correction_decision = {"decisions": decisions}

    # end_to_end
    corrections_count = len(old_drug_fixes) + len(old_flags)
    end_to_end = {
        "corrected": corrected,
        "corrections_count": corrections_count,
    }

    return {
        "id": new_id,
        "dialect": dialect,
        "category": cat,
        "input": transcript,
        "expected": {
            "scoring_and_flagging": {"flagged_spans": flagged_spans},
            "phonetic_retrieval": phonetic_retrieval,
            "correction_decision": correction_decision,
            "end_to_end": end_to_end,
        },
    }


def _load_jsonl(path: Path) -> list[dict]:
    """Load and convert old-format JSONL to new-format cases."""
    cases = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            old = json.loads(line)
            cases.append(_convert_old_case(old))
    return cases


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SpanResult:
    case_id: str
    gold_span_text: str
    predicted_spans: list[str]
    tp: bool = False   # gold span was found in predictions
    fp: int = 0        # number of predicted spans not in gold


@dataclass
class RetrievalResult:
    case_id: str
    gold_term: str
    candidates_returned: list[str]
    top_1_correct: bool = False
    top_k_correct: bool = False   # gold in any top-5 candidate


@dataclass
class DecisionResult:
    case_id: str
    gold_term: str | None
    predicted_term: str | None
    gold_path: str
    predicted_path: str
    correct: bool = False


@dataclass
class FalsePositiveResult:
    case_id: str
    input: str
    corrections_applied: int
    pass_: bool = False   # True if zero corrections applied on a clean case


@dataclass
class EndToEndResult:
    case_id: str
    gold_corrected: str
    predicted_corrected: str
    exact_match: bool = False
    corrections_count_gold: int = 0
    corrections_count_predicted: int = 0


@dataclass
class StageReport:
    stage_name: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    accuracy: float = 0.0   # used for stages that are exact-match
    notes: str = ""
    raw: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: call the /test-pipeline endpoint
# ---------------------------------------------------------------------------


def call_pipeline(endpoint: str, transcript: str, case_id: str, timeout: int = 120) -> dict:
    """POST to /test-pipeline and return the JSON response.

    Expected request schema:
        { "transcript": "<text>", "case_id": "<id>" }

    Expected response schema (your pipeline must return this from /test-pipeline):
        {
          "case_id": "TC-001",
          "original": "<input text>",
          "corrected": "<corrected text>",
          "corrections": [
            {
              "span_text": "...",
              "chosen": "...",
              "path": "llm|auto_fix|hitl_escalate|no_change",
              "confidence": 0.0
            }
          ],
          "flagged_spans": [
            {
              "text": "...",
              "start_index": 0,
              "end_index": 0,
              "max_suspicion": 0.0,
              "reason": "both|low_score|not_in_lexicon"
            }
          ],
          "retrieval_candidates": [
            {
              "span_text": "...",
              "candidates": [
                {"term": "...", "phonetic_score": 0.0}
              ]
            }
          ]
        }
    """
    url = endpoint.rstrip("/") + "/api/test-pipeline"
    try:
        resp = requests.post(
            url,
            json={"transcript": transcript, "case_id": case_id},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        return {"error": "connection_refused", "case_id": case_id}
    except requests.exceptions.Timeout:
        return {"error": "timeout", "case_id": case_id}
    except requests.exceptions.HTTPError as exc:
        return {"error": f"http_{exc.response.status_code}", "case_id": case_id}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc), "case_id": case_id}


# ---------------------------------------------------------------------------
# Stage 1 — Scoring & Flagging
# ---------------------------------------------------------------------------

def evaluate_scoring_and_flagging(
    cases: list[dict],
    pipeline_responses: dict[str, dict],
) -> StageReport:
    """
    Precision  = TP_spans / (TP_spans + FP_spans)
    Recall     = TP_spans / total_gold_spans

    A gold span is "found" (TP) if the predicted flagged spans contain
    a span whose text overlaps with the gold span text (substring match
    in either direction is accepted, to handle tokenisation differences).
    """
    total_gold_spans = 0
    tp_total = 0
    fp_total = 0
    raw: list[dict] = []

    for case in cases:
        case_id = case["id"]
        response = pipeline_responses.get(case_id, {})
        if "error" in response:
            raw.append({"case_id": case_id, "error": response["error"]})
            continue

        gold_spans: list[dict] = (
            case["expected"]["scoring_and_flagging"].get("flagged_spans", [])
        )
        pred_spans: list[dict] = response.get("flagged_spans", [])
        pred_texts = [s.get("text", "").strip() for s in pred_spans]

        case_tp = 0
        case_fn = 0
        case_fp = 0

        for gold_span in gold_spans:
            gold_text = gold_span["text"].strip()
            total_gold_spans += 1
            # Match: predicted span overlaps or is contained in gold span text
            matched = any(
                gold_text in p or p in gold_text
                for p in pred_texts
            )
            if matched:
                tp_total += 1
                case_tp += 1
            else:
                case_fn += 1

        # Count false positives: predicted spans that don't match any gold span
        gold_texts = [s["text"].strip() for s in gold_spans]
        for pred_text in pred_texts:
            if not any(gt in pred_text or pred_text in gt for gt in gold_texts):
                fp_total += 1
                case_fp += 1

        raw.append({
            "case_id": case_id,
            "gold_spans": [s["text"] for s in gold_spans],
            "pred_spans": pred_texts,
            "tp": case_tp,
            "fn": case_fn,
            "fp": case_fp,
        })

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / total_gold_spans if total_gold_spans > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return StageReport(
        stage_name="scoring_and_flagging",
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        notes=f"gold_spans={total_gold_spans} tp={tp_total} fp={fp_total}",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 2 — Phonetic Retrieval
# ---------------------------------------------------------------------------

def evaluate_phonetic_retrieval(
    cases: list[dict],
    pipeline_responses: dict[str, dict],
) -> StageReport:
    """
    top_1_accuracy = cases where top candidate is the gold term / total retrieval cases
    top_k_recall   = cases where gold term appears anywhere in top-K / total retrieval cases

    Only cases with at least one expected flagged span are included.
    Cases where expected_path is 'hitl_escalate' are scored separately and
    reported but not counted against the pipeline.
    """
    top_1_correct = 0
    top_k_correct = 0
    total_retrieval_cases = 0
    hitl_cases = 0
    raw: list[dict] = []

    for case in cases:
        case_id = case["id"]
        response = pipeline_responses.get(case_id, {})
        if "error" in response:
            raw.append({"case_id": case_id, "error": response["error"]})
            continue

        gold_retrieval = case["expected"].get("phonetic_retrieval", {})

        # Skip pure no-error cases
        if gold_retrieval.get("spans_processed", -1) == 0:
            continue

        # Handle single-span cases
        if "top_candidate" in gold_retrieval:
            gold_terms = [gold_retrieval["top_candidate"]]
            gold_spans_data = [{"span": case["expected"]["scoring_and_flagging"]["flagged_spans"][0]["text"], "top_candidate": gold_retrieval["top_candidate"]}]
        elif "spans" in gold_retrieval:
            gold_spans_data = gold_retrieval["spans"]
            gold_terms = [s["top_candidate"] for s in gold_spans_data]
        else:
            continue

        pred_candidates: list[dict] = response.get("retrieval_candidates", [])
        pred_by_span = {c.get("span_text", "").strip(): c.get("candidates", []) for c in pred_candidates}

        for gold_span_info in gold_spans_data:
            span_text = gold_span_info.get("span", gold_span_info.get("span_text", ""))
            gold_term = gold_span_info["top_candidate"].lower()

            # Check if this is a HITL escalation case
            decision = case["expected"].get("correction_decision", {})
            if isinstance(decision, dict):
                path = decision.get("path", "")
            else:
                path = ""

            if path == "hitl_escalate":
                hitl_cases += 1
                raw.append({
                    "case_id": case_id,
                    "span": span_text,
                    "note": "hitl_escalation — scored separately",
                    "gold_term": gold_term,
                })
                continue

            total_retrieval_cases += 1

            # Find matching predicted candidates for this span
            matched_candidates = []
            for pred_span_text, candidates in pred_by_span.items():
                if span_text in pred_span_text or pred_span_text in span_text:
                    matched_candidates = candidates
                    break

            if not matched_candidates:
                raw.append({
                    "case_id": case_id,
                    "span": span_text,
                    "gold_term": gold_term,
                    "top_1_correct": False,
                    "top_k_correct": False,
                    "note": "no candidates returned for this span",
                })
                continue

            pred_term_list = [c.get("term", "").lower() for c in matched_candidates]
            top1 = pred_term_list[0] if pred_term_list else ""
            t1 = top1 == gold_term
            tk = gold_term in pred_term_list

            if t1:
                top_1_correct += 1
            if tk:
                top_k_correct += 1

            raw.append({
                "case_id": case_id,
                "span": span_text,
                "gold_term": gold_term,
                "top_1_candidate": top1,
                "all_candidates": pred_term_list,
                "top_1_correct": t1,
                "top_k_correct": tk,
            })

    top_1_acc = top_1_correct / total_retrieval_cases if total_retrieval_cases > 0 else 0.0
    top_k_acc = top_k_correct / total_retrieval_cases if total_retrieval_cases > 0 else 0.0

    return StageReport(
        stage_name="phonetic_retrieval",
        precision=round(top_1_acc, 4),
        recall=round(top_k_acc, 4),
        f1=round((top_1_acc + top_k_acc) / 2, 4),
        accuracy=round(top_1_acc, 4),
        notes=(
            f"retrieval_cases={total_retrieval_cases} "
            f"top1_correct={top_1_correct} topk_correct={top_k_correct} "
            f"hitl_cases_excluded={hitl_cases}"
        ),
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 3 — Correction Decision
# ---------------------------------------------------------------------------

def evaluate_correction_decision(
    cases: list[dict],
    pipeline_responses: dict[str, dict],
) -> StageReport:
    """
    Exact-match accuracy on the chosen correction term.
    A decision is correct if chosen == gold_term (case-insensitive).
    NO_CHANGE / None decisions are correct only when gold also says no_change.
    HITL escalations are scored: correct if path == hitl_escalate.
    """
    total_decisions = 0
    correct_decisions = 0
    raw: list[dict] = []

    for case in cases:
        case_id = case["id"]
        response = pipeline_responses.get(case_id, {})
        if "error" in response:
            raw.append({"case_id": case_id, "error": response["error"]})
            continue

        gold_decision = case["expected"].get("correction_decision", {})

        # Multi-span decisions
        if "decisions" in gold_decision:
            gold_decisions = gold_decision["decisions"]
        elif "span" in gold_decision:
            gold_decisions = [gold_decision]
        else:
            # No decisions expected (clean case)
            gold_decisions = []

        pred_corrections: list[dict] = response.get("corrections", [])
        pred_by_span = {
            c.get("span_text", "").strip().lower(): c
            for c in pred_corrections
        }

        if not gold_decisions:
            # Clean case: no corrections expected
            if not pred_corrections:
                correct_decisions += 1
                total_decisions += 1
                raw.append({"case_id": case_id, "note": "clean_case_correctly_untouched", "correct": True})
            else:
                total_decisions += 1
                raw.append({
                    "case_id": case_id,
                    "note": "clean_case_but_corrections_applied",
                    "pred_corrections": [c.get("span_text") for c in pred_corrections],
                    "correct": False,
                })
            continue

        for gold_dec in gold_decisions:
            total_decisions += 1
            gold_span = gold_dec.get("span", gold_dec.get("text", "")).strip()
            gold_chosen = (gold_dec.get("chosen") or "").lower()
            gold_path = gold_dec.get("path", "")

            # Find matching predicted decision
            pred_dec = None
            for pred_span_text, pred in pred_by_span.items():
                if gold_span.lower() in pred_span_text or pred_span_text in gold_span.lower():
                    pred_dec = pred
                    break

            if pred_dec is None:
                raw.append({
                    "case_id": case_id,
                    "gold_span": gold_span,
                    "gold_chosen": gold_chosen,
                    "gold_path": gold_path,
                    "pred_chosen": None,
                    "pred_path": None,
                    "correct": False,
                    "note": "span not found in predictions",
                })
                continue

            pred_chosen = (pred_dec.get("chosen") or "").lower()
            pred_path = pred_dec.get("path", "")

            if gold_path == "hitl_escalate":
                correct = pred_path == "hitl_escalate"
            elif gold_chosen == "":
                correct = pred_chosen == "" or pred_chosen is None
            else:
                correct = pred_chosen == gold_chosen

            if correct:
                correct_decisions += 1

            raw.append({
                "case_id": case_id,
                "gold_span": gold_span,
                "gold_chosen": gold_chosen,
                "gold_path": gold_path,
                "pred_chosen": pred_chosen,
                "pred_path": pred_path,
                "correct": correct,
            })

    accuracy = correct_decisions / total_decisions if total_decisions > 0 else 0.0

    return StageReport(
        stage_name="correction_decision",
        accuracy=round(accuracy, 4),
        f1=round(accuracy, 4),  # for leaderboard sorting parity
        notes=f"total_decisions={total_decisions} correct={correct_decisions}",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 4 — False Positive Guard
# ---------------------------------------------------------------------------

def evaluate_false_positive_guard(
    cases: list[dict],
    pipeline_responses: dict[str, dict],
) -> StageReport:
    """
    Measures how well the pipeline avoids touching correct transcripts.
    Only evaluates cases where expected corrections_count == 0.
    Metric: specificity = correct_clean / total_clean
    """
    total_clean = 0
    correct_clean = 0
    raw: list[dict] = []

    for case in cases:
        gold_e2e = case["expected"].get("end_to_end", {})
        if gold_e2e.get("corrections_count", -1) != 0:
            continue  # Not a clean case

        case_id = case["id"]
        total_clean += 1
        response = pipeline_responses.get(case_id, {})
        if "error" in response:
            raw.append({"case_id": case_id, "error": response["error"]})
            continue

        pred_corrections: list[dict] = response.get("corrections", [])
        applied = len([c for c in pred_corrections if c.get("path") not in ("hitl_escalate", "no_change")])

        passed = applied == 0
        if passed:
            correct_clean += 1

        raw.append({
            "case_id": case_id,
            "input": case["input"],
            "corrections_applied": applied,
            "passed": passed,
        })

    specificity = correct_clean / total_clean if total_clean > 0 else 0.0

    return StageReport(
        stage_name="false_positive_guard",
        precision=round(specificity, 4),
        recall=1.0,
        f1=round(specificity, 4),
        accuracy=round(specificity, 4),
        notes=f"clean_cases={total_clean} correct_clean={correct_clean}",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Stage 5 — End-to-End
# ---------------------------------------------------------------------------


def _normalise(text: str) -> str:
    """Collapse whitespace and lowercase for lenient exact-match."""
    return " ".join(text.lower().split())


def evaluate_end_to_end(
    cases: list[dict],
    pipeline_responses: dict[str, dict],
) -> StageReport:
    """
    Two sub-metrics:
      exact_match      : normalised gold corrected == normalised pred corrected
      corrections_match: count of applied corrections matches gold

    Score reported is exact_match accuracy.
    """
    total = 0
    exact_matches = 0
    count_matches = 0
    raw: list[dict] = []

    for case in cases:
        case_id = case["id"]
        response = pipeline_responses.get(case_id, {})
        if "error" in response:
            total += 1
            raw.append({"case_id": case_id, "error": response["error"]})
            continue

        gold_e2e = case["expected"].get("end_to_end", {})
        gold_corrected = gold_e2e.get("corrected", case["input"])
        gold_count = gold_e2e.get("corrections_count", 0)

        pred_corrected = response.get("corrected", response.get("original", ""))
        pred_count = len([
            c for c in response.get("corrections", [])
            if c.get("path") not in ("hitl_escalate", "no_change")
        ])

        em = _normalise(gold_corrected) == _normalise(pred_corrected)
        cm = pred_count == gold_count

        total += 1
        if em:
            exact_matches += 1
        if cm:
            count_matches += 1

        raw.append({
            "case_id": case_id,
            "gold_corrected": gold_corrected,
            "pred_corrected": pred_corrected,
            "exact_match": em,
            "gold_count": gold_count,
            "pred_count": pred_count,
            "count_match": cm,
        })

    em_acc = exact_matches / total if total > 0 else 0.0
    cm_acc = count_matches / total if total > 0 else 0.0

    return StageReport(
        stage_name="end_to_end",
        accuracy=round(em_acc, 4),
        f1=round(em_acc, 4),
        notes=f"total={total} exact_matches={exact_matches} count_matches={count_matches} correction_count_acc={cm_acc:.4f}",
        raw=raw,
    )


# ---------------------------------------------------------------------------
# Leaderboard summary
# ---------------------------------------------------------------------------


def print_report(reports: list[StageReport], endpoint: str, elapsed: float) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  PIPELINE EVALUATION REPORT")
    print(f"  Endpoint : {endpoint}")
    print(f"  Date     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration : {elapsed:.1f}s")
    print("=" * width)

    header = f"{'Stage':<28} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Accuracy':>10}"
    print(header)
    print("-" * width)

    overall_f1 = 0.0
    for report in reports:
        row = (
            f"{report.stage_name:<28} "
            f"{report.precision:>10.4f} "
            f"{report.recall:>8.4f} "
            f"{report.f1:>8.4f} "
            f"{report.accuracy:>10.4f}"
        )
        print(row)
        overall_f1 += report.f1

    print("-" * width)
    mean_f1 = overall_f1 / len(reports) if reports else 0.0
    print(f"{'OVERALL MEAN F1':<28} {'':>10} {'':>8} {mean_f1:>8.4f}")
    print("=" * width)

    print()
    print("  STAGE NOTES")
    print("-" * width)
    for report in reports:
        print(f"  {report.stage_name}: {report.notes}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a /test-pipeline endpoint against the Gulf Arabic medical ASR test set."
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="Base URL of the pipeline server (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--test-set",
        default="eval/pipeline_testset.jsonl",
        help="Path to the test set file (.json or .jsonl; default: eval/pipeline_testset.jsonl)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the JSON report (default: eval_results_<timestamp>.json)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay in seconds between requests to avoid overloading the server (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load test set and print cases without hitting the endpoint",
    )
    parser.add_argument(
        "--filter-ids",
        nargs="*",
        default=None,
        help="Only run specific case IDs, e.g. --filter-ids TC-001 TC-008",
    )
    args = parser.parse_args()

    # Load test set (supports .jsonl and .json)
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        print(f"ERROR: test set file not found: {test_set_path}", file=sys.stderr)
        sys.exit(1)

    cases: list[dict] = _load_test_set(test_set_path)
    if args.filter_ids:
        cases = [c for c in cases if c["id"] in args.filter_ids]
        print(f"Filtered to {len(cases)} case(s): {args.filter_ids}")

    print(f"Loaded {len(cases)} test cases from {test_set_path}")
    print(f"Format: {test_set_path.suffix}")

    if args.dry_run:
        print("\nDRY RUN — cases:")
        for case in cases:
            print(f"  {case['id']} [{case['dialect']}] {case['category']}")
            flagged = case['expected']['scoring_and_flagging']['flagged_spans']
            if flagged:
                print(f"    Spans: {[s['text'] for s in flagged]}")
            else:
                print(f"    (clean — no flagged spans)")
        return

    # Run inference
    print(f"\nCalling endpoint: {args.endpoint.rstrip('/')}/test-pipeline")
    print(f"Timeout: {args.timeout}s  |  Inter-request delay: {args.delay}s\n")

    pipeline_responses: dict[str, dict] = {}
    errors = 0
    start_time = time.time()

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        transcript = case["input"]
        print(f"  [{i:02d}/{len(cases)}] {cid} ...", end=" ", flush=True)
        t0 = time.time()
        resp = call_pipeline(args.endpoint, transcript, cid, timeout=args.timeout)
        elapsed_case = time.time() - t0

        if "error" in resp:
            print(f"ERROR ({resp['error']}) [{elapsed_case:.1f}s]")
            errors += 1
        else:
            corrections_count = len(resp.get("corrections", []))
            print(f"OK — {corrections_count} correction(s) [{elapsed_case:.1f}s]")

        pipeline_responses[cid] = resp

        if i < len(cases) and args.delay > 0:
            time.sleep(args.delay)

    total_elapsed = time.time() - start_time
    print(f"\nCompleted {len(cases)} requests in {total_elapsed:.1f}s. Errors: {errors}/{len(cases)}\n")

    # Evaluate each stage
    reports = [
        evaluate_scoring_and_flagging(cases, pipeline_responses),
        evaluate_phonetic_retrieval(cases, pipeline_responses),
        evaluate_correction_decision(cases, pipeline_responses),
        evaluate_false_positive_guard(cases, pipeline_responses),
        evaluate_end_to_end(cases, pipeline_responses),
    ]

    # Print report
    print_report(reports, args.endpoint, total_elapsed)

    # Write JSON output
    output_path = args.output or f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        "meta": {
            "endpoint": args.endpoint,
            "test_set_path": str(test_set_path),
            "cases_run": len(cases),
            "errors": errors,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(total_elapsed, 2),
        },
        "stage_reports": [asdict(r) for r in reports],
        "overall_mean_f1": round(sum(r.f1 for r in reports) / len(reports), 4),
        "raw_responses": pipeline_responses,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"Detailed report written to: {output_path}\n")

    # Exit with non-zero if too many errors
    if errors > len(cases) // 2:
        print("WARNING: More than half the requests failed. Check endpoint connectivity.")
        sys.exit(1)


if __name__ == "__main__":
    main()
