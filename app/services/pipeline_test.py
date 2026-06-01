"""Pipeline evaluation service.

Loads the curated test set from eval/pipeline_testset.jsonl and runs each
test case through every pipeline stage, comparing the output against the
expected results.  The summary is split per stage so the user and their
colleague can compare whose phonetic/semantic pipeline performs better.

Stages
------
1. Scoring & flagging       flag.flag_suspicious()
2. Drug normalization       drug_normalize.normalize_drugs()
3. Auto-correction          flag.apply_high_confidence_corrections()
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import drug_normalize, flag

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TESET_PATH = PROJECT_ROOT / "eval" / "pipeline_testset.jsonl"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_test_set(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    path = path or DEFAULT_TESET_PATH
    if not path.exists():
        raise FileNotFoundError(f"test set not found: {path}")
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _normalize(s: str) -> str:
    """Collapse whitespace for comparison."""
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _compare_flags(
    actual_flags: List[Dict[str, Any]],
    expected_flags: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compare actual flag output against expected flags.

    For each expected flag, we check:
    - A flag exists at (or near) the expected index
    - The top candidate matches (or the expected candidate appears in top-3)
    - The similarity is above the minimum threshold
    """
    details: List[Dict[str, Any]] = []
    passed = 0
    total = len(expected_flags) if expected_flags else 1 if actual_flags else 0

    for exp in expected_flags:
        exp_idx = exp.get("index")
        exp_word = exp.get("word", "")
        exp_cand = exp.get("top_candidate", "")
        exp_sim = exp.get("min_similarity", 0.0)

        matching = [
            f
            for f in actual_flags
            if f.get("index") == exp_idx
            or (
                "span_indices" in f
                and isinstance(f["span_indices"], list)
                and exp_idx in f["span_indices"]
            )
        ]

        if not matching:
            details.append(
                {
                    "expected_word": exp_word,
                    "expected_index": exp_idx,
                    "status": "MISS",
                    "detail": "No flag found at expected index",
                    "actual_flags_nearby": [
                        f
                        for f in actual_flags
                        if abs(f.get("index", -1) - exp_idx) <= 2
                    ][:3],
                }
            )
            continue

        # Check top candidate
        best_match = matching[0]
        candidates = best_match.get("candidates", [])
        top_term = candidates[0].get("term", "") if candidates else ""
        top_sim = candidates[0].get("phonetic_similarity", 0.0) if candidates else 0.0

        cand_ok = False
        if exp_cand:
            if top_term.lower() == exp_cand.lower():
                cand_ok = True
            else:
                # Check if expected candidate is in top 3
                for c in candidates[:3]:
                    if c.get("term", "").lower() == exp_cand.lower():
                        cand_ok = True
                        break

        sim_ok = top_sim >= exp_sim if exp_sim > 0 else True

        if cand_ok and sim_ok:
            passed += 1
            details.append(
                {
                    "expected_word": exp_word,
                    "expected_index": exp_idx,
                    "status": "PASS",
                    "top_candidate": top_term,
                    "similarity": round(top_sim, 3),
                }
            )
        else:
            issues = []
            if not cand_ok:
                issues.append(
                    f"expected candidate '{exp_cand}' not in top 3 "
                    f"(got {[c.get('term','') for c in candidates[:3]]})"
                )
            if not sim_ok:
                issues.append(
                    f"similarity {top_sim:.3f} below threshold {exp_sim}"
                )
            details.append(
                {
                    "expected_word": exp_word,
                    "expected_index": exp_idx,
                    "status": "FAIL",
                    "top_candidate": top_term,
                    "similarity": round(top_sim, 3),
                    "issues": issues,
                }
            )

    # Penalty for extra flags the test didn't expect (false positives)
    false_positives = 0
    fp_details: List[Dict[str, Any]] = []
    expected_indices = {e.get("index") for e in expected_flags}
    for f in actual_flags:
        idx = f.get("index")
        span = f.get("span_indices") or [idx]
        if not any(i in expected_indices for i in span):
            false_positives += 1
            fp_details.append(
                {
                    "index": idx,
                    "word": f.get("word", ""),
                    "top_candidate": (
                        f["candidates"][0].get("term", "")
                        if f.get("candidates")
                        else ""
                    ),
                }
            )

    # If no flags expected and none found, it's a pass
    if not expected_flags and not actual_flags:
        passed = 1
        total = 1

    return {
        "passed": passed,
        "total": total,
        "false_positives": false_positives,
        "details": details,
        "false_positive_details": fp_details,
    }


def _compare_drug_corrections(
    actual_fixes: List[Dict[str, str]],
    expected_fixes: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Compare drug normalization output."""
    details: List[Dict[str, Any]] = []
    passed = 0
    total = len(expected_fixes) if expected_fixes else 1 if actual_fixes else 0

    if not expected_fixes and not actual_fixes:
        return {"passed": 1, "total": 1, "details": [], "mismatches": []}

    actual_map = {a.get("from", "").lower(): a.get("to", "") for a in actual_fixes}

    for exp in expected_fixes:
        exp_from = exp.get("from", "").lower()
        exp_to = exp.get("to", "")

        actual_to = actual_map.get(exp_from)
        if actual_to and actual_to.lower() == exp_to.lower():
            passed += 1
            details.append(
                {
                    "from": exp.get("from", ""),
                    "expected_to": exp_to,
                    "actual_to": actual_to,
                    "status": "PASS",
                }
            )
        elif actual_to:
            details.append(
                {
                    "from": exp.get("from", ""),
                    "expected_to": exp_to,
                    "actual_to": actual_to,
                    "status": "FAIL",
                    "issue": f"wrong target (expected '{exp_to}', got '{actual_to}')",
                }
            )
        else:
            details.append(
                {
                    "from": exp.get("from", ""),
                    "expected_to": exp_to,
                    "actual_to": None,
                    "status": "MISS",
                    "issue": "correction not applied",
                }
            )

    # Extra corrections not in expected
    extras = []
    for a in actual_fixes:
        if a.get("from", "").lower() not in {e.get("from", "").lower() for e in expected_fixes}:
            extras.append(a)

    return {
        "passed": passed,
        "total": total,
        "details": details,
        "mismatches": extras,
    }


def _compare_corrected(
    actual_corrected: str, expected_corrected: str
) -> Dict[str, Any]:
    """Compare the final corrected transcript."""
    a = _normalize(actual_corrected)
    e = _normalize(expected_corrected)
    if a == e:
        return {
            "passed": 1,
            "total": 1,
            "exact_match": True,
            "actual": actual_corrected,
            "expected": expected_corrected,
        }
    return {
        "passed": 0,
        "total": 1,
        "exact_match": False,
        "actual": actual_corrected,
        "expected": expected_corrected,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_test_set(
    test_set_path: Optional[Path] = None,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Run every test case through all pipeline stages and return per-stage
    results and aggregate statistics."""
    cases = _load_test_set(test_set_path)
    n = len(cases)

    stage_flags = {"passed": 0, "total": n, "detail_by_case": []}
    stage_drug = {"passed": 0, "total": n, "detail_by_case": []}
    stage_corrected = {"passed": 0, "total": n, "detail_by_case": []}
    detailed_results = []

    for case in cases:
        cid = case["id"]
        transcript = case["transcript"]
        expected = case.get("expected", {})
        desc = case.get("description", "")

        # ---- Stage 1: Scoring & Flagging ----
        try:
            flags_out = flag.flag_suspicious(transcript, use_llm=use_llm)
        except Exception as exc:
            flags_out = []
            print(f"[pipeline_test] case {cid}: flag_suspicious failed: {exc!r}")

        flag_result = _compare_flags(flags_out, expected.get("flags", []))

        # ---- Stage 2: Drug Normalization ----
        try:
            normalized_text, drug_fixes = drug_normalize.normalize_drugs(transcript)
        except Exception as exc:
            normalized_text = transcript
            drug_fixes = []
            print(f"[pipeline_test] case {cid}: normalize_drugs failed: {exc!r}")

        drug_result = _compare_drug_corrections(drug_fixes, expected.get("drug_corrections", []))

        # ---- Stage 3: Auto-correction ----
        try:
            corr_flags = flag.flag_suspicious(normalized_text, use_llm=use_llm)
            corr_result = flag.apply_high_confidence_corrections(normalized_text, corr_flags)
            final_text = corr_result["corrected_transcript"]
            auto_applied = corr_result["applied"]
        except Exception as exc:
            final_text = normalized_text
            auto_applied = []
            print(f"[pipeline_test] case {cid}: auto-correction failed: {exc!r}")

        corr_compare = _compare_corrected(final_text, expected.get("corrected", transcript))

        # ---- Aggregate ----
        if flag_result["passed"] >= flag_result["total"] and flag_result["false_positives"] <= 1:
            stage_flags["passed"] += 1
        if drug_result["passed"] >= drug_result["total"]:
            stage_drug["passed"] += 1
        if corr_compare["passed"]:
            stage_corrected["passed"] += 1

        stage_flags["detail_by_case"].append({"id": cid, "desc": desc, "passed": flag_result["passed"], "total": flag_result["total"], "false_positives": flag_result["false_positives"]})
        stage_drug["detail_by_case"].append({"id": cid, "desc": desc, "passed": drug_result["passed"], "total": drug_result["total"]})
        stage_corrected["detail_by_case"].append({"id": cid, "desc": desc, "passed": 1 if corr_compare["passed"] else 0, "total": 1})

        # Build detailed per-case result
        detailed_results.append({
            "id": cid,
            "description": desc,
            "transcript": transcript,
            "expected": expected,
            "stages": {
                "flagging": {
                    "passed": flag_result["passed"],
                    "total": flag_result["total"],
                    "false_positives": flag_result["false_positives"],
                    "details": flag_result["details"],
                    "false_positive_details": flag_result["false_positive_details"],
                    "actual_flags": [
                        {"index": f.get("index"), "word": f.get("word"), "reason": f.get("reason"),
                         "top_candidate": (f["candidates"][0] if f.get("candidates") else None)}
                        for f in flags_out
                    ],
                },
                "drug_normalization": {
                    "passed": drug_result["passed"],
                    "total": drug_result["total"],
                    "details": drug_result["details"],
                    "mismatches": drug_result["mismatches"],
                    "normalized_text": normalized_text,
                    "fixes_applied": drug_fixes,
                },
                "auto_correction": {
                    "passed": corr_compare["passed"],
                    "total": corr_compare["total"],
                    "exact_match": corr_compare.get("exact_match", False),
                    "actual": corr_compare.get("actual", ""),
                    "expected": corr_compare.get("expected", ""),
                    "corrections_applied": auto_applied,
                },
            },
        })

    # Build per-case detailed results
    detailed_results = []
    for case in cases:
        cid = case["id"]
        transcript = case["transcript"]
        expected = case.get("expected", {})
        desc = case.get("description", "")

        # ---- Stage 1: Scoring & Flagging ----
        try:
            flags_out = flag.flag_suspicious(transcript, use_llm=use_llm)
        except Exception as exc:
            flags_out = []
            print(f"[pipeline_test] case {cid}: flag_suspicious failed: {exc!r}")

        flag_result = _compare_flags(flags_out, expected.get("flags", []))

        # ---- Stage 2: Drug Normalization ----
        try:
            normalized_text, drug_fixes = drug_normalize.normalize_drugs(transcript)
        except Exception as exc:
            normalized_text = transcript
            drug_fixes = []
            print(f"[pipeline_test] case {cid}: normalize_drugs failed: {exc!r}")

        drug_result = _compare_drug_corrections(drug_fixes, expected.get("drug_corrections", []))

        # ---- Stage 3: Auto-correction ----
        try:
            corr_flags = flag.flag_suspicious(normalized_text, use_llm=use_llm)
            corr_result = flag.apply_high_confidence_corrections(normalized_text, corr_flags)
            final_text = corr_result["corrected_transcript"]
            auto_applied = corr_result["applied"]
        except Exception as exc:
            final_text = normalized_text
            auto_applied = []
            print(f"[pipeline_test] case {cid}: auto-correction failed: {exc!r}")

        corr_compare = _compare_corrected(final_text, expected.get("corrected", transcript))

        detailed_results.append({
            "id": cid,
            "description": desc,
            "transcript": transcript,
            "expected": expected,
            "stages": {
                "flagging": {
                    "passed": flag_result["passed"],
                    "total": flag_result["total"],
                    "false_positives": flag_result["false_positives"],
                    "details": flag_result["details"],
                    "false_positive_details": flag_result["false_positive_details"],
                    "actual_flags": [
                        {
                            "index": f.get("index"),
                            "word": f.get("word"),
                            "reason": f.get("reason"),
                            "top_candidate": (f["candidates"][0] if f.get("candidates") else None),
                        }
                        for f in flags_out
                    ],
                },
                "drug_normalization": {
                    "passed": drug_result["passed"],
                    "total": drug_result["total"],
                    "details": drug_result["details"],
                    "mismatches": drug_result["mismatches"],
                    "normalized_text": normalized_text,
                    "fixes_applied": drug_fixes,
                },
                "auto_correction": {
                    "passed": corr_compare["passed"],
                    "total": corr_compare["total"],
                    "exact_match": corr_compare.get("exact_match", False),
                    "actual": corr_compare.get("actual", ""),
                    "expected": corr_compare.get("expected", ""),
                    "corrections_applied": auto_applied,
                },
            },
        })

    # Summary
    summary = {
        "total_cases": n,
        "stages": {
            "scoring_and_flagging": {
                "label": "Scoring & Flagging",
                "passed": stage_flags["passed"],
                "total": stage_flags["total"],
                "accuracy": round(stage_flags["passed"] / max(1, stage_flags["total"]) * 100, 1),
            },
            "drug_normalization": {
                "label": "Drug Normalization (Arabic→Latin)",
                "passed": stage_drug["passed"],
                "total": stage_drug["total"],
                "accuracy": round(stage_drug["passed"] / max(1, stage_drug["total"]) * 100, 1),
            },
            "auto_correction": {
                "label": "Auto-Correction (Final Transcript)",
                "passed": stage_corrected["passed"],
                "total": stage_corrected["total"],
                "accuracy": round(stage_corrected["passed"] / max(1, stage_corrected["total"]) * 100, 1),
            },
        },
        "overall": {
            "passed_cases": sum(1 for c in stage_flags["detail_by_case"] if c["passed"] >= c["total"]),
            "total_cases": n,
        },
    }

    return {"summary": summary, "results": detailed_results}


def run_single_test(case: Dict[str, Any], use_llm: bool = False) -> Dict[str, Any]:
    """Run a single test case and return detailed per-stage results."""
    transcript = case["transcript"]
    expected = case.get("expected", {})
    cid = case.get("id", "?")

    # Stage 1: Flagging
    try:
        flags_out = flag.flag_suspicious(transcript, use_llm=use_llm)
    except Exception as exc:
        flags_out = []
        print(f"[pipeline_test] case {cid}: flag_suspicious failed: {exc!r}")

    flag_result = _compare_flags(flags_out, expected.get("flags", []))

    # Stage 2: Drug normalization
    try:
        normalized_text, drug_fixes = drug_normalize.normalize_drugs(transcript)
    except Exception as exc:
        normalized_text = transcript
        drug_fixes = []
        print(f"[pipeline_test] case {cid}: normalize_drugs failed: {exc!r}")

    drug_result = _compare_drug_corrections(
        drug_fixes, expected.get("drug_corrections", [])
    )

    # Stage 3: Auto-correction
    try:
        corr_flags = flag.flag_suspicious(normalized_text, use_llm=use_llm)
        corr_result = flag.apply_high_confidence_corrections(normalized_text, corr_flags)
        final_text = corr_result["corrected_transcript"]
        auto_applied = corr_result["applied"]
    except Exception as exc:
        final_text = normalized_text
        auto_applied = []
        print(f"[pipeline_test] case {cid}: auto-correction failed: {exc!r}")

    corr_compare = _compare_corrected(final_text, expected.get("corrected", transcript))

    return {
        "id": cid,
        "transcript": transcript,
        "stages": {
            "scoring_and_flagging": {
                "passed": flag_result["passed"],
                "total": flag_result["total"],
                "false_positives": flag_result["false_positives"],
                "details": flag_result["details"],
                "false_positive_details": flag_result["false_positive_details"],
                "actual_flags": [
                    {
                        "index": f.get("index"),
                        "word": f.get("word"),
                        "reason": f.get("reason"),
                        "top_candidate": f["candidates"][0] if f.get("candidates") else None,
                    }
                    for f in flags_out
                ],
            },
            "drug_normalization": {
                "passed": drug_result["passed"],
                "total": drug_result["total"],
                "details": drug_result["details"],
                "mismatches": drug_result["mismatches"],
                "normalized_text": normalized_text,
                "fixes_applied": drug_fixes,
            },
            "auto_correction": {
                "passed": corr_compare["passed"],
                "total": corr_compare["total"],
                "exact_match": corr_compare.get("exact_match", False),
                "actual": corr_compare.get("actual", ""),
                "expected": corr_compare.get("expected", ""),
                "corrections_applied": auto_applied,
            },
        },
    }
