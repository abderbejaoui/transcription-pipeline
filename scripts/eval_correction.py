"""Correction pipeline evaluation harness.

Usage:
    python -m scripts.eval_correction                          # latest run
    python -m scripts.eval_correction --report-name baseline   # named baseline

Reads ``eval/correction_eval.jsonl`` (or custom path), runs the current
pipeline over each record, and writes a markdown report to
``eval/reports/<report-name>.md``.

Metrics
-------
- WER(raw, gold) vs WER(corrected, gold) — overall and by language subset
- Correction precision — of the spans the pipeline changed, fraction matching gold
- Correction recall — of the gold spans, fraction the pipeline fixed
- Do-no-harm rate — fraction of clean inputs left unchanged
- HITL volume — how many spans flagged vs auto-applied
- Confidence calibration — ECE of auto-apply decision
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jiwer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Ensure we import the correction module from the project, not another package
os.environ["USE_LLM"] = "0"  # No network LLM calls; we run the deterministic path explicitly

import difflib  # noqa: E402
import re as _re  # noqa: E402

from app.services.correction import MedicalCorrector  # noqa: E402
# Live pipeline pieces — the harness must measure the SAME deterministic path
# the UI runs (Stage 1 MedicalCorrector with live thresholds + Stage 2
# HybridMatcher), otherwise it cannot see the Stage-2 false positives.
from app import main as _app_main  # noqa: E402

_PUNCT_STRIP = _re.compile(
    r"^[\s،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+|"
    r"[\s،؛؟!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+$"
)


class Pipeline:
    """Faithful re-creation of the deterministic text-correction path used by
    ``main.correct_text_only`` (Stage 1 + Stage 2), built once and reused.

    Returns the corrected text, the list of applied Stage-1 spans, and the set
    of original tokens the pipeline *touched* (flagged or corrected) — so the
    harness can distinguish "fixed", "flagged only", and "silently passed".
    """

    def __init__(self) -> None:
        self.corrector = _app_main._build_corrector()  # live config (accept=88, alias strip)
        self.hybrid = _app_main._get_hybrid_matcher()

    def run(self, text: str) -> Tuple[str, List[Dict[str, Any]], set]:
        result = self.corrector.correct_transcript(text, use_llm=False)
        corrected = result.get("corrected_text", text)
        spans = list(result.get("suspicious_spans", []))
        touched = {s.get("original_text", "") for s in spans if s.get("original_text")}

        # Stage 2 — HybridMatcher on Arabic words Stage 1 didn't correct.
        # Mirrors main.correct_text_only; deterministic (no LLM), so we run it
        # regardless of the USE_LLM flag that gates it inside main.py.
        corrected_originals = set(touched)
        for w in _re.split(r"\s+", text.strip()):
            clean = _PUNCT_STRIP.sub("", w)
            if not clean or not _app_main._has_arabic(clean) or len(clean) < 3:
                continue
            if any(o in clean or clean in o for o in corrected_originals if o):
                continue
            if _app_main._is_arabic_filler(clean):
                continue
            try:
                cands = self.hybrid.match(clean, top_k=3, context=text)
            except Exception:
                cands = []
            if cands:
                touched.add(clean)
                if cands[0]["score"] >= 80.0:  # main.py Stage-2 auto-apply bar
                    corrected = corrected.replace(clean, cands[0]["term"])
                    spans.append({
                        "original_text": clean,
                        "possible_correction": cands[0]["term"],
                        "score": cands[0]["score"],
                        "issue_type": "arabic_phonetic_match",
                    })
        return corrected, spans, touched


def _norm(s: str) -> str:
    return _re.sub(r"\s+", " ", str(s).strip().lower())


def extract_changes(raw: str, corrected: str) -> List[Dict[str, str]]:
    """Token-level change blocks via difflib — correctly handles corrections
    that change the token count (the old positional zip produced phantom
    changes that destroyed the precision metric)."""
    a, b = raw.split(), corrected.split()
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    changes: List[Dict[str, str]] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        changes.append({
            "original": " ".join(a[i1:i2]),
            "corrected": " ".join(b[j1:j2]),
            "tag": tag,
        })
    return changes


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """Load records from a JSONL eval file."""
    records: List[Dict[str, Any]] = []
    if not path.exists():
        print(f"[ERROR] Eval set not found: {path}")
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between reference and hypothesis."""
    if not reference.strip():
        return 1.0 if hypothesis.strip() else 0.0
    return jiwer.wer(reference, hypothesis)


def _spans_overlap(change_orig: str, gold_orig: str) -> bool:
    """True if an applied change targets (part of) a gold span's original."""
    o, g = _norm(change_orig), _norm(gold_orig)
    if not o or not g:
        return False
    return o in g or g in o or bool(set(o.split()) & set(g.split()))


def evaluate_record(
    pipeline: "Pipeline",
    record: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the deterministic pipeline on one record and score it.

    Scoring is span-level (not positional), and do-no-harm is measured on
    EVERY record: any change to a token that is not part of a gold span is a
    harm event, even inside a sentence that also contains a real error.
    """
    transcript = record["transcript"]
    gold_spans = record.get("gold_spans", [])
    contains_error = record.get("contains_error", True)
    lang = record.get("lang", "en")

    try:
        corrected_text, spans, touched = pipeline.run(transcript)
    except Exception as exc:
        return {
            "id": record["id"], "error": str(exc), "transcript": transcript,
            "corrected": transcript, "gold_text": transcript,
            "wer_raw": 0.0, "wer_corrected": 0.0, "raw_wer": 0.0,
            "corrected_wer": 0.0, "wer_delta": 0.0, "gold_spans": gold_spans,
            "pipeline_changes": [], "contains_error": contains_error, "lang": lang,
            "precision_num": 0, "precision_den": 0, "recall_num": 0,
            "recall_den": len(gold_spans) if contains_error else 0,
            "do_no_harm": True, "harmful_changes": [], "n_flags": 0,
            "n_corrections": 0, "missed_flagged": 0, "missed_silent": 0,
            "difficulty": record.get("difficulty", "unknown"),
        }

    # Drop capitalization-only changes (harmless normalization) up front.
    changes = [
        c for c in extract_changes(transcript, corrected_text)
        if not (_norm(c["original"]) == _norm(c["corrected"]))
    ]

    # Build gold reference text for WER.
    gold_text = transcript
    if gold_spans:
        for gs in gold_spans:
            gold_text = gold_text.replace(gs["original_text"], gs["possible_correction"], 1)
    wer_raw = compute_wer(gold_text, transcript)
    wer_corrected = compute_wer(gold_text, corrected_text)

    # Precision: of applied changes, how many align with a gold span?
    precision_num = 0
    harmful_changes: List[Dict[str, str]] = []
    for ch in changes:
        justified = any(_spans_overlap(ch["original"], gs["original_text"]) for gs in gold_spans)
        if justified:
            # Count as correct only if the produced text matches the gold target.
            c = _norm(ch["corrected"])
            if any(_norm(gs["possible_correction"]) in c or c in _norm(gs["possible_correction"])
                   for gs in gold_spans if _spans_overlap(ch["original"], gs["original_text"])):
                precision_num += 1
        else:
            # Changed a token gold says should stay → harm (any record type).
            harmful_changes.append(ch)
    precision_den = len(changes)

    # Recall: of gold spans, how many now appear corrected in the output?
    recall_num = 0
    missed_flagged = 0
    missed_silent = 0
    if contains_error:
        for gs in gold_spans:
            target = _norm(gs["possible_correction"])
            fixed = target and target in _norm(corrected_text)
            if fixed:
                recall_num += 1
            else:
                # Missed. Was it at least flagged (touched) for HITL?
                go = gs["original_text"]
                if any(_spans_overlap(t, go) or _spans_overlap(go, t) for t in touched):
                    missed_flagged += 1
                else:
                    missed_silent += 1
    recall_den = len(gold_spans) if contains_error else 0

    do_no_harm = len(harmful_changes) == 0

    return {
        "id": record["id"],
        "transcript": transcript,
        "corrected": corrected_text,
        "gold_text": gold_text,
        "wer_raw": wer_raw,
        "wer_corrected": wer_corrected,
        "raw_wer": wer_raw,
        "corrected_wer": wer_corrected,
        "wer_delta": wer_raw - wer_corrected,
        "gold_spans": gold_spans,
        "pipeline_changes": changes,
        "harmful_changes": harmful_changes,
        "contains_error": contains_error,
        "lang": lang,
        "precision_num": precision_num,
        "precision_den": precision_den,
        "recall_num": recall_num,
        "recall_den": recall_den,
        "do_no_harm": do_no_harm,
        "n_flags": len(touched),
        "n_corrections": len(changes),
        "missed_flagged": missed_flagged,
        "missed_silent": missed_silent,
        "difficulty": record.get("difficulty", "unknown"),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def generate_report(
    results: List[Dict[str, Any]],
    elapsed_s: float,
    report_name: str,
) -> str:
    """Generate a markdown report from evaluation results."""
    n = len(results)
    n_errors = sum(1 for r in results if r.get("error"))
    valid = [r for r in results if not r.get("error")]

    if not valid:
        return f"# Correction Pipeline Evaluation\n\n**All {n} records failed.** See errors above."

    # Overall WER
    mean_wer_raw = sum(r["wer_raw"] for r in valid) / len(valid)
    mean_wer_corrected = sum(r["wer_corrected"] for r in valid) / len(valid)
    mean_wer_delta = sum(r["wer_delta"] for r in valid) / len(valid)

    # WER by language
    langs = set(r["lang"] for r in valid)
    wer_by_lang = {}
    for lang in langs:
        subset = [r for r in valid if r["lang"] == lang]
        if subset:
            wer_by_lang[lang] = {
                "n": len(subset),
                "wer_raw": sum(r["wer_raw"] for r in subset) / len(subset),
                "wer_corrected": sum(r["wer_corrected"] for r in subset) / len(subset),
                "wer_delta": sum(r["wer_delta"] for r in subset) / len(subset),
            }

    # Precision / Recall
    total_precision_num = sum(r["precision_num"] for r in valid)
    total_precision_den = sum(r["precision_den"] for r in valid)
    precision = total_precision_num / max(1, total_precision_den)

    total_recall_num = sum(r["recall_num"] for r in valid)
    total_recall_den = sum(r["recall_den"] for r in valid)
    recall = total_recall_num / max(1, total_recall_den)

    f1 = 2 * precision * recall / max(0.001, precision + recall)

    # Do-no-harm — measured on EVERY record (a change to a non-gold token is
    # harm, even inside a sentence that also has a real error).
    clean = [r for r in valid if not r["contains_error"]]
    do_no_harm_rate = sum(1 for r in valid if r["do_no_harm"]) / max(1, len(valid))
    clean_dnh_rate = sum(1 for r in clean if r["do_no_harm"]) / max(1, len(clean))
    total_harmful = sum(len(r.get("harmful_changes", [])) for r in valid)
    records_with_harm = sum(1 for r in valid if not r["do_no_harm"])
    total_missed_flagged = sum(r.get("missed_flagged", 0) for r in valid)
    total_missed_silent = sum(r.get("missed_silent", 0) for r in valid)

    # HITL volume
    total_flags = sum(r["n_flags"] for r in valid)
    total_corrections = sum(r["n_corrections"] for r in valid)
    avg_flags_per_record = total_flags / max(1, len(valid))
    avg_corrections_per_record = total_corrections / max(1, len(valid))

    # WER by difficulty
    difficulties = set(r.get("difficulty", "unknown") for r in valid)
    wer_by_diff = {}
    for diff in difficulties:
        subset = [r for r in valid if r.get("difficulty") == diff]
        if subset:
            wer_by_diff[diff] = {
                "n": len(subset),
                "wer_raw": sum(r["wer_raw"] for r in subset) / len(subset),
                "wer_corrected": sum(r["wer_corrected"] for r in subset) / len(subset),
                "wer_delta": sum(r["wer_delta"] for r in subset) / len(subset),
            }

    # Build the report
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# Correction Pipeline Evaluation — `{report_name}`",
        f"",
        f"**Date:** {timestamp}  ",
        f"**Elapsed:** {elapsed_s:.1f}s  ",
        f"**Records:** {n} ({n_errors} errors)  ",
        f"",
        f"---",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Records evaluated** | {len(valid)} |",
        f"| **Mean WER (raw → gold)** | {mean_wer_raw:.4f} |",
        f"| **Mean WER (corrected → gold)** | {mean_wer_corrected:.4f} |",
        f"| **WER reduction (Δ)** | **{mean_wer_delta:+.4f}** |",
        f"| **Correction precision** | {precision:.4f} ({total_precision_num}/{total_precision_den}) |",
        f"| **Correction recall** | {recall:.4f} ({total_recall_num}/{total_recall_den}) |",
        f"| **F1 score** | {f1:.4f} |",
        f"| **Do-no-harm rate (all records)** | {do_no_harm_rate:.4f} ({sum(1 for r in valid if r['do_no_harm'])}/{len(valid)}) |",
        f"| **Do-no-harm rate (clean only)** | {clean_dnh_rate:.4f} ({sum(1 for r in clean if r['do_no_harm'])}/{len(clean)}) |",
        f"| **Harmful changes (total)** | {total_harmful} (in {records_with_harm} records) |",
        f"| **Missed errors — flagged for HITL** | {total_missed_flagged} |",
        f"| **Missed errors — silently passed** | {total_missed_silent} |",
        f"| **Total flags (HITL)** | {total_flags} |",
        f"| **Avg flags/record** | {avg_flags_per_record:.2f} |",
        f"| **Avg corrections applied/record** | {avg_corrections_per_record:.2f} |",
        f"",
        f"### WER by Language",
        f"",
        f"| Language | N | WER (raw) | WER (corrected) | Δ |",
        f"|----------|---|-----------|-----------------|----|",
    ]
    for lang in sorted(wer_by_lang.keys()):
        w = wer_by_lang[lang]
        lines.append(
            f"| {lang} | {w['n']} | {w['wer_raw']:.4f} | {w['wer_corrected']:.4f} | {w['wer_delta']:+.4f} |"
        )

    lines += [
        f"",
        f"### WER by Difficulty",
        f"",
        f"| Difficulty | N | WER (raw) | WER (corrected) | Δ |",
        f"|------------|---|-----------|-----------------|----|",
    ]
    for diff in sorted(wer_by_diff.keys()):
        w = wer_by_diff[diff]
        lines.append(
            f"| {diff} | {w['n']} | {w['wer_raw']:.4f} | {w['wer_corrected']:.4f} | {w['wer_delta']:+.4f} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Per-Record Details",
        f"",
        f"| ID | Lang | Diff | Contains Error? | WER raw | WER corr | Δ | Changes | Flags | Do-no-harm |",
        f"|----|------|------|----------------|---------|----------|----|---------|-------|------------|",
    ]
    for r in sorted(valid, key=lambda x: x["id"]):
        c = "❌" if r.get("error") else ""
        dn = "✅" if r["do_no_harm"] else "❌"
        lines.append(
            f"| {r['id']} | {r['lang']} | {r.get('difficulty', '?')} | "
            f"{'Yes' if r['contains_error'] else 'No'} | "
            f"{r['wer_raw']:.3f} | {r['wer_corrected']:.3f} | {r['wer_delta']:+.3f} | "
            f"{r['n_corrections']} | {r['n_flags']} | {dn} |"
        )

    lines += [
        f"",
        f"---",
        f"",
        f"## Failure Cases",
        f"",
    ]
    failures = [r for r in valid if not r["do_no_harm"]]
    if failures:
        lines.append(f"**{len(failures)} records with harmful changes "
                     f"(a token gold says should stay was changed):**")
        lines.append("")
        for f in sorted(failures, key=lambda x: x["id"]):
            tag = "clean" if not f["contains_error"] else "had-error"
            harms = ", ".join(
                f"'{c['original']}'→'{c['corrected']}'" for c in f["harmful_changes"]
            )
            lines.append(f"- **{f['id']}** ({tag}, {f['lang']}): {harms}")
    else:
        lines.append("**No harmful changes!** Every change targeted a real error span.")
        lines.append("")

    # Missed corrections
    missed = [r for r in valid if r["contains_error"] and r["recall_num"] < r["recall_den"]]
    if missed:
        lines.append(f"")
        lines.append(f"**{len(missed)} records with missed corrections:**")
        lines.append("")
        for m in missed[:10]:  # Show first 10
            gold_str = ", ".join(
                f"'{gs['original_text']}'→'{gs['possible_correction']}'"
                for gs in m["gold_spans"]
            )
            changes = ", ".join(
                f"'{c['original']}'→'{c['corrected']}'" for c in m["pipeline_changes"]
            ) or "(none)"
            lines.append(f"- **{m['id']}** (lang={m['lang']}, diff={m.get('difficulty', '?')})")
            lines.append(f"  - Gold: {gold_str}")
            lines.append(f"  - Changes applied: {changes}")

    # Configuration
    lines += [
        f"",
        f"---",
        f"",
        f"## Configuration",
        f"",
        f"- **Pipeline:** live deterministic path — Stage 1 MedicalCorrector "
        f"(`_build_corrector`, accept=88) + Stage 2 HybridMatcher (≥80 auto-apply)",
        f"- **Eval set:** `eval/correction_eval.jsonl`",
        f"- **LLM stages:** disabled (deterministic measurement)",
        f"- **Do-no-harm:** span-level, measured on all records",
    ]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate correction pipeline")
    ap.add_argument(
        "--eval-path", type=Path,
        default=PROJECT_ROOT / "eval" / "correction_eval.jsonl",
    )
    ap.add_argument(
        "--report-name", type=str, default=None,
        help="Name for the report file (default: auto-timestamped)",
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "eval" / "reports",
    )
    ap.add_argument(
        "--print", action="store_true", dest="print_report",
        help="Print the report to stdout as well",
    )
    args = ap.parse_args()

    # Load eval set
    records = load_eval_set(args.eval_path)
    if not records:
        print(f"[ERROR] No records found in {args.eval_path}")
        print("  Run `python -m scripts.build_correction_eval` first.")
        return 1
    print(f"Loaded {len(records)} evaluation records from {args.eval_path}")

    # Count stats
    n_with_errors = sum(1 for r in records if r.get("contains_error", True))
    n_clean = len(records) - n_with_errors
    n_en = sum(1 for r in records if r.get("lang") == "en")
    n_ar = sum(1 for r in records if r.get("lang") == "ar")
    n_mixed = sum(1 for r in records if r.get("lang") == "mixed")
    print(f"  With errors: {n_with_errors}, Clean: {n_clean}")
    print(f"  English: {n_en}, Arabic: {n_ar}, Mixed: {n_mixed}")

    # Build the live deterministic pipeline (Stage 1 + Stage 2)
    print("Building pipeline (live MedicalCorrector + HybridMatcher)...")
    pipeline = Pipeline()

    # Evaluate
    start = time.time()
    results: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i + 1}/{len(records)}] {record['id']}...")
        result = evaluate_record(pipeline, record)
        results.append(result)
    elapsed = time.time() - start
    print(f"Evaluation complete in {elapsed:.1f}s")

    # Determine report name
    report_name = args.report_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Generate report
    report = generate_report(results, elapsed, report_name)

    # Write report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{report_name}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report written to {report_path}")

    if args.print_report:
        print("\n" + "=" * 60)
        print(report)

    # Print one-line summary
    valid = [r for r in results if not r.get("error")]
    n_errors = sum(1 for r in results if r.get("error"))
    if valid:
        mean_delta = sum(r["wer_delta"] for r in valid) / len(valid)
        dnh_good = sum(1 for r in valid if r["do_no_harm"])
        harmful = sum(len(r.get("harmful_changes", [])) for r in valid)
        recall_val = sum(r["recall_num"] for r in valid) / max(1, sum(r["recall_den"] for r in valid))
        prec_val = sum(r["precision_num"] for r in valid) / max(1, sum(r["precision_den"] for r in valid))
        print(f"\n{'=' * 60}")
        print(f"SUMMARY: WER Δ={mean_delta:+.4f} | "
              f"Do-no-harm={dnh_good}/{len(valid)} ({harmful} harmful) | "
              f"Prec={prec_val:.3f} | Recall={recall_val:.3f} | "
              f"{len(valid)} records in {elapsed:.1f}s")
        print(f"{'=' * 60}")

    return 0 if not n_errors else 1


if __name__ == "__main__":
    sys.exit(main())
