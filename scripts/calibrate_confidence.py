"""Calibrated confidence model for the correction pipeline.

Phase 3 of the PROMPT.md spec: fits a logistic regression that maps features
to P(correction is right), then finds optimal auto-apply / HITL / leave-as-is
cut points on the eval set's precision/do-no-harm curve.

Usage:
    python -m scripts.calibrate_confidence                        # run all
    python -m scripts.calibrate_confidence --report-name phase3_cal

Outputs:
    - eval/reports/calibration_<name>.md   — markdown report with metrics
    - eval/models/confidence_model.pkl     — fitted logistic regression
    - eval/models/confidence_thresholds.json — auto-apply/HITL cut points
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score

# Ensure we can import from the project
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.correction import MedicalCorrector  # noqa: E402

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


@dataclass
class CorrectionSample:
    """Features + ground truth for one correction decision."""

    # --- Features (inputs to logistic regression) ---
    phonetic_score: float = 0.0          # 0-100: best candidate's _score_pair
    score_gap: float = 0.0               # 0-100: best - second_best score
    llm_confidence: float = 0.0          # 0-1: LLM self-reported confidence
    n_candidates: int = 0                # count of candidates considered
    is_mixed_script: float = 0.0         # 0 or 1
    span_length: float = 0.0             # len(span_text) chars
    is_arabic: float = 0.0               # 0 or 1
    best_retrieval_score: float = 0.0    # 0-1 from vector lexicon
    n_tokens_in_span: float = 1.0        # how many tokens the span covers
    is_multi_word: float = 0.0           # 0 or 1: multi-token span?

    # --- Ground truth ---
    is_correct: bool = False             # True if correction matches gold
    correction_matched: bool = False     # Did the pipeline produce a correction at all?

    # --- Metadata ---
    eval_id: str = ""
    span_text: str = ""
    gold_correction: str = ""
    pipeline_correction: str = ""
    lang: str = "en"

    @property
    def feature_vector(self) -> np.ndarray:
        """Return ordered feature vector for the logistic regression."""
        return np.array([
            self.phonetic_score / 100.0,        # Normalize 0-100 → 0-1
            self.score_gap / 100.0,              # Normalize 0-100 → 0-1
            self.llm_confidence,                  # Already 0-1
            min(1.0, self.n_candidates / 10.0),  # Cap at 10 → 1.0
            self.is_mixed_script,                 # 0 or 1
            min(1.0, self.span_length / 50.0),   # Cap at 50 chars
            self.is_arabic,                       # 0 or 1
            self.best_retrieval_score,            # Already 0-1
            min(1.0, self.n_tokens_in_span / 6.0),  # Cap at 6 tokens
            self.is_multi_word,                   # 0 or 1
        ], dtype=np.float32)

    @property
    def feature_names(self) -> List[str]:
        return [
            "phonetic_score_norm",
            "score_gap_norm",
            "llm_confidence",
            "n_candidates_norm",
            "is_mixed_script",
            "span_length_norm",
            "is_arabic",
            "best_retrieval_score",
            "n_tokens_norm",
            "is_multi_word",
        ]


# ---------------------------------------------------------------------------
# Run pipeline on eval set and extract features
# ---------------------------------------------------------------------------


def extract_samples(
    corrector: MedicalCorrector,
    records: List[Dict[str, Any]],
    use_llm: bool = False,
) -> List[CorrectionSample]:
    """Run the pipeline on eval records and extract features for each span.

    For each record that contains errors:
      1. Run corrector.correct_transcript()
      2. Get the suspicious_spans + what the pipeline changed
      3. For each gold_span, extract features from the pipeline's best candidate
      4. Record whether the correction was correct or not

    Returns a list of CorrectionSample (one per gold_span).
    """
    samples: List[CorrectionSample] = []

    for record in records:
        if not record.get("contains_error", True):
            continue  # Clean records are used for do-no-harm, not training

        transcript = record["transcript"]
        gold_spans = record.get("gold_spans", [])
        if not gold_spans:
            continue

        # Run the pipeline
        try:
            result = corrector.correct_transcript(transcript, use_llm=use_llm)
        except Exception as exc:
            logger.warning("Pipeline failed for %s: %s", record.get("id"), exc)
            continue

        corrected_text = result.get("corrected_text", transcript)
        suspicious = result.get("suspicious_spans", [])

        # Build a map of what the pipeline changed
        raw_words = transcript.split()
        corr_words = corrected_text.split()
        pipeline_changes: Dict[str, str] = {}
        for i in range(min(len(raw_words), len(corr_words))):
            if raw_words[i] != corr_words[i]:
                pipeline_changes[raw_words[i].lower()] = corr_words[i]

        lang = record.get("lang", "en")
        eval_id = record.get("id", "?")

        for gs in gold_spans:
            span_text = gs["original_text"]
            gold_corr = gs["possible_correction"]
            pipeline_corr = pipeline_changes.get(span_text.lower(), "")

            # Find the pipeline's best candidate for this span
            best_candidate = None
            for s in suspicious:
                if s.get("original_text", "").lower() == span_text.lower():
                    best_candidate = s
                    break

            if best_candidate:
                phonetic_score = float(best_candidate.get("score", 0.0))
                features = best_candidate.get("features", {})
                best_retrieval = features.get("ipa", 0.0) / 100.0
                # Score gap is now stored in features by the pipeline
                score_gap = features.get("score_gap", 0.0)
            else:
                phonetic_score = 0.0
                score_gap = 0.0
                best_retrieval = 0.0

            # Check if pipeline correction matches gold
            is_correct = (
                pipeline_corr.lower().strip() == gold_corr.lower().strip()
                or pipeline_corr.lower().strip() == gold_corr.lower().strip()
            )

            # Mixed-script detection
            has_arabic = bool(re.search(r"[\u0600-\u06FF]", span_text))
            has_latin = bool(re.search(r"[a-zA-Z]", span_text))
            is_mixed = 1.0 if (has_arabic and has_latin) else 0.0

            samples.append(CorrectionSample(
                phonetic_score=phonetic_score,
                score_gap=score_gap,
                llm_confidence=0.0,  # Will be filled when selector is integrated
                n_candidates=0,  # not tracked in serialized output
                is_mixed_script=is_mixed,
                span_length=float(len(span_text)),
                is_arabic=1.0 if has_arabic else 0.0,
                best_retrieval_score=best_retrieval,
                n_tokens_in_span=float(len(span_text.split())),
                is_multi_word=1.0 if len(span_text.split()) > 1 else 0.0,
                is_correct=is_correct,
                correction_matched=bool(pipeline_corr),
                eval_id=eval_id,
                span_text=span_text,
                gold_correction=gold_corr,
                pipeline_correction=pipeline_corr,
                lang=lang,
            ))

    return samples


# ---------------------------------------------------------------------------
# Train logistic regression + find cut points
# ---------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    model: LogisticRegression
    features: List[str]
    train_auc: float
    thresholds: Dict[str, float]  # auto_apply_threshold, hitl_threshold
    metrics_at_thresholds: Dict[str, Dict[str, float]]


def train_confidence_model(
    train_samples: List[CorrectionSample],
) -> CalibrationResult:
    """Train a logistic regression to predict P(correction is right).

    Args:
        train_samples: List of CorrectionSample with ground truth labels.

    Returns:
        CalibrationResult with fitted model, thresholds, and metrics.
    """
    if not train_samples:
        raise ValueError("No training samples provided")

    X = np.array([s.feature_vector for s in train_samples])
    y = np.array([1.0 if s.is_correct else 0.0 for s in train_samples], dtype=np.float32)

    # Train logistic regression with L2 regularization
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)

    # Predict probabilities
    y_prob = model.predict_proba(X)[:, 1]

    # AUC
    try:
        train_auc = float(roc_auc_score(y, y_prob))
    except Exception:
        train_auc = 0.0

    # Find optimal cut points from precision/do-no-harm trade-off
    # We want thresholds for:
    #   1. "auto_apply": high precision (>= 95% on corrections, >= 99% do-no-harm)
    #   2. "hitl_apply": moderate precision (>= 80%), flag for review
    #   3. "leave_as_is": below hitl threshold, don't change

    precision, recall, thresholds_pr = precision_recall_curve(y, y_prob)

    # Compute do-no-harm on clean records if available
    # (We don't have clean records in the training loop here, so we use
    # the precision as a proxy — high precision = low false positive rate)

    best_auto_threshold: float = 0.85  # default
    best_hitl_threshold: float = 0.60  # default

    # Find threshold where precision >= 0.95
    for p, t in zip(precision, thresholds_pr):
        if p >= 0.95:
            best_auto_threshold = float(t)
            break

    # Find threshold where precision >= 0.80
    for p, t in zip(precision, thresholds_pr):
        if p >= 0.80:
            best_hitl_threshold = float(t)
            break

    # Ensure auto >= hitl
    best_auto_threshold = max(best_auto_threshold, best_hitl_threshold + 0.05)

    # Compute metrics at chosen thresholds
    metrics: Dict[str, Dict[str, float]] = {}
    for name, thr in [("auto_apply", best_auto_threshold),
                       ("hitl", best_hitl_threshold),
                       ("permissive", 0.50)]:
        pred_at_thr = (y_prob >= thr).astype(float)
        tp = float(np.sum((pred_at_thr == 1) & (y == 1)))
        fp = float(np.sum((pred_at_thr == 1) & (y == 0)))
        fn = float(np.sum((pred_at_thr == 0) & (y == 1)))
        prec_at_thr = tp / max(1, tp + fp)
        rec_at_thr = tp / max(1, tp + fn)
        f1_at_thr = 2 * prec_at_thr * rec_at_thr / max(0.001, prec_at_thr + rec_at_thr)
        metrics[name] = {
            "threshold": float(thr),
            "precision": float(prec_at_thr),
            "recall": float(rec_at_thr),
            "f1": float(f1_at_thr),
            "n_applied": int(np.sum(pred_at_thr)),
            "n_total": len(y),
        }

    return CalibrationResult(
        model=model,
        features=train_samples[0].feature_names,
        train_auc=float(train_auc),
        thresholds={
            "auto_apply": float(best_auto_threshold),
            "hitl": float(best_hitl_threshold),
            "leave_as_is": 0.0,
        },
        metrics_at_thresholds=metrics,
    )


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_report(
    result: CalibrationResult,
    samples: List[CorrectionSample],
    eval_records: List[Dict[str, Any]],
    report_name: str,
) -> str:
    """Generate a markdown report of the calibration results."""
    n_total = len(samples)
    n_correct = sum(1 for s in samples if s.is_correct)
    n_incorrect = n_total - n_correct

    # Breakdown by language
    by_lang: Dict[str, int] = {}
    for s in samples:
        by_lang[s.lang] = by_lang.get(s.lang, 0) + 1

    # Breakdown by correct/incorrect per language
    correct_by_lang: Dict[str, int] = {}
    for s in samples:
        if s.is_correct:
            correct_by_lang[s.lang] = correct_by_lang.get(s.lang, 0) + 1

    lines = [
        f"# Confidence Model Calibration — `{report_name}`",
        f"",
        f"## Summary",
        f"",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| **Training samples** | {n_total} |",
        f"| **Correct corrections** | {n_correct} ({n_correct/max(1,n_total)*100:.1f}%) |",
        f"| **Incorrect corrections** | {n_incorrect} |",
        f"| **AUC (train)** | {result.train_auc:.4f} |",
        f"| **Features** | {len(result.features)} |",
        f"",
        f"### Features Used",
        f"",
    ]
    for i, name in enumerate(result.features):
        coef = result.model.coef_[0][i]
        lines.append(f"- `{name}`: coefficient = {coef:.4f}")

    lines += [
        f"",
        f"### Breakdown by Language",
        f"",
        f"| Language | Samples | Correct | Accuracy |",
        f"|----------|---------|---------|----------|",
    ]
    for lang in sorted(by_lang.keys()):
        total = by_lang[lang]
        corr = correct_by_lang.get(lang, 0)
        lines.append(f"| {lang} | {total} | {corr} | {corr/max(1,total)*100:.1f}% |")

    lines += [
        f"",
        f"## Thresholds & Metrics",
        f"",
        f"| Threshold Name | Value | Precision | Recall | F1 | Applied/Total |",
        f"|----------------|-------|-----------|--------|-----|--------------|",
    ]
    for name, m in sorted(result.metrics_at_thresholds.items()):
        lines.append(
            f"| {name} | {m['threshold']:.3f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['n_applied']}/{m['n_total']} |"
        )

    lines += [
        f"",
        f"## Recommended Operating Point",
        f"",
        f"- **Auto-apply threshold**: {result.thresholds['auto_apply']:.3f} "
        f"(P(correct) >= this → apply silently)",
        f"- **HITL threshold**: {result.thresholds['hitl']:.3f} "
        f"(P(correct) >= this → apply but flag for review)",
        f"- **Below HITL**: leave span unchanged, flag for human correction",
        f"",
        f"## Feature Coefficients",
        f"",
        f"| Feature | Coefficient | Direction |",
        f"|---------|-------------|-----------|",
    ]
    for i, name in enumerate(result.features):
        coef = result.model.coef_[0][i]
        direction = "positive" if coef > 0 else "negative"
        lines.append(f"| {name} | {coef:.4f} | {direction} |")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_eval_set(path: Path) -> List[Dict[str, Any]]:
    """Load eval records from JSONL."""
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate confidence model for correction pipeline")
    ap.add_argument(
        "--eval-path", type=Path,
        default=PROJECT_ROOT / "eval" / "correction_eval.jsonl",
    )
    ap.add_argument(
        "--report-name", type=str, default=None,
    )
    ap.add_argument(
        "--output-dir", type=Path,
        default=PROJECT_ROOT / "eval" / "reports",
    )
    ap.add_argument(
        "--model-dir", type=Path,
        default=PROJECT_ROOT / "eval" / "models",
    )
    args = ap.parse_args()

    # Load eval set
    records = load_eval_set(args.eval_path)
    if not records:
        logger.error("No eval records found at %s", args.eval_path)
        return 1

    # Split into dev (train) and test
    dev_records = [r for r in records if r.get("split") in ("dev", None, "")]
    test_records = [r for r in records if r.get("split") == "test"]

    logger.info(
        "Loaded %d records (dev=%d, test=%d)",
        len(records), len(dev_records), len(test_records),
    )

    # Build corrector
    logger.info("Building MedicalCorrector...")
    corrector = MedicalCorrector()

    # Extract features from dev split (train)
    logger.info("Extracting features from %d dev records...", len(dev_records))
    start = time.time()
    train_samples = extract_samples(corrector, dev_records, use_llm=False)
    elapsed = time.time() - start
    logger.info(
        "Extracted %d training samples in %.1fs",
        len(train_samples), elapsed,
    )

    if len(train_samples) < 10:
        logger.error("Too few training samples (%d) for calibration", len(train_samples))
        return 1

    # Train model
    logger.info("Training logistic regression...")
    result = train_confidence_model(train_samples)
    logger.info("Train AUC: %.4f", result.train_auc)

    # Save model
    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "confidence_model.pkl"
    thresholds_path = args.model_dir / "confidence_thresholds.json"

    import joblib
    joblib.dump(result.model, model_path)
    with thresholds_path.open("w", encoding="utf-8") as f:
        json.dump({
            "auto_apply": result.thresholds["auto_apply"],
            "hitl": result.thresholds["hitl"],
            "leave_as_is": 0.0,
            "features": result.features,
            "train_auc": result.train_auc,
            "n_train_samples": len(train_samples),
            "metrics_at_thresholds": result.metrics_at_thresholds,
        }, f, indent=2)

    logger.info("Model saved to %s", model_path)
    logger.info("Thresholds saved to %s", thresholds_path)

    # Also save thresholds in a format correction.py can read
    auto_path = args.model_dir / "auto_apply_threshold.txt"
    hitl_path = args.model_dir / "hitl_threshold.txt"
    auto_path.write_text(str(result.thresholds["auto_apply"]))
    hitl_path.write_text(str(result.thresholds["hitl"]))

    # Generate report
    report_name = args.report_name or f"phase3_calibration"
    report = generate_report(
        result, train_samples, dev_records, report_name,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f"{report_name}.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)
    logger.info("Report saved to %s", report_path)

    # Print summary
    print(f"\n{'=' * 60}")
    print(f"CALIBRATION COMPLETE")
    print(f"  Train AUC:    {result.train_auc:.4f}")
    print(f"  Auto-apply:   P >= {result.thresholds['auto_apply']:.3f}")
    print(f"  HITL:         P >= {result.thresholds['hitl']:.3f}")
    print(f"  Samples:      {len(train_samples)}")
    for name, m in sorted(result.metrics_at_thresholds.items()):
        print(f"  {name}: precision={m['precision']:.3f}, recall={m['recall']:.3f}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
