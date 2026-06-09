#!/usr/bin/env python3
"""
eval_project.py
===============
End-to-end evaluation: remote ASR → correction pipeline.

Loads every clip from eval/gulf_medical_v1/manifest.jsonl, sends each
audio file to the remote ASR endpoint (REMOTE_ASR_URL env var), then
feeds the transcript into the correction pipeline.

Metrics
-------
  WER         — Word Error Rate between ASR output and ground-truth transcript
  Term recall — fraction of expected medical terms identified/corrected
  FP rate     — fraction of clean clips (saudi_acoustic) that got corrections
  Pipeline F1 — how well the pipeline flagged+corrected the medical terms

Usage
-----
    # Set env var first (or export it in your shell)
    set REMOTE_ASR_URL=http://flavia-overmellow-uncially.ngrok-free.dev

    python scripts/eval_project.py
    python scripts/eval_project.py --category medical_vocab_ar
    python scripts/eval_project.py --asr-only          # skip pipeline, just WER
    python scripts/eval_project.py --pipeline-only     # skip ASR, use ground-truth text
    python scripts/eval_project.py --limit 10          # first N clips
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval" / "gulf_medical_v1"
MANIFEST_PATH = EVAL_DIR / "manifest.jsonl"
AUDIO_DIR = EVAL_DIR / "audio"

DEFAULT_ENDPOINT = "http://localhost:8000"
DEFAULT_REMOTE_ASR = os.environ.get("REMOTE_ASR_URL", "").rstrip("/")


# ---------------------------------------------------------------------------
# Text normalisation (WER)
# ---------------------------------------------------------------------------

_AR_DIAC = re.compile(r"[ً-ٰٟـ]")
_PUNCT   = re.compile(r"[^\w\s]", re.UNICODE)
_WS      = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = _AR_DIAC.sub("", text)
    text = _PUNCT.sub(" ", text)
    text = text.lower()
    return _WS.sub(" ", text).strip()


def wer(ref: str, hyp: str) -> float:
    """Compute Word Error Rate between reference and hypothesis."""
    r = normalize(ref).split()
    h = normalize(hyp).split()
    if not r:
        return 0.0 if not h else 1.0
    # Dynamic programming edit distance on word sequences
    d = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        prev = d[:]
        d[0] = i
        for j, hw in enumerate(h, 1):
            d[j] = prev[j - 1] if rw == hw else 1 + min(prev[j], d[j - 1], prev[j - 1])
    return d[len(h)] / len(r)


# ---------------------------------------------------------------------------
# Remote ASR call
# ---------------------------------------------------------------------------


def call_remote_asr(audio_path: Path, language: str, remote_url: str, timeout: int = 90, retries: int = 2) -> str | None:
    """POST audio to the remote ASR endpoint. Returns transcript text or None on error."""
    for attempt in range(1, retries + 2):
        try:
            with audio_path.open("rb") as fh:
                resp = requests.post(
                    f"{remote_url}/asr",
                    files={"audio": (audio_path.name, fh, "audio/wav")},
                    data={"language": language},
                    timeout=timeout,
                )
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "application/json" in ct:
                data = resp.json()
                return data.get("text") or data.get("transcript") or ""
            return resp.text.strip()
        except Exception as exc:
            if attempt <= retries:
                print(f"  ! ASR attempt {attempt} failed ({type(exc).__name__}), retrying…", end=" ", flush=True)
                time.sleep(1.5 * attempt)
            else:
                print(f"  ! ASR error for {audio_path.name}: {type(exc).__name__}")
                return None


# ---------------------------------------------------------------------------
# Pipeline call (reuse eval_pipeline logic)
# ---------------------------------------------------------------------------


def call_pipeline(endpoint: str, transcript: str, case_id: str, timeout: int = 120, use_llm: bool = False) -> dict:
    try:
        body = {"transcript": transcript, "case_id": case_id}
        if use_llm:
            body["use_llm"] = True
        resp = requests.post(
            f"{endpoint}/api/test-pipeline",
            json=body,
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Per-clip scoring
# ---------------------------------------------------------------------------


@dataclass
class ClipResult:
    clip_id: str
    category: str
    language: str
    ground_truth: str
    asr_transcript: str | None
    asr_wer: float | None
    expected_terms: list[str]
    pipeline_flagged: list[str]
    pipeline_corrections: list[dict]
    term_recall: float   # fraction of expected_terms found/corrected
    fp_corrections: int  # corrections made when expected_terms is empty
    error: str | None = None


def score_clip(
    manifest_entry: dict,
    asr_transcript: str | None,
    pipeline_response: dict | None,
) -> ClipResult:
    clip_id  = manifest_entry["id"]
    category = manifest_entry["category"]
    language = manifest_entry.get("language", "ar")
    gt       = manifest_entry.get("transcript", "")
    expected = [t.lower() for t in manifest_entry.get("medical_terms", [])]

    asr_wer = None
    if asr_transcript is not None:
        asr_wer = round(wer(gt, asr_transcript), 4)

    flagged     = []
    corrections = []
    error       = None

    if pipeline_response:
        if "error" in pipeline_response:
            error = pipeline_response["error"]
        else:
            flagged     = [s.get("text", "") for s in pipeline_response.get("flagged_spans", [])]
            corrections = pipeline_response.get("corrections", [])

    # Term recall: did the pipeline flag or correct each expected term?
    # A term is also considered "found" if it was already correctly spelled in
    # the input transcript — in that case the pipeline correctly left it alone.
    recall_hits = 0
    normalized_gt = normalize(gt)
    flagged_spans_raw = pipeline_response.get("flagged_spans", []) if pipeline_response else []
    for term in expected:
        found = term in normalized_gt
        if not found:
            found = any(
                term in (c.get("chosen") or "").lower() or
                term in (c.get("span_text") or "").lower()
                for c in corrections
            )
        if not found:
            found = any(term in (c.get("text") or "").lower() for c in flagged_spans_raw)
        if found:
            recall_hits += 1

    term_recall = recall_hits / len(expected) if expected else 1.0

    # False positives on clean clips: corrections applied when no terms expected
    fp_count = 0
    if not expected:
        fp_count = len([
            c for c in corrections
            if c.get("path") not in ("hitl_escalate", "no_change")
        ])

    return ClipResult(
        clip_id=clip_id,
        category=category,
        language=language,
        ground_truth=gt,
        asr_transcript=asr_transcript,
        asr_wer=asr_wer,
        expected_terms=expected,
        pipeline_flagged=flagged,
        pipeline_corrections=corrections,
        term_recall=round(term_recall, 4),
        fp_corrections=fp_count,
        error=error,
    )


# ---------------------------------------------------------------------------
# Report printing
# ---------------------------------------------------------------------------


def print_report(results: list[ClipResult], elapsed: float, args: argparse.Namespace) -> None:
    from collections import defaultdict
    by_cat: dict[str, list[ClipResult]] = defaultdict(list)
    for r in results:
        by_cat[r.category].append(r)

    width = 80
    print()
    print("=" * width)
    print("  EVAL PROJECT REPORT")
    print(f"  Remote ASR : {args.remote_asr or '(none — pipeline-only mode)'}")
    print(f"  Endpoint   : {args.endpoint}")
    print(f"  Date       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration   : {elapsed:.1f}s  |  Clips: {len(results)}")
    print("=" * width)

    hdr = f"{'Category':<22} {'Clips':>5} {'ASR WER':>8} {'TermRecall':>11} {'FP/clip':>8}"
    print(hdr)
    print("-" * width)

    for cat, cat_results in sorted(by_cat.items()):
        wers = [r.asr_wer for r in cat_results if r.asr_wer is not None]
        avg_wer = sum(wers) / len(wers) if wers else None
        recalls = [r.term_recall for r in cat_results if r.expected_terms]
        avg_recall = sum(recalls) / len(recalls) if recalls else None
        fp_clips = [r for r in cat_results if not r.expected_terms]
        fp_rate = sum(r.fp_corrections > 0 for r in fp_clips) / len(fp_clips) if fp_clips else None
        print(
            f"  {cat:<20} {len(cat_results):>5}"
            f"  {f'{avg_wer:.3f}' if avg_wer is not None else '  N/A':>8}"
            f"  {f'{avg_recall:.3f}' if avg_recall is not None else '  N/A':>11}"
            f"  {f'{fp_rate:.3f}' if fp_rate is not None else '  N/A':>8}"
        )

    print("-" * width)
    all_wers = [r.asr_wer for r in results if r.asr_wer is not None]
    all_recalls = [r.term_recall for r in results if r.expected_terms]
    all_fp_clips = [r for r in results if not r.expected_terms]
    print(
        f"  {'OVERALL':<20} {len(results):>5}"
        f"  {f'{sum(all_wers)/len(all_wers):.3f}' if all_wers else '  N/A':>8}"
        f"  {f'{sum(all_recalls)/len(all_recalls):.3f}' if all_recalls else '  N/A':>11}"
        f"  {f'{sum(r.fp_corrections>0 for r in all_fp_clips)/len(all_fp_clips):.3f}' if all_fp_clips else '  N/A':>8}"
    )
    print("=" * width)

    # Show per-clip details for errors and low-recall cases
    errors = [r for r in results if r.error]
    low_recall = [r for r in results if r.expected_terms and r.term_recall < 0.5]
    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for r in errors:
            print(f"    {r.clip_id}: {r.error}")
    if low_recall:
        print(f"\n  Low term recall (<50%) — {len(low_recall)} clips:")
        for r in low_recall[:10]:
            print(f"    {r.clip_id} [{r.category}]: expected={r.expected_terms}  flagged={r.pipeline_flagged}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end eval: remote ASR → pipeline")
    parser.add_argument("--manifest",      default=str(MANIFEST_PATH), help="Path to manifest.jsonl")
    parser.add_argument("--audio-dir",     default=str(AUDIO_DIR),     help="Directory containing WAV files")
    parser.add_argument("--remote-asr",    default=DEFAULT_REMOTE_ASR, help="Remote ASR base URL (env: REMOTE_ASR_URL)")
    parser.add_argument("--endpoint",      default=DEFAULT_ENDPOINT,   help="Pipeline server URL")
    parser.add_argument("--category",      default=None,               help="Filter to one category")
    parser.add_argument("--limit",         type=int, default=None,     help="Process only first N clips")
    parser.add_argument("--asr-only",      action="store_true",        help="Only run ASR, skip pipeline")
    parser.add_argument("--pipeline-only", action="store_true",        help="Skip ASR, feed ground-truth text to pipeline")
    parser.add_argument("--use-llm",       action="store_true",        help="Enable the LLM pass in the correction pipeline (non-deterministic)")
    parser.add_argument("--delay",         type=float, default=0.2,    help="Delay between requests (s)")
    parser.add_argument("--timeout",       type=int,   default=90,     help="Per-request timeout (s)")
    parser.add_argument("--output",        default=None,               help="JSON output path")
    args = parser.parse_args()

    if not args.pipeline_only and not args.remote_asr:
        print("WARNING: REMOTE_ASR_URL not set and --pipeline-only not specified.")
        print("         Use --pipeline-only to feed ground-truth text to the pipeline,")
        print("         or set REMOTE_ASR_URL / --remote-asr to use the remote ASR.")
        print()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    # Load manifest
    clips: list[dict] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                clips.append(json.loads(line))

    if args.category:
        clips = [c for c in clips if c.get("category") == args.category]
        print(f"Filtered to category '{args.category}': {len(clips)} clips")
    if args.limit:
        clips = clips[:args.limit]

    print(f"Loaded {len(clips)} clips from {manifest_path}")
    print(f"Remote ASR : {args.remote_asr or 'disabled (pipeline-only)'}")
    print(f"Pipeline   : {args.endpoint}")
    print()

    audio_dir = Path(args.audio_dir)
    results: list[ClipResult] = []
    start = time.time()
    errors = 0

    for i, entry in enumerate(clips, 1):
        clip_id   = entry["id"]
        language  = entry.get("language", "ar")
        if language == "mixed":
            language = "ar"

        audio_file = audio_dir / entry.get("audio_path", f"audio/{clip_id}.wav").lstrip("audio/")
        if not audio_file.exists():
            # try directly
            audio_file = audio_dir / f"{clip_id}.wav"

        print(f"  [{i:03d}/{len(clips)}] {clip_id} ...", end=" ", flush=True)
        t0 = time.time()

        # Step 1: ASR
        asr_transcript: str | None = None
        if args.pipeline_only:
            asr_transcript = entry.get("transcript", "")
        elif args.remote_asr and audio_file.exists():
            asr_transcript = call_remote_asr(audio_file, language, args.remote_asr, timeout=args.timeout)
        elif not audio_file.exists():
            print(f"MISSING AUDIO", end=" ")
            asr_transcript = entry.get("transcript", "")  # fall back to ground truth

        # Step 2: Pipeline
        pipeline_response: dict | None = None
        if not args.asr_only and asr_transcript:
            pipeline_response = call_pipeline(args.endpoint, asr_transcript, clip_id, timeout=args.timeout, use_llm=args.use_llm)
            if "error" in (pipeline_response or {}):
                errors += 1

        result = score_clip(entry, asr_transcript, pipeline_response)
        results.append(result)

        elapsed_clip = time.time() - t0
        wer_str = f"WER={result.asr_wer:.2f}" if result.asr_wer is not None else ""
        recall_str = f"recall={result.term_recall:.2f}" if result.expected_terms else "clean"
        print(f"{wer_str}  {recall_str}  [{elapsed_clip:.1f}s]")

        if i < len(clips) and args.delay > 0:
            time.sleep(args.delay)

    elapsed = time.time() - start
    print(f"\nCompleted {len(clips)} clips in {elapsed:.1f}s. Pipeline errors: {errors}")

    print_report(results, elapsed, args)

    # Write JSON output
    output_dir = PROJECT_ROOT / "eval_results"
    output_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output or str(output_dir / f"eval_project_{ts}.json")

    cat_label = f"_{args.category}" if args.category else ""
    if not args.output:
        output_path = str(output_dir / f"eval_project{cat_label}_{ts}.json")

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump({
            "meta": {
                "remote_asr": args.remote_asr,
                "endpoint": args.endpoint,
                "manifest": str(manifest_path),
                "clips": len(results),
                "errors": errors,
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": round(elapsed, 2),
            },
            "results": [asdict(r) for r in results],
        }, fh, ensure_ascii=False, indent=2)

    print(f"Report written to: {output_path}\n")


if __name__ == "__main__":
    main()
