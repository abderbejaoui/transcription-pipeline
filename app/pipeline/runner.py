"""Orchestrate the five-stage medical correction pipeline."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List

from .decider import decide_spans
from .flagger import flag_suspicious_spans
from .hitl import apply_human_correction, prompt_for_human_correction
from .models import Decision, PipelineResult, ScoredWord, SpanWithCandidates, SuspiciousSpan
from .retriever import retrieve_candidates
from .scorer import score_transcript, tokenize_transcript


def _span_char_offsets(transcript: str, scored_words: List[ScoredWord], span: SuspiciousSpan) -> tuple[int, int]:
    tokens = tokenize_transcript(transcript)
    if not tokens:
        return 0, 0
    start_index = span.start
    end_index = span.end
    start = tokens[start_index][1]
    end = tokens[end_index][2]
    return start, end


def _apply_replacements(transcript: str, scored_words: List[ScoredWord], decisions: List[Decision]) -> str:
    pieces = transcript
    offsets: List[tuple[int, int, str]] = []
    for decision in decisions:
        if not decision.chosen:
            continue
        start, end = _span_char_offsets(transcript, scored_words, decision.span)
        offsets.append((start, end, decision.chosen))
    for start, end, replacement in sorted(offsets, key=lambda item: item[0], reverse=True):
        pieces = pieces[:start] + replacement + pieces[end:]
    return pieces


def run_pipeline(transcript: str, interactive: bool = True) -> PipelineResult:
    scored_words = score_transcript(transcript)
    spans = flag_suspicious_spans(scored_words)
    span_candidates = [retrieve_candidates(span) for span in spans]
    decisions = decide_spans(transcript, span_candidates)

    corrected_text = _apply_replacements(transcript, scored_words, decisions)
    report: Dict[str, Any] = {
        "input": transcript,
        "scored_words": [asdict(word) for word in scored_words],
        "spans": [asdict(span) for span in spans],
        "candidates": [
            {"span": asdict(item.span), "candidates": [asdict(candidate) for candidate in item.candidates]}
            for item in span_candidates
        ],
        "decisions": [asdict(decision) for decision in decisions],
        "interactive": interactive,
    }

    if interactive:
        for item, decision in zip(span_candidates, decisions):
            if decision.path != "hitl_escalate":
                continue
            correction = prompt_for_human_correction(transcript, item.span.text, best_guess=item.candidates[0].term if item.candidates else None)
            if not correction:
                continue
            corrected_text, _ = apply_human_correction(corrected_text, item.span.text, correction)

    return PipelineResult(
        corrected_text=corrected_text,
        report=report,
        scored_words=scored_words,
        spans=spans,
        candidates=span_candidates,
        decisions=decisions,
    )