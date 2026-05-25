"""Orchestrate the five-stage medical correction pipeline."""

from __future__ import annotations

import os
import uuid
from dataclasses import asdict
from typing import Any, Dict, List

from app.services.llm import get_last_provider as _get_last_provider
from app.services.llm_config import get_llm_model as _get_llm_model

from .decider import decide_spans
from .flagger import flag_suspicious_spans
from .hitl import apply_human_correction, prompt_for_human_correction
from .models import Decision, PipelineResult, ScoredWord, SpanWithCandidates, SuspiciousSpan
from .retriever import retrieve_candidates
from .scorer import (
    last_scoring_used_bart as _scoring_used_bart,
    last_scoring_used_qwen as _scoring_used_qwen,
    score_transcript,
    tokenize_transcript,
)


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

    # ── Provider log (user-visible) ────────────────────────────────────
    if _scoring_used_qwen():
        print("[Stage 1] provider: Qwen2.5-1.5B-Instruct (system-prompt-conditioned log-prob)")
    elif _scoring_used_bart():
        print("[Stage 1] provider: BART (fill-mask fallback)")
    else:
        print("[Stage 1] provider: heuristic fallback")

    # ── Determine per-stage approach ────────────────────────────────────
    approaches: Dict[str, Dict[str, str]] = {}

    # Stage 1: Scoring — check which model was used
    if _scoring_used_qwen():
        approaches["scoring"] = {
            "mode": "qwen_system_prompt_logprob",
            "label": "Qwen (Qwen2.5-1.5B-Instruct)",
            "description": "Qwen2.5-1.5B-Instruct causal LM with medical-reviewer system prompt prefix — single forward pass, per-word log-probability scoring",
            "model": "Qwen/Qwen2.5-1.5B-Instruct",
            "status": "primary",
        }
    elif _scoring_used_bart():
        approaches["scoring"] = {
            "mode": "bart_masked_lm",
            "label": "BART (facebook/bart-large)",
            "description": "BART masked-language model — fill-mask scoring (fallback when Qwen unavailable)",
            "model": "facebook/bart-large",
            "status": "fallback",
        }
    else:
        approaches["scoring"] = {
            "mode": "heuristic",
            "label": "Heuristic",
            "description": "Character-level edit distance vs medical lexicon (both LM scorers unavailable)",
            "model": "none",
            "status": "fallback",
        }

    # Stage 2: Flagging
    approaches["flagging"] = {
        "mode": "rule_based",
        "label": "Rule-based",
        "description": "Adjacent suspicious-word merging with stop-word gap tolerance",
        "model": "none",
        "status": "primary",
    }

    # Stage 3: Retrieval — check alias vs phonetic
    alias_count = sum(
        1 for sc in span_candidates for c in sc.candidates if c.match_type == "alias"
    )
    phonetic_count = len(span_candidates) - alias_count
    if alias_count > 0 and phonetic_count == 0:
        approaches["retrieval"] = {
            "mode": "alias_lookup",
            "label": "Alias match",
            "description": "Exact alias lookup in medical lexicon (no IPA needed)",
            "model": "none",
            "status": "primary",
        }
    elif alias_count > 0:
        approaches["retrieval"] = {
            "mode": "mixed",
            "label": "Mixed",
            "description": f"{alias_count} alias + {phonetic_count} phonetic matches",
            "model": "espeak-ng",
            "status": "primary",
        }
    else:
        approaches["retrieval"] = {
            "mode": "phonetic_ipa",
            "label": "IPA phonetic",
            "description": "Phonetic matching via espeak-ng phonemizer (IPA edit distance)",
            "model": "espeak-ng",
            "status": "primary",
        }

    # Stage 4: Decision — check paths used
    llm_count = sum(1 for d in decisions if d.path == "llm")
    top_fallback_count = sum(1 for d in decisions if d.path == "top_fallback")
    auto_count = sum(1 for d in decisions if d.path == "auto_fix")
    hitl_count = sum(1 for d in decisions if "hitl" in d.path)

    # Determine active LLM provider for label — use the actual resolved
    # provider from llm_decide() which handles auto-detection internally.
    _actual_provider = _get_last_provider()
    print(f"[Stage 4] provider: {_actual_provider}")
    if _actual_provider == "groq":
        _model_name = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile").strip()
        _provider_label = f"Groq ({_model_name})"
        _model_label = _model_name
    elif _actual_provider == "openrouter":
        _model_name = _get_llm_model()
        _provider_label = f"OpenRouter ({_model_name})"
        _model_label = _model_name
    elif _actual_provider == "ollama":
        _model_name = _get_llm_model()
        _provider_label = f"Ollama ({_model_name})"
        _model_label = _model_name
    else:
        _provider_label = "Gemini LLM"
        _model_label = "Gemini 1.5 Flash"

    if llm_count > 0:
        approaches["decision"] = {
            "mode": "llm",
            "label": _provider_label,
            "description": f"{_provider_label} reranked {llm_count} span(s) (with top-candidate fallback for the rest)",
            "model": _model_label,
            "status": "primary",
        }
    elif auto_count > 0:
        approaches["decision"] = {
            "mode": "auto_fix",
            "label": "Voice auto-fix",
            "description": "User voice-match auto-fix (short-circuits LLM)",
            "model": "none",
            "status": "primary",
        }
    elif top_fallback_count > 0:
        approaches["decision"] = {
            "mode": "top_fallback",
            "label": "Top candidate",
            "description": f"LLM was unavailable or returned no change — used top-ranked phonetic candidate for {top_fallback_count} span(s)",
            "model": "none (LLM unavailable)",
            "status": "fallback",
        }
    elif hitl_count > 0:
        approaches["decision"] = {
            "mode": "escalated",
            "label": "Escalated",
            "description": "Not enough confidence — escalated for human review",
            "model": "none",
            "status": "fallback",
        }
    else:
        approaches["decision"] = {
            "mode": "fallback",
            "label": "No decision",
            "description": "No decision path matched (unexpected state)",
            "model": "none",
            "status": "fallback",
        }

    # Stage 5: Correction
    approaches["correction"] = {
        "mode": "string_replace",
        "label": "Token replace",
        "description": "Token-aware string replacement with character-offset alignment",
        "model": "none",
        "status": "primary",
    }

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
        "approaches": approaches,
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
        original=transcript,
        corrected_text=corrected_text,
        report=report,
        scored_words=scored_words,
        spans=spans,
        candidates=span_candidates,
        decisions=decisions,
        session_id=uuid.uuid4().hex[:12],
    )