"""Stage 4: decide whether to apply a candidate or escalate."""

from __future__ import annotations

from typing import Callable, List, Sequence

from app.services.llm import NO_CHANGE, llm_decide

from .config import LLM_MIN_CONFIDENCE, USER_AUTO_FIX_THRESHOLD
from .models import Candidate, Decision, SpanWithCandidates


def _valid_choice(choice: str, candidates: Sequence[Candidate]) -> bool:
    if not choice or choice == NO_CHANGE:
        return True
    return any(choice == candidate.term for candidate in candidates)


def decide_span(sentence: str, item: SpanWithCandidates, llm: Callable[[str, str, Sequence[Candidate]], str] = llm_decide) -> Decision:
    candidates = list(item.candidates)
    if not candidates:
        return Decision(span=item.span, chosen=None, confidence=0.0, path="hitl_escalate")

    top = candidates[0]
    if top.source == "user" and top.phonetic_score >= USER_AUTO_FIX_THRESHOLD:
        return Decision(span=item.span, chosen=top.term, confidence=top.phonetic_score, path="auto_fix")
    if top.phonetic_score < LLM_MIN_CONFIDENCE:
        return Decision(span=item.span, chosen=None, confidence=top.phonetic_score, path="hitl_escalate")

    # Try to use Gemini for context-aware reranking. If the key is missing,
    # the call returns NO_CHANGE immediately (saves a network round-trip).
    choice = llm(sentence, item.span.text, candidates)
    if choice != NO_CHANGE and _valid_choice(choice, candidates):
        selected = next(candidate for candidate in candidates if candidate.term == choice)
        return Decision(span=item.span, chosen=selected.term, confidence=selected.phonetic_score, path="llm")

    # LLM did not return a valid choice (key missing, quota exhausted, or
    # it determined NO_CHANGE). Fall back to the top-ranked candidate when
    # its phonetic_score is acceptable. This ensures offline / degraded runs
    # still produce corrections and match the canonical test output.
    # NOTE: path="top_fallback" distinguishes this from actual LLM decisions
    # so the UI can transparently report the decision method.
    if top.phonetic_score >= LLM_MIN_CONFIDENCE:
        return Decision(span=item.span, chosen=top.term, confidence=top.phonetic_score, path="top_fallback")
    return Decision(span=item.span, chosen=None, confidence=0.0, path="hitl_escalate")


def decide_spans(sentence: str, items: Sequence[SpanWithCandidates], llm: Callable[[str, str, Sequence[Candidate]], str] = llm_decide) -> List[Decision]:
    return [decide_span(sentence, item, llm=llm) for item in items]