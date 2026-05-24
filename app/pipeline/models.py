"""Dataclasses used across the correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ScoredWord:
    index: int
    text: str
    suspicion: float
    in_lexicon: bool
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class SuspiciousSpan:
    start: int
    end: int
    text: str
    suspicion: float
    reason: str


@dataclass(frozen=True)
class Candidate:
    term: str
    ipa: str
    term_type: str
    description: str
    phonetic_score: float
    source: str


@dataclass(frozen=True)
class SpanWithCandidates:
    span: SuspiciousSpan
    candidates: List[Candidate] = field(default_factory=list)


@dataclass(frozen=True)
class Decision:
    span: SuspiciousSpan
    chosen: Optional[str]
    confidence: float
    path: str


@dataclass(frozen=True)
class PipelineResult:
    corrected_text: str
    report: Dict[str, Any]
    scored_words: List[ScoredWord]
    spans: List[SuspiciousSpan]
    candidates: List[SpanWithCandidates]
    decisions: List[Decision]