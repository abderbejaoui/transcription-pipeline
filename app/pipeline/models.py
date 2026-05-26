"""Dataclasses used across the correction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Token:
    """Stage 0 token: a single word split from the input transcript.

    TODO: integrate this into the pipeline as a pre-processing step
    (stage 0) so the scorer can use structured Token objects instead
    of raw string parsing.
    """
    index: int
    text: str          # lowercased, no punctuation
    original: str      # exactly as it appeared in the input
    punct: str         # any trailing punctuation (". , ? !" etc.) or ""


@dataclass(frozen=True)
class ScoredWord:
    index: int
    text: str
    original: str = ""
    punct: str = ""
    suspicion: float = 0.0
    in_lexicon: bool = False
    score_source: str = ""
    """How the suspicion score was derived.

    - ``"zero"`` — function word / stop word, always 0.0.
    - ``"heuristic"`` — scored by dictionary/lexicon/enchant heuristic.
    - ``"modernbert"`` — refined by ModernBERT fill-mask.
    """
    has_close_dictionary_match: bool = False
    """True if suspicion > 0.50 AND the word is within Levenshtein distance 1-3
    of a known medical term (lexicon + medical_terms.txt).
    """
    # Legacy fields (kept for backward compat with existing tests)
    start: int = 0
    end: int = 0


@dataclass(frozen=True)
class SuspiciousSpan:
    start: int
    end: int
    text: str
    suspicion: float
    reason: str
    has_close_dictionary_match: bool = False
    """True if ANY scored word in this span has has_close_dictionary_match=True."""
    score_source: str = ""
    """Most authoritative score_source among words in this span.
    Precedence: modernbert > heuristic > zero."""


@dataclass(frozen=True)
class Candidate:
    term: str
    ipa: str
    term_type: str
    description: str
    phonetic_score: float
    source: str
    match_type: str = "phonetic"  # "alias" | "phonetic"


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
    original: str
    corrected_text: str
    report: Dict[str, Any]
    scored_words: List[ScoredWord]
    spans: List[SuspiciousSpan]
    candidates: List[SpanWithCandidates]
    decisions: List[Decision]
    hitl_required: List[Decision] = field(default_factory=list)
    session_id: str = ""

    @property
    def corrected(self) -> str:
        return self.corrected_text

    @property
    def corrections(self) -> List[Decision]:
        return [d for d in self.decisions if d.chosen is not None]