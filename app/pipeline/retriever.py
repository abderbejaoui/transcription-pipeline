"""Stage 3: retrieve phonetically similar candidates from the lexicon."""

from __future__ import annotations

from typing import List

from app.services import lexicon
from app.services.phonetics import ipa_edit_distance, text_to_ipa, fallback_text_to_ipa

from .config import TOP_K
from .models import Candidate, SpanWithCandidates, SuspiciousSpan


def _candidate_from_entry(entry: lexicon.LexiconEntry, score: float, match_type: str = "phonetic") -> Candidate:
    return Candidate(
        term=entry.term,
        ipa=entry.ipa,
        term_type=entry.term_type,
        description=entry.description,
        phonetic_score=round(max(0.0, min(1.0, score)), 6),
        source=entry.source,
        match_type=match_type,
    )


def retrieve_candidates(span: SuspiciousSpan, top_k: int = TOP_K) -> SpanWithCandidates:
    alias_hit = lexicon.find_by_alias(span.text)
    if alias_hit is not None:
        return SpanWithCandidates(
            span=span,
            candidates=[_candidate_from_entry(alias_hit, 1.0, match_type="alias")],
        )

    # Try real espeak-ng IPA via phonemizer; fall back to heuristic IPA.
    try:
        span_ipa = text_to_ipa(span.text)
        if not span_ipa:
            span_ipa = fallback_text_to_ipa(span.text)
    except Exception:
        span_ipa = fallback_text_to_ipa(span.text)
    ranked: List[Candidate] = []
    for entry in lexicon.load_lexicon():
        best = 0.0
        # Prefer stored IPA from the lexicon entry to avoid expensive
        # or unreliable runtime phonemizer calls (espeak-ng). Use the
        # entry's `ipa` for all variants; fall back to computing IPA
        # only if the stored `ipa` is empty.
        entry_ipa = (entry.ipa or "").strip()
        if entry_ipa:
            score = 1.0 - ipa_edit_distance(span_ipa, entry_ipa)
            best = max(best, score)
        else:
            variants = [entry.term, *entry.aliases]
            for variant in variants:
                variant_ipa = text_to_ipa(variant)
                score = 1.0 - ipa_edit_distance(span_ipa, variant_ipa)
                best = max(best, score)
        ranked.append(_candidate_from_entry(entry, best))
    ranked.sort(key=lambda candidate: (-candidate.phonetic_score, 0 if candidate.source == "user" else 1, candidate.term.lower()))
    return SpanWithCandidates(span=span, candidates=ranked[:top_k])