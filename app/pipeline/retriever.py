"""Stage 3: retrieve phonetically similar candidates from the lexicon."""

from __future__ import annotations

from typing import List

from Levenshtein import distance as _lev_distance
from spellchecker import SpellChecker

from app.services import lexicon
from app.services.phonetics import ipa_edit_distance, text_to_ipa, fallback_text_to_ipa

from .config import TOP_K
from .models import Candidate, SpanWithCandidates, SuspiciousSpan


# ── Module-level SpellChecker singleton (loaded once) ────────────────

_spell = SpellChecker()
"""pyspellchecker SpellChecker instance loaded once at import time.
Uses a built-in general English frequency dictionary — no custom lexicon
required. This enables open-world spell correction for words the medical
lexicon has never seen.
"""


def _spell_correct(span_text: str) -> list[str]:
    """Return up to 3 candidate corrections from the general English
    dictionary via pyspellchecker.

    Works on words the medical lexicon has never seen — no custom lexicon
    required. Returns an empty list when *span_text* is already recognised
    as correctly spelled, or when no plausible correction is found.
    """
    words = span_text.strip().lower().split()
    candidates: list[str] = []
    for word in words:
        if word not in _spell:  # word is unknown / misspelled
            suggestions = _spell.candidates(word)
            if suggestions:
                # pyspellchecker internally ranks by edit distance.
                # Take up to 3 per word.
                candidates.extend(list(suggestions)[:3])
    return candidates


# ── Confidence threshold for Path B → Path C fallback ────────────────

_SPELL_FALLBACK_THRESHOLD = 0.60
"""Path C fires when Path B's top candidate is below this threshold,
OR when the span has has_close_dictionary_match=False (open-world
spell correction supplements the weak IPA matches)."""


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
    candidates_out = ranked[:top_k]

    # ── Path C: Open-world spell correction ────────────────────────────
    # Fires when:
    #   1. No candidates from Path B at all, OR
    #   2. Top Path B candidate is weak (score < 0.60), OR
    #   3. The span has has_close_dictionary_match=False (no known
    #      vocabulary word is a close Levenshtein match) — this lets
    #      pyspellchecker find "dehydration" alongside the weak IPA
    #      match "respiration".
    span_hcd = getattr(span, "has_close_dictionary_match", False)
    if (
        not candidates_out
        or candidates_out[0].phonetic_score < _SPELL_FALLBACK_THRESHOLD
        or not span_hcd
    ):
        suggestions = _spell_correct(span.text)
        for suggestion in suggestions:
            if any(c.term.lower() == suggestion.lower() for c in candidates_out):
                continue
            lev_dist = _lev_distance(span.text.lower(), suggestion.lower())
            score = 1.0 - (lev_dist / max(len(span.text), len(suggestion)))
            entry = lexicon.find_by_alias(suggestion)
            candidates_out.append(
                Candidate(
                    term=suggestion,
                    ipa=entry.ipa if entry else "",
                    term_type=entry.term_type if entry else "",
                    description=entry.description if entry else "",
                    phonetic_score=round(max(0.0, min(1.0, score)), 6),
                    source="spell_checker",
                    match_type="spell_correct",
                )
            )
        candidates_out.sort(
            key=lambda c: (
                -c.phonetic_score,
                0 if c.source == "user" else 1,
                c.term.lower(),
            )
        )

    return SpanWithCandidates(span=span, candidates=candidates_out)