"""Stage 2: merge suspicious words into spans."""

from __future__ import annotations

from typing import List, Sequence

from .config import SUSPICION_THRESHOLD
from .models import ScoredWord, SuspiciousSpan


def flag_suspicious_spans(scored_words: Sequence[ScoredWord]) -> List[SuspiciousSpan]:
    """Merge adjacent suspicious words into contiguous spans.

    Two suspicious words are considered part of the same span only when they
    are directly adjacent (gap == 0).  Function words ("with", "and", "of",
    "the", etc.) act as span boundaries, keeping each misspelled word
    isolated for independent correction — especially important for the HITL
    review step where the user should correct each word separately.

    There is no limit on the number of adjacent suspicious words that can
    make up a span — e.g. "dolly prahn" merges into one span.
    """
    spans: List[SuspiciousSpan] = []

    def is_suspicious(word: ScoredWord) -> bool:
        return word.suspicion >= SUSPICION_THRESHOLD and not word.in_lexicon

    suspicious_indices = [index for index, word in enumerate(scored_words) if is_suspicious(word)]
    if not suspicious_indices:
        return spans

    current_start = suspicious_indices[0]
    current_end = suspicious_indices[0]

    def flush(start_index: int, end_index: int) -> None:
        text = " ".join(word.text for word in scored_words[start_index : end_index + 1])
        suspicion = max(word.suspicion for word in scored_words[start_index : end_index + 1])
        # Propagate has_close_dictionary_match: True if ANY word in the span has it True.
        hcd = any(
            getattr(word, "has_close_dictionary_match", False)
            for word in scored_words[start_index : end_index + 1]
        )
        # Propagate score_source: most authoritative source among span words.
        # Precedence: modernbert > heuristic > zero > ""
        source_priority = {"modernbert": 3, "heuristic": 2, "zero": 1}
        best_source = ""
        best_priority = 0
        for word in scored_words[start_index : end_index + 1]:
            src = getattr(word, "score_source", "") or ""
            prio = source_priority.get(src, 0)
            if prio > best_priority:
                best_priority = prio
                best_source = src
        spans.append(
            SuspiciousSpan(
                start=start_index,
                end=end_index,
                text=text,
                suspicion=suspicion,
                reason="both",
                has_close_dictionary_match=hcd,
                score_source=best_source,
            )
        )

    for index in suspicious_indices[1:]:
        gap = index - current_end - 1
        if gap == 0:
            # Adjacent suspicious word — extend the current span.
            current_end = index
            continue

        # Any gap (>= 1) between suspicious words creates a new span.
        # Function words like "with", "and", "of" act as boundaries,
        # not bridges — each misspelled word gets its own span for
        # independent HITL correction.
        flush(current_start, current_end)
        current_start = index
        current_end = index

    flush(current_start, current_end)
    return spans