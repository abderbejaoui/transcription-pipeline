"""Stage 2: merge suspicious words into spans."""

from __future__ import annotations

from typing import List, Sequence

from .config import SUSPICION_THRESHOLD
from .models import ScoredWord, SuspiciousSpan
from .scorer import is_function_word


def flag_suspicious_spans(scored_words: Sequence[ScoredWord]) -> List[SuspiciousSpan]:
    """Merge suspicious words into contiguous spans.

    Two suspicious words are considered part of the same span when they are
    either:
      - adjacent (gap == 0), or
      - separated by exactly one stop word (gap == 1 and middle is a stop word).

    There is no limit on the number of words that can make up a span — all
    adjacent / stop-gapped suspicious words merge into one.
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

        if gap == 1:
            middle_word = scored_words[current_end + 1]
            if is_function_word(middle_word.text):
                # Separated by a single function word — merge into the current span.
                current_end = index
                continue

        # Gap is too large (>= 2 content words between) — flush and start a new span.
        flush(current_start, current_end)
        current_start = index
        current_end = index

    flush(current_start, current_end)
    return spans