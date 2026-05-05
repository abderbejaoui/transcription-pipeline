"""Domain-agnostic transcript correction.

The single rule:
    If a span looks or sounds close enough to something in the user's
    vocabulary, replace it with the canonical form. Otherwise leave it.

There is NO medical-specific logic. Works for any vocabulary in any domain
(brands, names, products, jargon, drugs, anything).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jellyfish
from rapidfuzz import fuzz


DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "medical_lexicon.jsonl"
)


# Tokenizer: keep hyphenated words and apostrophes together. Numbers parsed
# as one token.
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*|\d+(?:\.\d+)?")
WORDISH_RE = re.compile(r"[a-z0-9]+")

# Spans that *start* or *end* with one of these are skipped because
# corrupting them produces noisy false positives.
COMMON_GLUE = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "had", "has", "have", "her", "him", "his", "i", "if", "in", "into",
    "is", "it", "its", "me", "my", "no", "not", "now", "of", "on",
    "or", "our", "she", "so", "than", "that", "the", "their", "them",
    "then", "there", "these", "they", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why",
    "will", "with", "you", "your", "yours", "twice", "once", "daily",
    "day", "days", "week", "weeks", "month", "months", "year", "years",
    "patient", "takes", "take", "taking", "every", "today", "tomorrow",
    "yesterday", "next", "last", "morning", "evening", "night",
}

# Tiny filler words ASR often inserts mid-word.
GLUE_TINY = {
    "a", "an", "the", "to", "of", "i", "is", "it", "at", "in", "on", "or",
    "and", "e", "uh", "um", "eh", "ah", "oh", "ya", "ye",
}

MIN_SPAN_CHARS = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Token:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Span:
    text: str
    start: int
    end: int
    token_start: int
    token_end: int


@dataclass(frozen=True)
class LexiconEntry:
    term: str
    type: str
    aliases: Tuple[str, ...]
    priority: float = 1.0

    @property
    def variants(self) -> Tuple[str, ...]:
        return (self.term, *self.aliases)


@dataclass
class Candidate:
    span: Span
    correction: str
    score: float
    confidence: float
    entry_type: str
    issue_type: str
    reason: str
    features: Dict[str, float]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("—", "-").replace("–", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", text).strip()


def compact(text: str) -> str:
    """Remove separators and the noise word 'and'."""
    words = WORDISH_RE.findall(normalize_text(text))
    words = [w for w in words if w != "and"]
    return "".join(words)


def token_words(text: str) -> List[str]:
    return WORDISH_RE.findall(normalize_text(text))


def metaphone_text(text: str) -> str:
    codes = []
    for word in token_words(text):
        if word == "and":
            continue
        code = jellyfish.metaphone(word)
        if code:
            codes.append(code)
    return " ".join(codes)


def is_capitalization_only(a: str, b: str) -> bool:
    return a.lower() == b.lower() and a != b


def _drop_glue(text: str) -> str:
    """Compact form that also drops short filler words."""
    words = [w for w in WORDISH_RE.findall(normalize_text(text)) if w not in GLUE_TINY]
    return "".join(words)


def _glueless_metaphone(text: str) -> str:
    """Metaphone of the glueless compact form."""
    g = _drop_glue(text)
    return jellyfish.metaphone(g) if g else ""


def load_lexicon(path: Path = DEFAULT_LEXICON_PATH) -> List[LexiconEntry]:
    entries: List[LexiconEntry] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            entries.append(
                LexiconEntry(
                    term=row["term"],
                    type=row.get("type", "term"),
                    aliases=tuple(row.get("aliases", [])),
                    priority=float(row.get("priority", 1.0)),
                )
            )
    return entries


def _score_pair(span: str, variant: str) -> Dict[str, float]:
    s_norm = normalize_text(span)
    v_norm = normalize_text(variant)
    s_compact = compact(span)
    v_compact = compact(variant)
    s_glueless = _drop_glue(span)
    v_glueless = _drop_glue(variant)
    s_phone = metaphone_text(span)
    v_phone = metaphone_text(variant)

    s_n_words = len(token_words(span))
    v_n_words = len(token_words(variant))

    # token_set_ratio is forgiving when one side is a strict subset of the
    # other; that lets a single common word match a multi-word variant.
    # For mismatched word counts use token_sort_ratio.
    if s_n_words != v_n_words:
        fuzzy = float(fuzz.token_sort_ratio(s_norm, v_norm))
    else:
        fuzzy = float(fuzz.token_set_ratio(s_norm, v_norm))

    compact_score = float(fuzz.ratio(s_compact, v_compact)) if s_compact and v_compact else 0.0
    phonetic = float(fuzz.ratio(s_phone, v_phone)) if s_phone and v_phone else 0.0

    # Partial alignment: how well does the variant fit somewhere inside the
    # span? Catches "target gin" containing "targin". Only fires when the
    # SPAN is at least as long as the variant.
    if s_compact and v_compact and len(s_compact) >= len(v_compact) and len(v_compact) >= 5:
        partial = float(fuzz.partial_ratio(s_compact, v_compact))
        len_ratio = len(v_compact) / max(1, len(s_compact))
        if partial >= 90 and len_ratio >= 0.55:
            compact_score = max(compact_score, partial * (0.7 + 0.3 * len_ratio))

    # Glueless compact: "doll a brain" -> "dollbrain" vs "doliprane".
    if s_glueless and v_glueless and (s_glueless != s_compact or v_glueless != v_compact):
        glueless_score = float(fuzz.ratio(s_glueless, v_glueless))
        compact_score = max(compact_score, glueless_score)

    # Glueless metaphone: catches splits like "doll e prane" whose glueless
    # form ("dollprane") has the same metaphone (TLPRN) as the canonical
    # ("doliprane"). This is the strongest signal for split-by-filler-word
    # ASR mistakes.
    s_gphone = _glueless_metaphone(span)
    v_gphone = _glueless_metaphone(variant)
    if s_gphone and v_gphone:
        gphonetic = float(fuzz.ratio(s_gphone, v_gphone))
        phonetic = max(phonetic, gphonetic)

    combined = max(
        0.50 * fuzzy + 0.20 * compact_score + 0.30 * phonetic,
        0.92 * compact_score,
        0.85 * phonetic,
    )

    # Length-mismatch penalty. If the span is much shorter than the variant,
    # we are likely matching a common English word against a longer specific
    # term (e.g. "open" against "OpenAI", "wing" against "WingSprint").
    # Apply a quadratic penalty proportional to how much of the variant the
    # span fails to cover.
    if s_compact and v_compact:
        len_ratio = len(s_compact) / max(1, len(v_compact))
        if len_ratio < 0.85:
            shortfall = 0.85 - len_ratio
            penalty = shortfall * shortfall * 200.0
            combined = max(0.0, combined - penalty)

    return {
        "fuzzy": fuzzy,
        "compact": compact_score,
        "phonetic": phonetic,
        "score": combined,
    }


# ---------------------------------------------------------------------------
# Corrector
# ---------------------------------------------------------------------------


class MedicalCorrector:
    """Domain-agnostic corrector. Class name kept for backwards compat."""

    def __init__(
        self,
        lexicon: Optional[Sequence[LexiconEntry]] = None,
        max_span_tokens: int = 6,
        accept_threshold: float = 80.0,
        single_word_phonetic_floor: float = 86.0,
        single_word_score_floor: float = 70.0,
    ) -> None:
        self.lexicon = list(lexicon or load_lexicon())
        self.max_span_tokens = max_span_tokens
        self.accept_threshold = accept_threshold
        self.single_word_phonetic_floor = single_word_phonetic_floor
        self.single_word_score_floor = single_word_score_floor

        self._canonical_forms = {entry.term for entry in self.lexicon}
        # Aliases that are trivially equivalent to their canonical (only
        # differ in case/whitespace) — these are "already correct".
        self._terminal_aliases = {
            a
            for entry in self.lexicon
            for a in entry.aliases
            if normalize_text(a) == normalize_text(entry.term)
        }
        # All known compact forms (canonical + every alias) -> canonical.
        # Lets us shortcut short-but-recognisable spans like "aws".
        self._known_compacts: Dict[str, str] = {}
        for entry in self.lexicon:
            for v in entry.variants:
                cv = compact(v)
                if cv:
                    self._known_compacts.setdefault(cv, entry.term)
                gv = _drop_glue(v)
                if gv:
                    self._known_compacts.setdefault(gv, entry.term)

    # ------------------ Public API ------------------

    def correct_transcript(self, transcript: str) -> Dict[str, Any]:
        tokens = self._tokenize(transcript)
        spans = self._generate_spans(transcript, tokens)

        candidates: List[Candidate] = []
        for span in spans:
            best = self._best_candidate_for_span(span)
            if best is not None:
                candidates.append(best)

        selected = self._select_non_overlapping(candidates)
        corrected_text = self._apply_corrections(transcript, selected)
        return {
            "corrected_text": corrected_text,
            "suspicious_spans": [self._serialize(c) for c in selected],
        }

    # ------------------ Tokenisation ----------------

    def _tokenize(self, text: str) -> List[Token]:
        return [Token(m.group(), m.start(), m.end()) for m in TOKEN_RE.finditer(text)]

    def _generate_spans(self, transcript: str, tokens: Sequence[Token]) -> List[Span]:
        spans: List[Span] = []
        for i in range(len(tokens)):
            for j in range(i + 1, min(len(tokens), i + self.max_span_tokens) + 1):
                text = transcript[tokens[i].start : tokens[j - 1].end]
                if self._bad_span_boundary(text, j - i):
                    continue
                spans.append(
                    Span(
                        text=text,
                        start=tokens[i].start,
                        end=tokens[j - 1].end,
                        token_start=i,
                        token_end=j,
                    )
                )
        return spans

    def _bad_span_boundary(self, span_text: str, n_tokens: int) -> bool:
        if re.search(r"[.!?;:]\s+\S", span_text):
            return True
        words = token_words(span_text)
        if not words:
            return True
        # Allow any span whose compact form exactly matches a known variant
        # (e.g. "aws", "a w s") — even short ones.
        if compact(span_text) in self._known_compacts:
            return False
        if _drop_glue(span_text) in self._known_compacts:
            return False
        if n_tokens == 1:
            if len(words[0]) < MIN_SPAN_CHARS:
                return True
            if words[0] in COMMON_GLUE:
                return True
            return False
        if words[0] in COMMON_GLUE or words[-1] in COMMON_GLUE:
            return True
        return False

    # ------------------ "Already correct" guard ----

    def _already_valid(self, span_text: str) -> bool:
        return span_text in self._canonical_forms or span_text in self._terminal_aliases

    # ------------------ Scoring ---------------------

    def _best_candidate_for_span(self, span: Span) -> Optional[Candidate]:
        if self._already_valid(span.text):
            return None

        words = token_words(span.text)
        n_words = len(words)
        if n_words == 0:
            return None
        if not compact(span.text):
            return None

        best: Optional[Candidate] = None
        best_features: Dict[str, float] = {}
        for entry in self.lexicon:
            for variant in entry.variants:
                feats = _score_pair(span.text, variant)
                if best is None or feats["score"] > best.score:
                    best = Candidate(
                        span=span,
                        correction=entry.term,
                        score=feats["score"],
                        confidence=0.0,
                        entry_type=entry.type,
                        issue_type="",
                        reason="",
                        features=feats,
                    )
                    best_features = feats

        if best is None:
            return None

        # Strong-phonetic relaxation: accept slightly below threshold ONLY
        # when phonetic match is high AND there is independent character-
        # level evidence AND the span actually covers most of the variant.
        # The length check stops short common English words ("Open") from
        # grabbing longer specific terms ("OpenAI") via phonetic-only luck.
        threshold = self.accept_threshold
        has_char_evidence = (
            best_features.get("fuzzy", 0.0) >= 70.0
            or best_features.get("compact", 0.0) >= 70.0
        )
        s_len = len(compact(span.text))
        v_len = len(compact(best.correction))
        good_coverage = s_len >= max(5, int(v_len * 0.85))
        if (
            best_features.get("phonetic", 0.0) >= self.single_word_phonetic_floor
            and best.score >= self.single_word_score_floor
            and has_char_evidence
            and good_coverage
        ):
            threshold = self.single_word_score_floor

        if best.score < threshold:
            return None

        if best.correction == span.text:
            return None

        # Reject pure substring corrections unless fuzzy is strong.
        if (
            best.features["fuzzy"] < 92
            and normalize_text(best.correction) in normalize_text(span.text)
            and normalize_text(best.correction) != normalize_text(span.text)
        ):
            return None

        conf = max(0.0, min(0.99, (best.score - 70.0) / 30.0))
        if " " in span.text and " " not in best.correction:
            issue = "split_phrase_should_merge"
            reason = "Span looks like one word split across multiple tokens."
        elif is_capitalization_only(span.text, best.correction):
            issue = "capitalization"
            reason = "Surface form differs only in case from a known term."
        elif best_features["phonetic"] >= 90 and best_features["fuzzy"] < 90:
            issue = "sound_alike"
            reason = f"Sounds like {best.correction!r}."
        else:
            issue = "single_word_misspelling" if n_words <= 2 else "wrong_term"
            reason = f"Close match to {best.correction!r}."

        best.confidence = conf
        best.issue_type = issue
        best.reason = reason
        return best

    # ------------------ Selection -------------------

    def _select_non_overlapping(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Pick non-overlapping winners.

        Prefer longer high-score spans. If a longer candidate fully contains
        a shorter one and scores within `dominate_margin` of it, the longer
        one wins. Stops "mohamad bin Rashid" from being reduced to "Rashid".
        """
        dominate_margin = 6.0
        ordered = sorted(
            candidates,
            key=lambda c: (c.score, c.span.token_end - c.span.token_start),
            reverse=True,
        )
        selected: List[Candidate] = []
        occupied: set = set()
        for c in ordered:
            ids = set(range(c.span.token_start, c.span.token_end))
            if ids & occupied:
                continue
            dominated = False
            for other in ordered:
                if other is c:
                    continue
                if (
                    other.span.token_start <= c.span.token_start
                    and other.span.token_end >= c.span.token_end
                    and (other.span.token_end - other.span.token_start)
                    > (c.span.token_end - c.span.token_start)
                    and other.score >= c.score - dominate_margin
                ):
                    other_ids = set(range(other.span.token_start, other.span.token_end))
                    if not (other_ids & occupied):
                        dominated = True
                        break
            if dominated:
                continue
            selected.append(c)
            occupied |= ids
        return sorted(selected, key=lambda c: c.span.start)

    def _apply_corrections(self, transcript: str, selected: Sequence[Candidate]) -> str:
        if not selected:
            return transcript
        out: List[str] = []
        last = 0
        for c in sorted(selected, key=lambda c: c.span.start):
            out.append(transcript[last : c.span.start])
            out.append(c.correction)
            last = c.span.end
        out.append(transcript[last:])
        return "".join(out)

    def _serialize(self, c: Candidate) -> Dict[str, Any]:
        return {
            "original_text": c.span.text,
            "start": c.span.start,
            "end": c.span.end,
            "issue_type": c.issue_type,
            "possible_correction": c.correction,
            "confidence": round(c.confidence, 4),
            "score": round(c.score, 2),
            "reason_short": c.reason,
            "entry_type": c.entry_type,
            "features": {k: round(v, 2) for k, v in c.features.items()},
        }


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Correct a transcript against the lexicon.")
    parser.add_argument("transcript", nargs="?")
    args = parser.parse_args()
    text = args.transcript or sys.stdin.read().strip()
    corrector = MedicalCorrector()
    print(json.dumps(corrector.correct_transcript(text), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
