"""Forced-prefix lexicon rescoring (the second Whisper pass).

The "Whisper-twice" architecture's core: detect suspect spans in the free
transcript, then for each suspect span force-score lexicon candidates
against the audio slice using Whisper's own decoder.

Public API
----------
suspect_word_indices(words, lexicon_terms) -> set[int]
    Return indices of words that are likely OOV mishears: low logprob OR
    not in standard English vocabulary AND not already matching the
    lexicon.

prune_candidates(span_word, lexicon_terms, k=30) -> list[str]
    Return the K lexicon terms most phonetically similar to the span
    word, so we don't have to score all 246 candidates against every
    audio slice.

score_span(audio_path, span_text, start_s, end_s, candidates, model_size,
           language) -> list[dict]
    Forced-decode every candidate. Returns ranked list with avg_logprob.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from . import asr, audio_verify


# A small built-in set of "obviously English" common words. If a word from
# Whisper is in this set, we skip it (won't ever be a medical term).
_COMMON_ENGLISH = set("""
a about above after again against all am an and any are as at be been before
being below between both but by can come could did do does doing don't down
during each few for from further had has have having he her here hers herself
him himself his how i i'm if in into is isn't it its itself just like me more
most my myself no nor not now of off on once only or other our ours ourselves
out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up us
very was we were what when where which while who whom why will with won't
would you your yours yourself yourselves
yes good okay maybe please thanks thank welcome hello hi bye day morning
evening night today tomorrow yesterday first second third year years month
months week weeks date hour hours minute minutes one two three four five six
seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen
eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety
hundred thousand million person people doctor patient family mother father
brother sister son daughter husband wife child children friend friends
hospital clinic question answer take takes took taken give gave given
help need want like feel felt make made go went gone come came back
left right back front side top bottom yes no much many few several
better worse best worst little much new old big small high low long short
""".split())


def _is_word(s: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z'\-]*$", s))


def _normalize_word(w: str) -> str:
    return w.strip().lower().strip(".,;:!?\"'()[]{}")


def suspect_word_indices(
    words: Sequence[Dict[str, Any]],
    lexicon_terms: Sequence[Dict[str, Any]],
    *,
    logprob_threshold: float = 0.55,
    min_word_chars: int = 3,
) -> List[int]:
    """Return indices of words worth force-rescoring.

    A word is suspect if:
      * It has a real word form (letters), AND
      * It's >= min_word_chars long (don't bother with "a", "of"), AND
      * It is NOT in the common-English shortlist, AND
      * It is NOT already an exact match for a lexicon term, AND either
          - Whisper's per-word probability < logprob_threshold, OR
          - it's a proper noun / capitalized non-English word
    """
    lex_compact = set()
    for e in lexicon_terms or []:
        for s in [e.get("term", "")] + list(e.get("aliases") or []):
            n = _normalize_word(s)
            if n:
                lex_compact.add(n)

    suspects: List[int] = []
    for i, tok in enumerate(words):
        word = (tok.get("word") or "").strip()
        if not word:
            continue
        norm = _normalize_word(word)
        if not norm or len(norm) < min_word_chars:
            continue
        if not _is_word(norm):
            continue
        if norm in _COMMON_ENGLISH:
            continue
        if norm in lex_compact:
            # Already correct.
            continue
        prob = tok.get("probability")
        # Two suspect conditions:
        #   1) Whisper itself is unsure (prob < threshold).
        #   2) Word looks like a non-English coinage (capitalized + not common).
        is_low = isinstance(prob, (int, float)) and prob < logprob_threshold
        is_coinage = word.strip()[:1].isupper() and norm not in _COMMON_ENGLISH
        if is_low or is_coinage:
            suspects.append(i)
    return suspects


def prune_candidates(
    span_word: str,
    lexicon_terms: Sequence[Dict[str, Any]],
    *,
    k: int = 30,
) -> List[str]:
    """Top-K lexicon terms phonetically closest to `span_word`.

    Uses the same IPA phonemizer that audio_verify already loads. If
    phonemizer is unavailable, falls back to character-level similarity.
    """
    if not span_word or not lexicon_terms:
        return []
    span_ipa = audio_verify.phonemize(span_word) or _normalize_word(span_word)
    span_chars = _normalize_word(span_word)

    scored: List[tuple] = []
    for entry in lexicon_terms:
        term = entry.get("term")
        if not term:
            continue
        # Two cheap distances: IPA Levenshtein and lowercased char Levenshtein.
        # Take the BEST of the two to be lenient.
        term_ipa = audio_verify.phonemize(term) or term.lower()
        d1 = _lev_norm(span_ipa, term_ipa)
        d2 = _lev_norm(span_chars, term.lower().replace(" ", ""))
        scored.append((min(d1, d2), term))
    scored.sort(key=lambda x: x[0])
    seen: Set[str] = set()
    out: List[str] = []
    for _d, term in scored:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(term)
        if len(out) >= k:
            break
    return out


def _lev_norm(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    cur = [0] * (m + 1)
    for i in range(1, n + 1):
        cur[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev, cur = cur, prev
    return prev[m] / max(n, m)


def score_whole(
    audio_path: str,
    full_transcript: str,
    span_text: str,
    candidates: Sequence[str],
    *,
    model_size: str = "small",
    language: str = "en",
) -> List[Dict[str, Any]]:
    """Score whole-utterance candidates by substituting `span_text` with
    each candidate in `full_transcript`.

    Empirically much more reliable than slicing: Whisper has the full
    acoustic+language context to score against. The original transcript
    is included as the baseline.
    """
    cands_text: List[str] = []
    seen: Set[str] = set()
    base = full_transcript.strip()
    if base:
        cands_text.append(base)
        seen.add(base.lower())
    for c in candidates:
        if not c:
            continue
        # Substitute the suspect span with the candidate. Use a case-
        # insensitive single replacement of the FIRST occurrence so we
        # don't mangle text if the span text appears multiple times.
        idx = base.lower().find(span_text.lower())
        if idx == -1:
            new_text = base + " " + c
        else:
            new_text = base[:idx] + c + base[idx + len(span_text):]
        if new_text.lower() in seen:
            continue
        cands_text.append(new_text)
        seen.add(new_text.lower())

    out: List[Dict[str, Any]] = []
    for cand in cands_text:
        try:
            r = asr.score_candidate(
                audio_path,
                cand,
                model_size=model_size,
                language=language,
                # No slicing — score against the full audio.
            )
            out.append({
                "candidate": cand,
                "avg_logprob": r["avg_logprob"],
                "returned_text": r["text"],
                "is_original": cand == base,
            })
        except Exception as exc:
            out.append({
                "candidate": cand,
                "avg_logprob": float("-inf"),
                "returned_text": "",
                "is_original": cand == base,
                "error": repr(exc),
            })
    out.sort(key=lambda x: -x["avg_logprob"])
    return out


def score_span(
    audio_path: str,
    span_text: str,
    start_s: float,
    end_s: float,
    candidates: Sequence[str],
    *,
    model_size: str = "small",
    language: str = "en",
    context_pad_s: float = 0.20,
) -> List[Dict[str, Any]]:
    """Forced-decode each candidate against the audio slice.

    Always includes the original span text as a candidate so we can
    measure margin. Returns a list ranked by avg_logprob (descending —
    higher is better).
    """
    cands: List[str] = list(dict.fromkeys([span_text] + list(candidates)))
    out: List[Dict[str, Any]] = []
    for cand in cands:
        try:
            r = asr.score_candidate(
                audio_path,
                cand,
                model_size=model_size,
                language=language,
                start_s=max(0.0, start_s - context_pad_s),
                end_s=end_s + context_pad_s,
            )
            out.append({
                "candidate": cand,
                "avg_logprob": r["avg_logprob"],
                "returned_text": r["text"],
                "is_original": cand == span_text,
            })
        except Exception as exc:
            out.append({
                "candidate": cand,
                "avg_logprob": float("-inf"),
                "returned_text": "",
                "is_original": cand == span_text,
                "error": repr(exc),
            })
    out.sort(key=lambda x: -x["avg_logprob"])
    return out


def lexicon_to_hotwords(lexicon_terms: Sequence[Dict[str, Any]], max_chars: int = 700) -> str:
    """Build a comma-separated hotwords string for Whisper biasing.

    Whisper's hotwords/initial_prompt has soft length limits — we keep
    the highest-priority terms and truncate.
    """
    if not lexicon_terms:
        return ""
    sorted_lex = sorted(
        lexicon_terms,
        key=lambda e: -float(e.get("priority", 1.0)),
    )
    parts: List[str] = []
    total = 0
    for e in sorted_lex:
        t = (e.get("term") or "").strip()
        if not t:
            continue
        nxt = (", " if parts else "") + t
        if total + len(nxt) > max_chars:
            break
        parts.append(t)
        total += len(nxt)
    return ", ".join(parts)
