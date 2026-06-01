"""Accent-mapped Levenshtein distance for Gulf-accented English terms."""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Sequence, Tuple


_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06ff]")

# Low-cost substitutions for common Gulf/Arabic-accented English swaps.
_DEFAULT_SUBSTITUTION_COSTS: Dict[Tuple[str, str], float] = {
    ("p", "b"): 0.1,
    ("b", "p"): 0.1,
    ("v", "f"): 0.1,
    ("f", "v"): 0.1,
    ("o", "u"): 0.1,
    ("u", "o"): 0.1,
    ("s", "c"): 0.1,
    ("c", "s"): 0.1,
    ("s", "z"): 0.1,
    ("z", "s"): 0.1,
}


def normalize_latin_word(word: str) -> str:
    """Lowercase and keep ASCII letters only."""
    return "".join(ch for ch in word.lower() if "a" <= ch <= "z")


def is_latin_word(word: str) -> bool:
    """True if the token contains Latin letters and no Arabic letters."""
    if _ARABIC_LETTER_RE.search(word):
        return False
    return any("a" <= ch.lower() <= "z" for ch in word)


def accent_mapped_levenshtein(
    a: str,
    b: str,
    *,
    substitution_costs: Mapping[Tuple[str, str], float] = _DEFAULT_SUBSTITUTION_COSTS,
    insertion_cost: float = 1.0,
    deletion_cost: float = 1.0,
) -> float:
    """Weighted Levenshtein distance with accent-aware substitutions."""
    if not a and not b:
        return 0.0
    if not a:
        return float(len(b)) * insertion_cost
    if not b:
        return float(len(a)) * deletion_cost

    n, m = len(a), len(b)
    prev = [j * insertion_cost for j in range(m + 1)]
    cur = [0.0] * (m + 1)

    for i in range(1, n + 1):
        cur[0] = i * deletion_cost
        ai = a[i - 1]
        for j in range(1, m + 1):
            bj = b[j - 1]
            if ai == bj:
                sub_cost = 0.0
            else:
                sub_cost = substitution_costs.get((ai, bj), 1.0)
            # DP step: min(delete, insert, substitute) for prefixes a[:i], b[:j].
            cur[j] = min(
                prev[j] + deletion_cost,
                cur[j - 1] + insertion_cost,
                prev[j - 1] + sub_cost,
            )
        prev, cur = cur, prev

    return prev[m]


PreparedDictionary = Sequence[Tuple[str, str]]


def prepare_dictionary(dictionary: Sequence[str]) -> List[Tuple[str, str]]:
    """Return (original, normalized) pairs for faster candidate scoring."""
    prepared: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for term in dictionary:
        if not term:
            continue
        norm = normalize_latin_word(term)
        if not norm or norm in seen:
            continue
        prepared.append((term, norm))
        seen.add(norm)
    return prepared


def get_phonetic_candidates_prepared(
    misspelled_word: str,
    prepared_dictionary: PreparedDictionary,
    *,
    max_distance: float = 2.0,
    top_k: int = 3,
) -> List[Dict[str, float | str]]:
    """Return top-k candidates from a pre-normalized dictionary."""
    needle = normalize_latin_word(misspelled_word)
    if not needle:
        return []

    results: List[Dict[str, float | str]] = []
    for term, norm in prepared_dictionary:
        if abs(len(needle) - len(norm)) > max_distance:
            continue
        dist = accent_mapped_levenshtein(needle, norm)
        if dist <= max_distance:
            denom = max(len(needle), len(norm))
            similarity = 1.0 - (dist / denom) if denom else 0.0
            results.append({
                "term": term,
                "distance": dist,
                "phonetic_similarity": similarity,
            })

    results.sort(key=lambda d: (d["distance"], -d["phonetic_similarity"]))
    return results[:max(1, int(top_k))]


def get_phonetic_candidates(
    misspelled_word: str,
    dictionary: Sequence[str],
    max_distance: float = 2.0,
    top_k: int = 3,
) -> List[Dict[str, float | str]]:
    """Return top-k medical terms within the accent-mapped distance.

    Args:
        misspelled_word: ASR token suspected to be a misspelling.
        dictionary: List of valid medical terms (canonical spellings).
        max_distance: Maximum weighted edit distance to accept.
        top_k: Number of closest candidates to return.
    """
    prepared = prepare_dictionary(dictionary)
    return get_phonetic_candidates_prepared(
        misspelled_word, prepared, max_distance=max_distance, top_k=top_k
    )


if __name__ == "__main__":
    sample_dict = ["paracetamol", "ibuprofen", "ventolin"]
    misspelled = "bacetamol"
    print(get_phonetic_candidates(misspelled, sample_dict, max_distance=2.0, top_k=3))
