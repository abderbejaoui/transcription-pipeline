"""pipeline/stage1_phonetic.py — Phonetic flagging pass.

Owns:
  - Drug-hint classification (suffix/name heuristics)
  - Phonetic candidate retrieval against medical_terms.txt
  - Single-word and n-gram (bigram/trigram) flagging
  - phonetic_pass() — the Stage 1 public entrypoint
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .arabic import (
    consonant_skeleton_ar,
    consonant_skeleton_latin,
    is_arabic_filler,
    is_pure_latin_or_digit,
    length_ratio_ok,
    lev_sim,
    longest_common_substring,
    translit,
)
from .lexicon import load_medical_lexicon

# ---------------------------------------------------------------------------
# Drug vs. disease classification
# ---------------------------------------------------------------------------

_DRUG_HINT_SUFFIXES = (
    "in", "ol", "ide", "ine", "ate", "ium", "one", "an", "el",
    "il", "etamol", "ofen", "azole", "cillin", "prazole",
    "statin", "sartan", "ipine", "formin", "tolin", "tarn", "tic",
    "amol", "ralgan",
)
_DRUG_HINT_TERMS = {
    "insulin", "panadol", "codeine", "morphine", "doliprane",
    "voltaren", "ventolin", "augmentin", "efferalgan", "flagyl",
    "warfarin", "heparin", "zithromax", "tramadol",
}

_PHONETIC_ALIAS: Dict[str, str] = {
    "afawlqn": "efferalgan",
    "afaqln": "efferalgan",
    "afawlqln": "efferalgan",
    "afywrqan": "efferalgan",
    "afywrqn": "efferalgan",
    "awqmntyn": "augmentin",
}


def _is_likely_drug(term: str) -> bool:
    term = term.lower().strip()
    if term in _DRUG_HINT_TERMS:
        return True
    return any(term.endswith(suf) for suf in _DRUG_HINT_SUFFIXES)


def _phonetic_alias_lookup(needle_translits: List[str]) -> Optional[str]:
    """Return the drug name if any needle matches a known English-mishearing
    alias (e.g. 'اف اول قن' = 'if all gone' → efferalgan)."""
    for n in needle_translits:
        if n in _PHONETIC_ALIAS:
            return _PHONETIC_ALIAS[n]
        sk = consonant_skeleton_ar(n)
        for key, drug in _PHONETIC_ALIAS.items():
            if consonant_skeleton_ar(key) == sk and len(sk) >= 3:
                return drug
    return None


def phonetic_candidates(
    word: str,
    lexicon: List[str],
    k: int = 3,
    *,
    threshold: float = 0.45,
    min_skeleton_len: int = 3,
) -> List[Dict[str, Any]]:
    """Find up to `k` lexicon entries phonetically similar to `word`.

    Compares consonant skeletons of the Arabic transliteration against
    consonant skeletons of the Latin lexicon entries. Drugs rank above
    diseases at equal similarity.
    """
    if len(word) < 2:
        return []
    needles = list({translit(word, strip_clitics=True),
                    translit(word, strip_clitics=False)})
    needles = [n for n in needles if len(n) >= 2]
    if not needles:
        return []
    needle_sks = [consonant_skeleton_ar(n) for n in needles]
    scored = []
    for term in lexicon:
        term_lat = re.sub(r"[^a-z]", "", term.lower())
        if not term_lat:
            continue
        term_sk = consonant_skeleton_latin(term_lat)
        if not term_sk:
            continue
        best = 0.0
        for n, n_sk in zip(needles, needle_sks):
            if length_ratio_ok(n, term_lat):
                best = max(best, lev_sim(n, term_lat))
            if (len(n_sk) >= min_skeleton_len
                    and len(term_sk) >= min_skeleton_len
                    and length_ratio_ok(n_sk, term_sk)):
                best = max(best, lev_sim(n_sk, term_sk))
        if best < threshold:
            continue
        scored.append({
            "term": term,
            "phonetic_similarity": round(best, 3),
            "_is_drug": _is_likely_drug(term),
        })

    alias_drug = _phonetic_alias_lookup(needles)
    if alias_drug:
        alias_idx = next(
            (i for i, c in enumerate(scored) if c["term"].lower() == alias_drug),
            None,
        )
        if alias_idx is not None:
            scored[alias_idx]["phonetic_similarity"] = max(
                scored[alias_idx]["phonetic_similarity"], 0.95
            )
        elif any(t.lower() == alias_drug for t in lexicon):
            scored.insert(0, {
                "term": alias_drug,
                "phonetic_similarity": 0.95,
                "_is_drug": True,
            })

    needle_skel = needles[0] if needles else ""
    needle_len = len(needle_skel)

    def _lcs_len(term: str) -> int:
        t = re.sub(r"[^a-z]", "", term.lower())
        if not t or not needle_skel:
            return 0
        n, m = len(needle_skel), len(t)
        prev = [0] * (m + 1)
        best = 0
        for i in range(1, n + 1):
            cur = [0] * (m + 1)
            ni = needle_skel[i - 1]
            for j in range(1, m + 1):
                if ni == t[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
            prev = cur
        return best

    scored.sort(key=lambda d: (
        -d["phonetic_similarity"],
        not d["_is_drug"],
        abs(len(re.sub(r"[^a-z]", "", d["term"].lower())) - needle_len),
        -_lcs_len(d["term"]),
    ))
    for s in scored:
        s.pop("_is_drug", None)
    return scored[:k]


def _is_known_medical(word: str, lexicon: List[str]) -> bool:
    """True if `word` is already a known medical term (exact or translit match)."""
    w = word.lower()
    if w in {t.lower() for t in lexicon}:
        return True
    tl = translit(word)
    for t in lexicon:
        if tl == translit(t):
            return True
    return False


def phonetic_pass(transcript: str) -> List[Dict[str, Any]]:
    """Stage 1: flag words/n-grams that are phonetically close to lexicon terms.

    Runs three sub-passes in order:
      1. Trigram windows (threshold 0.55)
      2. Bigram windows (threshold 0.50; higher 0.78 when a filler bridges)
      3. Single words (threshold 0.60)

    Returns one flag dict per suspicious span with keys:
      index, word, reason, candidates, [span_indices]
    """
    lexicon = load_medical_lexicon()
    if not lexicon:
        return []
    words = [w for w in re.split(r"\s+", transcript.strip()) if w]
    flags: List[Dict[str, Any]] = []
    consumed: set = set()

    single_results: List[Optional[List[Dict[str, Any]]]] = []
    for word in words:
        if is_arabic_filler(word):
            single_results.append(None)
            continue
        if _is_known_medical(word, lexicon):
            single_results.append(None)
            continue
        single_results.append(phonetic_candidates(word, lexicon, k=3))

    def _try_ngram(n: int, threshold: float, filler_threshold: float) -> None:
        for i in range(len(words) - n + 1):
            if any((i + off) in consumed for off in range(n)):
                continue
            window = words[i:i + n]
            if any(is_pure_latin_or_digit(w) for w in window):
                continue
            if n >= 2 and any(w == "و" for w in window):
                continue
            filler_count = sum(1 for w in window if is_arabic_filler(w))
            if n == 2 and filler_count >= 2:
                continue
            if n == 3 and filler_count >= 2:
                continue
            has_filler = filler_count > 0
            joined = "".join(window)
            candidates = phonetic_candidates(joined, lexicon, k=3, threshold=threshold)
            if not candidates:
                continue
            top = candidates[0]
            min_score = filler_threshold if has_filler else threshold
            if top["phonetic_similarity"] < min_score:
                continue
            joined_skel = consonant_skeleton_ar(translit(joined))
            term_skel = consonant_skeleton_latin(top["term"])
            lcs_len = longest_common_substring(joined_skel, term_skel)
            if top["phonetic_similarity"] < 0.65 and lcs_len < 3:
                continue
            should_skip = False
            for off in range(n):
                sc = single_results[i + off]
                if not sc:
                    continue
                single_top = sc[0]
                single_sim = single_top["phonetic_similarity"]
                if single_sim < 0.80:
                    continue
                from_word = words[i + off]
                from_translit = translit(from_word)
                term = single_top["term"]
                ratio = (min(len(from_translit), len(term)) /
                         max(len(from_translit), len(term)))
                if ratio < 0.65:
                    continue
                joined_translit = translit(joined)
                if len(joined_translit) > 1.7 * len(from_translit):
                    continue
                if single_sim > top["phonetic_similarity"]:
                    should_skip = True
                    break
            if should_skip:
                continue
            flags.append({
                "index": i,
                "word": " ".join(window),
                "reason": f"phonetic_near_medical_{n}gram",
                "candidates": candidates,
                "span_indices": list(range(i, i + n)),
            })
            for off in range(n):
                consumed.add(i + off)

    _try_ngram(3, threshold=0.55, filler_threshold=0.75)
    _try_ngram(2, threshold=0.50, filler_threshold=0.78)

    for i, cands in enumerate(single_results):
        if i in consumed or not cands:
            continue
        top = cands[0]
        if top["phonetic_similarity"] < 0.60:
            continue
        if top["phonetic_similarity"] < 0.65:
            word_skel = consonant_skeleton_ar(translit(words[i]))
            term_skel = consonant_skeleton_latin(top["term"])
            lcs = longest_common_substring(word_skel, term_skel)
            if lcs < 3:
                continue
        flags.append({
            "index": i,
            "word": words[i],
            "reason": "phonetic_near_medical",
            "candidates": cands,
        })

    flags.sort(key=lambda f: f["index"])
    return flags
