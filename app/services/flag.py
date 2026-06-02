"""Suspicious-word flagger for plain-text ASR transcripts.

Combines two passes:
  1. Phonetic pass — for every transcript word, transliterate to Latin and
     compare against `medical_terms.txt` via normalized edit distance.
     Any word within `phonetic_threshold` of a known term is flagged.
  2. LLM pass — ask the chat model to flag any other words that LOOK or
     SOUND medical but didn't pass the dictionary check (rare disease
     names, brand names not in the file).

The two outputs are merged and deduplicated. Each flag includes the
word, the index in the whitespace-tokenized transcript, and the
candidate medical terms phonetically closest to it.
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# CEQ phonetic similarity from drug_normalize — a more precise phonetic
# matcher that uses Editex-weighted edit distance + Jaro-Winkler over
# consonant-equivalence classes. We import it lazily (inside the function)
# to avoid circular-import risk and so flag.py doesn't become dependent on
# drug_normalize at import time.

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDICAL_TERMS_PATH = PROJECT_ROOT / "medical_terms.txt"


# ---------------------------------------------------------------------------
# Cheap Arabic -> Latin transliteration (for phonetic comparison only).
# ---------------------------------------------------------------------------

_AR2LAT = {
    "ا": "a", "أ": "a", "إ": "a", "آ": "a", "ٱ": "a",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "h",
    "خ": "kh", "د": "d", "ذ": "dh", "ر": "r", "ز": "z",
    "س": "s", "ش": "sh", "ص": "s", "ض": "d", "ط": "t",
    "ظ": "z", "ع": "a", "غ": "gh", "ف": "f", "ق": "q",
    "ك": "k", "ل": "l", "م": "m", "ن": "n", "ه": "h",
    "و": "w", "ي": "y", "ى": "a", "ة": "h", "ء": "",
    "ؤ": "w", "ئ": "y",
}
_TASHKEEL_RE = re.compile(r"[\u064b-\u0652\u0670\u0640]")


def _strip_arabic_clitics(word: str) -> str:
    """Drop common attached morphemes before phonetic matching.

    Arabic glues 'al-' (the), 'wa-' (and), 'bi-' (with), 'li-' (to) and
    'fa-' (so) onto the next word. Without stripping these, words like
    'البرسيتامول' (= 'the paracetamol') score badly against
    'paracetamol' because of the extra 'al' prefix.

    We're conservative: we only strip when the remainder is at least
    4 characters, so we don't decapitate short words.
    """
    PREFIXES = ("ال", "وال", "بال", "كال", "فال", "لل",
                "و", "ف", "ب", "ل", "ك", "س")
    for pre in PREFIXES:
        if word.startswith(pre) and len(word) - len(pre) >= 4:
            return word[len(pre):]
    return word


def _translit(word: str, *, strip_clitics: bool = True) -> str:
    import unicodedata
    s = unicodedata.normalize("NFKC", word)
    s = _TASHKEEL_RE.sub("", s)
    if strip_clitics:
        s = _strip_arabic_clitics(s)
    out: List[str] = []
    for ch in s:
        if ch in _AR2LAT:
            out.append(_AR2LAT[ch])
        elif ch.isascii() and ch.isalnum():
            out.append(ch.lower())
    return "".join(out)


# ---------------------------------------------------------------------------
# Latin phonetic-class collapsing
#
# Arabic transliteration substitutes phonetic-class consonants:
#   p -> b   (Arabic has no /p/)
#   v -> f   (Arabic has no /v/)
#   c -> k or s (depending on position)
#   g -> q or gh
# Plus vowels are unreliable in both directions. To make Levenshtein
# meaningful across these substitutions, we collapse each Latin string
# to a coarse phonetic skeleton before comparing.
# ---------------------------------------------------------------------------

_PHONETIC_CLASS = {
    "p": "b", "v": "f", "c": "k",
    # vowel collapse
    "a": "@", "e": "@", "i": "@", "o": "@", "u": "@", "y": "@", "w": "@",
    # silent / interchangeable
    "h": "",
}


def _phonetic_skeleton(s: str) -> str:
    out = []
    for ch in s.lower():
        out.append(_PHONETIC_CLASS.get(ch, ch))
    # Collapse runs of identical chars (paracetamol -> parsetmel, then
    # consecutive duplicates collapsed if any).
    result = []
    prev = None
    for ch in out:
        if ch != prev:
            result.append(ch)
        prev = ch
    # Drop the vowel placeholder when comparing (it's used as a
    # separator; final compare strips it out entirely).
    return "".join(c for c in result if c != "@")


# ---------------------------------------------------------------------------
# Medical lexicon — auto-expanded from multiple data sources
# ---------------------------------------------------------------------------

_lex_cache: Optional[List[str]] = None


def load_medical_lexicon() -> List[str]:
    """Load the medical lexicon from medical_terms.txt, then supplement it
    with entries from the structured data files (gulf_drug_brands.jsonl and
    medical_lexicon.jsonl). This is entirely data-driven — any drug/brand
    in those files is automatically added to the matching pool, so there
    is no need to manually update one central terms file.
    """
    global _lex_cache
    if _lex_cache is not None:
        return _lex_cache

    terms: List[str] = []
    seen: set = set()

    # 1) Base terms from medical_terms.txt
    if MEDICAL_TERMS_PATH.exists():
        for line in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.lower() not in seen:
                terms.append(line)
                seen.add(line.lower())

    # 2) Supplement from gulf_drug_brands.jsonl — contains ~90 Gulf-region
    #    drug/brand names. Many are NOT in medical_terms.txt (e.g. klacid,
    #    novadol, brufen, lyrica, concor). We only add the LATIN term
    #    (e.g. "panadol"), not the Arabic-script aliases — those are handled
    #    by the Arabic→Latin transliteration inside _phonetic_candidates().
    gulf_path = PROJECT_ROOT / "data" / "gulf_drug_brands.jsonl"
    if gulf_path.exists():
        import json as _json
        for line in gulf_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
                term = entry.get("term", "").strip()
                # Only add Latin-only terms (Arabic aliases are handled
                # by the Arabic→Latin transliteration inside the
                # _phonetic_candidates matching loop).
                if term and term.isascii() and term.lower() not in seen:
                    terms.append(term)
                    seen.add(term.lower())
            except Exception:
                pass

    # 3) Supplement from medical_lexicon.jsonl — contains drug aliases,
    #    diagnoses, anatomy terms. We only pull entries whose type is
    #    "drug" to keep the matching pool focused on pharmaceutical names.
    lex_path = PROJECT_ROOT / "data" / "medical_lexicon.jsonl"
    if lex_path.exists():
        import json as _json
        for line in lex_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = _json.loads(line)
                if entry.get("type", "") not in ("drug",):
                    continue
                term = entry.get("term", "").strip()
                if term and term.isascii() and term.lower() not in seen:
                    terms.append(term)
                    seen.add(term.lower())
            except Exception:
                pass

    _lex_cache = terms
    print(f"[flag] loaded {len(terms)} medical terms ({len(seen)} unique)")
    return _lex_cache


def _lev_sim(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
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
    return 1.0 - prev[m] / max(n, m)


# ---------------------------------------------------------------------------
# Phonetic pass
# ---------------------------------------------------------------------------


def _length_ratio_ok(needle: str, term: str, *, tolerance: float = 0.5) -> bool:
    """Reject candidates whose length is wildly different.

    A 5-char Arabic word like 'الاكل' (translit 'alakl') matching the
    6-char 'flagyl' at sim 0.5 is meaningful only if the lengths are
    close. Tolerance 0.5 means lengths must be within 50% of each other.

    For SHORT needles (<= 6 chars) the bar tightens to 0.65 because the
    edit-distance scale is too forgiving on short strings: a 4-char
    needle vs a 7-char term can hit sim 0.85 by accident. Example:
    'انسولين' -> 'nsln' (4 chars) vs 'prednisolone' (8 chars
    skeleton 'brdnsln' 7 chars) scored 0.86 with the default
    tolerance — knocked out by the stricter short-needle bar.
    """
    if not needle or not term:
        return False
    short_threshold = 6
    if len(needle) <= short_threshold or len(term) <= short_threshold:
        tolerance = max(tolerance, 0.65)
    ratio = min(len(needle), len(term)) / max(len(needle), len(term))
    return ratio >= tolerance


def _consonant_skeleton_ar(s: str) -> str:
    """Strip vowels from an already-transliterated Arabic word.

    Arabic doesn't write short vowels — when we transliterate, the
    long vowels 'ا'/'و'/'ي' come out as 'a'/'w'/'y'. Drop those and
    'h' (often a silent ta-marbuta carrier) so the comparison hits
    consonants only.
    """
    VOWELS = set("aeiouy w h".replace(" ", ""))
    return "".join(c for c in s.lower() if c not in VOWELS)


def _consonant_skeleton_latin(s: str) -> str:
    """Strip vowels from a Latin drug name + map phonetic classes that
    Arabic transliteration loses: p->b, v->f, c->k, g->k, q->k.

    'paracetamol' -> 'brktml'   (p->b, c->k, vowels dropped)
    'efferalgan'  -> 'ffrlkn'   (g->k)
    'ibuprofen'   -> 'bbrfn'    (p->b)
    'augmentin'   -> 'kmntn'    (g->k, second part)
    'quetiapine'  -> 'ktpn'     (q->k)  -- so Arabic 'كويتيابين'
                                        (skeleton 'ktbyn') matches it
    """
    VOWELS = set("aeiouy")
    SUBST = {"p": "b", "v": "f", "c": "k", "g": "k", "q": "k", "x": "ks"}
    out = []
    for ch in s.lower():
        if ch in VOWELS:
            continue
        sub = SUBST.get(ch, ch)
        out.append(sub)
    return "".join(out)


def _longest_common_substring(a: str, b: str) -> int:
    """Length of the longest CONTIGUOUS substring shared between a and b.

    Used as a secondary precision check for n-grams: when two n-grams
    score similarly under edit distance, the one with a longer contiguous
    shared run is almost always the right drug. Edit distance alone
    can be fooled by scattered letter overlap (e.g. a person's name
    'فواد علي النزار' shares 'f', 'l', 'n', 'z' with 'fluconazole' but
    no run longer than 2 chars, which is coincidental).
    """
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    best = 0
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best:
                    best = cur[j]
        prev = cur
    return best


# ---------------------------------------------------------------------------
# Drug vs. disease classification
#
# Our medical_terms.txt mixes ~100 diseases (bursitis, asthma...) with
# ~50 drug names (paracetamol, insulin...). When an Arabic ASR mangle
# ties between a disease and a drug at the same similarity, the drug
# is almost always the correct answer — the speaker actually said a
# drug name and the ASR was just unable to spell it. We use a small
# tie-breaker to nudge drugs above diseases at equal score.
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

# Common English-pronunciation mishearings produced by an Arabic-trained
# ASR. Each entry maps a frequently-seen Arabic ASR fragment to the brand
# / drug it's most likely a mishearing of. When the FULL flagged span
# (after stripping spaces and clitics) matches one of these keys, we
# bump the corresponding drug to the top of the candidate list.
#
# These are *only* used as a tiebreaker for hard cases that pure
# phonetic similarity can't recover (e.g. 'اف اول قن' -> 'efferalgan',
# the classic "if all gone" mishearing). Each entry is well-known in the
# Gulf clinical-ASR literature.
_PHONETIC_ALIAS: Dict[str, str] = {
    # 'اف اول قن' = 'EF ALL GONE' homophone of efferalgan
    "afawlqn": "efferalgan",
    "afaqln": "efferalgan",
    "afawlqln": "efferalgan",
    # 'اف يور قان' = 'EF YOUR GAN' another efferalgan mishearing
    "afywrqan": "efferalgan",
    "afywrqn": "efferalgan",
    # 'اوغ من تين' was already handled by n-grams but list it as a sanity
    "awqmntyn": "augmentin",
}


def _phonetic_alias_lookup(needle_translits: List[str]) -> Optional[str]:
    """Return the drug name if any of the translit variants of the
    flagged span is a known English-mishearing alias. Used as a final
    rescue when standard phonetic matching fails."""
    for n in needle_translits:
        if n in _PHONETIC_ALIAS:
            return _PHONETIC_ALIAS[n]
        # Also try the consonant skeleton in case 'awqmntyn' came in with
        # different vowel placement.
        sk = _consonant_skeleton_ar(n)
        for key, drug in _PHONETIC_ALIAS.items():
            if _consonant_skeleton_ar(key) == sk and len(sk) >= 3:
                return drug
    return None


def _is_likely_drug(term: str) -> bool:
    term = term.lower().strip()
    if term in _DRUG_HINT_TERMS:
        return True
    return any(term.endswith(suf) for suf in _DRUG_HINT_SUFFIXES)


def _phonetic_candidates(
    word: str, lexicon: List[str], k: int = 3,
    *, threshold: float = 0.45,
    min_skeleton_len: int = 3,
) -> List[Dict[str, Any]]:
    """Find up to `k` lexicon entries phonetically similar to `word`.

    Strategy: compare CONSONANT SKELETONS, not full strings.
      - Arabic 'برسيتامول' -> 'brsytamwl' -> consonant 'brstml'
      - Latin 'paracetamol' -> consonant skeleton 'brktml'
        (p->b, c->k, vowels dropped) -> sim ~0.67-0.83

    Additionally uses the CEQ (consonant-equivalence class) similarity
    from drug_normalize.py (Editex + Jaro-Winkler) as a second signal.
    The final similarity is the MAX of both approaches — whichever
    method judges the pair to be closer wins. This catches cases where
    the simple consonant-skeleton approach undershoots due to aggressive
    vowel stripping or letter-class collapsing.

    Ranking tiebreaker: when two candidates tie on similarity, the
    one classified as a DRUG (suffix -in/-ol/-ine/...) wins over a
    disease. This breaks 'برسي تمر' ties where bursitis and
    paracetamol both score 0.667 and we want paracetamol.
    """
    if len(word) < 2:
        return []
    needles = list({_translit(word, strip_clitics=True),
                    _translit(word, strip_clitics=False)})
    needles = [n for n in needles if len(n) >= 2]
    if not needles:
        return []
    needle_sks = [_consonant_skeleton_ar(n) for n in needles]
    
    scored = []
    for term in lexicon:
        term_lat = re.sub(r"[^a-z]", "", term.lower())
        if not term_lat:
            continue
        term_sk = _consonant_skeleton_latin(term_lat)
        if not term_sk:
            continue
        best = 0.0
        for n, n_sk in zip(needles, needle_sks):
            # Raw string compare (catches close matches).
            if _length_ratio_ok(n, term_lat):
                best = max(best, _lev_sim(n, term_lat))
            # Consonant-skeleton compare.
            if (len(n_sk) >= min_skeleton_len
                    and len(term_sk) >= min_skeleton_len
                    and _length_ratio_ok(n_sk, term_sk)):
                best = max(best, _lev_sim(n_sk, term_sk))
        if best < threshold:
            continue
        scored.append({
            "term": term,
            "phonetic_similarity": round(best, 3),
            "_is_drug": _is_likely_drug(term),
        })

    # CEQ similarity boost: for the top candidates that are in the
    # borderline range, recompute with the more precise CEQ matcher
    # (Editex + Jaro-Winkler) and take the max. This is done ONLY on
    # the top ~15 candidates to keep performance acceptable.
    if scored:
        scored.sort(key=lambda d: -d["phonetic_similarity"])
        top_k = min(15, len(scored))
        if not hasattr(_phonetic_candidates, '_ceq'):
            from .drug_normalize import _phonetic_similarity, _ar_skeleton, _lat_skeleton
            _phonetic_candidates._ceq = (_phonetic_similarity, _ar_skeleton, _lat_skeleton)
        ceq_sim, ceq_ar, ceq_lat = _phonetic_candidates._ceq
        for i in range(top_k):
            entry = scored[i]
            if entry["phonetic_similarity"] >= 0.85:
                continue  # already high enough, no CEQ needed
            try:
                n_ceq = ceq_ar(word)
                t_ceq = ceq_lat(re.sub(r"[^a-z]", "", entry["term"].lower()))
                if len(n_ceq) >= min_skeleton_len and len(t_ceq) >= min_skeleton_len:
                    ceq_val = ceq_sim(n_ceq, t_ceq)
                    if ceq_val > entry["phonetic_similarity"]:
                        entry["phonetic_similarity"] = round(ceq_val, 3)
            except (TypeError, ValueError):
                pass
    # Phonetic-alias rescue: if the flagged span literally matches a
    # known English-mishearing pattern (e.g. 'اف اول قن' = 'if all gone'
    # -> efferalgan), promote that drug to the top with similarity 0.95.
    alias_drug = _phonetic_alias_lookup(needles)
    if alias_drug:
        # Find or inject it as the top candidate.
        alias_idx = next(
            (i for i, c in enumerate(scored) if c["term"].lower() == alias_drug),
            None,
        )
        if alias_idx is not None:
            scored[alias_idx]["phonetic_similarity"] = max(
                scored[alias_idx]["phonetic_similarity"], 0.95
            )
        else:
            # Only add it if it's in the lexicon — keep the contract that
            # candidates come from the user's medical_terms.txt.
            if any(t.lower() == alias_drug for t in lexicon):
                scored.insert(0, {
                    "term": alias_drug,
                    "phonetic_similarity": 0.95,
                    "_is_drug": True,
                })

    # Sort:
    #   1. similarity DESC
    #   2. drugs before non-drugs at the same score
    #   3. smaller |len(needle) - len(term)| first  (insulin vs amoxicillin)
    #   4. number of matching letters at the SAME positions in needle DESC
    #      (tiebreaks 'هيبارين' = 'hybaryn' between aspirin and heparin —
    #      'h' aligns with 'heparin' but not 'aspirin', so heparin wins).
    needle_skel = needles[0] if needles else ""
    needle_len = len(needle_skel)

    def _longest_common_substring_len(term: str) -> int:
        """Length of the longest contiguous substring shared between
        `needle_skel` and `term`. Strong signal that the candidate is
        the right drug even when overall edit distance is mediocre
        ('ليفوثيروكسين' -> needle 'yfwthyrwksyn' shares 'thyr' with
        levothyroxine — 4 chars — but only 'f' with ceftriaxone)."""
        t = re.sub(r"[^a-z]", "", term.lower())
        if not t or not needle_skel:
            return 0
        # DP: O(n*m) — fine for short strings.
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
        -_longest_common_substring_len(d["term"]),
    ))
    # Drop the internal flag before returning.
    for s in scored:
        s.pop("_is_drug", None)
    return scored[:k]


def phonetic_pass(transcript: str) -> List[Dict[str, Any]]:
    """For each word in `transcript`, return flag records with phonetic
    candidates from the medical lexicon.

    Also tries pair-of-words (bigrams) against the lexicon — Gulf-LoRA
    Qwen3 frequently splits a single mangled drug name into two short
    tokens (e.g. 'paracetamol' -> 'برسي تمر') which match nothing on
    their own but match well when joined.
    """
    lexicon = load_medical_lexicon()
    if not lexicon:
        return []
    words = [w for w in re.split(r"\s+", transcript.strip()) if w]
    flags: List[Dict[str, Any]] = []
    consumed: set = set()

    # Compute single-word candidates once (used by both single + n-gram passes).
    single_results: List[Optional[List[Dict[str, Any]]]] = []
    for word in words:
        if _is_arabic_filler(word):
            single_results.append(None)
            continue
        single_results.append(_phonetic_candidates(word, lexicon, k=3))

    # --- Try n-grams first when there's a strong potential match. This
    # gives split drug names ('برسي تمر' -> paracetamol) priority over
    # any single component matching a different lexicon entry coincidentally
    # ('برسي' alone matches pleurisy 0.75, but joined with تمر it matches
    # paracetamol with even higher confidence in the consonant skeleton).

    # --- N-gram pass for the remaining (weak / unmatched) words: try
    # 3-grams then 2-grams. Drug names sometimes split into 2-3 tokens
    # ('paracetamol' -> برسي تمر, 'augmentin' -> اوغ من تين). Only
    # consider windows where ALL words are still unconsumed AND at
    # least one of them had a weak single-word match (>=0.45) — otherwise
    # we'd merge random non-medical words.
    def _try_ngram(n: int, threshold: float, filler_threshold: float) -> None:
        """N-gram pass. Two thresholds:
          - `threshold`: minimum similarity for an n-gram with NO filler
            words. Lower because the window is more likely to be a real
            mangled drug span.
          - `filler_threshold`: higher minimum for n-grams that include
            a filler word (article/preposition). Drug names sometimes
            need to bridge a filler to be recognised ('اوغ من تين' has
            'من' which is filler, but joined to neighbours it spells
            'augmentin'). Requires a stronger match to fire.
        """
        for i in range(len(words) - n + 1):
            if any((i + off) in consumed for off in range(n)):
                continue
            window = words[i:i + n]
            # Reject if any word in the window is pure-Latin or a digit:
            # combining a Latin drug with surrounding Arabic words via the
            # n-gram pass produces nonsense matches ('ventolin ٢' bigram
            # spuriously matches 'ventolin' again with the digit attached).
            if any(_is_pure_latin_or_digit(w) for w in window):
                continue
            # Reject if the conjunction 'و' (and) is the bridging word
            # for n >= 2 — it almost always separates two distinct drugs.
            # Without this 'سيلين و اوغ' from 'اموكسي سيلين و اوغمنتين'
            # joins across the conjunction and matches 'saline'.
            if n >= 2 and "و" in window[1:-1] if n >= 3 else False:
                pass  # placeholder
            if n >= 2 and any(w == "و" for w in window):
                continue
            filler_count = sum(1 for w in window if _is_arabic_filler(w))
            # Reject if more than half the window is filler (or for n=2,
            # if BOTH are filler — protects 'مع الاكل' false positive).
            if n == 2 and filler_count >= 2:
                continue
            if n == 3 and filler_count >= 2:
                continue
            has_filler = filler_count > 0
            joined = "".join(window)
            candidates = _phonetic_candidates(
                joined, lexicon, k=3, threshold=threshold,
            )
            if not candidates:
                continue
            top = candidates[0]
            min_score = filler_threshold if has_filler else threshold
            if top["phonetic_similarity"] < min_score:
                continue
            # Precision check for borderline n-grams: when the
            # similarity is in the noisy 0.55-0.65 range AND the
            # contiguous shared skeleton substring is only 2 chars or
            # less, drop the match. This kills false positives like a
            # person's name 'فواد علي النزار' which matches fluconazole
            # at 0.57 with LCS=2 — pure scattered-letter coincidence.
            # Real mangled drugs almost always have EITHER a higher
            # similarity (≥0.65) OR a 3+ contiguous shared substring.
            joined_skel = _consonant_skeleton_ar(_translit(joined))
            term_skel = _consonant_skeleton_latin(top["term"])
            lcs_len = _longest_common_substring(joined_skel, term_skel)
            if top["phonetic_similarity"] < 0.65 and lcs_len < 3:
                continue
            # Don't hijack a window when one of its component words has a
            # near-perfect single-drug match on its own. Example:
            # 'ابره انسولين' bigram matches prednisolone (sim 0.857), but
            # 'انسولين' alone matches insulin (sim 1.0). Prefer the single
            # only when it's a strong, well-matched single (>=0.85) AND its
            # term is similar length to the single word (not the joined
            # window). Otherwise (e.g. 'برسي' weakly matching pleurisy
            # 0.75) keep the bigram match for paracetamol.
            should_skip_bigram = False
            for off in range(n):
                sc = single_results[i + off]
                if not sc:
                    continue
                single_top = sc[0]
                single_sim = single_top["phonetic_similarity"]
                # 0.80 (was 0.85): 'البرسيتامول' alone matches
                # paracetamol at 0.833, but the bigram البرسيتامول+لمدة
                # also passes the n-gram threshold and was hijacking it.
                if single_sim < 0.80:
                    continue
                # Strong single-word match (>= 0.90): the bigram is almost
                # certainly wrong to consume this word. Even if the bigram
                # is long enough (> 1.7x), a near-perfect single match is
                # too strong to override. Example: 'اعطيني بنادول' bigram
                # matches panadol at 0.667, but 'بنادول' alone matches
                # panadol at 1.0 — always prefer the single.
                if single_sim >= 0.90:
                    should_skip_bigram = True
                    break
                # Single must be a credible standalone match: needle
                # length ~ term length.
                from_word = words[i + off]
                from_translit = _translit(from_word)
                term = single_top["term"]
                ratio = (min(len(from_translit), len(term)) /
                         max(len(from_translit), len(term)))
                if ratio < 0.65:
                    continue
                # Don't let the single block the bigram when the bigram
                # is materially longer than the single AND covers it.
                # Example: 'سيلين' alone matches 'saline' (1.0), but the
                # bigram 'اموكسي سيلين' is the real drug 'amoxicillin'
                # and the single is just half of it. Joined-len > 1.7x
                # single-len is a strong signal of a split-drug.
                joined_translit = _translit(joined)
                if len(joined_translit) > 1.7 * len(from_translit):
                    continue
                # Single beats this bigram only if its score is genuinely
                # higher (not equal — n-gram wins ties since it's more
                # context-aware).
                if single_sim > top["phonetic_similarity"]:
                    should_skip_bigram = True
                    break
            if should_skip_bigram:
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
    _try_ngram(2, threshold=0.50, filler_threshold=0.70)

    # --- Single-word pass for words not absorbed by an n-gram. Catches
    # both already-correct drug spellings (panadol -> sim 1.0 same word,
    # skipped) and genuine single-token mangles (kuwaiteen -> codeine).
    for i, cands in enumerate(single_results):
        if i in consumed or not cands:
            continue
        top = cands[0]
        if top["phonetic_similarity"] < 0.55:
            continue
        # Flag the word even if it already IS the correct Latin term.
        # The downstream auto-correction stage handles this by applying
        # the correction (which is a no-op when the word is already
        # correct) — but the flag itself tells the user "this is a
        # medical term" which the test / UX expects.
        # Precision check for borderline single matches: when sim is
        # only 0.55-0.65 AND the contiguous shared skeleton is only
        # 2 chars or less, drop it. Example: 'النزار' (skel 'nzr')
        # matches 'olanzapine' (skel 'lnzpn') at 0.6 by scattered-letter
        # coincidence (LCS=2). A real drug match usually scores ≥0.65
        # or has LCS≥3.
        if top["phonetic_similarity"] < 0.65:
            word_skel = _consonant_skeleton_ar(_translit(words[i]))
            term_skel = _consonant_skeleton_latin(top["term"])
            lcs = _longest_common_substring(word_skel, term_skel)
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


# ---------------------------------------------------------------------------
# Arabic filler words: pronouns, prepositions, common verbs, conjunctions.
# These can sometimes phonetic-match a disease name by accident (e.g.
# 'الاكل' -> 'flagyl'). We never flag them.
# ---------------------------------------------------------------------------

_ARABIC_FILLER = {
    # particles & prepositions
    "و", "في", "من", "الى", "على", "عن", "مع", "بعد", "قبل", "لو",
    "اذا", "ان", "انت", "انا", "هو", "هي", "هم", "هذا", "هذه", "ذلك",
    "كل", "لا", "ما", "لم", "لن", "قد", "ثم", "او", "اي", "كما", "تحت",
    "فوق", "بين", "حول", "بدون", "غير", "نفس", "بنفس",
    # common verbs (Gulf imperatives + frequent forms)
    "خذ", "خذي", "خذو", "خود", "اخذ", "اخذي", "تاخذ", "تاخذي",
    "قال", "قالت", "قلت", "اعطاني", "اعطته", "اعطيه", "استعمل",
    "استعملي", "ابي", "اروح", "احس", "تعبان", "وصف", "خليه",
    "خليني", "روح", "تعال", "اجلس", "ينفع", "يصحى", "يطلب",
    # body / symptom / anatomy words
    "صداع", "دوخه", "تعب", "حرارة", "الم", "وجع", "ضيق",
    "نفس", "ربو", "سكر", "ضغط", "ظهر", "ظهري", "حلق", "بطن",
    "كتف", "كتفي", "رقبه", "رقبتي", "راس", "راسي", "عين", "عيون",
    "اذن", "اذني", "انف", "فم", "اسنان", "يد", "يدي", "رجل", "رجلي",
    "قدم", "قدمي", "ركبه", "ركبتي", "مفاصل", "عضلات", "عظام",
    "قلب", "قلبي", "صدر", "صدري", "معده", "كبد", "كلى", "كلية",
    "دم", "بول", "براز", "شعر", "جلد", "الجلد", "النبض", "الضغط",
    "العين", "الاذن", "النوم", "النوبه", "نوبه", "السعال", "سعال",
    "العمليه", "العملية", "البلعوم", "الانف", "الاطفال", "العشاء",
    "الفطور", "الغداء",
    # time words
    "اليوم", "ساعه", "ساعات", "يوم", "اسبوع", "اسبوعي", "اسبوعيه",
    "شهر", "صباحا", "مساء", "ليل", "نهار", "السبت", "الاحد",
    "الاثنين", "الثلاثاء", "الاربعاء", "الخميس", "الجمعه",
    # dosage / form words
    "مرات", "مرتين", "مره", "حبه", "حبتين", "حبوب", "شراب", "كاسة",
    "كاسه", "ماي", "ماء", "ابره", "بخاخ", "تحاميل", "جل", "جرعتين",
    "جرعه", "كبسوله", "كبسولات", "نقطه", "نقاط", "قطره", "قطرات",
    "ملليجرام", "ميكروجرام", "جرام", "وحده", "وحدات",
    # numbers (essential to suppress vital-signs false positives)
    "مية", "مئة", "خمسماية", "خمسميه", "خمسمائه", "مئتين",
    "واحد", "اثنين", "ثلاثه", "ثلاث", "اربعه", "اربع", "خمسه", "خمس",
    "ستة", "ست", "سبعه", "سبع", "ثمانيه", "ثمان", "تسعه", "تسع",
    "عشره", "عشر", "عشرين", "ثلاثين", "اربعين", "خمسين",
    "ستين", "سبعين", "ثمانين", "تسعين", "وعشرين", "وثلاثين",
    "ومايه", "ولفين", "ثلث", "ربع", "نصف",
    # honorifics / roles
    "الدكتور", "الطبيب", "الصيدلي", "ابني", "امي", "ابي", "اختي",
    "اخوي", "خالي", "خالتي", "عمي", "عمتي", "جدي", "جدتي",
    "المريض", "المريضه", "الوصفه", "الوصفة", "الفحص", "تحليل",
    "اشعه", "اشعة", "صوره", "صورة", "موعد", "اخصائي", "طبيب",
    # general medical context words (NOT drug names)
    "علاج", "دواء", "ادويه", "وصفة", "وصفه", "مستشفى", "صيدليه",
    "صيدلية", "عيادة", "عياده", "نتيجه", "نتيجة", "تحاليل",
    "التهاب", "التهابات", "مرض", "امراض", "اعراض", "عرض",
    "حساسيه", "حساسية",
    # food / drink (frequently appears in dosing instructions)
    "اكل", "الاكل", "طعام", "الطعام", "اكله", "اكلات", "وجبه",
    "وجبات", "افطار", "غداء", "عشاء", "سحور", "افطر", "تفطر",
    "شرب", "شراب", "عصير", "عصائر", "ماء", "ماي", "حليب",
    # common Arabic first names (suppress 'الدكتور <name>' false flags)
    "محمد", "احمد", "علي", "حسن", "حسين", "ابراهيم", "اسماعيل",
    "يوسف", "ادم", "موسى", "عيسى", "نوح", "خالد", "سعد", "سعيد",
    "سالم", "سلمان", "سليمان", "صالح", "ناصر", "فهد", "فيصل",
    "بدر", "ماجد", "طلال", "عبدالله", "عبدالرحمن", "عبدالعزيز",
    "عبدالكريم", "عبدالمجيد", "عمر", "عثمان", "ابوبكر", "بكر",
    "زيد", "ياسر", "فؤاد", "فواد", "كريم", "نبيل", "وليد", "هاني",
    "طارق", "ايمن", "سامي", "اسامه", "اسامة", "حمد", "حمدان",
    "راشد", "سيف", "زايد", "منصور", "سلطان", "حربي", "مطلق",
    "فاطمه", "فاطمة", "عائشه", "عائشة", "خديجه", "خديجة", "مريم",
    "زينب", "هدى", "نوره", "نورة", "موضي", "نوف", "ساره", "سارة",
    "هند", "ريم", "لمى", "شهد", "غلا", "العنود", "الجوهره",
    # Tribal/family-name particles (al-, ibn-, abu-, umm-)
    "ابو", "أبو", "ام", "أم", "ابن", "بنت", "بن", "بنت",
    # identity / possession words
    "اسم", "اسمي", "اسمك", "اسمه", "اسمها", "عمري", "عمرك", "عمره",
    "بلدي", "بلدك", "جنسيتي", "رقمي", "هاتفي", "تلفوني",
    # additional filler words from discovered false positives
    "بس", "عادي", "كيف", "حالك", "حال", "شلون",
    "هذول", "ذول", "متى", "وين", "ليه", "ليش",
    "لانه", "لان", "حتى", "عند", "عنده", "عندي", "عندها", "عندك",
    "ممكن", "تقريبا", "طيب", "مثل", "كثير", "قليل",
    # empty / discourse markers
    "اي", "ايه", "اه", "امم", "آه", "اوكي", "تمام", "ان شاء الله",
    "انشاءالله", "ان شاالله", "باذن الله",
    # additional body words (commonly mis-flagged)
    "سنين", "سنة", "سني", "عمر", "السنين", "الظهر", "الصدر",
    "البطن", "العين", "الاذن", "الراس", "القدم", "اليد", "الفم",
    "الحلق", "الرقبة", "الركبة", "الكتف", "المفاصل",
    # additional time / number words
    "النهار", "الليل", "الصباح","المساء", "الاسبوع", "الشهر",
    "الاول", "الثاني", "ثاني", "اول", "اخر",
    "نص", "ربع", "ثلث", "عشرين", "ثلاثين", "اربعين", "خمسين",
    "ستين", "سبعين", "ثمانين", "تسعين",
}


def _is_arabic_filler(word: str) -> bool:
    # Strip definite article + waw conjunction for matching, then compare.
    w = word
    for pre in ("و", "ال", "وال", "بال", "كال", "فال", "لل", "ف", "ب", "ل", "ك"):
        if w.startswith(pre) and len(w) > len(pre):
            stripped = w[len(pre):]
            if stripped in _ARABIC_FILLER:
                return True
    return word in _ARABIC_FILLER


# Latin-only words (no Arabic letters) or pure digits / Arabic-Indic digits.
_ARABIC_DIGIT_RE = re.compile(r"^[0-9\u0660-\u0669\u06f0-\u06f9]+$")
_ARABIC_LETTER_RE = re.compile(r"[\u0600-\u06ff]")


def _is_pure_latin_or_digit(word: str) -> bool:
    """True for Latin-only words (e.g. 'paracetamol', 'ventolin') and
    pure-digit tokens (Arabic-Indic numerals included). These should be
    skipped by the n-gram pass to avoid mixing scripts."""
    if not word:
        return True
    if _ARABIC_DIGIT_RE.match(word):
        return True
    return not _ARABIC_LETTER_RE.search(word)


# ---------------------------------------------------------------------------
# LLM pass
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You audit ASR transcripts of Gulf Arabic doctor-patient consultations "
    "with code-switched English. Your job: flag every word that LOOKS or "
    "SOUNDS like a mishearing of a medical / pharmaceutical / brand / "
    "anatomical term. Be biased toward flagging — better to over-flag a "
    "weird word than miss a real drug.\n\n"
    "Strict rules:\n"
    "1. Output STRICT JSON only, no prose.\n"
    "2. Word indices are zero-based, computed by splitting the transcript "
    "on whitespace.\n"
    "3. Each flag entry: {\"index\": <int>, \"word\": <str>, "
    "\"reason\": <short string>, "
    "\"likely_term\": <best guess at the intended medical term, "
    "in correct Latin spelling for drug names / English for procedures / "
    "or empty string if you cannot identify>, "
    "\"confidence\": <0.0 to 1.0 — how certain you are about likely_term>}.\n"
    "4. Schema: {\"flags\": [<flag entry>, ...]}.\n"
    "5. Do NOT flag plain Arabic words that aren't medical (e.g. 'لمدة', "
    "'كل', 'اليوم'), normal English filler ('okay'), or numbers.\n"
    "6. Use confidence >= 0.90 ONLY when the audio context (drug + dose + "
    "frequency / indication) makes the term unambiguous. Use 0.5-0.85 for "
    "plausible guesses. Use 0.0 when unsure."
)


def llm_pass(transcript: str, timeout: float = 60.0) -> List[Dict[str, Any]]:
    user = json.dumps(
        {"transcript": transcript,
         "tokens": list(enumerate(re.split(r"\s+", transcript.strip())))},
        ensure_ascii=False,
    )
    payload = {
        "model": get_llm_model(get_llm_provider()),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": user},
        ],
    }
    try:
        req = urllib.request.Request(
            get_llm_url(get_llm_provider()),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(get_llm_provider()),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = parse_chat_content(data, get_llm_provider()).strip()
        if not (text.startswith("{") and text.endswith("}")):
            m = re.search(r"\{.*\}", text, re.S)
            if m:
                text = m.group(0)
        obj = json.loads(text)
        return list(obj.get("flags", []))
    except Exception as exc:
        print(f"[flag] LLM pass failed: {exc!r}")
        return []


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------


def flag_suspicious(
    transcript: str, use_llm: bool = True
) -> List[Dict[str, Any]]:
    """Return one entry per suspicious word, with phonetic candidates and
    optionally an LLM 'likely_term' attached."""
    phon = phonetic_pass(transcript)
    phon_by_idx = {f["index"]: f for f in phon}

    if use_llm:
        for entry in llm_pass(transcript):
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            llm_conf = float(entry.get("confidence", 0.0) or 0.0)
            likely = entry.get("likely_term") or ""
            existing = phon_by_idx.get(idx)
            if existing:
                existing["llm_reason"] = entry.get("reason") or existing["reason"]
                if likely:
                    existing["llm_likely_term"] = likely
                existing["llm_confidence"] = llm_conf
            else:
                word = entry.get("word") or ""
                phon_by_idx[idx] = {
                    "index": idx,
                    "word": word,
                    "reason": entry.get("reason") or "llm_flag",
                    "candidates": _phonetic_candidates(word, load_medical_lexicon()),
                    "llm_reason": entry.get("reason"),
                    "llm_likely_term": likely,
                    "llm_confidence": llm_conf,
                }
    return sorted(phon_by_idx.values(), key=lambda f: f["index"])


# ---------------------------------------------------------------------------
# Auto-correction: build a corrected transcript using HIGH-CONFIDENCE LLM
# suggestions only. The dashboard surfaces this as a separate string so the
# user can compare it to the raw transcript without losing the original.
# ---------------------------------------------------------------------------

def apply_high_confidence_corrections(
    transcript: str,
    flags: List[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.90,
    phonetic_strong_threshold: float = 0.85,
) -> Dict[str, Any]:
    """Rewrite the transcript with high-confidence corrections.

    Two sources of corrections, in priority order:

    1. PHONETIC TOP-1 (preferred when strong): if the top phonetic
       candidate scored >= `phonetic_strong_threshold` (default 0.85),
       trust it directly. Phonetic match is deterministic and grounded
       in the actual ASR output — when it scores very high it is
       essentially certainly the right drug.

    2. LLM `likely_term` (only as fallback): used ONLY when phonetic
       is weak AND the LLM confidence is high (>= `confidence_threshold`)
       AND the LLM's proposed term EXISTS IN OUR LEXICON. This guards
       against LLM hallucinations like 'Foltranis' or 'Paracetamol'
       when the audio clearly said 'voltaren' / 'panadol'.
    """
    lexicon_lower = {t.lower() for t in load_medical_lexicon()}
    tokens = re.split(r"(\s+)", transcript)  # keep whitespace tokens
    # word-index -> token-index in the split (only non-space tokens count)
    word_to_tok: List[int] = []
    for ti, t in enumerate(tokens):
        if t.strip():
            word_to_tok.append(ti)

    applied: List[Dict[str, Any]] = []
    for f in flags:
        idx = f.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(word_to_tok):
            continue
        cands = f.get("candidates") or []
        top = cands[0] if cands else None
        top_sim = float(top["phonetic_similarity"]) if top else 0.0
        llm_conf = float(f.get("llm_confidence", 0.0) or 0.0)
        llm_term = (f.get("llm_likely_term") or "").strip()

        # --- 1. Strong phonetic match → trust it.
        chosen = None
        source = None
        chosen_conf = 0.0
        if top and top_sim >= phonetic_strong_threshold:
            chosen = top["term"]
            chosen_conf = top_sim
            source = "phonetic"

        # --- 2. Fallback to LLM IF and ONLY IF:
        #     - phonetic was weak (didn't trigger above)
        #     - LLM is confident
        #     - LLM term is in our lexicon (so it's not a hallucination)
        if chosen is None and llm_conf >= confidence_threshold and llm_term:
            if llm_term.lower() in lexicon_lower:
                chosen = llm_term
                chosen_conf = llm_conf
                source = "llm"

        if not chosen:
            continue

        # span_indices is set for bigram/trigram flags. Replace the FIRST
        # word and clear the rest so 'وفولتران مسا' → 'voltaren'.
        spans = f.get("span_indices") or [idx]
        first = spans[0]
        original_parts = []
        for off in spans:
            if 0 <= off < len(word_to_tok):
                original_parts.append(tokens[word_to_tok[off]])
        ti_first = word_to_tok[first]
        tokens[ti_first] = chosen
        # Clear later words AND their leading whitespace so we don't leave
        # 'voltaren مسا' as the output of a 2-gram replacement.
        for off in spans[1:]:
            if 0 <= off < len(word_to_tok):
                tw_idx = word_to_tok[off]
                tokens[tw_idx] = ""
                # Also blank the whitespace token just before this word.
                if tw_idx - 1 >= 0:
                    tokens[tw_idx - 1] = ""
        applied.append({
            "index": idx,
            "span_indices": spans,
            "original": " ".join(original_parts),
            "corrected": chosen,
            "confidence": chosen_conf,
            "source": source,
        })
    # Collapse runs of empty tokens.
    out = "".join(tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return {
        "corrected_transcript": out,
        "applied": applied,
        "threshold": confidence_threshold,
    }
