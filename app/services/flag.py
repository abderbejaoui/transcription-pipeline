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

import functools
import json
import os
import re
import time
import unicodedata
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MEDICAL_TERMS_PATH = PROJECT_ROOT / "medical_terms.txt"
# Dedicated store for clinician-confirmed alias->term mappings taught at
# runtime via the HITL loop. Kept separate from the seed lexicon so only
# explicit human teachings are ever auto-applied.
HITL_ALIASES_PATH = PROJECT_ROOT / "data" / "hitl_aliases.json"


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
# Punctuation that ASR/transcripts glue onto word edges. Natural dictation
# is full of commas/periods (Arabic \u060c \u061b \u061f and Latin), and a trailing \u060c
# makes a real Arabic word fail morphological analysis \u2014 so it slips past
# the filler check and gets phonetically matched as a fake drug. Strip it
# from word EDGES before analysis (interior punctuation is left untouched).
_EDGE_PUNCT = "\u060c\u061b\u061f.,;:!\u061f?\"'`()[]{}\u00ab\u00bb\u2026\u201c\u201d\u2018\u2019-\u2014_ "


# ---------------------------------------------------------------------------
# CAMeL Tools morphological analyzers (Gulf Arabic + MSA).
# Loaded once at import time; silently absent if the DBs aren't downloaded.
# ---------------------------------------------------------------------------

def _load_morph_analyzers():
    glf, msa = None, None
    try:
        from camel_tools.morphology.database import MorphologyDB
        from camel_tools.morphology.analyzer import Analyzer
        try:
            glf = Analyzer(MorphologyDB.builtin_db("calima-glf-01"))
        except Exception:
            pass
        try:
            msa = Analyzer(MorphologyDB.builtin_db("calima-msa-r13"))
        except Exception:
            pass
    except ImportError:
        pass
    return glf, msa


_GLF_ANALYZER, _MSA_ANALYZER = _load_morph_analyzers()


@functools.lru_cache(maxsize=8192)
def _morph_is_real_arabic(word: str) -> bool:
    """Return True if `word` has at least one valid morphological analysis.

    Gulf Arabic + MSA databases together cover all standard Arabic words,
    colloquial Gulf forms, and MSA medical vocabulary. Drug mangles
    (e.g. \u0644\u0627\u064a\u0632\u064a\u0646\u0648 for lisinopril) produce zero analyses; genuine Arabic
    words always produce at least one.
    """
    for analyzer in (_GLF_ANALYZER, _MSA_ANALYZER):
        if analyzer is None:
            continue
        try:
            if any(a.get("pos") not in ("PUNC", None) for a in analyzer.analyze(word)):
                return True
        except Exception:
            continue
    return False


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
# Medical lexicon
# ---------------------------------------------------------------------------

_lex_cache: Optional[List[str]] = None
# Exact clinician-confirmed alias -> canonical term (normalised key).
_alias_cache: Optional[Dict[str, str]] = None


def load_medical_lexicon() -> List[str]:
    """Return the candidate-retrieval lexicon (medical_terms.txt).

    HITL-taught terms are appended to this very file by add_retrieval_term(),
    so anything a clinician teaches becomes retrievable on the next run.
    Result is cached; invalidate_lexicon_cache() clears it after a teach so
    no server restart is needed.
    """
    global _lex_cache
    if _lex_cache is not None:
        return _lex_cache
    if not MEDICAL_TERMS_PATH.exists():
        _lex_cache = []
        return _lex_cache
    terms: List[str] = []
    for line in MEDICAL_TERMS_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    _lex_cache = terms
    return _lex_cache


def add_retrieval_term(term: str) -> bool:
    """Append a canonical term to the candidate-retrieval dataset.

    Used by the HITL / teach feedback loop: the clinician-confirmed term is
    written into medical_terms.txt so it can be retrieved as a candidate for
    future similar-sounding mishearings. Returns True if newly added.
    Only Latin canonical terms are useful (the retrieval matcher folds Arabic
    needles against Latin skeletons), so non-Latin terms are skipped.
    """
    term = (term or "").strip()
    if not term or not re.search(r"[A-Za-z]", term):
        return False
    existing = {t.lower() for t in load_medical_lexicon()}
    if term.lower() in existing:
        return False
    with MEDICAL_TERMS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(term + "\n")
    invalidate_lexicon_cache()
    return True


def _norm_alias(s: str) -> str:
    """Normalise an alias/span for exact HITL matching: NFKC, drop tashkeel
    and all whitespace, lowercase. So 'ريزيدرونيك اسيد' (two tokens) and the
    flagged span match regardless of spacing."""
    s = unicodedata.normalize("NFKC", s)
    s = _TASHKEEL_RE.sub("", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _load_taught_alias_map() -> Dict[str, str]:
    """Map normalised clinician-taught aliases -> canonical term.

    Sourced ONLY from the dedicated HITL file (data/hitl_aliases.json), which
    /api/teach writes to. This is kept separate from the seed corrector
    lexicon so only mappings a human explicitly confirmed at runtime are
    auto-applied — the seed vocabulary never silently rewrites text."""
    global _alias_cache
    if _alias_cache is not None:
        return _alias_cache
    amap: Dict[str, str] = {}
    if HITL_ALIASES_PATH.exists():
        try:
            raw = json.loads(HITL_ALIASES_PATH.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for key, term in raw.items():
                    if isinstance(key, str) and isinstance(term, str) and len(key) >= 4:
                        amap[key] = term
        except Exception:
            pass
    _alias_cache = amap
    return amap


def record_taught_aliases(term: str, aliases: List[str]) -> int:
    """Persist clinician-confirmed alias -> term mappings for auto-apply.

    Called by the HITL / teach feedback loop. Returns the number of new
    mappings written. Keys are normalised for spacing/diacritic-insensitive
    exact matching."""
    term = (term or "").strip()
    if not term:
        return 0
    amap: Dict[str, str] = {}
    if HITL_ALIASES_PATH.exists():
        try:
            loaded = json.loads(HITL_ALIASES_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                amap = {str(k): str(v) for k, v in loaded.items()}
        except Exception:
            amap = {}
    added = 0
    for a in aliases or []:
        key = _norm_alias(str(a))
        if len(key) >= 4 and key not in amap:
            amap[key] = term
            added += 1
    if added:
        HITL_ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
        HITL_ALIASES_PATH.write_text(
            json.dumps(amap, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        invalidate_lexicon_cache()
    return added


def apply_taught_aliases(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Replace any clinician-taught alias occurrence with its canonical term.

    Deterministic exact-match pass (1-3 token windows) that runs before
    flagging. This is the auto-apply arm of the HITL loop: a mapping a human
    already confirmed is applied with full confidence. Returns
    (new_text, [{"from":..., "to":...}, ...])."""
    amap = _load_taught_alias_map()
    if not amap:
        return text, []
    tokens = re.split(r"(\s+)", text)
    word_pos = [i for i, t in enumerate(tokens) if t.strip()]
    replacements: List[Dict[str, str]] = []
    n = len(word_pos)
    i = 0
    while i < n:
        matched = False
        for size in (3, 2, 1):
            if i + size > n:
                continue
            positions = word_pos[i:i + size]
            key = _norm_alias("".join(tokens[p] for p in positions))
            if key in amap:
                canonical = amap[key]
                original = " ".join(tokens[p].strip() for p in positions)
                tokens[positions[0]] = canonical
                for p in positions[1:]:
                    tokens[p] = ""
                    if p - 1 >= 0:
                        tokens[p - 1] = ""
                replacements.append({"from": original, "to": canonical})
                i += size
                matched = True
                break
        if not matched:
            i += 1
    out = re.sub(r"\s+", " ", "".join(tokens)).strip()
    return out, replacements


def invalidate_lexicon_cache() -> None:
    """Drop cached lexicon + alias map so newly-taught terms take effect
    immediately (no restart). Called by /api/teach and /api/learn_from_edit."""
    global _lex_cache, _alias_cache
    _lex_cache = None
    _alias_cache = None


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
    # سفيجموميتر / بالسفيجموميتر → sphygmomanometer
    # Skeleton similarity is too low for normal matching; alias rescues it.
    "sfyjmwmytr": "sphygmomanometer",
    "fyjmwmytr": "sphygmomanometer",
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


# Correctly-spelled Arabic medical vocabulary — not to be flagged as suspicious.
# These are the stable, well-established Arabic forms for common clinical terms.
# The skeleton approach can't distinguish a correct Arabic form from a misspelling
# with the same consonants (e.g. نيكسيوم and a correct nexium spelling share the
# same skeleton). This set grows at runtime via register_known_arabic_form().
_KNOWN_ARABIC_MEDICAL_FORMS: set = {
    # cholesterol (كوليسترول) — common clitic variants
    "كوليسترول", "الكوليسترول", "للكوليسترول", "وكوليسترول", "بالكوليسترول",
    # insulin (أنسولين)
    "أنسولين", "الأنسولين", "للأنسولين", "وأنسولين", "بالأنسولين",
    # creatinine (كرياتينين)
    "كرياتينين", "الكرياتينين", "للكرياتينين", "وكرياتينين",
    # ibuprofen (إيبوبروفين)
    "إيبوبروفين", "الإيبوبروفين", "للإيبوبروفين", "بالإيبوبروفين",
}


def register_known_arabic_form(arabic_alias: str) -> None:
    """Add an Arabic alias to the known-correct set at runtime.

    Called by the HITL endpoint when a clinician registers a correctly-spelled
    Arabic alias for a lexicon term. This way the set grows automatically as
    the lexicon grows — no manual maintenance needed.
    """
    stripped = arabic_alias.strip()
    if stripped and any("؀" <= c <= "ۿ" for c in stripped):
        _KNOWN_ARABIC_MEDICAL_FORMS.add(stripped)


def _is_known_medical(word: str, lexicon: List[str]) -> bool:
    """Return True if `word` is already a correctly-spelled medical term.

    Checks the explicit known-Arabic-forms set first, then falls back to
    a direct lexicon match. The set grows at runtime via
    register_known_arabic_form() when clinicians register Arabic aliases.
    """
    word = word.strip(_EDGE_PUNCT)
    if word in _KNOWN_ARABIC_MEDICAL_FORMS:
        return True
    w = word.lower()
    if w in {t.lower() for t in lexicon}:
        return True
    tl = _translit(word)
    for t in lexicon:
        if tl == _translit(t):
            return True
    return False


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
        # Skip already-correct medical terms — they should NOT be flagged.
        if _is_known_medical(word, lexicon):
            single_results.append(None)
            continue
        single_results.append(_phonetic_candidates(word, lexicon, k=5))

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
            if n >= 2 and any(w in _ARABIC_SHORT_PARTICLES for w in window):
                continue
            # Reject if any word in the window is a known correct Arabic
            # medical term (e.g. الأنسولين, للكوليسترول) — combining a
            # correctly-spelled Arabic drug word with its neighbours
            # produces false-positive bigram flags like 'غيّر الأنسولين'.
            if any(_is_known_medical(w, lexicon) for w in window):
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
                joined, lexicon, k=5, threshold=threshold,
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
    # 2-gram filler_threshold raised 0.70 -> 0.78: a 2-gram that bridges ONE
    # filler word (e.g. 'الدكتور لايزينو', 'تحليل إيتش') is almost always a
    # spurious match unless the phonetic score is very high. Real split drugs
    # that include a filler ('له نيكسيوم', 'له برولوسيك') score ~1.0 and are
    # unaffected; filler-free splits use the lower base threshold.
    _try_ngram(2, threshold=0.50, filler_threshold=0.78)

    # --- Single-word pass for words not absorbed by an n-gram. Catches
    # both already-correct drug spellings (panadol -> sim 1.0 same word,
    # skipped) and genuine single-token mangles (kuwaiteen -> codeine).
    for i, cands in enumerate(single_results):
        if i in consumed or not cands:
            continue
        top = cands[0]
        if top["phonetic_similarity"] < 0.60:
            continue

        # Precision check for borderline single matches: when sim is
        # only 0.60-0.65 AND the contiguous shared skeleton is only
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
# Arabic filler detection — principled morphological approach.
#
# We keep only a tiny hardcoded set of 1-3 character particles that are
# too short for reliable morphological analysis. For all longer words,
# we delegate to the CAMeL Tools Gulf Arabic + MSA morphological analyzers:
# a word with ≥1 valid analysis is a real Arabic word and is never flagged.
# Drug mangles (Arabic-script transliterations of drug names) have zero
# valid analyses in both databases, so they correctly pass through to the
# phonetic similarity check.
# ---------------------------------------------------------------------------

_ARABIC_SHORT_PARTICLES = {
    # 1-3 char structural particles too short for reliable morphology
    "و", "أو", "او", "إذ",
    "في", "من", "إلى", "الى", "على", "عن", "مع", "لو",
    "لا", "ما", "لم", "لن", "قد", "ثم", "هو", "هي",
    "له", "لها", "لهم", "لنا", "به", "بها", "بهم",
    "ان", "إن", "أن",
    # very short dosage / unit abbreviations
    "ملغ", "مجم",
    # pure conjunction/preposition single chars
    "ف", "ب", "ل", "ك",
    # 3-char Arabic words the V2.1 short-word rule would otherwise
    # treat as potential drug fragments (len ≤ 3 → not filler).
    # These are all structural / verbal words that can never be drug names.
    "بدل",  # "instead of"
    "اخذ",  # "take / took" — verb that frequently precedes drug names
    "خذ",   # "take!" (imperative)
    "هل",   # question particle (do / is / did?)
    "بس",   # Gulf Arabic "just / only"
    "كم",   # "how much / how many"
    "عم",   # Gulf Arabic continuous marker ("doing")
    "قبل",  # "before" — preposition; collides with flagyl skeleton at sim 0.60
    "بعد",  # "after" — preposition; similar collision risk
    "آمن",  # "safe/secure" — adjective; skeleton mntn ≈ augmentin kmntn at 0.80
    "كل",   # "every/all" — adjective/determiner; too short to be a drug fragment
}

# Fallback for when morphology DBs are unavailable: a compact set covering
# the most common Arabic words that phonetically collide with drug skeletons.
_ARABIC_FILLER_FALLBACK = {
    "صداع", "دوخه", "دوار", "تعب", "حرارة", "الم", "وجع",
    "نفس", "ربو", "سكر", "ضغط", "التهاب", "مستشفى",
    "يحتاج", "عشان", "الحين", "استمر", "النتائج", "نتائج",
    "عدوى", "حمية", "كوليسترول", "الكوليسترول",
    "الدكتور", "الطبيب", "علاج", "دواء", "تحليل",
    "اليوم", "ساعه", "يوم", "شهر", "مرات", "حبوب",
}


def _is_arabic_filler(word: str) -> bool:
    """Return True if `word` is a real Arabic word that should never be flagged.

    Strategy (in order):
    1. Empty string → skip.
    2. Explicit particle set: confirmed short structural words → always skip.
    3. Short words (≤3 chars) NOT in the particle set → NOT filler. Drug
       name mangles often produce short fragments (e.g. 'با' in كارباmazepine,
       'كار' in كاربامازيبين) that must NOT be blocked from n-gram joining.
    4. Longer words (4+ chars): morphological check via Gulf + MSA analyzers.
    5. Fallback if morphology DBs unavailable: compact hardcoded set.
    """
    w = _TASHKEEL_RE.sub("", unicodedata.normalize("NFKC", word))
    w = w.strip(_EDGE_PUNCT)
    if not w:
        return True
    if w in _ARABIC_SHORT_PARTICLES:
        return True
    # Short words not in the particle list may be drug mangle fragments —
    # don't classify them as filler or they get excluded from n-gram joining.
    if len(w) <= 3:
        return False
    if _GLF_ANALYZER is not None or _MSA_ANALYZER is not None:
        return _morph_is_real_arabic(w)
    # Fallback: compact hardcoded set
    if w in _ARABIC_FILLER_FALLBACK:
        return True
    for pre in ("ال", "وال", "بال", "كال", "فال", "لل"):
        if w.startswith(pre) and len(w) > len(pre):
            if w[len(pre):] in _ARABIC_FILLER_FALLBACK:
                return True
    return False


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
    "with code-switched English. Your job: catch medical terms the "
    "automated phonetic checker might have missed.\n\n"
    "The phonetic checker already handles simple cases: Arabic drug names "
    "that phonetically resemble a known brand (بنادول -> panadol), "
    "and exact-match English drug names (augmentin, insulin).\n\n"
    "Focus on THESE hard cases:\n"
    "  1. English mishearing patterns: \"if all gone\" -> efferalgan, "
    "\"ef your gan\" -> efferalgan, \"all ergic\" -> allergic, etc.\n"
    "  2. Split drug names: when a single drug name was split across "
    "multiple tokens by the ASR. Usually these form a COMPLETE drug name "
    "when joined. Example tokens like 'برسي تمر' joined = 'برسيتمر' which "
    "is paracetamol.\n"
    "  3. Near-miss English: an English word that nearly matches a drug "
    "name (e.g. \"augmenta\" for augmentin, \"panadol\" is correct).\n"
    "  4. Novel drug names NOT in the medical lexicon.\n"
    "\n"
    "CRITICAL: Do NOT confirm or echo back the same term the phonetic "
    "pass already suggested when its similarity is low. The phonetic pass "
    "lists its best guesses for each flagged index. If its top guess is "
    "correct the score will be high (>=0.85). When the score is low "
    "(0.50-0.80), the flagged span is likely a FALSE POSITIVE — a common "
    "Arabic word that coincidentally shares consonants with a drug name. "
    "DO NOT return a flag for those indices unless you have an entirely "
    "different, clearly correct term to suggest.\n"
    "\n"
    "Correct example (skip): phonetic pass says index 5 -> pregabalin "
    "0.75 for \"بالركبة اليمنى\" (Arabic for 'the right knee'). This is a "
    "false positive anatomy phrase — do NOT flag it.\n"
    "\n"
    "Do NOT flag:\n"
    "  - Plain Arabic conversation (كيف حالك, لازم ترتاح, etc.)\n"
    "  - Common non-drug English words (patient, history, today, etc.)\n"
    "  - Anatomical words unless clearly a mishearing\n"
    "  - Words already caught by the phonetic pass (listed below)\n"
    "\n"
    "Strict rules:\n"
    "1. Output STRICT JSON only, no prose, no markdown.\n"
    "2. Word indices are zero-based, computed by splitting the transcript "
    "on whitespace. For a multi-token span (n-gram), use the index of "
    "the FIRST word in the span.\n"
    "3. Each flag entry: {\"index\": <int>, \"word\": <str>, "
    "\"reason\": <short string describing why>, "
    "\"likely_term\": <the correct drug name in Latin, "
    "or empty string if uncertain>, "
    "\"confidence\": <0.0 to 1.0>}.\n"
    "4. Schema: {\"flags\": [<flag entry>, ...]}.\n"
    "5. Use confidence >= 0.90 ONLY when the intended drug is clear from "
    "the medical/dosage context. Use 0.5-0.85 for plausible-but-uncertain. "
    "Use 0.0 if you have no idea.\n"
    "6. For split drug names (n-grams), set confidence to 0.0 and do NOT "
    "provide a likely_term unless you are CERTAIN of the merged result."
)


def _extract_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
    """Extract a JSON object from LLM output, handling markdown fences
    and stray prose."""
    # Try direct parse first
    text = text.strip()
    # Remove markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    # Try to find {...} anywhere in the text
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    # Try to find a "flags" array
    m = re.search(r'"flags"\s*:\s*\[.*?\]', text, re.S)
    if m:
        try:
            return json.loads("{" + m.group(0) + "}")
        except json.JSONDecodeError:
            pass
    return None


_LLM_SELECTION_SYSTEM = (
    "You are selecting the correct medical term for an Arabic Gulf dialect ASR transcript. "
    "You receive a flagged span (likely a mis-transcribed drug name) and a ranked list of "
    "phonetically similar candidates from the medical lexicon. "
    "Your task: pick ONE candidate that best fits the flagged span given the transcript context, "
    "or respond with 'NONE' if no candidate makes clinical sense.\n\n"
    "Rules:\n"
    "1. Output STRICT JSON: {\"chosen\": \"<term>\"} or {\"chosen\": \"NONE\"}.\n"
    "2. The chosen term MUST come exactly from the candidates list (no new terms).\n"
    "3. Use the surrounding transcript for clinical context "
    "(e.g. 'للضغط' = blood pressure → prefer ACE inhibitors; "
    "'للسكري' = diabetes → prefer metformin/insulin).\n"
    "4. Prefer drug names over disease names.\n"
    "5. If uncertain, output {\"chosen\": \"NONE\"} — do not guess."
)


def _llm_select_candidate(
    transcript: str,
    span_text: str,
    candidates: List[Dict[str, Any]],
    timeout: float = 20.0,
) -> Optional[str]:
    """Ask the LLM to pick the best candidate from a short list.

    Used when phonetic score is in the uncertain range (0.55–0.84): good
    enough to retrieve the right drug, but below the auto-correction
    threshold. The LLM uses full transcript context to make the call.
    Returns the chosen term string, or None if LLM is unavailable / uncertain.
    """
    if not candidates:
        return None
    cand_list = [
        f"  {c['term']} (phonetic_score={c['phonetic_similarity']:.2f})"
        for c in candidates[:5]
    ]
    user_msg = {
        "transcript": transcript,
        "flagged_span": span_text,
        "candidates": cand_list,
    }
    payload = {
        "model": get_llm_model(get_llm_provider()),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _LLM_SELECTION_SYSTEM},
            {"role": "user", "content": json.dumps(user_msg, ensure_ascii=False)},
        ],
    }
    obj = _call_llm(payload, timeout)
    if obj is None:
        return None
    chosen = (obj.get("chosen") or "").strip()
    if not chosen or chosen.upper() == "NONE":
        return None
    valid_terms = {c["term"].lower() for c in candidates}
    if chosen.lower() not in valid_terms:
        return None
    return chosen


def _call_llm(payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """Make an LLM API call with retry on failure."""
    last_error = None
    for attempt in range(2):
        try:
            req = urllib.request.Request(
                get_llm_url(get_llm_provider()),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(get_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = parse_chat_content(data, get_llm_provider()).strip()
            obj = _extract_json_from_llm(text)
            if obj is not None:
                return obj
            last_error = ValueError(f"failed to extract JSON from: {text[:200]}")
        except Exception as exc:
            last_error = exc
            if attempt == 0:
                print(f"[flag] LLM attempt {attempt + 1} failed: {exc!r}, retrying...")
    print(f"[flag] LLM pass failed after retries: {last_error!r}")
    return None


def llm_pass(
    transcript: str, *,
    phonetic_flags: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 60.0,
) -> List[Dict[str, Any]]:
    """Ask the LLM to flag medical terms the phonetic pass may have missed.

    `phonetic_flags` is passed as context so the LLM knows what was already
    caught and can focus on novel/missed cases instead of re-flagging.
    """
    words = re.split(r"\s+", transcript.strip())
    tokens_with_indices = [[i, w] for i, w in enumerate(words)]

    already_flagged_indices = set()
    flagged_summary: List[str] = []
    if phonetic_flags:
        for f in phonetic_flags:
            idx = f.get("index", -1)
            span = f.get("span_indices") or [idx]
            already_flagged_indices.update(span)
            word = f.get("word", "")
            cands = f.get("candidates", [])
            top_term = cands[0]["term"] if cands else "?"
            flagged_summary.append(f"  - index {idx}: '{word}' -> {top_term}")

    context = {
        "transcript": transcript,
        "tokens": tokens_with_indices,
        "phonetic_pass_flags": flagged_summary or ["(none)"],
    }
    payload = {
        "model": get_llm_model(get_llm_provider()),
        "stream": False,
        "format": "json",
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": _LLM_SYSTEM},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }
    obj = _call_llm(payload, timeout)
    if obj is None:
        return []
    flags = list(obj.get("flags", []))
    # Filter: exclude flags that point at indices the phonetic pass already
    # caught and that don't add new info (no likely_term or low confidence).
    filtered = []
    for f in flags:
        try:
            idx = int(f.get("index", -1))
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx >= len(words):
            continue
        likely = (f.get("likely_term") or "").strip()
        conf = float(f.get("confidence", 0.0) or 0.0)
        # Skip if phonetic pass already caught this index AND the LLM isn't
        # offering a meaningful correction.
        if idx in already_flagged_indices and (not likely or conf < 0.50):
            continue
        filtered.append(f)
    return filtered


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
        # Build a set of ALL indices covered by existing phonetic flags
        # (including n-gram spans) so we can skip overlapping LLM flags.
        covered_indices: set = set()
        for f in phon:
            spans = f.get("span_indices") or [f["index"]]
            covered_indices.update(spans)

        llm_results = llm_pass(transcript, phonetic_flags=phon)
        for entry in llm_results:
            try:
                idx = int(entry.get("index"))
            except (TypeError, ValueError):
                continue
            llm_conf = float(entry.get("confidence", 0.0) or 0.0)
            likely = (entry.get("likely_term") or "").strip()
            existing = phon_by_idx.get(idx)
            if existing:
                # Guard A: Don't let the LLM confirm the SAME term the
                # phonetic pass already suggested at low confidence.
                # The LLM often echoes the phonetic top candidate when it
                # should reject it as a false positive.
                top_phonetic = (existing.get("candidates") or [None])[0]
                phonetic_term = top_phonetic["term"].lower() if top_phonetic else ""
                phonetic_sim = float(top_phonetic["phonetic_similarity"]) if top_phonetic else 0.0
                if (
                    likely
                    and likely.lower() == phonetic_term
                    and phonetic_sim < 0.85
                ):
                    # LLM is parroting a weak phonetic match — don't boost it.
                    existing["llm_reason"] = "llm_rejected_weak_phonetic"
                    continue
                existing["llm_reason"] = entry.get("reason") or existing["reason"]
                if likely:
                    existing["llm_likely_term"] = likely
                existing["llm_confidence"] = max(existing.get("llm_confidence", 0.0), llm_conf)
            else:
                # Guard B: Skip LLM flags whose index falls within an
                # existing phonetic n-gram span. The phonetic pass already
                # handles multi-word spans; LLM single-word flags overlapping
                # them create duplicate corrections.
                if idx in covered_indices:
                    continue

                word = entry.get("word") or ""
                # Get phonetic candidates first; if empty but LLM has a
                # likely_term, inject it as a synthetic candidate so the
                # auto-correction stage can use it.
                cands = _phonetic_candidates(word, load_medical_lexicon())
                if not cands and likely:
                    cands = [{"term": likely, "phonetic_similarity": round(llm_conf, 3)}]
                # Guard C: no phonetic candidates and no concrete LLM term →
                # unactionable flag, almost certainly a hallucination on an
                # innocent Arabic context word. Drop it.
                if not cands and not likely:
                    continue
                entry_data = {
                    "index": idx,
                    "word": word,
                    "reason": entry.get("reason") or "llm_flag",
                    "candidates": cands,
                    "llm_reason": entry.get("reason"),
                    "llm_likely_term": likely,
                    "llm_confidence": llm_conf,
                }
                # For n-gram flags from the LLM, set span_indices so the
                # auto-correction stage handles multi-word replacement correctly.
                span_text = entry.get("word", "")
                span_tokens = span_text.split()
                if len(span_tokens) > 1:
                    entry_data["span_indices"] = list(range(idx, idx + len(span_tokens)))
                phon_by_idx[idx] = entry_data
    return sorted(phon_by_idx.values(), key=lambda f: f["index"])


# ---------------------------------------------------------------------------
# Auto-correction: build a corrected transcript using HIGH-CONFIDENCE LLM
# suggestions only. The dashboard surfaces this as a separate string so the
# user can compare it to the raw transcript without losing the original.
# ---------------------------------------------------------------------------

_AR_WAW = "و"  # Arabic conjunction 'and', commonly cliticised onto the next word


def apply_high_confidence_corrections(
    transcript: str,
    flags: List[Dict[str, Any]],
    *,
    confidence_threshold: float = 0.90,
    phonetic_strong_threshold: float = 0.85,
    phonetic_select_threshold: float = 0.55,
    include_hitl: bool = False,
    use_llm: bool = False,
) -> Dict[str, Any]:
    """Rewrite the transcript with high-confidence corrections.

    Sources of corrections, in priority order:

    1. PHONETIC TOP-1 (strong): top phonetic candidate scored >=
       `phonetic_strong_threshold` (0.85). Applied automatically.

    2. LLM SELECTION (borderline): when score is in
       [phonetic_select_threshold, phonetic_strong_threshold) and
       use_llm=True, the LLM chooses among top-5 candidates using the
       full transcript context. Answer is constrained to the candidate
       list — no hallucination risk.

    3. LLM DETECTION fallback: the existing llm_likely_term from the
       detection pass, used when phonetic is too weak.

    Conjunction preservation: if the original span starts with the Arabic
    conjunction 'و' (waw) cliticised to a drug mangle, the 'و' is
    prepended to the Latin correction so 'وسيمفاستاتن' → 'وsimvastatin'
    rather than dropping the conjunction.
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

        # Detect leading Arabic conjunction 'و' on the first span word.
        span_word = f.get("word", "")
        first_token = span_word.split()[0] if span_word else ""
        waw_prefix = first_token.startswith(_AR_WAW) and len(first_token) > 1

        # --- 1. Strong phonetic match → apply automatically.
        chosen = None
        source = None
        chosen_conf = 0.0
        if top and top_sim >= phonetic_strong_threshold:
            chosen = top["term"]
            chosen_conf = top_sim
            source = "phonetic"

        # --- 2. Borderline phonetic score → ask LLM to select among candidates.
        if chosen is None and use_llm and cands and phonetic_select_threshold <= top_sim < phonetic_strong_threshold:
            selected = _llm_select_candidate(transcript, span_word, cands)
            if selected and selected.lower() in lexicon_lower:
                chosen = selected
                chosen_conf = top_sim
                source = "llm_select"

        # --- 3. LLM detection fallback (llm_likely_term from the flagging pass).
        if chosen is None and llm_conf >= confidence_threshold and llm_term:
            if llm_term.lower() in lexicon_lower:
                chosen = llm_term
                chosen_conf = llm_conf
                source = "llm"

        if not chosen:
            continue

        # Preserve the Arabic conjunction if the span started with 'و<drug>'.
        # But NOT when the drug itself begins with a 'و'/w-sound — e.g.
        # 'وار فارين' is warfarin (و is part of the name), not 'و'+'arfarin'.
        # In that case the leading 'و' is consonant, so prepending it would
        # produce 'وwarfarin'. Latin drugs starting w/o/u correspond to a
        # leading Arabic و, so treat 'و' as a conjunction only otherwise.
        drug_starts_with_waw_sound = chosen[:1].lower() in ("w", "o", "u")
        if waw_prefix and not chosen.startswith(_AR_WAW) and not drug_starts_with_waw_sound:
            chosen = _AR_WAW + chosen

        # span_indices is set for bigram/trigram flags. Replace the FIRST
        # word and clear the rest so 'وفولتران مسا' → 'وvoltaren'.
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
    # --- 4. HITL escalation: flagged spans that couldn't be auto-corrected
    #     but have phonetic candidates are marked for human review.
    if include_hitl:
        corrected_indices = {a.get("index") for a in applied}
        for f in flags:
            idx = f.get("index")
            if not isinstance(idx, int) or idx < 0 or idx >= len(word_to_tok):
                continue
            if idx in corrected_indices:
                continue
            cands = f.get("candidates") or []
            if not cands:
                # No candidates at all — likely a false positive, don't escalate.
                continue
            spans = f.get("span_indices") or [idx]
            original_parts = []
            for off in spans:
                if 0 <= off < len(word_to_tok):
                    original_parts.append(tokens[word_to_tok[off]])
            applied.append({
                "index": idx,
                "span_indices": spans,
                "original": " ".join(original_parts),
                "corrected": "",
                "confidence": 0.0,
                "source": "hitl_escalate",
                "path": "hitl_escalate",
            })

    # Collapse runs of empty tokens.
    out = "".join(tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return {
        "corrected_transcript": out,
        "applied": applied,
        "threshold": confidence_threshold,
    }
