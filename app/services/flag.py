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
# Medical lexicon
# ---------------------------------------------------------------------------

_lex_cache: Optional[List[str]] = None


def load_medical_lexicon() -> List[str]:
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
    close. Tolerance 0.5 means lengths must be within 50% of each other:
    e.g. needle=5 -> term must be 3..10. Rejects the worst false
    positives without hurting real matches (where Arabic translit
    drops at most ~30% of the letters).
    """
    if not needle or not term:
        return False
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
    Arabic transliteration loses: p->b, v->f, c->k, g->k.

    'paracetamol' -> 'brktml'   (p->b, c->k, vowels dropped)
    'efferalgan'  -> 'ffrlkn'   (g->k)
    'ibuprofen'   -> 'bbrfn'    (p->b)
    'augmentin'   -> 'kmntn'    (g->k, second part)
    """
    VOWELS = set("aeiouy")
    SUBST = {"p": "b", "v": "f", "c": "k", "g": "k"}
    out = []
    for ch in s.lower():
        if ch in VOWELS:
            continue
        out.append(SUBST.get(ch, ch))
    return "".join(out)


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
    # Sort: similarity DESC, then drugs before non-drugs at the same score.
    scored.sort(key=lambda d: (-d["phonetic_similarity"], not d["_is_drug"]))
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
            has_filler = any(_is_arabic_filler(w) for w in window)
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
        # Skip only when the literal word IS already the Latin term.
        if words[i].lower() == top["term"].lower():
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
    "كل", "لا", "ما", "لم", "لن", "قد", "ثم", "او", "اي", "كما",
    # common verbs (Gulf imperatives + frequent forms)
    "خذ", "خذي", "خذو", "خود", "اخذ", "اخذي", "تاخذ", "تاخذي",
    "قال", "قالت", "قلت", "اعطاني", "اعطته", "اعطيه", "استعمل",
    "استعملي", "ابي", "اروح", "احس", "تعبان", "وصف", "خليه",
    "خليني", "روح", "تعال", "اجلس",
    # body / symptom words (not flagged: these are valid Arabic, not drugs)
    "صداع", "دوخه", "تعب", "حرارة", "حرارة", "الم", "وجع", "ضيق",
    "نفس", "ربو", "سكر", "ضغط", "ظهر", "ظهري", "حلق", "بطن",
    # time words
    "اليوم", "اليوم", "ساعه", "ساعات", "يوم", "اسبوع", "شهر", "صباحا",
    "مساء", "ليل", "نهار", "السبت", "الاحد", "الاثنين", "الثلاثاء",
    # dosage words
    "مرات", "مرتين", "مره", "حبه", "حبتين", "حبوب", "شراب", "كاسة",
    "كاسه", "ماي", "ماء", "ابره", "بخاخ", "تحاميل", "جل", "جرعتين",
    "ملليجرام", "مية", "خمسماية", "ثلاث", "خمس", "اربع", "ست",
    # honorifics / roles
    "الدكتور", "الطبيب", "الصيدلي", "ابني", "امي", "ابي",
    "المريض", "الوصفه", "الوصفة",
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
) -> Dict[str, Any]:
    """Rewrite the transcript using only LLM suggestions where
    `llm_confidence >= confidence_threshold` AND `llm_likely_term` is set.
    """
    tokens = re.split(r"(\s+)", transcript)  # keep whitespace tokens
    # Build a mapping word-index -> token-index in the split (only non-space
    # tokens count as words).
    word_to_tok: List[int] = []
    for ti, t in enumerate(tokens):
        if t.strip():
            word_to_tok.append(ti)

    applied: List[Dict[str, Any]] = []
    for f in flags:
        idx = f.get("index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(word_to_tok):
            continue
        conf = float(f.get("llm_confidence", 0.0) or 0.0)
        likely = f.get("llm_likely_term") or ""
        if conf < confidence_threshold or not likely:
            continue
        ti = word_to_tok[idx]
        original = tokens[ti]
        tokens[ti] = likely
        applied.append({
            "index": idx,
            "original": original,
            "corrected": likely,
            "confidence": conf,
        })
    return {
        "corrected_transcript": "".join(tokens),
        "applied": applied,
        "threshold": confidence_threshold,
    }
