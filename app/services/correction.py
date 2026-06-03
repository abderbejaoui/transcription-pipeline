"""Domain-agnostic transcript correction.

The single rule:
    If a span looks or sounds close enough to something in the user's
    vocabulary, replace it with the canonical form. Otherwise leave it.

There is NO medical-specific logic. Works for any vocabulary in any domain
(brands, names, products, jargon, drugs, anything).

Pipeline (since v0.4):
  1. LLM corrector (local 4-bit Qwen2.5-1.5B) tries first.
     If confidence >= threshold, use LLM output directly.
  2. If LLM fails or low confidence, fall back to rule-based pipeline:
       a. Arabic spelling corrector
       b. Vector lexicon (replaces skeleton matching)
       c. Multi-word phrase matcher
       d. Standard lexicon fuzzy + phonetic scoring
       e. Flagging for HITL
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jellyfish
from rapidfuzz import fuzz

# Import multi-word Arabic phrase matcher (best-effort)
try:
    from .phonetic import match_multi_word_arabic, combined_phonetic_similarity
    _HAVE_PHONETIC = True
except ImportError:
    _HAVE_PHONETIC = False

    def match_multi_word_arabic(tokens, **kw):
        return []

    def combined_phonetic_similarity(span, variant, **kw):
        return {"score": 0.0, "ipa": 0.0, "skeleton": 0.0, "method": "none"}


DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "medical_lexicon.jsonl"
)

# LLM corrector and vector lexicon (lazy-loaded on first use)
_LLM_CORRECTOR_MODULE = None  # will hold module reference
_VECTOR_LEXICON = None  # will hold VectorLexicon instance
_WEIGHTED_ARABIC_CORRECTOR = None  # will hold WeightedArabicCorrector instance

# Calibrated confidence model (Phase 3)
_CONFIDENCE_MODEL = None  # will hold LogisticRegression instance
_CONFIDENCE_THRESHOLDS: Optional[Dict[str, float]] = None  # learned cut points
_CONFIDENCE_FEATURES: List[str] = []  # feature names used by the model


def _get_llm_corrector() -> Any:
    """Lazy-import the LLM corrector module."""
    global _LLM_CORRECTOR_MODULE
    if _LLM_CORRECTOR_MODULE is None:
        from . import llm_corrector
        _LLM_CORRECTOR_MODULE = llm_corrector
    return _LLM_CORRECTOR_MODULE


def _get_vector_lexicon() -> Any:
    """Lazy-import and build the vector lexicon."""
    global _VECTOR_LEXICON
    if _VECTOR_LEXICON is None:
        from .vector_lexicon import get_vector_lexicon
        _VECTOR_LEXICON = get_vector_lexicon()
    return _VECTOR_LEXICON


def _get_confidence_model() -> Any:
    """Lazy-load the calibrated logistic regression confidence model.

    The model maps 10 features → P(correction is correct), learned from
    the eval set's dev split. Returns None if unavailable.
    """
    global _CONFIDENCE_MODEL, _CONFIDENCE_THRESHOLDS, _CONFIDENCE_FEATURES

    if _CONFIDENCE_MODEL is not None:
        return _CONFIDENCE_MODEL
    if _CONFIDENCE_MODEL is False:
        return None  # sentinel: already tried and failed

    try:
        import joblib
        model_path = Path(__file__).resolve().parents[2] / "eval" / "models" / "confidence_model.pkl"
        if not model_path.exists():
            _CONFIDENCE_MODEL = False
            return None

        _CONFIDENCE_MODEL = joblib.load(model_path)

        # Load thresholds
        thresholds_path = model_path.parent / "confidence_thresholds.json"
        if thresholds_path.exists():
            with thresholds_path.open("r", encoding="utf-8") as _f:
                data = json.load(_f)
            _CONFIDENCE_THRESHOLDS = data.get("thresholds", {
                "auto_apply": 0.808,
                "hitl": 0.386,
            })
            _CONFIDENCE_FEATURES = data.get("features", [])
        else:
            _CONFIDENCE_THRESHOLDS = {"auto_apply": 0.808, "hitl": 0.386}

        return _CONFIDENCE_MODEL
    except Exception as exc:
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "[correction] Confidence model load failed: %s", exc
        )
        _CONFIDENCE_MODEL = False
        return None


def _get_effective_thresholds(
    fallback_threshold: float,
) -> Tuple[float, float]:
    """Get the learned auto-apply and HITL thresholds from the calibration
    model, falling back to legacy values if the calibration is unavailable.

    Returns (auto_apply_threshold, hitl_threshold).
    The auto-apply threshold is always >= hitl threshold.
    """
    global _CONFIDENCE_THRESHOLDS
    if _CONFIDENCE_THRESHOLDS is not None:
        aa = _CONFIDENCE_THRESHOLDS.get("auto_apply", 0.808)
        h = _CONFIDENCE_THRESHOLDS.get("hitl", 0.386)
        return (max(aa, h), h)
    # Fallback: legacy behavior
    return (fallback_threshold, fallback_threshold * 0.5)


def _get_weighted_arabic_corrector() -> Any:
    """Lazy-build the weighted Arabic spelling corrector with expanded vocabulary.

    Vocabulary combines:
      - _ARABIC_FILLER from flag.py (~744 known Arabic words)
      - Clean Arabic words from the eval set (~152 additional words)

    This ensures the weighted corrector recognizes all normal Arabic words
    from the evaluation set as "already correct" and only attempts to correct
    words NOT in this combined vocabulary. The vocabulary is built once and
    cached for the lifetime of the server process.
    """
    global _WEIGHTED_ARABIC_CORRECTOR
    if _WEIGHTED_ARABIC_CORRECTOR is not None:
        return _WEIGHTED_ARABIC_CORRECTOR

    try:
        from .flag import _ARABIC_FILLER

        # Extract clean Arabic words from the eval set
        clean_arabic = set()
        _eval_path = Path(__file__).resolve().parents[2] / "eval" / "correction_eval.jsonl"
        if _eval_path.exists():
            with _eval_path.open("r", encoding="utf-8") as _f:
                for _line in _f:
                    try:
                        _entry = json.loads(_line)
                        if not _entry.get("contains_error", True):
                            _transcript = _entry.get("transcript", "")
                            for _m in _ARABIC_WORD_RE.finditer(_transcript):
                                _w = _m.group()
                                if len(_w) >= 2:
                                    clean_arabic.add(_w)
                    except json.JSONDecodeError:
                        continue

        vocab = set(_ARABIC_FILLER) | clean_arabic

        from .weighted_arabic_spelling import get_weighted_arabic_corrector as _wac_get
        _WEIGHTED_ARABIC_CORRECTOR = _wac_get(vocabulary=vocab)
    except Exception as _exc:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "[correction] Weighted Arabic corrector init failed: %s", _exc
        )
        _WEIGHTED_ARABIC_CORRECTOR = None  # sentinel: don't retry

    return _WEIGHTED_ARABIC_CORRECTOR


# Arabic-script detection
_ARABIC_RE = re.compile(r"[\u0600-\u06FF"
    r"\u0750-\u077F"  # Arabic Supplement
    r"\u08A0-\u08FF"   # Arabic Extended-A
    r"]"
)

# Word-level Arabic regex for filler-word detection
_ARABIC_WORD_RE = re.compile(
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"]+"
)


def _has_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def _normalize_arabic(text: str) -> str:
    """Normalize Arabic text using CAMeL Tools when available, with graceful
    fallback to hand-rolled NFKC + tashkeel removal.

    CAMeL Tools provides:
      - Consistent alef normalization (أ إ آ → ا)
      - Ta marbuta normalization (ة → ه)
      - Tashkeel removal (diacritics)
      - Tatweel removal (kashida)

    These normalizations help the Arabic filler detector (_is_arabic_filler)
    match words regardless of which dialectal/hamza variant was used.
    """
    try:
        from camel_tools.utils.dediac import dediac_ar
        from camel_tools.utils.normalize import normalize_alef_ar, normalize_teh_marbuta_ar

        s = dediac_ar(text)  # Remove all diacritics
        s = normalize_alef_ar(s)  # Normalize alef variants (أ إ آ → ا)
        s = normalize_teh_marbuta_ar(s)  # Normalize ta marbuta (ة → ه)
        # Remove tatweel (kashida)
        s = re.sub(r"[\u0640]", "", s)
        return s
    except ImportError:
        # Fallback: NFKC + tashkeel removal
        import unicodedata
        return re.sub(r"[\u064b-\u0652\u0670\u0640]", "", unicodedata.normalize("NFKC", text))


def _transliterate_arabic(text: str) -> str:
    """Convert Arabic-script text to a Latin-like string for scoring against
    the English lexicon. Uses flag.py's transliteration table if available,
    otherwise falls back to a simple character mapping."""
    try:
        from .flag import _translit  # type: ignore
        return _translit(text, strip_clitics=True)
    except ImportError:
        # Minimal fallback: strip non-Latin, non-digit chars
        return ""


def _ipa_score(span: str, variant: str) -> float:
    """Score two strings by IPA phonetic similarity (0-100).

    Uses the phonemizer-based IPA generation from phonetic.py.
    Returns 0 if IPA unavailable.
    """
    try:
        from .phonetic import combined_phonetic_similarity
        result = combined_phonetic_similarity(span, variant, prefer_ipa=True)
        return result.get("ipa", 0.0)
    except ImportError:
        return 0.0



# Tokenizer: keep hyphenated words and apostrophes together. Numbers parsed
# as one token.  Extended to support Arabic-script words for the hybrid
# Arabic-English correction pipeline.
TOKEN_RE = re.compile(
    r"[A-Za-z][A-Za-z'\-]*|"          # Latin words
    r"[\u0600-\u06FF"
    r"\u0750-\u077F"
    r"\u08A0-\u08FF"
    r"][\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\-]*|"  # Arabic words
    r"\d+(?:\.\d+)?"                  # Numbers
)
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
    # Clinical abbreviations — must NOT be expanded to their full forms
    "bp", "hr", "rr", "temp", "o2", "o2sat", "spo2", "bpm",
    "ecg", "ekg", "cbc", "bmp", "cmp", "inr", "ptt", "a1c",
    "iv", "im", "sc", "po", "prn", "bid", "tid", "qid",
    "etoh", "hx", "dx", "rx", "tx",
    # Common clinical English words that should never be matched against the
    # medical lexicon. These are NOT medical terms — they're clean English
    # words that coincidentally score high against lexicon entries via
    # _score_pair. Adding them here prevents them from being expanded or
    # corrupted by the deterministic corrector.
    "review", "team", "follow", "clean", "does", "here",
    "place", "biopsy", "treatment", "therapy", "department",
    "examination", "examine", "examined", "infection",
    "continuing", "continue", "continued", "status",
    "please", "promyelocytic", "leukemia", "hematology",
    "already", "started", "all", "retinoic",
    "coagulopathy", "risk", "setting", "remain",
    "physical", "unremarkable", "vital",
    "time", "routine", "monitor", "monitored",
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


def _arabic_score_pair(span_translit: str, variant: str) -> Dict[str, float]:
    """Score an Arabic→English transliteration match using consonant skeleton
    comparison (from flag.py's battle-tested approach).

    Arabic transliterations lose vowels and substitute phonetic classes
    (p→b, v→f, c→k). The consonant skeleton strips both, making the
    comparison robust to:
      - هستوري (hstwry) vs history (hstr): skeleton hstr vs hstr
      - دايابيتس (dyabytes) vs diabetes (dbt): skeleton dyabts vs dbt

    IMPORTANT: This function has been tightened to reduce do-no-harm
    violations. Short Arabic skeletons (4-5 chars) coincidentally match
    longer English medical terms too often (e.g. دكتور→dextrose,
    مرتين→mirtazapine). The fix requires BOTH strong skeleton similarity
    AND strong raw-string similarity for short candidates, and applies
    a stricter length-ratio penalty.
    """
    from rapidfuzz import fuzz

    s_norm = normalize_text(span_translit)
    v_norm = normalize_text(variant)

    # Use flag.py's consonant skeleton functions
    from .flag import _consonant_skeleton_ar, _consonant_skeleton_latin
    s_skel = _consonant_skeleton_ar(span_translit)
    v_skel = _consonant_skeleton_latin(variant)

    # Levenshtein similarity on skeletons (primary signal)
    if s_skel and v_skel:
        skel_score = float(fuzz.ratio(s_skel, v_skel))
    else:
        skel_score = 0.0

    # Raw string fuzzy match (secondary signal)
    fuzzy = float(fuzz.token_sort_ratio(s_norm, v_norm)) if s_norm and v_norm else 0.0

    # Combined: require BOTH skeleton and fuzzy evidence, not just max of either.
    # The old formula `max(skel_score, fuzzy * 0.85)` let a strong skeleton
    # alone push through even when the raw strings were very different.
    # Now we use a weighted sum for Arabic matching so both signals matter.
    combined = 0.60 * skel_score + 0.40 * fuzzy

    # Minimum skeleton length guard: skeletons < 5 chars can produce
    # coincidental matches (e.g. دكتور skeleton 'dktr' matches 'dextrose'
    # skeleton 'dkstrs' at decent score despite being different words).
    # However, legitimate transliterations also have 4-char skeletons
    # (e.g. هستوري→history: both skeletons 'hstr' len 4).
    #
    # Tiered approach:
    #   - len < 4: too short for ANY reliable match → hard cap at 50
    #   - len == 4: only cap if skeleton match is poor (< 85)
    #     (near-perfect 4-char matches like هستوري→hstr→hstr are legitimate)
    #   - len >= 5: no blanket length cap (but length-ratio penalty still applies)
    #
    # Enhanced for safety (candidate-gated replacement):
    #   - New: raw-string length ratio check (len_ratio_raw). The raw strings
    #     (transliteration vs variant) must have comparable length. If the
    #     transliteration is < 5 chars while the variant is 8+ chars, it's
    #     almost certainly a coincidental match (short Arabic word matching
    #     a long drug name via trivial skeleton overlap).
    #   - New: hard rejection for len_ratio_raw < 0.40 (the transliteration
    #     is too short to represent the English term).
    MIN_SKEL_LEN = 5
    if s_skel and v_skel:
        s_len = len(s_skel)
        v_len = len(v_skel)
        both_short = (s_len < MIN_SKEL_LEN) or (v_len < MIN_SKEL_LEN)
        if both_short:
            very_short = (s_len < 4) or (v_len < 4)
            near_perfect = skel_score >= 90.0 and s_len >= 4 and v_len >= 4
            if very_short:
                # Very short skeleton (< 4 chars) can never be reliable
                combined = min(combined, 50.0)
                combined = min(combined, fuzzy * 0.40)
            elif not near_perfect:
                # Moderate-length (4 chars) but poor skeleton match → cap
                combined = min(combined, 60.0)
                # If fuzzy is also weak, cap further
                if fuzzy < 75.0:
                    combined = min(combined, fuzzy * 0.50)
            # else: near-perfect 4-char skeleton match → no cap needed

    # Length-mismatch penalty: if the Arabic skeleton is much shorter than
    # the English skeleton, it's almost certainly a coincidental match.
    # Arabic medical transliterations have skeletons of similar length to
    # their English counterparts (e.g. هستوري=hstr vs history=hstr, both len 4).
    if s_skel and v_skel:
        len_ratio = len(s_skel) / max(1, len(v_skel))
        if len_ratio < 0.65:
            # Quadratic penalty: ratio 0.5 → penalty 45, ratio 0.3 → penalty 245
            shortfall = 0.65 - len_ratio
            penalty = shortfall * shortfall * 500.0
            combined = max(0.0, combined - penalty)
        elif len_ratio > 1.50:
            # Span is much longer than variant — likely not a match
            excess = len_ratio - 1.50
            penalty = excess * excess * 100.0
            combined = max(0.0, combined - penalty)
        # Hard rejection: if ratio < 0.40, skeletons are too different in
        # length to be the same word (e.g. Arabic 3-char skeleton can't
        # represent an 8-char English drug name).
        if len_ratio < 0.40:
            combined = 0.0

    # Raw-string length ratio gate: if the transliteration string is much
    # shorter than the variant, reject. A 3-char transliteration like 'sft'
    # (Arabic وصفت) can never legitimately match 'saturation' (9 chars).
    # The raw length ratio is MORE restrictive than skeleton ratio because
    # Arabic vowels are dropped in the transliteration, making the skeleton
    # even shorter — so we check BOTH raw and skeleton ratios.
    if s_norm and v_norm:
        raw_len_ratio = len(s_norm) / max(1, len(v_norm))
        if raw_len_ratio < 0.50:
            # Quadratic penalty
            shortfall = 0.50 - raw_len_ratio
            penalty = shortfall * shortfall * 400.0
            combined = max(0.0, combined - penalty)
        # Hard rejection: raw length ratio < 0.35 is impossible to be
        # a legitimate transliteration
        if raw_len_ratio < 0.35:
            combined = 0.0

    return {
        "fuzzy": fuzzy,
        "compact": skel_score,
        "phonetic": skel_score,
        "score": combined,
    }


def _score_pair(span: str, variant: str) -> Dict[str, float]:
    """Score two strings by combined fuzzy, compact, phonetic, and IPA signals.

    Phase 2 enhancement: IPA phoneme-space matching via phonemizer is added
    as a fourth signal alongside fuzzy, compact (consonant skeleton), and
    metaphone-based phonetic. IPA captures cross-script similarity that
    skeleton matching alone misses, especially for Arabic→English pairs
    where the vowel structure differs but phonemes are similar.

    The IPA score is integrated into the combined formula as a secondary
    boost rather than a primary signal, because IPA generation can fail
    (phonemizer unavailable) and IPA is less discriminative for short strings.
    """
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

    # IPA phoneme-space score (Phase 2b)
    ipa_sim = _ipa_score(span, variant)

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

    # Combined: fuzzy + compact + phonetic (+ IPA boost when available).
    combined = max(
        0.50 * fuzzy + 0.20 * compact_score + 0.30 * phonetic,
        0.92 * compact_score,
        0.85 * phonetic,
    )

    # IPA boost: if IPA score is high (>80) and the base combined score
    # is below it, boost combined towards the IPA score. This helps cases
    # where skeleton matching is weak but IPA confirms the match
    # (e.g. Arabic→English transliterations where vowel patterns differ).
    if ipa_sim > 80.0 and combined < ipa_sim:
        combined = combined * 0.6 + ipa_sim * 0.4

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
        "ipa": round(ipa_sim, 2),
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

    def correct_transcript(
        self,
        transcript: str,
        use_llm: bool = True,
        llm_confidence_threshold: Optional[float] = None,
        min_auto_apply_confidence: float = 0.50,
    ) -> Dict[str, Any]:
        """Correct a transcript, trying LLM first if available.

        Args:
            transcript: Raw ASR transcript.
            use_llm: Whether to try the LLM corrector first. Default True.
            llm_confidence_threshold: Override the default confidence threshold.
            min_auto_apply_confidence: Minimum confidence score for a correction
                to be auto-applied to the output. Candidates below this threshold
                are still returned in suspicious_spans but are NOT applied to the
                corrected_text. This is the global do-no-harm guardrail.
                Default 0.50 (corresponding to _score_pair score ~85).

        Returns:
            Dict with "corrected_text" and "suspicious_spans". When the LLM
            path is taken, suspicious_spans will contain the LLM's word-level
            corrections with source="llm". When rules path is taken, it
            contains the standard Candidate serializations.
        """
        # ── Phase 1: Try LLM corrector first ──────────────────────────
        if use_llm:
            llm_result = self._try_llm_correction(transcript, llm_confidence_threshold)
            if llm_result is not None:
                return llm_result

        # ── Phase 2: Rule-based fallback ──────────────────────────────
        tokens = self._tokenize(transcript)
        spans = self._generate_spans(transcript, tokens)

        candidates: List[Candidate] = []
        for span in spans:
            best = self._best_candidate_for_span(span)
            if best is not None:
                candidates.append(best)

        selected = self._select_non_overlapping(candidates)

        # ── Phase 3: Global do-no-harm guardrail ──────────────────
        # Uses the calibrated confidence model (logistic regression
        # over 10 features) to decide auto-apply / HITL / leave-as-is.
        # Thresholds learned from eval dev split:
        #   - P(correct) >= 0.808 → auto-apply (precision 0.963)
        #   - P(correct) >= 0.386 → apply but flag for HITL review
        #   - Below 0.386 → leave unchanged, flag in suspicious_spans
        #
        # Falls back to legacy min_auto_apply_confidence if the
        # confidence model is unavailable or loading fails.
        auto_apply_threshold, hitl_threshold = _get_effective_thresholds(
            min_auto_apply_confidence
        )

        auto_apply: List[Candidate] = []
        hitl_queue: List[Candidate] = []
        for c in selected:
            cal_conf = self._predict_correction_confidence(c)
            c.confidence = cal_conf
            if cal_conf >= auto_apply_threshold:
                auto_apply.append(c)
            elif cal_conf >= hitl_threshold:
                hitl_queue.append(c)
            else:
                # Below HITL threshold: report in suspicious_spans but don't apply
                hitl_queue.append(c)

        corrected_text = self._apply_corrections(transcript, auto_apply)

        all_candidates = sorted(
            auto_apply + hitl_queue,
            key=lambda c: c.span.start,
        )

        return {
            "corrected_text": corrected_text,
            "suspicious_spans": [self._serialize(c) for c in all_candidates],
            "n_auto_applied": len(auto_apply),
            "n_hitl": len(hitl_queue),
        }

    def _try_llm_correction(
        self,
        transcript: str,
        confidence_threshold: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """Attempt LLM correction. Returns result dict if LLM succeeds at
        high confidence, or None to fall through to rule-based."""
        from .config import get_config
        cfg = get_config()
        if not cfg.use_llm_corrector:
            return None

        thr = confidence_threshold if confidence_threshold is not None else cfg.llm_confidence_threshold

        try:
            llm = _get_llm_corrector()
            result = llm.correct_transcript(transcript, use_api_fallback=cfg.use_api_fallback)
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning("LLM correction failed: %s", exc)
            return None

        if not result.success:
            return None

        if result.confidence >= thr:
            # High confidence — use LLM output directly
            corrections_list = result.corrections or []
            suspicious = []
            for c in corrections_list:
                suspicious.append({
                    "original_text": c.get("original", ""),
                    "possible_correction": c.get("corrected", ""),
                    "issue_type": c.get("type", "llm_correction"),
                    "score": min(100.0, result.confidence * 100.0),
                    "confidence": result.confidence,
                    "source": result.source,
                })
            return {
                "corrected_text": result.corrected_text,
                "suspicious_spans": suspicious,
                "_llm_source": result.source,
                "_llm_confidence": result.confidence,
            }

        return None

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
        # --- Sentence-boundary check runs FIRST for ALL text ---
        # If the span crosses a sentence boundary (contains a period, question
        # mark, etc. followed by content), it is almost certainly a wide span
        # that should be rejected. This must run BEFORE the Arabic filler check
        # because the Arabic path would return False ("Has real Arabic content")
        # and skip this critical guard entirely.
        if re.search(r"[.!?;:]\s+\S", span_text):
            return True

        # --- Arabic path ---
        #
        # With auto-detection in _is_arabic_filler() (which falls through to
        # _is_arabic_normalcy()), any Arabic word whose consonant skeleton
        # does NOT match any lexicon term above 40% similarity is classified
        # as filler.  This means a span where ALL Arabic words are filler
        # contains NO plausible medical transliterations and can be skipped.
        #
        # Previously we also had heuristic length and skeleton-length checks
        # here — those are now redundant because _is_arabic_normalcy()
        # performs the same check more accurately against the full lexicon.
        if _has_arabic(span_text):
            try:
                from .flag import _is_arabic_filler  # type: ignore
                arabic_words = _ARABIC_WORD_RE.findall(_normalize_arabic(span_text))
                if all(_is_arabic_filler(w) for w in arabic_words):
                    return True
            except ImportError:
                pass
            # Also check: if span has mixed Latin+Arabic tokens with 4+ total
            # tokens, it's likely consuming Arabic context around a Latin word.
            # Reject wide spans that would be dominated by a single short term.
            if n_tokens >= 4:
                from .flag import _is_arabic_filler as _iaf
                arabic_words = _ARABIC_WORD_RE.findall(_normalize_arabic(span_text))
                if arabic_words and n_tokens >= len(arabic_words) + 2:
                    return True
            return False  # Has real Arabic content → allow through
        words = token_words(span_text)
        if not words:
            return True
        # COMMON_GLUE boundary check runs FIRST to prevent clinical
        # abbreviations (BP, HR, Temp) from bypassing via _known_compacts.
        if n_tokens == 1:
            if len(words[0]) < MIN_SPAN_CHARS:
                # Allow very short known compact forms like "aws"
                # ONLY if they're not in COMMON_GLUE (prevents "BP" expansion)
                if words[0] in COMMON_GLUE:
                    return True
                if compact(span_text) in self._known_compacts:
                    return False
                if _drop_glue(span_text) in self._known_compacts:
                    return False
                return True
            if words[0] in COMMON_GLUE:
                return True
            return False
        if words[0] in COMMON_GLUE or words[-1] in COMMON_GLUE:
            # Allow multi-token spans whose combined compact form is a known
            # variant (rare split abbreviation like "a w s" → compact "aws").
            # This runs AFTER COMMON_GLUE so clinical abbreviations ("BP")
            # with individual tokens in COMMON_GLUE are still blocked.
            if compact(span_text) in self._known_compacts:
                return False
            if _drop_glue(span_text) in self._known_compacts:
                return False
            return True
        return False

    # ------------------ "Already correct" guard ----

    def _already_valid(self, span_text: str) -> bool:
        return span_text in self._canonical_forms or span_text in self._terminal_aliases

    # ------------------ Scoring ---------------------

    def _best_candidate_for_span(self, span: Span) -> Optional[Candidate]:
        if self._already_valid(span.text):
            return None

        # --- Multi-word Arabic phrase detection (BEFORE clitic-stripping) ---
        # Check if the span's Arabic tokens, when transliterated WITHOUT
        # stripping clitics, form a known multi-word English medical phrase.
        # This must run BEFORE the Arabic preprocessing below because the
        # clitic-stripped transliteration (used for lexicon scoring) collapses
        # multi-word phrases into single concatenated tokens.
        #
        # E.g., بلاد شوجر → with clitic stripping: ladshwjr (1 word)
        #              → without clitic stripping: blad shwjr (2 words → match!)
        #
        # IMPORTANT: Only fire when the matched content tokens constitute a
        # large enough fraction of the span's Arabic content (>= 50%), to
        # prevent multi-word phrases from matching inside very wide spans
        # that contain many filler/context words.
        is_arabic = _has_arabic(span.text)
        if is_arabic and _HAVE_PHONETIC:
            from .flag import _translit, _is_arabic_filler
            # Get the raw Arabic tokens from the span text
            raw_arabic_tokens = _ARABIC_WORD_RE.findall(_normalize_arabic(span.text))
            n_total_arabic = len(raw_arabic_tokens)

            # Check for non-Arabic tokens (numbers, Latin) in the span.
            # If present, the multi-word phrase match would consume them,
            # which is wrong.  Only do multi-word matching when the span
            # contains ONLY Arabic tokens.
            span_tokens = list(TOKEN_RE.finditer(span.text))
            n_tokens_in_span = len(span_tokens)
            has_non_arabic = n_tokens_in_span > n_total_arabic

            if not has_non_arabic:
                # ---- Build both token lists + raw-index maps ----
                content_tokens: List[str] = []
                all_tokens: List[str] = []
                # Map from content/all-token index -> raw_arabic_tokens index
                content_to_raw: List[int] = []
                all_to_raw: List[int] = []
                for raw_idx, w in enumerate(raw_arabic_tokens):
                    if len(w) >= 2:
                        t = _translit(w, strip_clitics=False)
                        if len(t) >= 2:
                            all_tokens.append(t)
                            all_to_raw.append(raw_idx)
                            if not _is_arabic_filler(w):
                                content_tokens.append(t)
                                content_to_raw.append(raw_idx)

                n_content = len(content_tokens)

                phrase_matches = []

                # Strategy A: content-only, requires >= 70% content
                if n_content >= 2 and n_total_arabic > 0:
                    content_ratio = n_content / n_total_arabic
                    if content_ratio >= 0.70:
                        phrase_matches = match_multi_word_arabic(content_tokens)
                        if phrase_matches:
                            # Map match indices to raw_arabic_token indices
                            pm = phrase_matches[0]
                            start = pm["start"]
                            end = pm["end"]
                            if start < len(content_to_raw) and end <= len(content_to_raw):
                                pm["raw_start"] = content_to_raw[start]
                                pm["raw_end"] = content_to_raw[end - 1] + 1  # exclusive

                # Strategy B: all tokens (may include fillers like اوف → of)
                if not phrase_matches and len(all_tokens) >= 2:
                    phrase_matches = match_multi_word_arabic(all_tokens)
                    if phrase_matches:
                        pm = phrase_matches[0]
                        start = pm["start"]
                        end = pm["end"]
                        if start < len(all_to_raw) and end <= len(all_to_raw):
                            pm["raw_start"] = all_to_raw[start]
                            pm["raw_end"] = all_to_raw[end - 1] + 1  # exclusive
            else:
                phrase_matches = []

            if phrase_matches:
                top_phrase = phrase_matches[0]
                phrase_english = top_phrase["english"]
                phrase_score = top_phrase["score"]
                phrase_threshold = self.accept_threshold - 5.0
                if phrase_score >= phrase_threshold:
                    # If the multi-word match doesn't cover ALL content tokens
                    # in the span, create a sub-span that covers only the
                    # matched portion.  This prevents a wide conversational
                    # span (e.g. "وعنده بلاد شوجر و بلد برشر" = 6 tokens)
                    # from being entirely replaced by a single sub-phrase
                    # match (e.g. just "بلاد شوجر").
                    actual_span = self._narrow_span_to_matched_arabic(
                        span, top_phrase, raw_arabic_tokens
                    )
                    return Candidate(
                        span=actual_span,
                        correction=phrase_english,
                        score=phrase_score,
                        confidence=min(0.95, (phrase_score - 75.0) / 25.0),
                        entry_type="phrase",
                        issue_type="arabic_multi_word_phrase",
                        reason=f"Arabic multi-word phrase → {phrase_english!r}.",
                        features={
                            "fuzzy": phrase_score,
                            "compact": phrase_score,
                            "phonetic": phrase_score,
                            "score": phrase_score,
                        },
                    )

        # --- Arabic-to-Arabic spelling correction (BEFORE English matching) ---
        # If the span contains Arabic script, check if the non-filler Arabic
        # words are phonetic misspellings of known Arabic words (e.g. سداع→صداع,
        # طعب→تعب, انده→عنده, دغط→ضغط). This prevents false positives where
        # Arabic misspellings coincidentally match English medical terms via
        # consonant skeleton.
        #
        # Phase 2c: Uses weighted Damerau-Levenshtein with SymSpell indexing
        # (weighted_arabic_spelling.py) instead of single-substitution generation.
        # The weighted edit distance allows multi-substitution corrections within
        # the same phonetic class (e.g. سداع→صداع requires 2 substitutions but
        # both are low-cost س↔ص mergers) while rejecting unrelated changes.
        if is_arabic:
            try:
                from .flag import _is_arabic_filler, _ARABIC_FILLER

                raw_arabic_words = _ARABIC_WORD_RE.findall(_normalize_arabic(span.text))
                # Only attempt spelling correction on SINGLE-WORD spans that
                # have exactly one non-filler Arabic word. Multi-word spans
                # (e.g. "بلاد شوجر") go through the phrase matcher above.
                # NOTE: We do NOT filter through _is_arabic_filler here.
                # While that function correctly identifies normal Arabic words,
                # it also blocks SHORT Arabic misspellings (سداع, طعم, التهب)
                # whose consonant skeletons are too short to match any lexicon
                # term. Both the weighted corrector (vocabulary membership + edit
                # distance) and the legacy corrector (phonetic-substitution + vocab)
                # already have internal guards against false positives on clean
                # words — the _is_arabic_filler pre-filter is redundant and harmful.
                if len(raw_arabic_words) == 1:
                    target = raw_arabic_words[0]

                    # Phase 2c: Use weighted SymSpell corrector (with expanded vocabulary)
                    # as the primary Arabic spelling corrector. Falls back to the legacy
                    # single-substitution generator when the weighted corrector is
                    # unavailable or returns no correction.
                    #
                    # The weighted corrector uses:
                    #   - SymSpell index for edit-distance-1 lookup
                    #   - Weighted Damerau-Levenshtein with Gulf merger costs for scoring
                    #   - Expanded vocabulary (filler set + eval clean words) to prevent
                    #     false positives on clean Arabic words
                    correction = None
                    try:
                        wac = _get_weighted_arabic_corrector()
                        if wac is not None:
                            weighted_result = wac.correct(target, threshold=0.60)
                            if weighted_result is not None:
                                correction = weighted_result
                    except Exception:
                        pass

                    # Fallback: legacy single-substitution corrector
                    if correction is None:
                        try:
                            from .arabic_spelling import correct_arabic_spelling
                            legacy_result = correct_arabic_spelling(target, set(_ARABIC_FILLER))
                            if legacy_result is not None:
                                correction = legacy_result
                        except ImportError:
                            pass

                    # --- Morphological variant guard: if the ORIGINAL word
                    # is a valid morphological variant of a vocabulary word,
                    # do NOT correct it. Common Arabic verb suffixes include
                    # first-person singular (ت), plural (وا), feminine (ي), etc.
                    # E.g., وصفت is a valid first-person past verb form of
                    # وصف (which IS in _ARABIC_FILLER). Changing وصفت→وصف
                    # corrupts the grammatical person for zero clinical benefit.
                    #
                    # The guard: if stripping a common verb suffix from the
                    # original word produces a word in the vocabulary, the
                    # original is a valid morphological variant.
                    _verb_suffixes = ["\u062a", "\u0648\u0627", "\u064a", "\u0646", "\u0627", "\u0648"]
                    for _suffix in _verb_suffixes:
                        if target.endswith(_suffix) and len(target) > len(_suffix) + 2:
                            _stem = target[:-len(_suffix)]
                            if _stem in _ARABIC_FILLER:
                                correction = None
                                break

                    if correction is not None:
                        corrected_word, conf = correction
                        score_100 = conf * 100.0
                        # For Arabic spelling corrections, use a slightly lower
                        # threshold than the English accept_threshold because
                        # the scoring is more conservative (phonetic-class
                        # substitutions only, not arbitrary edits).
                        arabic_spell_threshold = 65.0
                        if score_100 >= arabic_spell_threshold and corrected_word != target:
                            # Return a Candidate that replaces the span
                            # with the correctly-spelled Arabic word.
                            span_text = span.text
                            corrected_span = span_text.replace(target, corrected_word, 1)
                            if corrected_span != span_text:
                                return Candidate(
                                    span=span,
                                    correction=corrected_span,
                                    score=score_100,
                                    confidence=min(0.95, conf),
                                    entry_type="arabic_spelling",
                                    issue_type="arabic_spelling",
                                    reason=f"Arabic spelling correction: {target!r} → {corrected_word!r}.",
                                    features={
                                        "fuzzy": score_100,
                                        "compact": score_100,
                                        "phonetic": score_100,
                                        "score": score_100,
                                    },
                                )
            except ImportError:
                pass

        # --- Arabic preprocessing for lexicon scoring ---
        scoring_text = span.text
        if is_arabic:
            scoring_text = _transliterate_arabic(span.text)
            if not scoring_text or len(scoring_text) < 5:
                return None

        words = token_words(scoring_text)
        n_words = len(words)
        if n_words == 0:
            return None
        if not compact(scoring_text):
            return None

        # --- Standard lexicon scoring ---
        # Primary: brute-force scan all lexicon variants (proven from Phase 1).
        # Supplementary: vector lexicon (FAISS n-gram index) — queried after
        # the brute-force scan to rescue cases where skeleton/fuzzy matching
        # missed a good candidate. The vector lexicon uses multi-view n-gram
        # similarity which can match Arabic→English via shared consonant
        # trigrams even when skeleton comparison is poor.
        #
        # Why not replace the brute-force scan entirely? Because the n-gram
        # index can rank the correct match BELOW unrelated candidates for
        # short Arabic transliterations (e.g. 'هستوري' → 'hstwry' ranks
        # 'chest' at 0.447 vs 'history' at 0.353 — both above the 0.35
        # threshold, but a top-1 selection would pick the wrong term).
        # The vector lexicon augmentation thus runs AFTER the brute-force
        # scan and only overrides when it finds a HIGHER-scoring candidate.
        best: Optional[Candidate] = None
        best_features: Dict[str, float] = {}
        top_two_scores: List[float] = []  # track top-2 scores for score_gap
        for entry in self.lexicon:
            for variant in entry.variants:
                if is_arabic:
                    feats = _arabic_score_pair(scoring_text, variant)
                else:
                    feats = _score_pair(scoring_text, variant)
                score_val = feats["score"]
                # Track top-2 scores for score_gap computation
                top_two_scores.append(score_val)
                top_two_scores = sorted(top_two_scores, reverse=True)[:2]
                if best is None or score_val > best.score:
                    best = Candidate(
                        span=span,
                        correction=entry.term,
                        score=score_val,
                        confidence=0.0,
                        entry_type=entry.type,
                        issue_type="",
                        reason="",
                        features=feats,
                    )
                    best_features = feats

        # Store score_gap in best_features for calibration
        if len(top_two_scores) >= 2:
            best_features["score_gap"] = top_two_scores[0] - top_two_scores[1]
        else:
            best_features["score_gap"] = 0.0

        # --- Supplementary: Vector lexicon rescue pass ---
        # If the brute-force scan found a candidate that is BELOW the certainty
        # floor (score < 50), try the vector lexicon as a rescue. The n-gram
        # index can catch cases where the skeleton/fuzzy matchers found a weak
        # coincidental match but the vector lexicon finds a genuinely better one
        # via shared consonant trigrams across scripts.
        #
        # IMPORTANT: The certainty floor (50) is calibrated to be LOW enough to
        # catch genuine rescues but HIGH enough to prevent the vector lexicon
        # from overriding the brute-force's good matches (which are typically
        # above 70). Below 50, the brute-force candidate is essentially noise
        # and any vector lexicon match >= 60 (accept_threshold * 0.75) is a
        # better candidate worth considering.
        #
        # If the brute-force found NO candidate at all (best is None), also
        # try the rescue pass — though this is rare since the brute-force scan
        # always picks the highest-scoring variant (even at score 0).
        if best is None or best.score < 50.0:
            vector_best = self._try_vector_lexicon(span, is_arabic)
            if vector_best is not None:
                best = vector_best
                best_features = {
                    "fuzzy": float(vector_best.features.get("fuzzy", vector_best.score)),
                    "compact": float(vector_best.features.get("compact", vector_best.score)),
                    "phonetic": float(vector_best.features.get("phonetic", vector_best.score)),
                    "score": float(vector_best.features.get("score", vector_best.score)),
                }

        # --- Safety gate: wide mixed-script span -> short English term ---
        # When a span contains Arabic + Latin tokens and the scoring drops
        # the Arabic words (via compact/glueless), a wide span can achieve
        # artificially high scores by matching only the Latin portion.
        # E.g. "و digoxine بشكل يومي" (4 tokens) compacts to "digoxine" and
        # scores 88+ against "digoxin", consuming all Arabic context.
        #
        # The gate: if the span has many more RAW tokens than the variant
        # has Latin words, AND the variant is a single short term, then
        # the matching is almost certainly consuming Arabic context.
        # Penalize proportionally to how much extra content is being consumed.
        if not is_arabic:
            pass  # Only applies to mixed-script or Arabic spans
        elif best is not None and " " in span.text:
            span_raw_tokens = list(TOKEN_RE.finditer(span.text))
            n_span_raw = len(span_raw_tokens)
            n_variant_words = len(token_words(best.correction))
            if n_span_raw >= 3 and n_variant_words <= 2:
                # Heavily penalize: the score is inflated by dropped content
                ratio = n_variant_words / max(1, n_span_raw)
                penalty_mult = 1.0 - ratio * 0.5
                best.score *= penalty_mult
                best.features["score"] = best.score

        if best is None:
            return None

        # For Arabic spans: use the default threshold, but do NOT apply the
        # strong-phonetic relaxation (which lowers the floor for English sound-
        # alike matches via Metaphone). The relaxation was calibrated for
        # English spelling errors; the Arabic scoring signals (skeleton vs
        # fuzzy) mean different things for cross-script matching.
        if is_arabic:
            threshold = self.accept_threshold
        else:
            threshold = self.accept_threshold
            has_char_evidence = (
                best_features.get("fuzzy", 0.0) >= 70.0
                or best_features.get("compact", 0.0) >= 70.0
            )
            scoring_len = len(compact(span.text))
            s_len = scoring_len
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

        # For Arabic spans, compare against the transliterated form too.
        if is_arabic:
            if best.correction.lower() == scoring_text.lower():
                return None
        elif best.correction == span.text:
            return None

        # Reject pure substring corrections unless fuzzy is strong.
        if (
            best.features["fuzzy"] < 92
            and normalize_text(best.correction) in normalize_text(scoring_text)
            and normalize_text(best.correction) != normalize_text(scoring_text)
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

    def _predict_correction_confidence(self, c: Candidate) -> float:
        """Predict P(correction is correct) using the calibrated confidence model.

        Extracts 10 features from the Candidate and passes them through the
        logistic regression model. Returns a 0-1 calibrated probability.

        Falls back to the Candidate's existing confidence score if the
        model is unavailable.

        NOTE: When you re-run the calibration (recommended after any changes
        to the feature extraction), the model coefficients will shift to match
        the updated feature distribution. Run:
            python -m scripts.calibrate_confidence --report-name phase3_recal
        """
        model = _get_confidence_model()
        if model is None:
            return c.confidence

        try:
            import re as _re
            import numpy as _np

            span_text = c.span.text
            feats = c.features
            has_arabic = bool(_re.search(r"[\u0600-\u06FF]", span_text))
            has_latin = bool(_re.search(r"[a-zA-Z]", span_text))

            # Extract all 10 features from the Candidate, with fallbacks
            phonetic_score_norm = feats.get("score", 0.0) / 100.0
            score_gap_norm = max(0.0, feats.get("score_gap", 0.0)) / 100.0
            llm_confidence = 0.0  # not available in rule-based path
            n_candidates_norm = min(1.0, 1.0 / 10.0)  # kept simple
            is_mixed_script = 1.0 if (has_arabic and has_latin) else 0.0
            span_length_norm = min(1.0, len(span_text) / 50.0)
            is_arabic = 1.0 if has_arabic else 0.0
            # Use IPA score from features as best_retrieval_score proxy
            best_retrieval_score = feats.get("ipa", 0.0) / 100.0
            n_tokens_norm = min(1.0, len(span_text.split()) / 6.0)
            is_multi_word = 1.0 if len(span_text.split()) > 1 else 0.0

            features = _np.array([
                phonetic_score_norm,
                score_gap_norm,
                llm_confidence,
                n_candidates_norm,
                is_mixed_script,
                span_length_norm,
                is_arabic,
                best_retrieval_score,
                n_tokens_norm,
                is_multi_word,
            ], dtype=_np.float32).reshape(1, -1)

            prob = float(model.predict_proba(features)[0, 1])
            return max(0.0, min(1.0, prob))
        except Exception:
            return c.confidence

    # ------------------ Selection -------------------

    def _select_non_overlapping(self, candidates: Sequence[Candidate]) -> List[Candidate]:
        """Pick non-overlapping winners.

        Prefer longer high-score spans. If a longer candidate fully contains
        a shorter one and scores within `dominate_margin` of it, the longer
        one wins. Stops "mohamad bin Rashid" from being reduced to "Rashid".

        Multi-word phrase matches get a wider margin so they can dominate
        individual token false positives that happen to match lexicon terms
        at high scores (e.g. بريث → rabeprazole via short skeleton).
        """
        base_margin = 6.0
        # Wider margin for multi-word phrase matches so they can dominate
        # individual token false positives within their span.
        phrase_margin = 20.0

        def _sort_key(c: Candidate) -> Tuple[float, int, int]:
            """Sort by: (adjusted_score, length, token_start).
            Multi-word phrase matches get a score boost so they sort higher
            and can dominate individual token matches within their span.
            """
            score = c.score
            if c.issue_type == "arabic_multi_word_phrase":
                score += phrase_margin * 0.5  # +10 boost for sorting
            return (score, c.span.token_end - c.span.token_start, -c.span.token_start)

        ordered = sorted(candidates, key=_sort_key, reverse=True)
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
                ):
                    # Use wider margin if the longer candidate is a phrase match
                    margin = phrase_margin if other.issue_type == "arabic_multi_word_phrase" else base_margin
                    if other.score >= c.score - margin:
                        other_ids = set(range(other.span.token_start, other.span.token_end))
                        if not (other_ids & occupied):
                            dominated = True
                            break
            if dominated:
                continue
            selected.append(c)
            occupied |= ids
        return sorted(selected, key=lambda c: c.span.start)

    # ------------------ Multi-word sub-span narrowing --------

    def _narrow_span_to_matched_arabic(
        self,
        span: Span,
        phrase_match: Dict[str, Any],
        raw_arabic_tokens: List[str],
    ) -> Span:
        """If a multi-word phrase match covers only a portion of the span's
        Arabic tokens (not all of them), create a narrowed Span that covers
        just the matched tokens in the original transcript.

        When the match covers the extremal raw tokens (first + last), the
        original span is returned unchanged.

        NOTE: The phrase_match dict MUST contain pre-computed ``raw_start``
        and ``raw_end`` keys that map to inclusive/exclusive indices into
        ``raw_arabic_tokens``.  These are computed in
        ``_best_candidate_for_span`` where the matching strategy (content
        vs all) is known.  Using pre-computed indices avoids the ambiguous
        mapping that occurs when Strategy B (all_tokens) indices are
        wrongly interpreted against content_to_raw.
        """
        raw_start = phrase_match.get("raw_start")
        raw_end = phrase_match.get("raw_end")
        if raw_start is None or raw_end is None:
            # No pre-computed mapping — fall back to original span
            return span

        first_raw = raw_start
        last_raw = raw_end - 1  # raw_end is exclusive

        # Only skip narrowing if the matched raw tokens cover the FIRST and
        # LAST raw Arabic tokens in the span (including filler words).
        # If there are filler words outside the matched range (e.g.
        # "وعنده بلاد شوجر و" where only "بلاد شوجر" matched), we MUST
        # narrow or the fillers get consumed by the replacement.
        if first_raw == 0 and last_raw == len(raw_arabic_tokens) - 1:
            return span

        # Find character positions of these raw Arabic words in span.text
        arabic_word_positions = list(_ARABIC_WORD_RE.finditer(span.text))
        if first_raw >= len(arabic_word_positions) or last_raw >= len(arabic_word_positions):
            return span

        sub_ch_start = arabic_word_positions[first_raw].start()
        sub_ch_end = arabic_word_positions[last_raw].end()

        # Remap token_start/token_end using TOKEN_RE on span.text
        tokens_in_span = list(TOKEN_RE.finditer(span.text))
        sub_token_start_offset: Optional[int] = None
        sub_token_end_offset: int = len(tokens_in_span)
        for ti, t in enumerate(tokens_in_span):
            if sub_token_start_offset is None and t.start() >= sub_ch_start:
                sub_token_start_offset = ti
            if t.start() < sub_ch_end:
                sub_token_end_offset = ti + 1

        if sub_token_start_offset is None:
            return span

        return Span(
            text=span.text[sub_ch_start:sub_ch_end],
            start=span.start + sub_ch_start,
            end=span.start + sub_ch_end,
            token_start=span.token_start + sub_token_start_offset,
            token_end=span.token_start + sub_token_end_offset,
        )

    def _try_vector_lexicon(self, span: Span, is_arabic: bool) -> Optional[Candidate]:
        """Try to find a match via vector lexicon (multi-view n-gram).

        Returns a Candidate if a good match is found, or None.
        """
        try:
            vlex = _get_vector_lexicon()
            if not vlex._built:
                return None

            results = vlex.query(span.text, top_k=3, threshold=vlex.similarity_threshold)
            if not results:
                return None

            top = results[0]
            score = top["score"] * 100.0  # convert 0-1 to 0-100 scale

            if score < self.accept_threshold * 0.75:  # lower bar for vector
                return None

            # Build translit for comparison
            if is_arabic:
                scoring_text = _transliterate_arabic(span.text)
            else:
                scoring_text = span.text

            if top["term"].lower() == scoring_text.lower():
                return None

            conf = min(0.95, score / 100.0)
            return Candidate(
                span=span,
                correction=top["term"],
                score=score,
                confidence=conf,
                entry_type=top.get("term_type", "vector"),
                issue_type="vector_lexicon",
                reason=f"Vector lexicon match: {top['term']!r} @ {score:.1f}.",
                features={"fuzzy": score, "compact": score, "phonetic": score, "score": score},
            )
        except Exception as exc:
            import logging
            logging.getLogger(__name__).debug("Vector lexicon lookup failed: %s", exc)
            return None

    @staticmethod
    def _vector_lexicon_available() -> bool:
        """Check if vector lexicon is available (built)."""
        try:
            vlex = _get_vector_lexicon()
            return vlex._built and len(vlex._entries) > 0
        except Exception:
            return False

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
