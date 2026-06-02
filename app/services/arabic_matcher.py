"""Arabic-English medical term matcher — multi-strategy correction pipeline.

Stages
------
1. Arabic → Latin transliteration + consonant skeleton matching (deterministic)
   (Primary path lives in correction.py's _score_pair / _best_candidate_for_span;
    this module provides a supplementary SkeletonMatcher for direct use.)

2. LaBSE / multilingual embedding similarity (requires sentence-transformers).
   Embeds both the suspicious span and all lexicon terms into a shared
   multilingual vector space and returns the closest neighbours by cosine
   similarity.  Gracefully degrades when sentence-transformers is unavailable.

3. LLM Open Correction — for terms the lexicon doesn't cover, ask the LLM
   to suggest the intended medical term directly.  Output is constrained to
   either a recognised medical term or "UNSURE".

Usage
-----
    matcher = HybridMatcher(lexicon_entries)
    candidates = matcher.match("هستوري")
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 1 — Deterministic transliteration + consonant skeleton matching
# ---------------------------------------------------------------------------

class SkeletonMatcher:
    """Supplementary Arabic→English matcher using transliteration + consonant
    skeleton comparison.

    This duplicates some logic from correction.py's _score_pair but is
    optimised for direct Arabic→English matching with a low acceptance
    threshold (floored at 0.45) so it can serve as a "loose" first pass
    before the embedding and LLM stages tighten the results.
    """

    def __init__(self, lexicon: Sequence[Dict[str, Any]]):
        # Build a flat variant index: (translit, skeleton, term)
        # The skeleton is computed using the LATIN consonant skeleton
        # (drops vowels, maps p→b, v→f, c→k, etc.) for comparison
        # against Arabic consonant skeletons later in match().
        self._variants: List[Tuple[str, str, str]] = []  # (latin_form, skeleton, term)
        self._canonical_set: set = set()
        for entry in lexicon:
            term = entry.get("term", "")
            if not term:
                continue
            self._canonical_set.add(term.lower())
            for variant in [term] + entry.get("aliases", []):
                if not variant:
                    continue
                lat = variant.lower()
                # Skip very short abbreviation aliases (e.g. 'sle', 'aml',
                # 'gpa', 'asa').  Their 2-3 char consonant skeletons match the
                # backbone of normal Arabic words by pure coincidence
                # (السلام→sle, العملية→aml, الكبد→gpa), producing high-confidence
                # garbage corrections.  Mirrors main._build_corrector's alias
                # stripping for the Stage-1 corrector.
                if len(re.sub(r"[^a-z0-9]", "", lat)) <= 3:
                    continue
                sk = self._latin_skeleton(lat)
                self._variants.append((lat, sk, term))

    @staticmethod
    def _latin_skeleton(s: str) -> str:
        """Latin consonant skeleton, matching flag.py's _consonant_skeleton_latin.

        Strips vowels, maps phonetic classes that Arabic transliteration
        loses: p→b, v→f, c→k, g→k, q→k, x→ks.
        'paracetamol' → 'brktml'
        'history' → 'hstr'
        """
        from .flag import _consonant_skeleton_latin
        return _consonant_skeleton_latin(s)

    @staticmethod
    def _arabic_skeleton(s: str) -> str:
        """Arabic consonant skeleton, matching flag.py's _consonant_skeleton_ar.

        Strips vowels AND 'h', 'w', 'y' from an already-transliterated
        Arabic word. Arabic doesn't write short vowels — 'h' and 'w' are
        often silent carriers that shouldn't count as consonants.
        'hstwry' → 'str'
        'dyabts' → 'dyabts' (no dropped chars)
        'bryth' → 'brt'
        """
        from .flag import _consonant_skeleton_ar
        return _consonant_skeleton_ar(s)

    def _transliterate(self, text: str) -> str:
        """Convert Arabic text to Latin using flag.py if available."""
        try:
            from .flag import _translit  # type: ignore
            return _translit(text, strip_clitics=True)
        except ImportError:
            return ""

    def match(self, span_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Return up to `top_k` candidates from the lexicon, scored 0-100."""
        # Detect Arabic script
        has_arabic = bool(re.search(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", span_text))
        if not has_arabic:
            return []

        # Pre-filter: skip common Arabic filler words that have no business
        # being matched against English medical terms. Short filler words
        # like 'لاحظنا', 'يمتد', 'بينت' have short consonant skeletons that
        # coincidentally match substrings of English medical terms.
        try:
            from .flag import _is_arabic_filler
            if _is_arabic_filler(span_text):
                return []
        except ImportError:
            pass

        translit = self._transliterate(span_text)
        if not translit or len(translit) < 3:
            return []

        # Compute Arabic consonant skeleton — this aggressively drops 'h',
        # 'w', 'y' (silent carriers in Arabic transliteration) so that
        # only the true consonant backbone remains. CRITICALLY different
        # from the Latin skeleton: without this, هستوري → 'hstwry' (Arabic
        # side) gets skeleton 'hstwr' (5 chars) which accidentally overlaps
        # with unrelated English terms. With the Arabic skeleton, هستوري →
        # 'str' (3 chars) which correctly matches only 'history' → 'hstr'.
        arabic_sk = self._arabic_skeleton(translit)
        if not arabic_sk or len(arabic_sk) < 2:
            return []

        candidates: List[Dict[str, Any]] = []

        for var_lat, var_sk, term in self._variants:
            # Don't return the same term as the span (no-op)
            if term.lower() == translit.lower():
                continue

            from rapidfuzz import fuzz
            # Compare transliterated string (raw signal)
            str_score = float(fuzz.ratio(translit, var_lat))

            # Compare consonant skeletons — Arabic skeleton vs Latin skeleton.
            # Using appropriate skeletons for each side (Arabic drops h/w/y,
            # Latin maps p→b, v→f, c→k, etc.) ensures a fair comparison.
            if arabic_sk and var_sk:
                sk_score = float(fuzz.ratio(arabic_sk, var_sk))
            else:
                sk_score = 0.0

            # Combined: skeleton is the primary signal (it's more robust
            # to vowel mismatches across scripts).
            combined = max(str_score, sk_score * 1.15)

            # Length-mismatch penalty: Arabic skeletons are typically 2-5
            # chars while Latin skeletons are 3-12 chars. A short Arabic
            # skeleton like 'str' (3 chars) should only match Latin
            # skeletons of similar length (e.g. 'hstr' = 4 chars).
            # Matching 'str' (3 chars) against 'dktksl' (6 chars) should
            # be heavily penalized.
            #
            # RAISED threshold from 0.55 to 0.75 + multiplier 300→1500 to
            # kill coincidental matches where a 4-char Arabic skeleton
            # (e.g. 'ksjn' for oxygen) matches a longer Latin skeleton
            # (e.g. 'ksljnz' for xeljanz alias of tofacitinib) via pure
            # subsequence coincidence — share 4/4 chars but the Latin
            # skeleton has 2 extra insertions, scoring 92.0 after 1.15x
            # boost and incorrectly beating the genuine oxygen match
            # (86.25).  The tighter threshold forces the len_ratio to
            # be >= 0.75 (i.e. skeletons within 25% of each other in
            # length) before the penalty is waived.
            if arabic_sk and var_sk:
                len_ratio = len(arabic_sk) / max(1, len(var_sk))
                if len_ratio < 0.75:
                    # Arabic skeleton is significantly shorter than Latin —
                    # the match may be coincidental overlap.
                    shortfall = 0.75 - len_ratio
                    penalty = shortfall * shortfall * 1500.0
                    combined = max(0.0, combined - penalty)
                elif len_ratio > 2.0:
                    # Arabic skeleton much longer than Latin — unlikely
                    excess = len_ratio - 2.0
                    penalty = excess * excess * 100.0
                    combined = max(0.0, combined - penalty)

            # Short-skeleton floor: Arabic skeletons <= 3 chars cannot
            # independently reach the threshold via skeleton alone.
            if len(arabic_sk) <= 3 and combined >= 85.0:
                # Require some raw string corroboration for super-short
                # skeletons to prevent 'str' (3 chars) matching unrelated
                # terms like 'statin' (Latin skel 'sttn', 4 chars).
                if str_score < 55.0:
                    combined = min(combined, 84.9)

            # Acceptance gate — accept only when the evidence is strong enough
            # to be a real transliteration, not a consonant-backbone
            # coincidence on a normal Arabic word.  Two ways to qualify:
            #   (a) a NEAR-PERFECT skeleton match with raw corroboration, or
            #   (b) strong raw string overlap on its own.
            # Genuine transliterations clear (a): history (sk100/str77),
            # diabetes (sk100/str63), hypertension (sk100/str64).
            # Coincidences are rejected because their skeletons are merely
            # similar, not identical (سيتم→asthma sk86, التحاليل→sotalol sk86,
            # المعدل→medrol sk86), and their raw overlap is weak
            # (الفحوصات→avastin str31, دكتور→dextrose str46).
            strong = (sk_score >= 90.0 and str_score >= 55.0) or (str_score >= 80.0)
            if combined >= 85.0 and strong:
                candidates.append({
                    "term": term,
                    "score": round(combined, 2),
                    "match_type": "arabic_skeleton",
                    "transliteration": translit,
                })

        candidates.sort(key=lambda c: -c["score"])
        return candidates[:top_k]


# ---------------------------------------------------------------------------
# Stage 2 — LaBSE / multilingual embedding similarity
# ---------------------------------------------------------------------------

_EMBEDDING_MODEL: Any = None


def _lazy_load_embedding_model() -> Any:
    """Load LaBSE model on first call; return None if unavailable.

    Uses a global cache so the model is loaded at most once per process
    even if multiple matchers reference it.
    """
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("[arabic_matcher] Loading LaBSE embedding model...")
        _EMBEDDING_MODEL = SentenceTransformer(
            "sentence-transformers/LaBSE",
            device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        )
        logger.info("[arabic_matcher] LaBSE model ready.")
    except Exception as exc:
        logger.warning(f"[arabic_matcher] LaBSE load failed: {exc!r}")
        _EMBEDDING_MODEL = False  # sentinel for "not available"
    return _EMBEDDING_MODEL if _EMBEDDING_MODEL is not False else None


class EmbeddingMatcher:
    """Cross-lingual embedding similarity using LaBSE.

    Embeds all lexicon terms on initialisation, then compares each
    suspicious span via cosine similarity in the shared multilingual space.
    """

    def __init__(self, lexicon: Sequence[Dict[str, Any]], batch_size: int = 64):
        self._model: Any = _lazy_load_embedding_model()
        self._terms: List[str] = []
        self._embeddings: Any = None  # numpy array
        self._batch_size = batch_size

        if self._model is not None:
            self._build_index(lexicon)

    def _build_index(self, lexicon: Sequence[Dict[str, Any]]) -> None:
        """Embed all lexicon terms into a flat index."""
        import numpy as np

        seen: set = set()
        for entry in lexicon:
            term = entry.get("term", "")
            if not term or term.lower() in seen:
                continue
            seen.add(term.lower())
            self._terms.append(term)
            for alias in entry.get("aliases", []):
                al = str(alias).strip()
                if al and al.lower() not in seen:
                    seen.add(al.lower())
                    self._terms.append(al)

        if not self._terms:
            self._embeddings = np.array([], dtype=np.float32)
            return

        # Encode in batches to avoid OOM for large lexicons
        all_embs: List[Any] = []
        for i in range(0, len(self._terms), self._batch_size):
            batch = self._terms[i : i + self._batch_size]
            embs = self._model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embs.append(embs)
        self._embeddings = np.vstack(all_embs) if len(all_embs) > 1 else all_embs[0]

    def match(self, span_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find closest lexicon terms by cosine similarity."""
        import numpy as np

        if self._model is None or self._embeddings is None or not self._terms:
            return []

        span_emb = self._model.encode(
            [span_text], convert_to_numpy=True, show_progress_bar=False
        )
        # Cosine similarity
        norms = np.linalg.norm(self._embeddings, axis=1)
        span_norm = np.linalg.norm(span_emb)
        if span_norm == 0 or np.any(norms == 0):
            return []
        sims = (self._embeddings @ span_emb.T).flatten() / (norms * span_norm)

        # Top-K
        top_indices = np.argsort(-sims)[:top_k]
        results = []
        for idx in top_indices:
            if float(sims[idx]) >= 0.30:  # low threshold, this is a loose pass
                results.append({
                    "term": self._terms[idx],
                    "score": round(float(sims[idx]) * 100.0, 2),
                    "match_type": "embedding",
                })
        return results


# ---------------------------------------------------------------------------
# Stage 4 — LLM Open Correction
# ---------------------------------------------------------------------------

_LLM_SYSTEM = (
    "You are a medical term correction assistant.  You are given a "
    "suspicious word or phrase from an Arabic-English medical transcript.  "
    "Your job: suggest the MOST LIKELY correct medical term (drug, disease, "
    "procedure, anatomical term).\n\n"
    "Rules:\n"
    "1. Output STRICT JSON only: {\"correction\": \"term\" or \"UNSURE\", "
    "\"confidence\": 0.0-1.0, \"reason\": \"short explanation\"}\n"
    "2. If the word is a known drug / disease / medical term that was simply "
    "misspelled (e.g. 'amoxicilin' -> 'amoxicillin'), correct it.\n"
    "3. If the word looks like an Arabic transliteration of a medical term "
    "(e.g. 'هستوري' -> 'history', 'دايابيتس' -> 'diabetes'), suggest the "
    "English canonical term.\n"
    "4. If you are genuinely unsure, return \"UNSURE\" with confidence 0.\n"
    "5. Do NOT invent terms or guess wildly.  Better to return UNSURE than "
    "to make something up.\n"
    "6. Consider the clinical context: a word that appears near 'insulin', "
    "'glucose', 'HbA1c' is likely diabetes-related.\n"
    "7. Normal English words, Arabic filler, numbers, and correctly-spelled "
    "terms: return UNSURE.\n"
    "8. For mixed Arabic-English terms like 'بلاد شوجر' → 'blood sugar', "
    "suggest the full English medical term."
)


class LLMOpenCorrector:
    """Final-resort LLM-based correction for terms the deterministic and
    embedding matchers couldn't handle."""

    def __init__(self, confidence_threshold: float = 0.60):
        self._threshold = confidence_threshold

    def correct(
        self,
        span_text: str,
        context: str = "",
        timeout: float = 30.0,
    ) -> Optional[Dict[str, Any]]:
        """Ask the LLM to suggest a correction for one span.

        Returns None if the LLM is unavailable or returns UNSURE.
        """
        from .llm_config import (
            get_llm_headers,
            get_llm_model,
            get_llm_provider,
            get_llm_url,
            parse_chat_content,
        )

        user = json.dumps({
            "suspicious_span": span_text,
            "context": context[:500] if context else "",
        }, ensure_ascii=False)

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

        import urllib.request
        try:
            req = urllib.request.Request(
                get_llm_url(get_llm_provider()),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(get_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = parse_chat_content(data, get_llm_provider()).strip()
            # Extract JSON from the response
            if not (raw.startswith("{") and raw.endswith("}")):
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                raw = m.group(0) if m else "{}"
            result = json.loads(raw)
            correction = (result.get("correction") or "").strip()
            confidence = float(result.get("confidence", 0.0) or 0.0)
            reason = (result.get("reason") or "").strip()

            if not correction or correction.upper() == "UNSURE":
                return None
            if confidence < self._threshold:
                return None

            return {
                "term": correction,
                "score": round(confidence * 100.0, 2),
                "match_type": "llm_open",
                "confidence": confidence,
                "reason": reason,
            }
        except Exception as exc:
            logger.warning(f"[LLMOpenCorrector] LLM call failed: {exc!r}")
            return None

    def correct_batch(
        self,
        spans: List[str],
        context: str = "",
        timeout: float = 60.0,
    ) -> List[Optional[Dict[str, Any]]]:
        """Correct multiple spans in a single batched LLM call."""
        from .llm_config import (
            get_llm_headers,
            get_llm_model,
            get_llm_provider,
            get_llm_url,
            parse_chat_content,
        )

        if not spans:
            return []

        batched_system = (
            "You are a medical term correction assistant.  For EACH suspicious "
            "span in the list, suggest the correct medical term or UNSURE.\n\n"
            "Output strict JSON: {\"corrections\": [{\"index\": int, "
            "\"correction\": \"term\" or \"UNSURE\", "
            "\"confidence\": 0.0-1.0, \"reason\": \"...\"}, ...]}\n"
        )

        user = json.dumps({
            "spans": [{"index": i, "text": s} for i, s in enumerate(spans)],
            "context": context[:500] if context else "",
        }, ensure_ascii=False)

        payload = {
            "model": get_llm_model(get_llm_provider()),
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0},
            "messages": [
                {"role": "system", "content": batched_system},
                {"role": "user", "content": user},
            ],
        }

        import urllib.request
        try:
            req = urllib.request.Request(
                get_llm_url(get_llm_provider()),
                data=json.dumps(payload).encode("utf-8"),
                headers=get_llm_headers(get_llm_provider()),
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = parse_chat_content(data, get_llm_provider()).strip()
            if not (raw.startswith("{") and raw.endswith("}")):
                m = re.search(r"\{.*\}", raw, re.DOTALL)
                raw = m.group(0) if m else "{}"
            result = json.loads(raw)
            corrections_raw = result.get("corrections") or []
            corrections_map = {}
            for c in corrections_raw:
                idx = c.get("index")
                corr = (c.get("correction") or "").strip()
                conf = float(c.get("confidence", 0.0) or 0.0)
                reason = (c.get("reason") or "").strip()
                if (
                    isinstance(idx, int)
                    and 0 <= idx < len(spans)
                    and corr
                    and corr.upper() != "UNSURE"
                    and conf >= self._threshold
                ):
                    corrections_map[idx] = {
                        "term": corr,
                        "score": round(conf * 100.0, 2),
                        "match_type": "llm_open",
                        "confidence": conf,
                        "reason": reason,
                    }

            return [corrections_map.get(i) for i in range(len(spans))]
        except Exception as exc:
            logger.warning(f"[LLMOpenCorrector] batch LLM call failed: {exc!r}")
            return [None] * len(spans)


# ---------------------------------------------------------------------------
# Orchestrator — runs all stages in sequence
# ---------------------------------------------------------------------------

class HybridMatcher:
    """Orchestrates the multi-stage correction pipeline.

    Stages (in order):
      1. SkeletonMatcher — transliteration + consonant skeleton
      2. EmbeddingMatcher — LaBSE multilingual similarity (optional)
      3. LLM Open Correction — for remaining unmatched spans (optional)
    """

    def __init__(
        self,
        lexicon: Sequence[Dict[str, Any]],
        enable_embedding: bool = True,
        enable_llm_open: bool = True,
    ):
        self._skeleton = SkeletonMatcher(lexicon)
        self._embedding: Optional[EmbeddingMatcher] = (
            EmbeddingMatcher(lexicon) if enable_embedding else None
        )
        self._llm_open = LLMOpenCorrector() if enable_llm_open else None

    def match(
        self,
        span_text: str,
        top_k: int = 5,
        context: str = "",
    ) -> List[Dict[str, Any]]:
        """Run all available stages and return merged, deduplicated candidates.

        Each candidate dict: {term, score, match_type, ...}
        """
        seen_terms: set = set()
        all_candidates: List[Dict[str, Any]] = []

        # Stage 1: Deterministic skeleton matching
        for c in self._skeleton.match(span_text, top_k=top_k):
            if c["term"].lower() not in seen_terms:
                seen_terms.add(c["term"].lower())
                all_candidates.append(c)

        # Stage 2: Embedding matching
        if self._embedding is not None:
            for c in self._embedding.match(span_text, top_k=top_k):
                if c["term"].lower() not in seen_terms:
                    seen_terms.add(c["term"].lower())
                    all_candidates.append(c)

        # Sort by score descending
        all_candidates.sort(key=lambda c: -c["score"])
        return all_candidates[:top_k]

    def llm_open_correct(
        self,
        span_text: str,
        context: str = "",
    ) -> Optional[Dict[str, Any]]:
        """Final-resort LLM correction for a single span."""
        if self._llm_open is None:
            return None
        return self._llm_open.correct(span_text, context=context)

    def llm_open_correct_batch(
        self,
        spans: List[str],
        context: str = "",
    ) -> List[Optional[Dict[str, Any]]]:
        """Final-resort LLM correction for multiple spans in one call."""
        if self._llm_open is None:
            return [None] * len(spans)
        return self._llm_open.correct_batch(spans, context=context)
