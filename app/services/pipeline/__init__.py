"""app/services/pipeline — Staged Gulf Arabic medical ASR correction pipeline.

Public API (mirrors flag.py so existing call sites can switch cleanly):

    from app.services.pipeline import (
        flag_suspicious,
        apply_high_confidence_corrections,
        apply_taught_aliases,
        load_medical_lexicon,
        add_retrieval_term,
        record_taught_aliases,
        invalidate_lexicon_cache,
    )

Stage layout
------------
  stage1_phonetic.py  — phonetic flagging (consonant-skeleton + edit distance)
  stage2_llm.py       — LLM detection (novel terms) + LLM selection (borderline)
  stage3_apply.py     — correction application + HITL escalation
  lexicon.py          — medical_terms.txt + HITL alias management
  arabic.py           — transliteration, skeleton, filler detection
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .lexicon import (
    add_retrieval_term,
    apply_taught_aliases,
    invalidate_lexicon_cache,
    load_medical_lexicon,
    record_taught_aliases,
)
from .stage1_phonetic import phonetic_candidates, phonetic_pass
from .stage2_llm import llm_pass, llm_select_candidate
from .stage3_apply import apply_high_confidence_corrections


def flag_suspicious(
    transcript: str, use_llm: bool = True
) -> List[Dict[str, Any]]:
    """Run Stage 1 + Stage 2 and return merged, deduplicated flag records.

    Each record contains:
      index        — zero-based word index of the first flagged token
      word         — the flagged span text (may be multi-word for n-grams)
      reason       — "phonetic_near_medical" | "phonetic_near_medical_2gram" | ...
      candidates   — ranked list of {term, phonetic_similarity} dicts
      [span_indices] — for n-gram flags, the indices of all consumed words
      [llm_reason]   — Stage 2 LLM annotation
      [llm_likely_term] — LLM's suggested canonical term
      [llm_confidence]  — LLM confidence score

    Args:
        transcript: Raw (or drug-normalised) transcript text.
        use_llm:    Whether to run the Stage 2 LLM detection pass.
                    Set False for deterministic / offline operation.
    """
    # Stage 1: deterministic phonetic flagging.
    phon = phonetic_pass(transcript)
    phon_by_idx = {f["index"]: f for f in phon}

    if not use_llm:
        return sorted(phon_by_idx.values(), key=lambda f: f["index"])

    # Build the set of ALL word indices already covered by Stage 1 flags
    # (including n-gram spans) so Stage 2 doesn't re-flag them.
    covered_indices: set = set()
    for f in phon:
        spans = f.get("span_indices") or [f["index"]]
        covered_indices.update(spans)

    # Stage 2: LLM detection for novel/missed terms.
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
            # Guard: don't let the LLM echo back the phonetic top candidate
            # when phonetic similarity was already low — that's just noise.
            top_phonetic = (existing.get("candidates") or [None])[0]
            phonetic_term = top_phonetic["term"].lower() if top_phonetic else ""
            phonetic_sim = float(top_phonetic["phonetic_similarity"]) if top_phonetic else 0.0
            if (
                likely
                and likely.lower() == phonetic_term
                and phonetic_sim < 0.85
            ):
                existing["llm_reason"] = "llm_rejected_weak_phonetic"
                continue
            existing["llm_reason"] = entry.get("reason") or existing["reason"]
            if likely:
                existing["llm_likely_term"] = likely
            existing["llm_confidence"] = max(existing.get("llm_confidence", 0.0), llm_conf)
        else:
            # Guard: skip LLM flags that land inside an existing n-gram span.
            if idx in covered_indices:
                continue

            word = entry.get("word") or ""
            cands = phonetic_candidates(word, load_medical_lexicon())
            if not cands and likely:
                cands = [{"term": likely, "phonetic_similarity": round(llm_conf, 3)}]
            entry_data: Dict[str, Any] = {
                "index": idx,
                "word": word,
                "reason": entry.get("reason") or "llm_flag",
                "candidates": cands,
                "llm_reason": entry.get("reason"),
                "llm_likely_term": likely,
                "llm_confidence": llm_conf,
            }
            span_text = entry.get("word", "")
            span_tokens = span_text.split()
            if len(span_tokens) > 1:
                entry_data["span_indices"] = list(range(idx, idx + len(span_tokens)))
            phon_by_idx[idx] = entry_data

    return sorted(phon_by_idx.values(), key=lambda f: f["index"])


__all__ = [
    "flag_suspicious",
    "apply_high_confidence_corrections",
    "apply_taught_aliases",
    "load_medical_lexicon",
    "add_retrieval_term",
    "record_taught_aliases",
    "invalidate_lexicon_cache",
    "phonetic_pass",
    "llm_pass",
]
