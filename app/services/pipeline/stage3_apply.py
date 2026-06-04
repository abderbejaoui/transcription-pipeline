"""pipeline/stage3_apply.py — Correction application and HITL escalation.

Owns:
  - apply_high_confidence_corrections(): rewrites the transcript using
    high-confidence phonetic matches, LLM-selected candidates, and
    LLM detection fallback; escalates borderline spans to HITL.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .lexicon import load_medical_lexicon
from .stage2_llm import llm_select_candidate

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
    """Stage 3: rewrite the transcript with high-confidence corrections.

    Sources of corrections, in priority order:

    1. PHONETIC TOP-1 (strong): top phonetic candidate ≥ phonetic_strong_threshold
       (0.85). Applied automatically.

    2. LLM SELECTION (borderline): phonetic score in
       [phonetic_select_threshold, phonetic_strong_threshold) and use_llm=True.
       The LLM picks among top-5 candidates using full transcript context.
       Answer is constrained to the candidate list — no hallucination risk.

    3. LLM DETECTION fallback: llm_likely_term from the Stage 2 detection
       pass, used when phonetic is too weak to auto-apply.

    4. HITL escalation (include_hitl=True): flagged spans that couldn't be
       auto-corrected are marked for human review.

    Conjunction preservation: if the original span starts with 'و' cliticised
    onto a drug mangle (e.g. 'وسيمفاستاتن'), the 'و' is prepended to the
    Latin correction so it isn't dropped.

    Returns:
        {
          "corrected_transcript": str,
          "applied": [{"index", "span_indices", "original", "corrected",
                        "confidence", "source"}, ...],
          "threshold": float,
        }
    """
    lexicon_lower = {t.lower() for t in load_medical_lexicon()}
    tokens = re.split(r"(\s+)", transcript)
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

        span_word = f.get("word", "")
        first_token = span_word.split()[0] if span_word else ""
        waw_prefix = first_token.startswith(_AR_WAW) and len(first_token) > 1

        chosen: Optional[str] = None
        source: Optional[str] = None
        chosen_conf = 0.0

        # 1. Strong phonetic match → auto-apply.
        if top and top_sim >= phonetic_strong_threshold:
            chosen = top["term"]
            chosen_conf = top_sim
            source = "phonetic"

        # 2. Borderline phonetic → LLM selection.
        if chosen is None and use_llm and cands and phonetic_select_threshold <= top_sim < phonetic_strong_threshold:
            selected = llm_select_candidate(transcript, span_word, cands)
            if selected and selected.lower() in lexicon_lower:
                chosen = selected
                chosen_conf = top_sim
                source = "llm_select"

        # 3. LLM detection fallback.
        if chosen is None and llm_conf >= confidence_threshold and llm_term:
            if llm_term.lower() in lexicon_lower:
                chosen = llm_term
                chosen_conf = llm_conf
                source = "llm"

        if not chosen:
            continue

        if waw_prefix and not chosen.startswith(_AR_WAW):
            chosen = _AR_WAW + chosen

        spans = f.get("span_indices") or [idx]
        first = spans[0]
        original_parts = []
        for off in spans:
            if 0 <= off < len(word_to_tok):
                original_parts.append(tokens[word_to_tok[off]])
        ti_first = word_to_tok[first]
        tokens[ti_first] = chosen
        for off in spans[1:]:
            if 0 <= off < len(word_to_tok):
                tw_idx = word_to_tok[off]
                tokens[tw_idx] = ""
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

    # 4. HITL escalation.
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

    out = "".join(tokens)
    out = re.sub(r"\s+", " ", out).strip()
    return {
        "corrected_transcript": out,
        "applied": applied,
        "threshold": confidence_threshold,
    }
