"""pipeline/stage2_llm.py — LLM detection and candidate-selection passes.

Owns:
  - LLM detection pass: ask the LLM to flag medical terms the phonetic pass
    missed (novel drug names, English mishearings, split-token mangles).
  - LLM selection: when phonetic score is borderline (0.55–0.84), ask the
    LLM to pick among the top-5 candidates using full transcript context.
  - Raw LLM call with retry.
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional

from ..llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)
from .lexicon import load_medical_lexicon
from .stage1_phonetic import phonetic_candidates

# ---------------------------------------------------------------------------
# System prompts
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


# ---------------------------------------------------------------------------
# LLM call infrastructure
# ---------------------------------------------------------------------------

def _extract_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    m = re.search(r'"flags"\s*:\s*\[.*?\]', text, re.S)
    if m:
        try:
            return json.loads("{" + m.group(0) + "}")
        except json.JSONDecodeError:
            pass
    return None


def _call_llm(payload: Dict[str, Any], timeout: float) -> Optional[Dict[str, Any]]:
    """Make an LLM API call with one retry on failure."""
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
                print(f"[pipeline/stage2] LLM attempt {attempt + 1} failed: {exc!r}, retrying...")
    print(f"[pipeline/stage2] LLM pass failed after retries: {last_error!r}")
    return None


# ---------------------------------------------------------------------------
# Stage 2a: LLM selection (borderline phonetic scores)
# ---------------------------------------------------------------------------

def llm_select_candidate(
    transcript: str,
    span_text: str,
    candidates: List[Dict[str, Any]],
    timeout: float = 20.0,
) -> Optional[str]:
    """Ask the LLM to pick the best candidate from the phonetic top-5.

    Used when phonetic score is in 0.55–0.84: good enough to retrieve the
    right drug but below the auto-correction threshold. The LLM uses full
    transcript context. Answer is constrained to the candidate list.
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


# ---------------------------------------------------------------------------
# Stage 2b: LLM detection (novel/missed terms)
# ---------------------------------------------------------------------------

def llm_pass(
    transcript: str,
    *,
    phonetic_flags: Optional[List[Dict[str, Any]]] = None,
    timeout: float = 60.0,
) -> List[Dict[str, Any]]:
    """Stage 2: ask the LLM to flag medical terms the phonetic pass missed.

    `phonetic_flags` is passed as context so the LLM knows what was already
    caught and can focus on novel/missed cases instead of re-flagging known
    spans.
    """
    words = re.split(r"\s+", transcript.strip())
    tokens_with_indices = [[i, w] for i, w in enumerate(words)]

    already_flagged_indices: set = set()
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
        if idx in already_flagged_indices and (not likely or conf < 0.50):
            continue
        filtered.append(f)
    return filtered
