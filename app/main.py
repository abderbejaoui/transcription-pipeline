"""FastAPI app — LLM-driven medical transcript correction grounded by voice.

Pipeline
--------
1. /api/transcribe
   - Save uploaded audio under data/sessions/<session_id>.<ext>
   - faster-whisper -> text + word-level timestamps + per-word probability
   - LLM DETECT (one call): identify suspicious medical-term spans by index
   - For each span:
       slice audio at timestamps -> wav2vec2 embedding -> top-K voice index hits
       attach the description metadata of each hit
   - LLM DECIDE (one batched call): pick the candidate that fits the
     patient's clinical context, or NO_CHANGE
   - Apply chosen replacements
   - Return raw_text, corrected_text, session_id, words, suspicious

2. /api/learn_from_edit
   - Word-level diff vs the cached session audio
   - For each replaced span, slice audio -> wav2vec2 -> store in voice index
     under the new canonical term (with an LLM-generated description if
     not already cached).

Endpoints
---------
GET  /                       single-page UI
GET  /api/healthz
GET  /api/lexicon
POST /api/teach              (kept for compatibility; updates lexicon only)
POST /api/transcribe         audio in -> corrected transcript out
POST /api/learn_from_edit    teach the system from a user edit
GET  /api/voices             list voice index (without embeddings)
POST /api/voices/reset       wipe voice index (testing)
"""

from __future__ import annotations

import csv
import difflib
import json
import math
import os
import re
import shutil
import threading
import urllib.request
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastapi.responses import StreamingResponse

from .services import asr, asr_dual, asr_correction_pipeline, descriptions, lexicon, llm_decide, llm_detect, suspect, tracing, voice_match
from .services.llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WHISPER_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
DEFAULT_LANGUAGE = os.environ.get("ASR_LANGUAGE", "")  # "" = auto-detect (Arabic+English)
USE_LLM = os.environ.get("USE_LLM", "1") == "1"
# When 1, transcription runs the dual-ASR (Gulf LoRA + base Qwen3) with an
# LLM judge merging the two outputs. Costs 2x GPU memory + one extra LLM call.
USE_DUAL_ASR = os.environ.get("USE_DUAL_ASR", "0") == "1"

# Audio-retrieval thresholds, calibrated for the CTC phonetic similarity
# scale (normalized Levenshtein over greedy wav2vec2-base-960h transcripts).
# Empirically: same word / different voice -> 0.55-0.85, same word / same
# voice -> 0.85-1.00, different words -> < 0.40. The LLM remains the
# final filter; we just keep candidates loose enough to feed it.
AUDIO_RETRIEVE_THRESHOLD_USER = 0.55
AUDIO_RETRIEVE_THRESHOLD_SEED = 0.45
AUDIO_AUTOFIX_THRESHOLD = 0.85  # short-circuit when very strong USER match

_LEARN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


@lru_cache(maxsize=1)
def _load_medical_dictionary() -> List[str]:
    """Load canonical medical terms + aliases (deduped, case-insensitive)."""
    terms: List[str] = []
    medical_file = PROJECT_ROOT / "medical_terms.txt"
    if medical_file.exists():
        for line in medical_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                terms.append(line)
    for entry in lexicon.list_terms():
        term = entry.get("term")
        if term:
            terms.append(term)
        for alias in entry.get("aliases", []) or []:
            if alias:
                terms.append(str(alias))

    deduped: List[str] = []
    seen: set[str] = set()
    for term in terms:
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(term)
    return deduped


def _suspects_to_spans(suspects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert suspect.detect output into span dicts used by the pipeline."""
    spans: List[Dict[str, Any]] = []
    for s in suspects:
        idx = s.get("index")
        if not isinstance(idx, int):
            continue
        prob = s.get("probability")
        prob_val = float(prob) if isinstance(prob, (int, float)) else 1.0
        spans.append(
            {
                "index_start": idx,
                "index_end": idx + 1,
                "text": s.get("text", ""),
                "start_s": float(s.get("start") or 0.0),
                "end_s": float(s.get("end") or 0.0),
                "probability_min": prob_val,
                "reason": s.get("reason", "low_confidence_english"),
                "candidates": s.get("candidates", []),
            }
        )
    return spans


@lru_cache(maxsize=1)
def _load_drug_dictionary_manager() -> asr_correction_pipeline.MedicalDictionaryManager:
    """Load drug names from cleaned JSON and uses from the CSV file."""
    names: List[str] = []
    meta: Dict[str, Dict[str, Any]] = {}
    line_map: Dict[str, int] = {}

    json_path = PROJECT_ROOT / "data" / "medicine_details_cleaned.json"
    if json_path.exists():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                seen: set[str] = set()
                for item in payload:
                    line_no = None
                    if isinstance(item, str):
                        name = item
                    elif isinstance(item, dict):
                        name = (
                            item.get("name")
                            or item.get("Medicine Name")
                            or item.get("medicine_name")
                            or item.get("drug")
                            or ""
                        )
                        line_no = item.get("line") or item.get("row")
                    else:
                        continue

                    name = str(name).strip()
                    if not name:
                        continue
                    clean = asr_correction_pipeline.sanitize_name(name)
                    if not clean or clean in seen:
                        continue
                    seen.add(clean)
                    names.append(name)
                    if line_no is not None:
                        try:
                            line_map[clean] = int(line_no)
                        except (TypeError, ValueError):
                            pass
        except Exception as exc:
            print(f"[phonetic] failed to read cleaned JSON: {exc!r}")

    csv_path = PROJECT_ROOT / "data" / "Medicine_Details.csv"
    if csv_path.exists():
        try:
            uses_by_line: Dict[int, str] = {}
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                for line_no, row in enumerate(reader, start=1):
                    uses = (row.get("Uses") or "").strip()
                    if uses:
                        uses_by_line[line_no] = uses

            if line_map:
                for clean, line_no in line_map.items():
                    uses = uses_by_line.get(line_no)
                    if uses:
                        meta.setdefault(clean, {})["uses"] = uses

            csv_names, csv_meta = asr_correction_pipeline.load_medicine_details_csv(csv_path)
            for clean, data in csv_meta.items():
                meta.setdefault(clean, data)
            if not names:
                names = csv_names
        except Exception as exc:
            print(f"[phonetic] failed to read CSV: {exc!r}")

    if not names:
        raise RuntimeError("No medical dictionary entries found")
    return asr_correction_pipeline.MedicalDictionaryManager(names, metadata=meta)


def _replace_first(text: str, old: str, new: str) -> str:
    idx = text.find(old)
    if idx < 0:
        return text
    return text[:idx] + new + text[idx + len(old):]


def _match_candidate(raw: str, candidates: Sequence[str]) -> Optional[str]:
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{") and text.endswith("}"):
        try:
            obj = json.loads(text)
            choice = obj.get("choice") or obj.get("term")
            if isinstance(choice, str):
                text = choice.strip()
        except Exception:
            pass

    text = text.splitlines()[0].strip()
    if len(text) >= 2 and text[0] in "'\"`" and text[-1] == text[0]:
        text = text[1:-1].strip()
    text = text.strip(" \t\n\r.,;:!؟")

    for c in candidates:
        if c == text:
            return c
    for c in candidates:
        if c.lower() == text.lower():
            return c
    return None


def _llm_choose_term(system: str, user: str, candidates: Sequence[str], timeout: float = 60.0) -> Optional[str]:
    provider = get_llm_provider()
    payload = {
        "model": get_llm_model(provider),
        "stream": False,
        "options": {"temperature": 0.0},
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    try:
        req = urllib.request.Request(
            get_llm_url(provider),
            data=json.dumps(payload).encode("utf-8"),
            headers=get_llm_headers(provider),
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        content = parse_chat_content(data, provider)
        return _match_candidate(content, candidates)
    except Exception as exc:
        print(f"[phonetic] LLM call failed: {exc!r}")
        return None


def _apply_span_replacements(
    tokens: Sequence[str],
    spans: Sequence[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    """Replace tokens within spans and return corrected text + applied list."""
    out_tokens = list(tokens)
    applied: List[Dict[str, Any]] = []
    for s in spans:
        chosen = s.get("chosen")
        if not chosen:
            continue
        i0 = s.get("index_start")
        i1 = s.get("index_end")
        if not isinstance(i0, int) or not isinstance(i1, int):
            continue
        if i0 < 0 or i1 <= i0 or i1 > len(out_tokens):
            continue
        original = " ".join(t for t in out_tokens[i0:i1] if t).strip()
        out_tokens[i0] = chosen
        for j in range(i0 + 1, i1):
            out_tokens[j] = ""
        applied.append(
            {"index_start": i0, "index_end": i1, "original": original, "corrected": chosen, "source": "llm"}
        )

    corrected = " ".join(t for t in out_tokens if t).strip()
    return corrected, applied


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class TeachRequest(BaseModel):
    term: str
    type: str = Field("drug")
    aliases: List[str] = Field(default_factory=list)
    priority: float = Field(1.0, ge=0.0, le=2.0)


class LearnFromEditRequest(BaseModel):
    raw_text: str
    corrected_text: str
    session_id: Optional[str] = None
    type: str = Field("drug")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Medical Voice Corrector — Demo", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.middleware("http")
async def _no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static"):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


@app.on_event("startup")
def _prewarm() -> None:
    def _bg() -> None:
        try:
            asr._load_model()
            print("[startup] Gulf Arabic ASR model ready.")
        except Exception as exc:
            print(f"[startup] ASR warmup failed: {exc}")

    threading.Thread(target=_bg, daemon=True).start()
    voice_match.warm_up()


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/healthz")
def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/lexicon")
def get_lexicon() -> Dict[str, Any]:
    entries = lexicon.list_terms()
    return {"count": len(entries), "entries": entries}


@app.get("/api/voices")
def get_voices() -> Dict[str, Any]:
    voices = voice_match.list_voices()
    return {"count": len(voices), "voices": voices}


@app.post("/api/voices/reset")
def reset_voices() -> Dict[str, Any]:
    voice_match.reset()
    return {"ok": True}


@app.post("/api/teach")
def teach(req: TeachRequest) -> Dict[str, Any]:
    entry = lexicon.add_term(
        term=req.term, type_=req.type, aliases=req.aliases, priority=req.priority
    )
    # Pre-cache description so the next /transcribe DECIDE has it.
    threading.Thread(
        target=lambda: descriptions.get_or_generate(req.term, type_hint=req.type),
        daemon=True,
    ).start()
    return {"ok": True, "entry": entry}


class CorrectRequest(BaseModel):
    text: str = Field(..., description="Raw text without audio.")


@app.post("/api/correct")
def correct_text_only(req: CorrectRequest) -> Dict[str, Any]:
    """Text-only quick path. No audio means no voice retrieval; the LLM
    DECIDE step will see only NO_CHANGE candidates per span — so this
    endpoint is mostly for showing the DETECT output. Real corrections
    require audio."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    fake_words = []
    for tok in req.text.split():
        fake_words.append({"word": " " + tok, "start": 0.0, "end": 0.0, "probability": 0.0})
    spans = []
    if fake_words:
        try:
            # Text-only mode: treat all tokens as low-confidence to surface candidates.
            dictionary = _load_medical_dictionary()
            suspects = suspect.detect(fake_words, dictionary)
            spans = _suspects_to_spans(suspects)
        except Exception as exc:
            print(f"[correct] DETECT failed: {exc!r}")
    return {
        "raw_text": req.text,
        "corrected_text": req.text,
        "suspicious": spans,
        "note": "text-only mode: voice retrieval is disabled without audio",
    }


class PhoneticCorrectRequest(BaseModel):
    text: str = Field(..., description="Raw mixed Arabic/English text.")
    confidence_threshold: float = Field(0.80, ge=0.0, le=1.0)
    top_k: int = Field(5, ge=1, le=10)


@app.post("/api/phonetic_correct")
def phonetic_correct(req: PhoneticCorrectRequest) -> Dict[str, Any]:
    """Run the phonetic+semantic post-ASR correction on typed text."""
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    tokens = [t for t in re.split(r"\s+", text) if t]
    words = [
        {"word": tok, "start": float(i), "end": float(i + 1), "probability": 1.0}
        for i, tok in enumerate(tokens)
    ]

    pipeline_steps: List[Dict[str, Any]] = [
        {"step": "Text input", "output": f"tokens={len(tokens)}"},
    ]

    try:
        manager = _load_drug_dictionary_manager()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dictionary load failed: {exc}")

    try:
        detect_spans = llm_detect.detect(words)
    except Exception as exc:
        print(f"[phonetic_correct] LLM detect failed: {exc!r}")
        detect_spans = []

    detect_list = "; ".join(s.get("text", "") for s in detect_spans) if detect_spans else "none"
    pipeline_steps.append({
        "step": "LLM detect",
        "output": f"spans={len(detect_spans)} | {detect_list}",
    })

    corrected = text
    auto_corrections: List[Dict[str, Any]] = []
    flags: List[Dict[str, Any]] = []
    choose_lines: List[str] = []
    for s in detect_spans:
        raw_text = s.get("text", "")
        raw_candidates = asr_correction_pipeline.find_top_k_candidates(
            raw_text, manager, top_k=req.top_k
        )
        candidate_terms = [str(c.get("term")) for c in raw_candidates if c.get("term")]

        candidates = []
        for c in raw_candidates:
            uses = None
            meta = c.get("meta")
            if isinstance(meta, dict):
                uses = meta.get("uses")
            candidates.append(
                {
                    "term": c.get("term"),
                    "phonetic_similarity": float(c.get("score") or 0.0),
                    "uses": uses,
                    "score": float(c.get("score") or 0.0),
                }
            )

        prompt = asr_correction_pipeline.build_llm_correction_prompt(
            text, raw_text, raw_candidates
        )
        chosen = _llm_choose_term(prompt["system"], prompt["user"], candidate_terms)
        s["chosen"] = chosen
        choose_lines.append(
            f"{raw_text} -> [{', '.join(candidate_terms)}] | chosen={chosen or 'NO_CHANGE'}"
        )

        flags.append(
            {
                "index": int(s.get("index_start", 0)),
                "word": raw_text,
                "reason": "near_medical",
                "candidates": candidates,
                "llm_prompt": prompt,
                "llm_likely_term": chosen or "",
                "llm_confidence": 1.0 if chosen else 0.0,
                "index_start": s.get("index_start"),
                "index_end": s.get("index_end"),
                "chosen": chosen,
            }
        )

    pipeline_steps.append({
        "step": "Phonetic candidates + LLM choose",
        "output": " ; ".join(choose_lines) if choose_lines else "no spans",
    })

    corrected_text, applied = _apply_span_replacements(tokens, flags)
    corrected = corrected_text or text
    for item in applied:
        auto_corrections.append(
            {"original": item["original"], "corrected": item["corrected"], "source": "llm"}
        )

    pipeline_steps.append({
        "step": "Corrected output",
        "output": corrected,
    })

    return {
        "session_id": None,
        "transcript": text,
        "corrected_transcript": corrected,
        "auto_corrections": auto_corrections,
        "flags": flags,
        "words": [],
        "audio_url": None,
        "pipeline_steps": pipeline_steps,
    }


class PhoneticCandidatesRequest(BaseModel):
    text: str = Field(..., description="Medicine name to correct.")
    top_k: int = Field(5, ge=1, le=10)


@app.post("/api/phonetic_candidates")
def phonetic_candidates(req: PhoneticCandidatesRequest) -> Dict[str, Any]:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    try:
        manager = _load_drug_dictionary_manager()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"dictionary load failed: {exc}")

    raw_candidates = asr_correction_pipeline.find_top_k_candidates(
        text, manager, top_k=req.top_k
    )

    candidates = []
    for c in raw_candidates:
        uses = None
        meta = c.get("meta")
        if isinstance(meta, dict):
            uses = meta.get("uses")
        candidates.append(
            {
                "term": c.get("term"),
                "score": float(c.get("score") or 0.0),
                "uses": uses,
            }
        )

    return {
        "query": text,
        "best": candidates[0]["term"] if candidates else "",
        "candidates": candidates,
    }


# ---------------------------------------------------------------------------
# /api/transcribe (JSON)  and  /api/transcribe_stream (NDJSON traces)
# ---------------------------------------------------------------------------


def _run_transcribe_pipeline(
    session_path: Path, session_id: str, language: Optional[str], model_size: str
) -> Dict[str, Any]:
    """The full pipeline. Calls into services that emit trace events via
    `app.services.tracing`, so this function can be run with or without an
    active Tracer."""
    # 1) ASR. If USE_DUAL_ASR=1, run both Gulf LoRA + base Qwen3 in parallel
    # and merge their outputs with an LLM judge. Otherwise just the LoRA.
    tracing.emit("asr.start", {
        "session_id": session_id,
        "audio_path": str(session_path),
        "model": "dual_asr" if USE_DUAL_ASR else model_size,
        "language": language or "auto",
        "size_bytes": session_path.stat().st_size,
    })
    if USE_DUAL_ASR:
        asr_result = asr_dual.transcribe_and_merge(session_path, language=language)
    else:
        asr_result = asr.transcribe(session_path, model_size=model_size, language=language)
    raw_text = asr_result["text"]
    words = list(asr_result["words"])
    tracing.emit("asr.done", {
        "raw_text": raw_text,
        "language": asr_result["language"],
        "duration_s": asr_result["duration"],
        "words": words,
        "extra": asr_result.get("extra", {}),
    })
    print(f"[transcribe] raw text: {raw_text!r}  ({len(words)} tokens)")

    # 2a) Voice-first scan: for every SINGLE word, check the voice DB.
    # Similarity is now phonetic (CTC transcript distance), so we can be
    # more confident: same word should score >= 0.75 even across speakers,
    # while unrelated words drop below 0.45. We still only scan single
    # words to keep the scan fast and unambiguous.
    voice_first_spans: List[Dict[str, Any]] = []
    if voice_match.list_voices() and words:
        VF_THRESHOLD = 0.80
        for i, tok in enumerate(words):
            start = tok.get("start")
            end = tok.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if end - start < 0.15:
                # too short to embed reliably
                continue
            try:
                hits = voice_match.match(
                    session_path,
                    start_s=float(start),
                    end_s=float(end),
                    threshold=VF_THRESHOLD,
                    top_k=1,
                )
            except Exception:
                hits = []
            if not hits or hits[0].get("source") != "user":
                continue
            voice_first_spans.append({
                "index_start": i,
                "index_end": i + 1,
                "text": (tok.get("word") or "").strip(),
                "start_s": float(start),
                "end_s": float(end),
                "probability_min": 1.0,
                "reason": f"voice_match:{hits[0]['term']}@{hits[0]['similarity']:.2f}",
            })

    # 2b) DETECT — LLM call #1.
    detect_spans: List[Dict[str, Any]] = []
    if words:
        try:
            dictionary = _load_medical_dictionary()
            suspects = suspect.detect(words, dictionary)
            detect_spans = _suspects_to_spans(suspects)
        except Exception as exc:
            print(f"[transcribe] DETECT failed: {exc!r}")
            detect_spans = []

    # 2c) Combine voice-first + DETECT spans WITHOUT losing words.
    #
    # A confident voice-first hit is a stop sign: it pinpoints exactly which
    # token an existing fingerprint matches. If the LLM returned a wider span
    # that engulfs that token (e.g. "gripex, maxilas" overlapping a voice hit
    # on just "gripex"), we MUST NOT merge them — that would allow the
    # voice-chosen replacement to swallow the other token. Instead we split
    # the LLM span around the voice-first tokens.
    voice_indices = {s["index_start"] for s in voice_first_spans}

    def _make_span(i0: int, i1: int, reason: str) -> Optional[Dict[str, Any]]:
        if i1 <= i0:
            return None
        toks = words[i0:i1]
        starts = [t.get("start") for t in toks if isinstance(t.get("start"), (int, float))]
        ends = [t.get("end") for t in toks if isinstance(t.get("end"), (int, float))]
        if not starts or not ends:
            return None
        probs = [t.get("probability") for t in toks if isinstance(t.get("probability"), (int, float))]
        return {
            "index_start": i0,
            "index_end": i1,
            "text": " ".join((t.get("word") or "").strip() for t in toks).strip(),
            "start_s": float(min(starts)),
            "end_s": float(max(ends)),
            "probability_min": float(min(probs)) if probs else 1.0,
            "reason": reason,
        }

    # Carry voice-first spans through unchanged.
    final_spans: List[Dict[str, Any]] = list(voice_first_spans)
    # Split each DETECT span around any voice indices it covers.
    for d in detect_spans:
        i0, i1 = d["index_start"], d["index_end"]
        cur = i0
        for vi in sorted(voice_indices):
            if vi < cur or vi >= i1:
                continue
            piece = _make_span(cur, vi, d.get("reason", ""))
            if piece is not None:
                final_spans.append(piece)
            cur = vi + 1
        if cur < i1:
            piece = _make_span(cur, i1, d.get("reason", ""))
            if piece is not None:
                final_spans.append(piece)

    # Dedup: if voice-first and a DETECT piece end up on the exact same
    # range, prefer the voice-first version.
    seen_ranges: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for s in final_spans:
        key = (s["index_start"], s["index_end"])
        prev = seen_ranges.get(key)
        if prev is None:
            seen_ranges[key] = s
        else:
            prefer = prev if prev.get("reason", "").startswith("voice_match") else s
            other = s if prev is prefer else prev
            seen_ranges[key] = prefer
            if other.get("reason") and other["reason"] not in prefer.get("reason", ""):
                prefer["reason"] = (prefer.get("reason", "") + "+" + other["reason"]).strip("+")

    spans = sorted(seen_ranges.values(), key=lambda s: s["index_start"])
    tracing.emit("voice_first.spans", {"spans": voice_first_spans})
    tracing.emit("detect.spans", {"spans": detect_spans})
    tracing.emit("spans.merged", {"spans": spans})
    print(f"[transcribe] voice_first: {[(s['index_start'], s['index_end'], s['text']) for s in voice_first_spans]}")
    print(f"[transcribe] DETECT:      {[(s['index_start'], s['index_end'], s['text']) for s in detect_spans]}")
    print(f"[transcribe] FINAL spans: {[(s['index_start'], s['index_end'], s['text']) for s in spans]}")

    # 3) For each span: voice retrieval -> candidates with descriptions.
    items_for_decide: List[Dict[str, Any]] = []
    span_meta: List[Dict[str, Any]] = []
    auto_choices: Dict[str, str] = {}
    tracing.emit("retrieve.start", {"n_spans": len(spans)})
    for s in spans:
        try:
            user_hits = voice_match.match(
                session_path,
                start_s=s["start_s"],
                end_s=s["end_s"],
                threshold=AUDIO_RETRIEVE_THRESHOLD_USER,
                top_k=8,
            )
        except Exception as exc:
            print(f"[transcribe] voice user match failed: {exc!r}")
            user_hits = []
        # Strong user-fingerprint match -> short-circuit, skip the LLM.
        item_id = f"s{len(items_for_decide)}"
        if user_hits and user_hits[0].get("source") == "user" and user_hits[0]["similarity"] >= AUDIO_AUTOFIX_THRESHOLD:
            auto_choices[item_id] = user_hits[0]["term"]
            span_meta.append({"id": item_id, "span": s, "hits": user_hits, "auto": True})
            items_for_decide.append({"id": item_id, "span": s["text"], "candidates": []})
            tracing.emit("retrieve.span", {
                "span_id": item_id,
                "span_text": s["text"],
                "start_s": s["start_s"],
                "end_s": s["end_s"],
                "user_hits": user_hits,
                "auto": True,
                "chosen": user_hits[0]["term"],
            })
            continue

        # Otherwise also pull seed (TTS-derived) hits at a more lenient floor.
        try:
            all_hits = voice_match.match(
                session_path,
                start_s=s["start_s"],
                end_s=s["end_s"],
                threshold=AUDIO_RETRIEVE_THRESHOLD_SEED,
                top_k=8,
            )
        except Exception as exc:
            print(f"[transcribe] voice fallback match failed: {exc!r}")
            all_hits = []
        # Dedup: keep best similarity per term, prefer user > seed.
        seen: Dict[str, Dict[str, Any]] = {}
        for h in (user_hits + all_hits):
            term = h["term"]
            prev = seen.get(term)
            if prev is None or h["similarity"] > prev["similarity"]:
                seen[term] = h
        ranked = sorted(seen.values(), key=lambda h: -h["similarity"])
        top = ranked[:5]
        # Attach description (use cached one if metadata didn't have it).
        candidates = []
        for h in top:
            desc = h.get("description") or descriptions.get(h["term"]) or ""
            candidates.append({
                "term": h["term"],
                "similarity": h["similarity"],
                "description": desc,
                "source": h.get("source", "user"),
            })
        items_for_decide.append({"id": item_id, "span": s["text"], "candidates": candidates})
        span_meta.append({"id": item_id, "span": s, "hits": top, "auto": False})
        tracing.emit("retrieve.span", {
            "span_id": item_id,
            "span_text": s["text"],
            "start_s": s["start_s"],
            "end_s": s["end_s"],
            "user_hits": user_hits,
            "all_hits": all_hits,
            "candidates": candidates,
            "auto": False,
        })

    # 4) DECIDE — LLM call #2.
    decisions: Dict[str, Optional[str]] = dict(auto_choices)
    decide_items = [it for it in items_for_decide if it["id"] not in auto_choices and it["candidates"]]
    if decide_items and USE_LLM:
        try:
            results = llm_decide.decide(raw_text, decide_items)
            for r in results:
                decisions[r["id"]] = r["choice"]
        except Exception as exc:
            print(f"[transcribe] DECIDE failed: {exc!r}")
            tracing.emit("decide.error", {"error": repr(exc)})

    tracing.emit("decide.done", {"decisions": decisions, "auto": auto_choices})
    print(f"[transcribe] decisions: {decisions}")

    # 5) Apply replacements at word-token level.
    word_replacements: Dict[int, str] = {}
    drop_indices: set = set()
    suspicious_out: List[Dict[str, Any]] = []
    for item, meta in zip(items_for_decide, span_meta):
        choice = decisions.get(item["id"])
        suspicious_out.append({
            "span": meta["span"]["text"],
            "start_s": meta["span"]["start_s"],
            "end_s": meta["span"]["end_s"],
            "reason": meta["span"].get("reason", "near_medical"),
            "auto_via_voice": bool(meta.get("auto")),
            "candidates": item["candidates"],
            "voice_hits": meta["hits"],
            "chosen": choice,
        })
        if not choice:
            continue
        i0 = meta["span"]["index_start"]
        i1 = meta["span"]["index_end"]
        word_replacements[i0] = choice
        for idx in range(i0 + 1, i1):
            drop_indices.add(idx)

    corrected_text = _apply_word_replacements(words, word_replacements, drop_indices) if words else raw_text
    if not words:
        corrected_text = raw_text

    return {
        "session_id": session_id,
        "raw_text": raw_text,
        "corrected_text": corrected_text.strip(),
        "suspicious": suspicious_out,
        "asr": {
            "language": asr_result["language"],
            "language_probability": asr_result["language_probability"],
            "duration": asr_result["duration"],
            "model_size": "dual_asr" if USE_DUAL_ASR else model_size,
            "words": words,
            # When USE_DUAL_ASR=1 this exposes both raw ASR outputs and the
            # LLM merge reason so the UI can show them side-by-side.
            "dual": asr_result.get("extra"),
        },
    }


def _apply_word_replacements(
    words: List[Dict, Any],
    replacements: Dict[int, str],
    drop_indices: set,
) -> str:
    pieces: List[str] = []
    for i, w in enumerate(words):
        if i in drop_indices:
            continue
        token_text = w.get("word") or ""
        if i in replacements:
            replacement = replacements[i]
            m = re.match(r"^(\s*)(.*?)(\s*)$", token_text, re.S)
            lead, _, trail = (m.group(1), m.group(2), m.group(3)) if m else ("", token_text, "")
            pieces.append(f"{lead}{replacement}{trail}")
        else:
            pieces.append(token_text)
    return "".join(pieces)


def _save_upload(audio: UploadFile) -> Tuple[str, Path, int]:
    session_id = uuid.uuid4().hex
    suffix = Path(audio.filename or "audio").suffix or ".webm"
    session_path = SESSIONS_DIR / f"{session_id}{suffix}"
    with session_path.open("wb") as fh:
        shutil.copyfileobj(audio.file, fh)
    size = session_path.stat().st_size
    return session_id, session_path, size


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model_size: str = Form(DEFAULT_WHISPER_SIZE),
) -> Dict[str, Any]:
    session_id, session_path, size = _save_upload(audio)
    effective_lang = language or DEFAULT_LANGUAGE
    print(
        f"[transcribe] session={session_id} type={audio.content_type!r} "
        f"size={size}B model={model_size} lang={effective_lang}"
    )
    if size < 200:
        return JSONResponse(
            status_code=400, content={"error": f"audio file is too small ({size} bytes)"}
        )
    try:
        return _run_transcribe_pipeline(session_path, session_id, effective_lang, model_size)
    except Exception as exc:
        print(f"[transcribe] pipeline error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.post("/api/transcribe_stream")
async def transcribe_stream(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model_size: str = Form(DEFAULT_WHISPER_SIZE),
):
    """NDJSON-streamed pipeline: one JSON event per line. Each event has
    {t, stage, payload}. The last event has stage='final' with the full
    transcribe result."""
    session_id, session_path, size = _save_upload(audio)
    effective_lang = language or DEFAULT_LANGUAGE
    print(
        f"[transcribe_stream] session={session_id} type={audio.content_type!r} "
        f"size={size}B model={model_size} lang={effective_lang}"
    )

    tracer = tracing.Tracer()

    def _runner():
        token = tracing.set_active(tracer)
        try:
            if size < 200:
                tracer.close({"error": f"audio file is too small ({size} bytes)"})
                return
            try:
                result = _run_transcribe_pipeline(session_path, session_id, effective_lang, model_size)
                tracer.close(result)
            except Exception as exc:
                print(f"[transcribe_stream] pipeline error: {exc!r}")
                tracer.close({"error": str(exc)})
        finally:
            tracing.reset_active(token)

    threading.Thread(target=_runner, daemon=True).start()

    return StreamingResponse(
        tracer.stream_lines(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# /api/transcribe_debug — transcript + flags + per-flag audio slice info
# ---------------------------------------------------------------------------
#
# Pipeline: ASR -> phonetic+LLM flagging -> CTC forced alignment of each
# flagged word back to (start_s, end_s). The UI shows all three so the
# user can see exactly where in the audio each suspicious word lives.

from .services import alignment_v2 as _alignment


@app.post("/api/transcribe_debug")
async def transcribe_debug(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    use_llm_flag: bool = Form(True),
) -> Dict[str, Any]:
    """Run ASR + flagging + alignment and return everything for the UI."""
    session_id, session_path, size = _save_upload(audio)
    effective_lang = language or DEFAULT_LANGUAGE
    print(
        f"[transcribe_debug] session={session_id} type={audio.content_type!r} "
        f"size={size}B lang={effective_lang}"
    )
    if size < 200:
        return JSONResponse(
            status_code=400, content={"error": f"audio file is too small ({size} bytes)"}
        )
    try:
        # 1) ASR (same path as /api/transcribe)
        asr_result = asr.transcribe(session_path, model_size=DEFAULT_WHISPER_SIZE,
                                    language=effective_lang)
        transcript = asr_result.get("text", "")
        duration_s = float(asr_result.get("duration", 0.0))
        words = list(asr_result.get("words") or [])
        pipeline_steps: List[Dict[str, Any]] = []
        pipeline_steps.append({
            "step": "ASR",
            "output": f"text={transcript} | tokens={len(words)} | duration_s={duration_s:.2f}",
        })

        # 2) Word-level CTC forced alignment of the full transcript.
        try:
            words_aligned = _alignment.align_words(session_path, transcript)
        except Exception as exc:
            print(f"[transcribe_debug] alignment failed: {exc!r}")
            words_aligned = []
        pipeline_steps.append({
            "step": "Alignment",
            "output": f"aligned_words={len(words_aligned)}",
        })

        # 3) LLM detect suspicious spans by context.
        try:
            detect_spans = llm_detect.detect(words)
        except Exception as exc:
            print(f"[transcribe_debug] LLM detect failed: {exc!r}")
            detect_spans = []
        if detect_spans:
            detect_list = "; ".join(s.get("text", "") for s in detect_spans)
        else:
            detect_list = "none"
        pipeline_steps.append({
            "step": "LLM detect",
            "output": f"spans={len(detect_spans)} | {detect_list}",
        })

        # 4) Phonetic candidate search + LLM choose.
        try:
            manager = _load_drug_dictionary_manager()
        except Exception as exc:
            print(f"[transcribe_debug] dictionary load failed: {exc!r}")
            manager = None

        flags: List[Dict[str, Any]] = []
        choose_lines: List[str] = []
        for s in detect_spans:
            span_text = s.get("text", "")
            candidates_raw = (
                asr_correction_pipeline.find_top_k_candidates(span_text, manager, top_k=5)
                if manager is not None else []
            )
            prompt = asr_correction_pipeline.build_llm_correction_prompt(
                transcript, span_text, candidates_raw
            )
            chosen = _llm_choose_term(
                prompt["system"],
                prompt["user"],
                [str(c.get("term")) for c in candidates_raw if c.get("term")],
            )
            s["chosen"] = chosen
            s["llm_prompt"] = prompt
            s["llm_likely_term"] = chosen or ""
            s["llm_confidence"] = 1.0 if chosen else 0.0

            cand_out = []
            for c in candidates_raw:
                uses = None
                meta = c.get("meta")
                if isinstance(meta, dict):
                    uses = meta.get("uses")
                cand_out.append(
                    {
                        "term": c.get("term"),
                        "phonetic_similarity": float(c.get("score") or 0.0),
                        "uses": uses,
                        "score": float(c.get("score") or 0.0),
                    }
                )
            top_terms = ", ".join(c.get("term", "") for c in candidates_raw[:5])
            choose_lines.append(f"{span_text} -> [{top_terms}] | chosen={chosen or 'NO_CHANGE'}")
            flags.append(
                {
                    "index": int(s.get("index_start", 0)),
                    "word": span_text,
                    "start_s": s.get("start_s"),
                    "end_s": s.get("end_s"),
                    "alignment_confidence": 0.0,
                    "reason": s.get("reason") or "near_medical",
                    "candidates": cand_out,
                    "llm_prompt": prompt,
                    "llm_likely_term": chosen or "",
                    "llm_confidence": 1.0 if chosen else 0.0,
                    "index_start": s.get("index_start"),
                    "index_end": s.get("index_end"),
                    "chosen": chosen,
                }
            )

        pipeline_steps.append({
            "step": "Phonetic candidates + LLM choose",
            "output": " ; ".join(choose_lines) if choose_lines else "no spans",
        })

        # 5) Apply LLM choices to build corrected transcript.
        raw_tokens = [(w.get("word") or "").strip() for w in words]
        corrected_text, applied = _apply_span_replacements(raw_tokens, flags)
        corrected = {
            "corrected_transcript": corrected_text or transcript,
            "applied": applied,
            "threshold": 0.0,
        }
        pipeline_steps.append({
            "step": "Corrected output",
            "output": corrected["corrected_transcript"],
        })

        return {
            "session_id": session_id,
            "audio_url": f"/api/session_audio/{session_id}",
            "transcript": transcript,
            "corrected_transcript": corrected["corrected_transcript"],
            "auto_corrections": corrected["applied"],
            "correction_threshold": corrected["threshold"],
            "duration_s": duration_s,
            "words": words_aligned,
            "flags": flags,
            "pipeline_steps": pipeline_steps,
        }
    except Exception as exc:
        print(f"[transcribe_debug] error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


@app.get("/api/session_audio/{session_id}")
def get_session_audio(session_id: str):
    """Serve the raw session audio so the browser can <audio>-play it
    and the UI can seek to flagged-word offsets."""
    # session_id is opaque; only allow files we actually have.
    for ext in (".webm", ".wav", ".mp3", ".m4a", ".ogg", ".flac"):
        p = SESSIONS_DIR / f"{session_id}{ext}"
        if p.exists():
            return FileResponse(p)
    return JSONResponse(status_code=404, content={"error": "audio not found"})


# ---------------------------------------------------------------------------
# /api/learn_from_edit
# ---------------------------------------------------------------------------


def _word_tokens(text: str) -> List[str]:
    return _LEARN_TOKEN_RE.findall(text)


def _diff_replacements(raw: str, corrected: str) -> List[Tuple[str, str]]:
    raw_words = _word_tokens(raw)
    corr_words = _word_tokens(corrected)
    raw_low = [w.lower() for w in raw_words]
    corr_low = [w.lower() for w in corr_words]
    pairs: List[Tuple[str, str]] = []
    sm = difflib.SequenceMatcher(a=raw_low, b=corr_low, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        new_phrase = " ".join(corr_words[j1:j2]).strip()
        old_phrase = " ".join(raw_words[i1:i2]).strip()
        if new_phrase:
            pairs.append((new_phrase, old_phrase))
    return pairs


def _find_session_audio(session_id: str) -> Optional[Path]:
    matches = list(SESSIONS_DIR.glob(f"{session_id}.*"))
    return matches[0] if matches else None


def _locate_words_in_raw(
    raw_text: str, old_phrase: str, asr_words: List[Dict[str, Any]]
) -> List[int]:
    raw_tokens = _word_tokens(raw_text)
    old_tokens = _word_tokens(old_phrase)
    if not old_tokens or not raw_tokens:
        return []
    raw_low = [t.lower() for t in raw_tokens]
    old_low = [t.lower() for t in old_tokens]
    start_in_raw = -1
    for i in range(len(raw_low) - len(old_low) + 1):
        if raw_low[i : i + len(old_low)] == old_low:
            start_in_raw = i
            break
    if start_in_raw < 0:
        return []
    word_to_indices: List[int] = []
    for asr_idx, w in enumerate(asr_words):
        toks = _word_tokens(w.get("word") or "")
        for _ in toks:
            word_to_indices.append(asr_idx)
    if start_in_raw + len(old_low) > len(word_to_indices):
        return []
    asr_indices = word_to_indices[start_in_raw : start_in_raw + len(old_low)]
    out: List[int] = []
    for idx in asr_indices:
        if not out or out[-1] != idx:
            out.append(idx)
    return out


@app.post("/api/learn_from_edit")
def learn_from_edit(req: LearnFromEditRequest) -> Dict[str, Any]:
    pairs = _diff_replacements(req.raw_text, req.corrected_text)
    if not pairs:
        return {"ok": True, "learned_text": [], "learned_voices": []}

    session_path = _find_session_audio(req.session_id) if req.session_id else None
    asr_words: List[Dict[str, Any]] = []
    if session_path:
        try:
            asr_result = asr.transcribe(session_path, model_size=DEFAULT_WHISPER_SIZE, language=None)
            asr_words = list(asr_result["words"])
        except Exception as exc:
            print(f"[learn] re-ASR failed: {exc!r}")

    known_text = {e["term"].lower() for e in lexicon.list_terms()}
    for e in lexicon.list_terms():
        for a in e.get("aliases") or []:
            known_text.add(str(a).lower())

    learned_text: List[Dict[str, Any]] = []
    learned_voices: List[Dict[str, Any]] = []

    for new_phrase, old_phrase in pairs:
        new_lower = new_phrase.lower()

        # 1) Save term in lexicon (text side).
        if new_lower not in known_text:
            aliases = []
            if old_phrase and old_phrase.lower() != new_lower:
                aliases.append(old_phrase)
            entry = lexicon.add_term(
                term=new_phrase, type_=req.type, aliases=aliases, priority=1.0
            )
            learned_text.append({"entry": entry, "from_alias": old_phrase or None})
            known_text.add(new_lower)

        # 2) Generate description (best-effort, cached).
        try:
            descriptions.get_or_generate(new_phrase, type_hint=req.type)
        except Exception as exc:
            print(f"[learn] description gen failed: {exc!r}")
        desc = descriptions.get(new_phrase)

        # 3) Save voice fingerprint if we have audio + matching ASR words.
        if session_path and asr_words and old_phrase:
            indices = _locate_words_in_raw(req.raw_text, old_phrase, asr_words)
            if indices:
                first_idx = indices[0]
                last_idx = indices[-1]
                start_s = float(asr_words[first_idx].get("start") or 0.0)
                end_s = float(asr_words[last_idx].get("end") or start_s + 0.5)
                if end_s > start_s:
                    try:
                        v = voice_match.register(
                            term=new_phrase,
                            audio_path=session_path,
                            start_s=start_s,
                            end_s=end_s,
                            description=desc,
                            source="user",
                        )
                        learned_voices.append({"voice": v, "from_phrase": old_phrase})
                    except Exception as exc:
                        print(f"[learn] voice register failed: {exc!r}")

    return {"ok": True, "learned_text": learned_text, "learned_voices": learned_voices}
