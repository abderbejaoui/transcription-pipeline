"""FastAPI app — ASR post-correction pipeline for medical transcripts.

Pipeline
--------
Stage 1: Transcription & flagging
    - ASR returns text + word-level confidence
    - Words below the confidence threshold are flagged
    - Confident words never go to the LLM

Stage 2: Error lexicon (no LLM)
    - SQLite lexicon from doctor-confirmed corrections
    - Lookup: exact match -> Double Metaphone -> fuzzy similarity
    - Apply corrections immediately when matched

Stage 3: Routing & LLM correction
    - Suspected medical entities: fuzzy KG lookup first
    - If KG misses, call Calme with a strict medical prompt
    - General language errors go to Calme with full sentence context
    - Ambiguous/short words are queued for human review
    - LLM corrections below the confidence threshold are not auto-applied

Stage 4: Verification & doctor queue
    - Calme verifies coherence between raw and corrected text
    - If coherence confidence is low, revert to original transcript
    - Any correction touching drug names is queued for doctor review

Endpoints
---------
GET  /                       single-page UI
GET  /api/healthz
GET  /api/lexicon
POST /api/teach              (kept for compatibility; updates lexicon only)
POST /api/transcribe         audio in -> corrected transcript out
POST /api/learn_from_edit    write confirmed corrections to the error lexicon
GET  /api/voices             list voice index (legacy)
POST /api/voices/reset       wipe voice index (legacy)
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from fastapi.responses import StreamingResponse

from .services import (
    asr,
    correction,
    descriptions,
    error_lexicon,
    kg_lookup,
    lexicon,
    llm_correct,
    llm_runtime,
    llm_verify,
    medspeakian,
    review_queue,
    suspect,
    tracing,
    voice_match,
)


# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = PROJECT_ROOT / "app" / "static"
SESSIONS_DIR = PROJECT_ROOT / "data" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_WHISPER_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "large-v3")
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
USE_LLM = os.environ.get("USE_LLM", "1") == "1"

WORD_CONFIDENCE_THRESHOLD = float(os.environ.get("WORD_CONFIDENCE_THRESHOLD", "0.70"))
LLM_CONFIDENCE_THRESHOLD = float(os.environ.get("LLM_CONFIDENCE_THRESHOLD", "0.70"))
COHERENCE_THRESHOLD = float(os.environ.get("COHERENCE_THRESHOLD", "0.60"))
KG_AUTOFIX_THRESHOLD = float(os.environ.get("KG_AUTOFIX_THRESHOLD", "90"))
KG_SUSPECT_THRESHOLD = float(os.environ.get("KG_SUSPECT_THRESHOLD", "80"))
MEDSPEAK_AUTO_THRESHOLD = float(os.environ.get("MEDSPEAK_AUTO_THRESHOLD", "0.60"))
MEDSPEAK_MIN_SCORE = float(os.environ.get("MEDSPEAK_MIN_SCORE", "0.60"))
FLAG_MERGE_GAP_S = float(os.environ.get("FLAG_MERGE_GAP_S", "0.10"))
MIN_AMBIGUOUS_CHARS = int(os.environ.get("MIN_AMBIGUOUS_CHARS", "3"))

TEXT_CORRECTOR = correction.MedicalCorrector()

AMBIGUOUS_WORDS = {
    "uh",
    "um",
    "er",
    "ah",
    "eh",
    "hmm",
}

_LEARN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")


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
            asr._load_model(DEFAULT_WHISPER_SIZE)
            print(f"[startup] faster-whisper '{DEFAULT_WHISPER_SIZE}' ready.")
        except Exception as exc:
            print(f"[startup] Whisper warmup failed: {exc}")
        if USE_LLM:
            llm_runtime.warm_up()

    threading.Thread(target=_bg, daemon=True).start()


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
    """Text-only correction path for pasted transcription.

    This uses the same vocabulary-driven correction engine as the medical
    pipeline, but skips audio-only steps because no word confidences exist.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    result = TEXT_CORRECTOR.correct_transcript(req.text)
    steps: List[Dict[str, Any]] = [
        {
            "step": "input",
            "message": "Received pasted text without audio.",
            "text": req.text,
        },
        {
            "step": "analysis",
            "message": "Generated candidate correction spans using the medical lexicon.",
            "count": len(result.get("suspicious_spans") or []),
        },
    ]
    for span in result.get("suspicious_spans") or []:
        steps.append(
            {
                "step": "match",
                "message": f"Matched {span.get('original_text')!r} -> {span.get('possible_correction')!r}.",
                "original_text": span.get("original_text"),
                "possible_correction": span.get("possible_correction"),
                "issue_type": span.get("issue_type"),
                "confidence": span.get("confidence"),
                "score": span.get("score"),
                "reason_short": span.get("reason_short"),
                "features": span.get("features"),
            }
        )
    steps.append(
        {
            "step": "apply",
            "message": "Applied the selected replacements to produce the corrected text.",
            "changed": result["corrected_text"] != req.text,
        }
    )
    return {
        "raw_text": req.text,
        "corrected_text": result["corrected_text"],
        "suspicious_spans": result["suspicious_spans"],
        "correction_steps": steps,
        "note": "text-only correction mode: no audio signals available",
    }


def _clean_token(text: str) -> str:
    return re.sub(r"^[\s\W_]+|[\s\W_]+$", "", text)


def _flag_low_confidence(
    words: List[Dict[str, Any]],
    *,
    threshold: float,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, w in enumerate(words):
        prob = w.get("probability")
        prob_val = float(prob) if isinstance(prob, (int, float)) else 1.0
        if prob_val >= threshold:
            continue
        raw = w.get("word") or ""
        clean = _clean_token(raw)
        if not clean:
            continue
        out.append(
            {
                "index": i,
                "text": clean,
                "start": float(w.get("start") or 0.0),
                "end": float(w.get("end") or 0.0),
                "probability": prob_val,
                "reason": "low_confidence",
            }
        )
    return out


def _is_sentence_break(token_text: str) -> bool:
    return bool(re.search(r"[.!?]", token_text))


def _sentence_bounds(words: List[Dict[str, Any]], i0: int, i1: int) -> Tuple[int, int]:
    start = max(0, i0)
    end = min(len(words) - 1, i1)
    while start > 0 and not _is_sentence_break(words[start - 1].get("word") or ""):
        start -= 1
    while end < len(words) - 1 and not _is_sentence_break(words[end].get("word") or ""):
        end += 1
    return start, end


def _sentence_text(words: List[Dict[str, Any]], i0: int, i1: int) -> str:
    if not words:
        return ""
    start, end = _sentence_bounds(words, i0, i1)
    return "".join((w.get("word") or "") for w in words[start : end + 1]).strip()


def _is_ambiguous(span_text: str) -> bool:
    stripped = _clean_token(span_text)
    alpha = re.sub(r"[^A-Za-z]", "", stripped)
    if len(alpha) < MIN_AMBIGUOUS_CHARS:
        return True
    if alpha.lower() in AMBIGUOUS_WORDS:
        return True
    return False


def _norm_simple(text: str) -> str:
    return " ".join(text.lower().strip().split())


# ---------------------------------------------------------------------------
# /api/transcribe (JSON)  and  /api/transcribe_stream (NDJSON traces)
# ---------------------------------------------------------------------------


def _run_transcribe_pipeline(
    session_path: Path, session_id: str, language: Optional[str], model_size: str
) -> Dict[str, Any]:
    """The full pipeline. Calls into services that emit trace events via
    `app.services.tracing`, so this function can be run with or without an
    active Tracer."""
    tracing.emit("asr.start", {
        "session_id": session_id,
        "audio_path": str(session_path),
        "model": model_size,
        "language": language or "auto",
        "size_bytes": session_path.stat().st_size,
    })
    asr_result = asr.transcribe(session_path, model_size=model_size, language=language)
    raw_text = asr_result["text"]
    words = list(asr_result["words"])
    tracing.emit("asr.done", {
        "raw_text": raw_text,
        "language": asr_result["language"],
        "duration_s": asr_result["duration"],
        "words": words,
    })
    print(f"[transcribe] raw text: {raw_text!r}  ({len(words)} tokens)")

    suspicious_words = _flag_low_confidence(words, threshold=WORD_CONFIDENCE_THRESHOLD)
    spans = suspect.merge_adjacent(suspicious_words, max_gap_s=FLAG_MERGE_GAP_S)
    tracing.emit("flagging.done", {
        "threshold": WORD_CONFIDENCE_THRESHOLD,
        "spans": spans,
    })
    print(f"[transcribe] low-conf spans: {[(s['index_first'], s['index_last'], s['text']) for s in spans]}")

    word_replacements: Dict[int, str] = {}
    drop_indices: set = set()
    suspicious_out: List[Dict[str, Any]] = []
    queued_out: List[Dict[str, Any]] = []

    def _apply(i0: int, i1: int, replacement: str) -> None:
        word_replacements[i0] = replacement
        for idx in range(i0 + 1, i1):
            drop_indices.add(idx)

    def _queue(
        *,
        span_text: str,
        correction: str,
        confidence: float,
        stage: str,
        reason: str,
        route: Optional[str],
        sentence: str,
        entity_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        item = {
            "session_id": session_id,
            "span": span_text,
            "suggested": correction,
            "confidence": round(confidence, 4),
            "stage": stage,
            "route": route,
            "reason": reason,
            "sentence": sentence,
            "entity_type": entity_type,
        }
        queued = review_queue.enqueue(item)
        queued_out.append(queued)
        tracing.emit("review.queue", {"item": queued})
        return queued

    for s in spans:
        span_text = s["text"]
        i0 = int(s["index_first"])
        i1 = int(s["index_last"]) + 1
        start_s = float(s.get("start") or 0.0)
        end_s = float(s.get("end") or 0.0)
        sentence = _sentence_text(words, i0, i1 - 1)

        lex_match = error_lexicon.lookup(span_text)
        if lex_match is not None:
            correction = lex_match.correct_text
            confidence = lex_match.similarity / 100.0
            queued = None
            applied = False
            if kg_lookup.is_drug(correction) and lex_match.source != "doctor":
                queued = _queue(
                    span_text=span_text,
                    correction=correction,
                    confidence=confidence,
                    stage="lexicon",
                    reason="drug_review",
                    route=None,
                    sentence=sentence,
                    entity_type="drug",
                )
            else:
                _apply(i0, i1, correction)
                applied = True
            suspicious_out.append({
                "span": span_text,
                "start_s": start_s,
                "end_s": end_s,
                "stage": "lexicon",
                "match_type": lex_match.match_type,
                "chosen": correction,
                "confidence": round(confidence, 4),
                "applied": applied,
                "queued": queued is not None,
                "queue_id": queued.get("id") if queued else None,
                "reason": f"lexicon_{lex_match.match_type}",
                "source": lex_match.source,
            })
            tracing.emit("lexicon.match", {
                "span": span_text,
                "correction": correction,
                "match_type": lex_match.match_type,
                "similarity": lex_match.similarity,
            })
            continue

        if _is_ambiguous(span_text):
            queued = _queue(
                span_text=span_text,
                correction="",
                confidence=0.0,
                stage="ambiguous",
                reason="ambiguous",
                route=None,
                sentence=sentence,
            )
            suspicious_out.append({
                "span": span_text,
                "start_s": start_s,
                "end_s": end_s,
                "stage": "ambiguous",
                "chosen": None,
                "confidence": 0.0,
                "applied": False,
                "queued": True,
                "queue_id": queued.get("id"),
                "reason": "ambiguous",
            })
            tracing.emit("span.ambiguous", {"span": span_text})
            continue

        kg_match = kg_lookup.find_best(span_text)
        if kg_match and kg_match["score"] >= KG_AUTOFIX_THRESHOLD:
            correction = str(kg_match["term"])
            confidence = float(kg_match["score"]) / 100.0
            _apply(i0, i1, correction)
            suspicious_out.append({
                "span": span_text,
                "start_s": start_s,
                "end_s": end_s,
                "stage": "local_kg",
                "chosen": correction,
                "confidence": round(confidence, 4),
                "kg_score": round(float(kg_match["score"]), 2),
                "applied": True,
                "queued": False,
                "queue_id": None,
                "reason": "local_kg_exact",
                "entity_type": str(kg_match.get("type") or "term"),
            })
            tracing.emit("kg.match", {
                "span": span_text,
                "correction": correction,
                "score": kg_match["score"],
            })
            continue

        if kg_match and kg_match["score"] >= KG_SUSPECT_THRESHOLD:
            correction = str(kg_match["term"])
            confidence = float(kg_match["score"]) / 100.0
            variant = str(kg_match.get("variant") or "")
            exact_alias = _norm_simple(variant) == _norm_simple(span_text)
            if exact_alias:
                _apply(i0, i1, correction)
                suspicious_out.append({
                    "span": span_text,
                    "start_s": start_s,
                    "end_s": end_s,
                    "stage": "local_kg",
                    "chosen": correction,
                    "confidence": round(confidence, 4),
                    "kg_score": round(float(kg_match["score"]), 2),
                    "applied": True,
                    "queued": False,
                    "queue_id": None,
                    "reason": "local_kg_alias",
                    "entity_type": str(kg_match.get("type") or "term"),
                })
                tracing.emit("kg.match", {
                    "span": span_text,
                    "correction": correction,
                    "score": kg_match["score"],
                    "alias_match": True,
                })
            else:
                queued = _queue(
                    span_text=span_text,
                    correction=correction,
                    confidence=confidence,
                    stage="local_kg",
                    reason="local_kg_low_confidence",
                    route="medical",
                    sentence=sentence,
                    entity_type=str(kg_match.get("type") or "term"),
                )
                suspicious_out.append({
                    "span": span_text,
                    "start_s": start_s,
                    "end_s": end_s,
                    "stage": "local_kg",
                    "chosen": correction,
                    "confidence": round(confidence, 4),
                    "kg_score": round(float(kg_match["score"]), 2),
                    "applied": False,
                    "queued": True,
                    "queue_id": queued.get("id"),
                    "reason": "local_kg_low_confidence",
                    "entity_type": str(kg_match.get("type") or "term"),
                })
                tracing.emit("kg.match", {
                    "span": span_text,
                    "correction": correction,
                    "score": kg_match["score"],
                    "alias_match": False,
                })
            continue

        medspeak_match: Optional[Dict[str, Any]] = None
        try:
            medspeak_match = medspeakian.retrieve(span_text)
        except Exception as exc:
            print(f"[transcribe] MedSpeak retrieval failed: {exc!r}")
            medspeak_match = None

        if medspeak_match is not None and medspeak_match.get("score") is not None:
            med_score = float(medspeak_match.get("score") or 0.0)
            if med_score >= MEDSPEAK_MIN_SCORE:
                correction = str(medspeak_match.get("term") or "").strip()
                applied = False
                queued = None
                if correction and med_score >= MEDSPEAK_AUTO_THRESHOLD:
                    _apply(i0, i1, correction)
                    applied = True
                else:
                    queued = _queue(
                        span_text=span_text,
                        correction=correction,
                        confidence=med_score,
                        stage="medspeak",
                        reason="medspeak_low_confidence",
                        route="medical",
                        sentence=sentence,
                    )
                suspicious_out.append({
                    "span": span_text,
                    "start_s": start_s,
                    "end_s": end_s,
                    "stage": "medspeak",
                    "chosen": correction or None,
                    "confidence": round(med_score, 4),
                    "phonetic_score": round(float(medspeak_match.get("phonetic_score") or 0.0), 4),
                    "semantic_score": round(float(medspeak_match.get("semantic_score") or 0.0), 4),
                    "applied": applied,
                    "queued": queued is not None,
                    "queue_id": queued.get("id") if queued else None,
                    "reason": "medspeak_retrieval",
                })
                tracing.emit("medspeak.match", {
                    "span": span_text,
                    "correction": correction,
                    "score": med_score,
                    "phonetic_score": medspeak_match.get("phonetic_score"),
                    "semantic_score": medspeak_match.get("semantic_score"),
                })
                continue

        route = "medical" if medspeak_match is not None else "general"
        llm_result = {"replacement": "", "confidence": 0.0, "reason": "llm_disabled"}
        if USE_LLM:
            try:
                if route == "medical":
                    llm_result = llm_correct.correct_medical(span_text, sentence)
                else:
                    llm_result = llm_correct.correct_general(span_text, sentence)
            except Exception as exc:
                print(f"[transcribe] LLM correct failed: {exc!r}")
                llm_result = {"replacement": "", "confidence": 0.0, "reason": "llm_error"}

        replacement = str(llm_result.get("replacement") or "").strip()
        llm_conf = float(llm_result.get("confidence") or 0.0)
        if replacement and replacement.lower() == span_text.lower():
            replacement = ""

        queued = None
        applied = False
        if route == "medical":
            if replacement:
                queued = _queue(
                    span_text=span_text,
                    correction=replacement,
                    confidence=llm_conf,
                    stage="llm",
                    reason="medical_llm_review",
                    route=route,
                    sentence=sentence,
                )
        else:
            if replacement and llm_conf >= LLM_CONFIDENCE_THRESHOLD:
                _apply(i0, i1, replacement)
                applied = True
            else:
                if replacement or llm_conf > 0.0:
                    queued = _queue(
                        span_text=span_text,
                        correction=replacement,
                        confidence=llm_conf,
                        stage="llm",
                        reason="low_confidence",
                        route=route,
                        sentence=sentence,
                    )

        suspicious_out.append({
            "span": span_text,
            "start_s": start_s,
            "end_s": end_s,
            "stage": "llm",
            "route": route,
            "chosen": replacement or None,
            "confidence": round(llm_conf, 4),
            "applied": applied,
            "queued": queued is not None,
            "queue_id": queued.get("id") if queued else None,
            "reason": str(llm_result.get("reason") or ""),
        })
        tracing.emit("llm.correct", {
            "span": span_text,
            "route": route,
            "replacement": replacement,
            "confidence": llm_conf,
        })

    corrected_text = _apply_word_replacements(words, word_replacements, drop_indices) if words else raw_text
    if not words:
        corrected_text = raw_text

    coherence = {"confidence": 1.0, "issues": []}
    reverted = False
    if USE_LLM and corrected_text.strip() and corrected_text.strip() != raw_text.strip():
        try:
            coherence = llm_verify.verify(raw_text, corrected_text)
        except Exception as exc:
            print(f"[transcribe] coherence check failed: {exc!r}")
            coherence = {"confidence": 0.0, "issues": ["llm_error"]}
        if coherence.get("confidence", 0.0) < COHERENCE_THRESHOLD:
            corrected_text = raw_text
            reverted = True
    coherence["threshold"] = COHERENCE_THRESHOLD
    coherence["reverted"] = reverted
    tracing.emit("verify.done", {"coherence": coherence})

    return {
        "session_id": session_id,
        "raw_text": raw_text,
        "corrected_text": corrected_text.strip(),
        "suspicious": suspicious_out,
        "review_queue": queued_out,
        "coherence": coherence,
        "asr": {
            "language": asr_result["language"],
            "language_probability": asr_result["language_probability"],
            "duration": asr_result["duration"],
            "model_size": model_size,
            "words": words,
        },
    }


def _apply_word_replacements(
    words: List[Dict[str, Any]],
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


@app.post("/api/learn_from_edit")
def learn_from_edit(req: LearnFromEditRequest) -> Dict[str, Any]:
    pairs = _diff_replacements(req.raw_text, req.corrected_text)
    if not pairs:
        return {"ok": True, "learned_text": [], "learned_voices": []}
    learned_text: List[Dict[str, Any]] = []
    for new_phrase, old_phrase in pairs:
        if not old_phrase:
            continue
        try:
            entry = error_lexicon.add_correction(
                wrong_text=old_phrase,
                correct_text=new_phrase,
                source="doctor",
            )
            learned_entry = {
                "entry": {"term": new_phrase, "type": req.type},
                "from_alias": old_phrase,
                "error_lexicon_id": entry.get("id"),
            }
            if req.type in {"drug", "diagnosis", "procedure"}:
                try:
                    kg_res = kg_lookup.add_entity(
                        new_phrase,
                        entity_type=req.type,
                        alias=old_phrase if req.type == "drug" else None,
                    )
                    learned_entry["kg_updated"] = bool(kg_res.get("ok"))
                except Exception as exc:
                    print(f"[learn] kg update failed: {exc!r}")
            learned_text.append(learned_entry)
        except Exception as exc:
            print(f"[learn] error lexicon update failed: {exc!r}")

    return {"ok": True, "learned_text": learned_text, "learned_voices": []}
