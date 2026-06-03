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
    asr, asr_benchmark, asr_dual, descriptions, drug_normalize, flag, lexicon,
    llm_decide, llm_detect, tracing, voice_match,
)
from .services.correction import MedicalCorrector, LexiconEntry as _LexiconEntry, compact


def _build_corrector() -> MedicalCorrector:
    """Build a MedicalCorrector from the current lexicon on disk.

    Short abbreviation aliases (<= 3 compact characters, e.g. 'asa', 'aml')
    are stripped here because they match too many common English short words
    when applied to free-form medical conversation text.  The LLM DETECT
    / DECIDE pipeline handles those cases when the context is clear.
    """
    raw = lexicon.list_terms()
    entries = []
    for e in raw:
        aliases = [
            a for a in (e.get("aliases") or [])
            if len(compact(a)) > 3  # drop 2-3 char abbreviations
        ]
        entries.append(
            _LexiconEntry(
                term=e["term"],
                type=e.get("type", ""),
                aliases=tuple(aliases),
                priority=float(e.get("priority", 1.0)),
            )
        )
    return MedicalCorrector(
        lexicon=entries,
        accept_threshold=88.0,         # was 80 — tighter to reduce false positives
        single_word_score_floor=80.0,  # floor for the strong-phonetic path
        single_word_phonetic_floor=92.0,
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
# When 1, skip word-level forced alignment in /api/transcribe_debug. Alignment
# only produces per-word TIMESTAMPS (it does not change the transcript text),
# so disabling it avoids loading the MMS aligner / whisper-small timing model.
DISABLE_ALIGNMENT = os.environ.get("DISABLE_ALIGNMENT", "0") == "1"

# Audio-retrieval thresholds, calibrated for the CTC phonetic similarity
# scale (normalized Levenshtein over greedy wav2vec2-base-960h transcripts).
# Empirically: same word / different voice -> 0.55-0.85, same word / same
# voice -> 0.85-1.00, different words -> < 0.40. The LLM remains the
# final filter; we just keep candidates loose enough to feed it.
AUDIO_RETRIEVE_THRESHOLD_USER = 0.55
AUDIO_RETRIEVE_THRESHOLD_SEED = 0.45
AUDIO_AUTOFIX_THRESHOLD = 0.85  # short-circuit when very strong USER match

_LEARN_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9'-]+")

# When 1, run both Gulf LoRA and base Qwen3 in parallel then merge via LLM judge.
# Only affects the /api/transcribe path; the /api/test-pipeline (eval) does not
# use ASR and is unaffected.
USE_DUAL_ASR = os.environ.get("USE_DUAL_ASR", "0") == "1"

# Lazy-loading globals for the /pipeline page (optional front-end features).
# Independent of the main /api/transcribe pipeline.
_TEXT_CORRECTOR: Optional[MedicalCorrector] = None


def _get_text_corrector() -> MedicalCorrector:
    global _TEXT_CORRECTOR
    if _TEXT_CORRECTOR is None:
        _TEXT_CORRECTOR = _build_corrector()
    return _TEXT_CORRECTOR


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


@app.get("/pipeline", include_in_schema=False)
def pipeline_tester() -> FileResponse:
    """Standalone page to test the correction pipeline on text, without ASR."""
    return FileResponse(STATIC_DIR / "pipeline.html")


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
    # Close the HITL feedback loop: add the clinician-confirmed term to the
    # candidate-retrieval dataset (medical_terms.txt) and invalidate the
    # caches so it (and its aliases) take effect on the next run — no restart.
    flag.add_retrieval_term(req.term)
    flag.record_taught_aliases(req.term, req.aliases)
    flag.invalidate_lexicon_cache()
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
        fake_words.append({"word": " " + tok, "start": 0.0, "end": 0.0, "probability": 1.0})
    spans = []
    if USE_LLM and fake_words:
        try:
            spans = llm_detect.detect(fake_words)
        except Exception as exc:
            print(f"[correct] DETECT failed: {exc!r}")
    return {
        "raw_text": req.text,
        "corrected_text": req.text,
        "suspicious": spans,
        "note": "text-only mode: voice retrieval is disabled without audio",
    }


BENCHMARK_PROGRESS: Dict[str, Any] = {}

@app.get("/api/benchmark_progress/{session_id}")
def get_benchmark_progress(session_id: str) -> Dict[str, Any]:
    return BENCHMARK_PROGRESS.get(session_id, {"status": "unknown"})

@app.post("/api/benchmark_asr")
def benchmark_asr(
    file: UploadFile = File(...),
    models: Optional[str] = Form(None),
    client_session_id: Optional[str] = Form(None)
) -> JSONResponse:
    target_models = []
    if models:
        target_models = [m.strip() for m in models.split(",") if m.strip()]
    else:
        target_models = list(asr_benchmark.MODELS.keys())
        
    session_id = client_session_id or uuid.uuid4().hex
    
    BENCHMARK_PROGRESS[session_id] = {
        "status": "Receiving audio file...",
        "completed": 0,
        "total": len(target_models),
        "current_model": None
    }
    print(f"[{session_id}] Upload received: {file.filename}")
    
    ext = ".wav"
    if file.filename:
        _, ext = os.path.splitext(file.filename)
    audio_path = SESSIONS_DIR / f"{session_id}{ext}"
    
    try:
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        print(f"[{session_id}] Audio saved to {audio_path}. Starting digestion...")
        BENCHMARK_PROGRESS[session_id]["status"] = "Audio saved. Initializing models..."
            
        results = []
        for index, model_key in enumerate(target_models):
            print(f"[{session_id}] Running {index + 1}/{len(target_models)}: {model_key}")
            BENCHMARK_PROGRESS[session_id].update({
                "status": f"Running inference on {model_key}...",
                "current_model": model_key
            })
            try:
                res = asr_benchmark.run_asr(model_key, str(audio_path))
                print(f"[{session_id}] -> Finished {model_key} in {res.get('duration_s', 0)}s")
                results.append(res)
            except Exception as e:
                 print(f"[{session_id}] -> Failed {model_key}: {e}")
                 results.append({
                    "model_key": model_key,
                    "transcript": "",
                    "language": None,
                    "duration_s": 0.0,
                    "word_timestamps": [],
                    "error": str(e),
                })
                 
            BENCHMARK_PROGRESS[session_id]["completed"] = index + 1
                 
        BENCHMARK_PROGRESS[session_id]["status"] = "Processing completed."

        # Audio duration for extra telemetry, if we can read it
        audio_duration_s = 0.0
        try:
             import soundfile as sf
             info = sf.info(str(audio_path))
             audio_duration_s = info.duration
        except Exception:
             pass

        return JSONResponse({
            "audio_filename": file.filename or "uploaded_audio",
            "audio_duration_s": round(audio_duration_s, 2),
            "results": results
        })

    finally:
        pass


# ---------------------------------------------------------------------------
# /api/transcribe (JSON)  and  /api/transcribe_stream (NDJSON traces)
# ---------------------------------------------------------------------------


def _run_transcribe_pipeline(
    session_path: Path, session_id: str, language: Optional[str], model_size: str
) -> Dict[str, Any]:
    """The full pipeline. Calls into services that emit trace events via
    `app.services.tracing`, so this function can be run with or without an
    active Tracer."""
    # 1) ASR. ONLY the 900h Gulf Arabic fine-tune. There is no other model and
    # no fallback: if it can't run, no transcript is produced.
    tracing.emit("asr.start", {
        "session_id": session_id,
        "audio_path": str(session_path),
        "model": model_size,
        "language": language or "auto",
        "size_bytes": session_path.stat().st_size,
    })
    if USE_DUAL_ASR:
        asr_result = asr_dual.transcribe_and_merge(
            session_path, language=language
        )
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
    if USE_LLM and words:
        try:
            detect_spans = llm_detect.detect(words)
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
            "model_size": model_size,
            "words": words,
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
# /api/asr — ASR-only endpoint for the /pipeline page
# ---------------------------------------------------------------------------
#
# Just ASR (Qwen3-ASR-1.7B Gulf LoRA), no correction pipeline. Used by the
# standalone /pipeline page to get a raw transcript for manual correction.


@app.post("/api/asr")
async def asr_only(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    model_size: str = Form(DEFAULT_WHISPER_SIZE),
) -> Dict[str, Any]:
    """ASR-only: transcribe audio without the correction pipeline."""
    session_id, session_path, size = _save_upload(audio)
    effective_lang = language or DEFAULT_LANGUAGE
    if size < 200:
        return JSONResponse(
            status_code=400, content={"error": f"audio file is too small ({size} bytes)"}
        )
    try:
        result = asr.transcribe(session_path, model_size=model_size, language=effective_lang)
        return {
            "text": result["text"],
            "raw_text": result["text"],
            "language": result["language"],
            "duration": result["duration"],
            "session_id": session_id,
        }
    except Exception as exc:
        print(f"[asr] error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# /api/transcribe_debug — transcript + flags + per-flag audio slice info
# ---------------------------------------------------------------------------
#
# Pipeline: ASR -> phonetic+LLM flagging -> CTC forced alignment of each
# flagged word back to (start_s, end_s). The UI shows all three so the
# user can see exactly where in the audio each suspicious word lives.

from .services import alignment_v2 as _alignment, flag as _flag


# ---------------------------------------------------------------------------
# /api/test-pipeline — run the pipeline on a single text case and return
# the full response schema for the standalone eval script.
# ---------------------------------------------------------------------------
#
# Expected request:
#   POST /api/test-pipeline  { "transcript": "...", "case_id": "TC-001" }
#
# Expected response:
#   {
#     "case_id": "TC-001",
#     "original": "...",
#     "corrected": "...",
#     "corrections": [{ "span_text": "...", "chosen": "...", "path": "auto_fix", "confidence": 0.0 }],
#     "flagged_spans": [{ "text": "...", "start_index": 0, "end_index": 1, "max_suspicion": 0.0, "reason": "..." }],
#     "retrieval_candidates": [{ "span_text": "...", "candidates": [{ "term": "...", "phonetic_score": 0.0 }] }]
#   }


@app.post("/api/test-pipeline")
async def test_pipeline(req: dict) -> Dict[str, Any]:
    """Run the pipeline on one text input and return the structured response.

    The standalone eval_pipeline.py script calls this endpoint for each
    test case and evaluates the response against expected data across
    5 stages: scoring & flagging, phonetic retrieval, correction decision,
    false positive guard, and end-to-end exact match.
    """
    transcript = (req.get("transcript") or "").strip()
    case_id = req.get("case_id") or "unknown"
    # Optional: enable the LLM pass for higher-accuracy correction at the
    # cost of latency and non-determinism. Default False for reproducibility.
    use_llm_flag = req.get("use_llm", False)
    if not isinstance(use_llm_flag, bool):
        use_llm_flag = False
    if not transcript:
        return JSONResponse(status_code=400, content={"error": "transcript is required", "case_id": case_id})

    try:
        # 1) Drug normalization first (Arabic → Latin for known drug variants).
        normalized_text, drug_fixes = drug_normalize.normalize_drugs(transcript)

        # 1b) HITL feedback: apply any clinician-confirmed alias mappings
        #     (exact matches a human taught) deterministically before flagging.
        normalized_text, taught_fixes = flag.apply_taught_aliases(normalized_text)

        # 2) Flag suspicious spans on the NORMALIZED text.
        #    The phonetic pass is deterministic and catches phonetic mishearings
        #    plus split drug names. When use_llm=True, the LLM pass can attach
        #    likely_term / confidence to help correct borderline cases.
        flags_out = flag.flag_suspicious(normalized_text, use_llm=use_llm_flag)

        # 3) Apply high-confidence corrections based on phonetic flags.
        #    include_hitl=True produces escalation entries for spans that
        #    were flagged but couldn't be auto-corrected (e.g. low-confidence
        #    phonetic candidates).
        corr_result = flag.apply_high_confidence_corrections(
            normalized_text, flags_out,
            include_hitl=True,
            use_llm=use_llm_flag,
        )
        final_text = corr_result["corrected_transcript"]
        auto_applied = corr_result["applied"]

        # 4) Build the response schema.
        words = [w for w in re.split(r"\s+", transcript.strip()) if w]

        # flagged_spans: from flag_suspicious output on the normalized text.
        flagged_spans = []
        for f in flags_out:
            idx = f.get("index", 0)
            span_indices = f.get("span_indices") or [idx]
            word = f.get("word", "")
            cands = f.get("candidates", [])
            top_sim = cands[0]["phonetic_similarity"] if cands else 0.0
            flagged_spans.append({
                "text": word,
                "start_index": span_indices[0],
                "end_index": span_indices[-1] + 1,
                "max_suspicion": round(top_sim, 4),
                "reason": f.get("reason", "phonetic"),
            })

        # corrections: from apply_high_confidence_corrections (incl. HITL).
        corrections = []
        for a in auto_applied:
            path = a.get("path", "auto_fix")
            corrections.append({
                "span_text": a.get("original", ""),
                "chosen": a.get("corrected", ""),
                "path": path,
                "confidence": round(a.get("confidence", 0.0), 4),
            })
        # Add HITL-taught alias fixes (clinician-confirmed exact mappings).
        for tf in taught_fixes:
            tf_from = tf.get("from", "")
            if not any(c["span_text"].strip() == tf_from.strip() for c in corrections):
                corrections.append({
                    "span_text": tf_from,
                    "chosen": tf.get("to", ""),
                    "path": "hitl_applied",
                    "confidence": 1.0,
                })
        # Also add drug_normalize fixes that weren't already applied.
        for df in drug_fixes:
            df_from = df.get("from", "")
            if not any(c["span_text"].strip() == df_from.strip() for c in corrections):
                corrections.append({
                    "span_text": df_from,
                    "chosen": df.get("to", ""),
                    "path": "auto_fix",
                    "confidence": 1.0,
                })

        # A span that drug_normalize / HITL already corrected was, by
        # definition, suspicious — surface it in flagged_spans too so the
        # flagging metric credits it (otherwise pre-normalized drugs like
        # 'فولتران'->voltaren vanish before flag_suspicious runs).
        orig_words = [w for w in re.split(r"\s+", transcript.strip()) if w]

        def _locate(span_from: str) -> tuple:
            toks = span_from.split()
            if toks:
                for i in range(len(orig_words) - len(toks) + 1):
                    if orig_words[i:i + len(toks)] == toks:
                        return i, i + len(toks)
            return 0, 1

        covered = {fs["text"].strip() for fs in flagged_spans}
        for fx in list(taught_fixes) + list(drug_fixes):
            frm = (fx.get("from") or "").strip()
            if not frm or any(frm in c or c in frm for c in covered if c):
                continue
            i0, i1 = _locate(frm)
            flagged_spans.append({
                "text": frm,
                "start_index": i0,
                "end_index": i1,
                "max_suspicion": 1.0,
                "reason": "normalized",
            })
            covered.add(frm)

        # retrieval_candidates: for each flag, list the candidates.
        retrieval_candidates = []
        for f in flags_out:
            cands = f.get("candidates", [])
            retrieval_candidates.append({
                "span_text": f.get("word", ""),
                "candidates": [
                    {"term": c["term"], "phonetic_score": round(c["phonetic_similarity"], 4)}
                    for c in cands
                ],
            })

        return {
            "case_id": case_id,
            "original": transcript,
            "corrected": final_text,
            "corrections": corrections,
            "flagged_spans": flagged_spans,
            "retrieval_candidates": retrieval_candidates,
        }

    except Exception as exc:
        print(f"[test-pipeline] error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc), "case_id": case_id})


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

        # 2) Word-level CTC forced alignment of the full transcript.
        # Set DISABLE_ALIGNMENT=1 to skip this entirely — it ONLY produces
        # per-word timestamps (it does NOT affect the transcript text). When
        # disabled, neither the MMS aligner nor the whisper-small timing
        # fallback is ever loaded, which is faster and removes those logs.
        if DISABLE_ALIGNMENT:
            words_aligned = []
        else:
            try:
                words_aligned = _alignment.align_words(session_path, transcript)
            except Exception as exc:
                print(f"[transcribe_debug] alignment failed: {exc!r}")
                words_aligned = []

        # 3) Flag suspicious words (phonetic + optional LLM).
        try:
            flags = _flag.flag_suspicious(transcript, use_llm=use_llm_flag)
        except Exception as exc:
            print(f"[transcribe_debug] flagging failed: {exc!r}")
            flags = []

        # 4) Stitch alignment into each flag so the UI knows where to slice.
        for f in flags:
            idx = f.get("index")
            if isinstance(idx, int) and 0 <= idx < len(words_aligned):
                f["start_s"] = words_aligned[idx].get("start_s")
                f["end_s"] = words_aligned[idx].get("end_s")
                f["alignment_confidence"] = words_aligned[idx].get("confidence", 0.0)
            else:
                f["start_s"] = None
                f["end_s"] = None
                f["alignment_confidence"] = 0.0

        # 5) Auto-apply HIGH-confidence LLM corrections (conf >= 0.90).
        # Surfaced as a separate string so the user can compare to raw.
        try:
            corrected = _flag.apply_high_confidence_corrections(transcript, flags)
        except Exception as exc:
            print(f"[transcribe_debug] auto-correct failed: {exc!r}")
            corrected = {"corrected_transcript": transcript, "applied": [], "threshold": 0.90}

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
        }
    except Exception as exc:
        print(f"[transcribe_debug] error: {exc!r}")
        return JSONResponse(status_code=500, content={"error": str(exc)})


# ---------------------------------------------------------------------------
# /api/transcribe_ab — qualitative A/B test of the two v2 medical LoRA arms.
# ---------------------------------------------------------------------------
#
# Independent from the production ASR mode above. Records your own voice and
# returns BOTH arms' transcripts side by side. Models are loaded lazily and
# cached in app.services.asr_ab the first time this endpoint is hit.


@app.post("/api/transcribe_ab")
async def transcribe_ab(
    audio: UploadFile = File(...),
    language: Optional[str] = Form(None),
    run_pipeline: bool = Form(False),
) -> Dict[str, Any]:
    """Transcribe one clip with both v2 medical LoRA arms (A and B).

    When `run_pipeline` is set, each arm also runs the full downstream
    pipeline (alignment + flagging + auto-correction), so the A/B view can
    show exactly what happens to each model's transcript."""
    # DISABLED: the A/B/C arms load other models (vanilla base + medical LoRA
    # adapters). This service is pinned to the single 900h Gulf Arabic
    # fine-tune, so the A/B test endpoint is turned off.
    return JSONResponse(
        status_code=410,
        content={
            "error": "A/B/C arms are disabled. This service only uses the "
            "900h Gulf Arabic fine-tune. Use /api/transcribe_debug."
        },
    )

    from .services import asr_ab

    session_id, session_path, size = _save_upload(audio)
    effective_lang = language or DEFAULT_LANGUAGE
    print(
        f"[transcribe_ab] session={session_id} type={audio.content_type!r} "
        f"size={size}B lang={effective_lang} pipeline={run_pipeline}"
    )
    if size < 200:
        return JSONResponse(
            status_code=400, content={"error": f"audio file is too small ({size} bytes)"}
        )
    try:
        result = asr_ab.transcribe_ab(
            session_path, language=effective_lang, run_pipeline=run_pipeline
        )
        result["session_id"] = session_id
        result["audio_url"] = f"/api/session_audio/{session_id}"
        return result
    except Exception as exc:
        print(f"[transcribe_ab] error: {exc!r}")
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

    if learned_text:
        # Newly-learned terms must enter the candidate-retrieval dataset.
        flag.invalidate_lexicon_cache()
    return {"ok": True, "learned_text": learned_text, "learned_voices": learned_voices}
