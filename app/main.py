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
import math
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
    audio_verify,
    descriptions,
    forced_score,
    lexicon,
    llm_decide,
    llm_detect,
    tracing,
    voice_match,
)
from .services.correction import MedicalCorrector, LexiconEntry as _LexiconEntry, compact


# Forced-decoding margin: the winning candidate must beat the original
# Whisper output by at least this many nats of avg_logprob to override.
FORCED_MARGIN = 0.05
# Tiebreaker zone: if multiple candidates are within this distance of the
# top, fall back to LLM DECIDE.
FORCED_TIE_BAND = 0.02
# How many lexicon candidates to phoneme-prune to before forced scoring.
FORCED_TOPK = 3


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
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "en")
USE_LLM = os.environ.get("USE_LLM", "1") == "1"

# Audio-retrieval thresholds, calibrated for the CTC phonetic similarity
# scale (normalized Levenshtein over greedy wav2vec2-base-960h transcripts).
# Empirically: same word / different voice -> 0.55-0.85, same word / same
# voice -> 0.85-1.00, different words -> < 0.40. The LLM remains the
# final filter; we just keep candidates loose enough to feed it.
AUDIO_RETRIEVE_THRESHOLD_USER = 0.55
AUDIO_RETRIEVE_THRESHOLD_SEED = 0.45
AUDIO_AUTOFIX_THRESHOLD = 0.85  # short-circuit when very strong USER match

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
    # Optional: words from the original /api/transcribe response. If
    # provided, we use them directly instead of re-running ASR (which would
    # produce different word boundaries because the original used
    # hotwords-biased decoding).
    words: Optional[List[Dict[str, Any]]] = None


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


# ---------------------------------------------------------------------------
# /api/transcribe (JSON)  and  /api/transcribe_stream (NDJSON traces)
# ---------------------------------------------------------------------------


def _run_transcribe_pipeline(
    session_path: Path, session_id: str, language: Optional[str], model_size: str
) -> Dict[str, Any]:
    """The full pipeline. Calls into services that emit trace events via
    `app.services.tracing`, so this function can be run with or without an
    active Tracer."""
    # 1) Whisper free pass — with `hotwords` biasing toward the lexicon.
    #
    # Step 1 of the Whisper-twice architecture. Just by feeding the lexicon
    # as soft hot-words, Whisper recovers most OOV medical terms in one
    # shot (Doliprane, Acitrom, Efferalgan, …). The remaining mishears
    # (rare/new words, accent confusion) are caught by step 2 below.
    lex_entries = lexicon.list_terms()
    hotwords = forced_score.lexicon_to_hotwords(lex_entries)
    tracing.emit("asr.start", {
        "session_id": session_id,
        "audio_path": str(session_path),
        "model": model_size,
        "language": language or "auto",
        "size_bytes": session_path.stat().st_size,
        "hotwords_chars": len(hotwords),
    })
    asr_result = asr.transcribe(
        session_path,
        model_size=model_size,
        language=language,
        hotwords=hotwords or None,
    )
    raw_text = asr_result["text"]
    words = list(asr_result["words"])
    tracing.emit("asr.done", {
        "raw_text": raw_text,
        "language": asr_result["language"],
        "duration_s": asr_result["duration"],
        "words": words,
    })
    print(f"[transcribe] raw text: {raw_text!r}  ({len(words)} tokens)")

    # 2) Voice-DB instant fix: for any single word that exactly matches a
    # learned voice fingerprint, autofix it before forced-scoring runs.
    # This is the "system gets better as users correct it" loop — once a
    # term has a stored fingerprint, repeated audio is replaced for free.
    voice_first_replacements: Dict[int, str] = {}
    if voice_match.list_voices() and words:
        VF_THRESHOLD = 0.80
        for i, tok in enumerate(words):
            start = tok.get("start")
            end = tok.get("end")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                continue
            if end - start < 0.15:
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
            if hits and hits[0].get("source") == "user" and hits[0]["similarity"] >= AUDIO_AUTOFIX_THRESHOLD:
                voice_first_replacements[i] = hits[0]["term"]
    tracing.emit("voice_first.replacements", {"replacements": voice_first_replacements})

    # 3) Suspect detection: text-only (Whisper logprob + OOV vs lexicon).
    # We DO NOT use audio_verify here — its scoring was unreliable on real
    # audio. Suspect detection only triggers FORCED scoring; it doesn't
    # decide anything by itself, so being a bit lenient is safe.
    suspect_indices = forced_score.suspect_word_indices(words, lex_entries)
    # Don't re-score words that voice-DB already locked in.
    suspect_indices = [i for i in suspect_indices if i not in voice_first_replacements]
    tracing.emit("suspect.indices", {
        "indices": suspect_indices,
        "words": [(i, (words[i].get("word") or "").strip(), words[i].get("probability")) for i in suspect_indices],
    })
    print(f"[transcribe] suspect: {[(i, (words[i].get('word') or '').strip()) for i in suspect_indices]}")

    # 4) Forced whole-utterance scoring per suspect span.
    # For each suspect word, build candidate sentences (raw_text with the
    # suspect token replaced by each phoneme-similar lexicon term).
    # Whisper scores all of them at full-utterance scope; whichever
    # sentence beats the raw transcript by FORCED_MARGIN wins.
    forced_replacements: Dict[int, str] = {}
    for i in suspect_indices:
        tok = words[i]
        word_text = (tok.get("word") or "").strip()
        if not word_text:
            continue
        # Phoneme-prune to top-K lexicon candidates.
        candidates = forced_score.prune_candidates(word_text, lex_entries, k=FORCED_TOPK)
        if not candidates:
            continue
        ranked = forced_score.score_whole(
            str(session_path),
            raw_text,
            word_text,
            candidates,
            model_size=model_size,
            language=language or "en",
        )
        tracing.emit("forced.score", {
            "word_index": i,
            "span_text": word_text,
            "ranked": ranked[:5],
        })
        # The first entry of `ranked` is the highest avg_logprob.
        # The "original" entry (is_original=True) is our baseline.
        original_score = next(
            (r["avg_logprob"] for r in ranked if r.get("is_original")),
            float("-inf"),
        )
        non_original = [r for r in ranked if not r.get("is_original")]
        if not non_original:
            continue
        winner = non_original[0]
        margin = winner["avg_logprob"] - original_score
        if margin > FORCED_MARGIN:
            # We picked a NEW transcript. Look up which lexicon term made
            # it win — the substitution we performed is recorded as the
            # `candidate` text. Find the term that's INSIDE winner["candidate"]
            # but NOT in raw_text — that's the replacement.
            chosen_term = None
            for cand_term in candidates:
                if cand_term.lower() in winner["candidate"].lower() and cand_term.lower() not in raw_text.lower():
                    chosen_term = cand_term
                    break
            if chosen_term:
                forced_replacements[i] = chosen_term
                print(f"[transcribe] forced[{i}] {word_text!r} -> {chosen_term!r} (margin={margin:.4f})")

    # 6) Build the suspicious_out list (what the UI shows).
    suspicious_out: List[Dict[str, Any]] = []
    for i, choice in voice_first_replacements.items():
        suspicious_out.append({
            "span": (words[i].get("word") or "").strip(),
            "start_s": float(words[i].get("start") or 0.0),
            "end_s": float(words[i].get("end") or 0.0),
            "reason": "voice_db",
            "auto_via_voice": True,
            "chosen": choice,
        })
    for i, choice in forced_replacements.items():
        if i in voice_first_replacements:
            continue
        suspicious_out.append({
            "span": (words[i].get("word") or "").strip(),
            "start_s": float(words[i].get("start") or 0.0),
            "end_s": float(words[i].get("end") or 0.0),
            "reason": "forced_score",
            "auto_via_voice": False,
            "chosen": choice,
        })

    # 7) Apply word-level replacements (voice DB + forced-scoring winners).
    word_replacements: Dict[int, str] = {**voice_first_replacements, **forced_replacements}
    drop_indices: set = set()

    corrected_text = _apply_word_replacements(words, word_replacements, drop_indices) if words else raw_text
    if not words:
        corrected_text = raw_text

    print(f"[transcribe] applied: {word_replacements}")
    tracing.emit("apply.done", {
        "replacements": word_replacements,
        "corrected_text": corrected_text.strip(),
    })

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
    # Prefer the client-supplied words from the original transcribe call.
    # Re-running ASR here would use different decoding parameters (no
    # hotwords) and produce mismatched word boundaries, which breaks the
    # diff -> slice mapping below.
    if req.words:
        asr_words = list(req.words)
    elif session_path:
        try:
            lex_for_re = forced_score.lexicon_to_hotwords(lexicon.list_terms())
            asr_result = asr.transcribe(
                session_path,
                model_size=DEFAULT_WHISPER_SIZE,
                language=None,
                hotwords=lex_for_re or None,
            )
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
