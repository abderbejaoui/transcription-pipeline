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

from .services import asr, asr_dual, descriptions, lexicon, llm_decide, llm_detect, tracing, voice_match

try:
    from .services import asr_benchmark  # optional: only used by /api/benchmark_asr
except ImportError:
    asr_benchmark = None
from .services.correction import MedicalCorrector, LexiconEntry as _LexiconEntry, compact, _has_arabic
from .services.flag import _is_arabic_filler, _clear_lexicon_skeleton_cache
from .services.arabic_matcher import HybridMatcher, LLMOpenCorrector


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
    # ASR model is loaded lazily on first /api/transcribe request.
    # Do NOT try to pre-load it here — the transformers C extensions
    # can segfault (DLL init failure on Windows) which kills the
    # entire server process and cannot be caught with try/except.
    #
    # Voice-match warmup uses safe numpy/scipy operations:
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
    # Pre-cache description so the next /transcribe DECIDE has it.
    threading.Thread(
        target=lambda: descriptions.get_or_generate(req.term, type_hint=req.type),
        daemon=True,
    ).start()
    # Invalidate the cached corrector and hybrid matcher so the next
    # /api/correct call picks up the newly added term.
    global _TEXT_CORRECTOR, _HYBRID_MATCHER
    _TEXT_CORRECTOR = None
    _HYBRID_MATCHER = None
    # Invalidate the flag.py lexicon skeleton cache so auto-normalcy
    # detection picks up the new term.
    _clear_lexicon_skeleton_cache()
    return {"ok": True, "entry": entry}


class CorrectRequest(BaseModel):
    text: str = Field(..., description="Raw text without audio.")


# Cached corrector for text-only correction path.
_TEXT_CORRECTOR: Optional[MedicalCorrector] = None
_HYBRID_MATCHER: Optional[HybridMatcher] = None
_LLM_OPEN_CORRECTOR: Optional[LLMOpenCorrector] = None


def _get_text_corrector() -> MedicalCorrector:
    global _TEXT_CORRECTOR
    if _TEXT_CORRECTOR is None:
        _TEXT_CORRECTOR = _build_corrector()
    return _TEXT_CORRECTOR


def _get_hybrid_matcher() -> HybridMatcher:
    """Lazy-build the hybrid matcher (Stages 2-3: skeleton + embedding).

    Embedding matching (LaBSE) is DISABLED because it produces many
    false positives for Arabic text: LaBSE matches Arabic words to
    English medical terms via cross-lingual semantic similarity
    (e.g. 'قلب' → 'cardiac'), which is the OPPOSITE of what we want.
    We want to match Arabic TRANSLITERATIONS (هستوري → history) not
    Arabic TRANSLATIONS (قلب → cardiac). The skeleton matcher handles
    transliteration matching correctly.
    """
    global _HYBRID_MATCHER
    if _HYBRID_MATCHER is None:
        raw = lexicon.list_terms()
        _HYBRID_MATCHER = HybridMatcher(
            lexicon=raw,
            enable_embedding=False,
            # LLM open correction is handled separately
            enable_llm_open=False,
        )
    return _HYBRID_MATCHER


def _get_llm_open_corrector() -> LLMOpenCorrector:
    """Lazy-build the LLM open corrector (Stage 4)."""
    global _LLM_OPEN_CORRECTOR
    if _LLM_OPEN_CORRECTOR is None:
        _LLM_OPEN_CORRECTOR = LLMOpenCorrector(confidence_threshold=0.60)
    return _LLM_OPEN_CORRECTOR


def _build_scored_words(
    transcript: str,
    flags: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build scored_words array for the UI from the raw transcript and flags."""
    tokens = re.split(r"\s+", transcript.strip())
    flag_map: Dict[str, Dict[str, Any]] = {}
    for f in flags:
        fw = f.get("word", "")
        if fw:
            flag_map[fw] = f

    scored: List[Dict[str, Any]] = []
    for idx, token in enumerate(tokens):
        if not token:
            continue
        flag = flag_map.get(token)
        suspicion = 0.0
        if flag:
            candidates = flag.get("candidates", [])
            if candidates:
                suspicion = min(1.0, candidates[0].get("phonetic_similarity", 0.6))
            else:
                suspicion = 0.7
        in_lexicon = suspicion < 0.3 and not flag
        scored.append({
            "text": token,
            "index": idx,
            "suspicion": round(suspicion, 4),
            "in_lexicon": in_lexicon,
        })
    return scored


def _build_spans(flags: List[Dict[str, Any]], tokens: List[str]) -> List[Dict[str, Any]]:
    """Build spans array for the UI from flags, mapping each flag to its token indices."""
    spans: List[Dict[str, Any]] = []
    for f in flags:
        word = f.get("word", "")
        candidates = f.get("candidates", [])
        score = 0.6
        if candidates:
            score = min(1.0, candidates[0].get("phonetic_similarity", 0.6))

        # Map the flag word back to its token indices in the original text
        word_parts = word.split()
        start_idx = -1
        end_idx = -1
        for i in range(len(tokens) - len(word_parts) + 1):
            if tokens[i:i + len(word_parts)] == word_parts:
                start_idx = i
                end_idx = i + len(word_parts) - 1
                break

        if start_idx == -1:
            # Fallback: use position in flags array
            start_idx = len(spans)
            end_idx = len(spans)

        spans.append({
            "text": word,
            "start": start_idx,
            "end": end_idx,
            "suspicion": round(score, 4),
            "reason": f.get("reason", "both"),
        })
    return spans


def _build_candidates_list(flags: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build candidates_list grouped by span for the UI."""
    candidates_list: List[Dict[str, Any]] = []
    for f in flags:
        word = f.get("word", "")
        raw_candidates = f.get("candidates", [])
        cands = [
            {
                "term": c.get("term", ""),
                "phonetic_score": min(1.0, c.get("phonetic_similarity", 0.5)),
                "term_type": c.get("match_type", f.get("reason", "")),
                "source": c.get("match_type", "pipeline"),
                "description": "",
            }
            for c in raw_candidates
        ]
        candidates_list.append({
            "span": {"text": word},
            "candidates": cands,
        })
    return candidates_list


def _build_decisions(
    flags: List[Dict[str, Any]],
    auto_corrections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build decisions array from flags + auto_corrections."""
    # Map original word -> corrected word
    auto_map: Dict[str, str] = {}
    for ac in auto_corrections:
        auto_map[ac["original"]] = ac["corrected"]

    decisions: List[Dict[str, Any]] = []
    for f in flags:
        word = f.get("word", "")
        candidates = f.get("candidates", [])
        corrected = auto_map.get(word)
        if corrected:
            decisions.append({
                "span": {"text": word},
                "chosen": corrected,
                "path": "auto_fix",
                "confidence": 1.0,
            })
        else:
            top = candidates[0]["term"] if candidates else None
            decisions.append({
                "span": {"text": word},
                "chosen": top,
                "path": top and "auto_fix" or "hitl",
                "confidence": top and (candidates[0].get("phonetic_similarity", 0.6)) or 0.0,
            })
    return decisions


def _find_uncorrected_spans(
    transcript: str,
    corrected_text: str,
    existing_flags: List[Dict[str, Any]],
) -> List[str]:
    """Find words/phrases in the transcript that weren't changed and
    weren't already flagged — these are candidates for Stage 4 LLM
    open correction.

    Arabic-script words are SKIPPED entirely because:
      1. Genuine Arabic medical transliterations are already caught by
         Stage 1 (MedicalCorrector with Arabic skeleton matching).
      2. Normal Arabic words (filler, context, anatomy terms) when sent
         to an English-biased LLM produce hallucinated English medical
         terms (e.g. 'المريض' -> 'amaryl', 'لسان' -> 'lasix').
      3. The Arabic filler list catches common non-medical words, but
         there will always be Arabic words NOT in the filler list that
         are still not medical transliterations (e.g. 'لاحظنا', 'يمتد',
         'بينت'). Sending them to the LLM is harmful.
    """
    if transcript.strip() == corrected_text.strip():
        # Nothing changed; find suspicious-looking words (English only)
        words = [w for w in re.split(r"\s+", transcript.strip()) if w]
        already_flagged = set()
        for f in existing_flags:
            already_flagged.add(f.get("word", "").lower())
        return [
            w for w in words
            if w.lower() not in already_flagged
            and len(w) >= 4
            and not _has_arabic(w)  # Skip Arabic-script words entirely
        ]
    return []


@app.post("/api/correct")
def correct_text_only(req: CorrectRequest) -> Dict[str, Any]:
    """Multi-stage text-only correction pipeline.

    Stage 1 — MedicalCorrector: deterministic fuzzy + phonetic matching
               (now Arabic-aware via transliteration + consonant skeletons).
    Stage 2 — SkeletonMatcher: supplementary Arabic→Latin skeleton matching
               for Arabic spans the MedicalCorrector missed.
    Stage 3 — EmbeddingMatcher: LaBSE multilingual similarity (if available).
    Stage 4 — LLM Open Correction: final-resort LLM correction for terms
               the lexicon doesn't cover.
    """
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")

    # ------------------------------------------------------------------
    # Stage 1: Deterministic correction via MedicalCorrector
    # ------------------------------------------------------------------
    corrector = _get_text_corrector()
    result = corrector.correct_transcript(req.text)
    corrected_text = result["corrected_text"]
    suspicious = list(result["suspicious_spans"])

    # Track which original spans were corrected (for dedup)
    corrected_originals = {}
    for s in suspicious:
        orig = s.get("original_text", "")
        corr = s.get("possible_correction", "")
        if orig and corr:
            corrected_originals[orig] = corr

    # ------------------------------------------------------------------
    # Build initial flags + auto_corrections from Stage 1
    # ------------------------------------------------------------------
    flags: List[Dict[str, Any]] = []
    auto_corrections: List[Dict[str, Any]] = []
    for s in suspicious:
        original = s.get("original_text", "")
        correction = s.get("possible_correction", "")
        score = s.get("score", 0.0)
        issue_type = s.get("issue_type", "misspelling")
        flags.append({
            "index": 0,
            "word": original,
            "reason": issue_type,
            "candidates": [{
                "term": correction,
                "phonetic_similarity": score / 100.0,
                "match_type": "deterministic",
            }] if correction else [],
            "start_s": None,
            "end_s": None,
        })
        if correction:
            auto_corrections.append({
                "original": original,
                "corrected": correction,
                "type": issue_type,
            })

    # ------------------------------------------------------------------
    # Stage 2-3: Hybrid matcher for additional Arabic/non-Latin spans
    # that MedicalCorrector might have missed (e.g. terms not in lexicon
    # but phonetically close via skeleton matching).
    # ------------------------------------------------------------------
    if USE_LLM:
        # Collect Arabic-script tokens from the original text that weren't corrected
        words = re.split(r"\s+", req.text.strip())
        missed_arabic: List[str] = []
        for w in words:
            if not _has_arabic(w):
                continue
            # Strip leading/trailing punctuation that defeats filler detection
            # (e.g. "دكتور،" with comma won't match the filler "دكتور")
            clean_w = re.sub(r"^[\s\u060c\u061b\u061f!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+", "", w)
            clean_w = re.sub(r"[\s\u060c\u061b\u061f!\"#$%&'()*+,\-./:;<=>?@\[\]^_`{|}~]+$", "", clean_w)
            if not clean_w or not _has_arabic(clean_w):
                continue
            is_corrected = any(
                orig in clean_w or clean_w in orig for orig in corrected_originals
            )
            if not is_corrected and len(clean_w) >= 3:
                missed_arabic.append(clean_w)

        if missed_arabic:
            try:
                hybrid = _get_hybrid_matcher()
                # Pre-filter: skip Arabic filler words (words from common
                # Gulf Arabic vocabulary that happen to look like they
                # could be English medical transliterations via skeleton).
                filtered_arabic = [
                    w for w in missed_arabic
                    if not _is_arabic_filler(w)
                ]
                for w in filtered_arabic:
                    candidates = hybrid.match(w, top_k=3, context=req.text)
                    if candidates:
                        # Check if already flagged
                        already_flagged = any(
                            f.get("word", "") == w for f in flags
                        )
                        if not already_flagged:
                            flags.append({
                                "index": 0,
                                "word": w,
                                "reason": "arabic_phonetic_match",
                                "candidates": [
                                    {
                                        "term": c["term"],
                                        "phonetic_similarity": c["score"] / 100.0,
                                        "match_type": c.get("match_type", "hybrid"),
                                    }
                                    for c in candidates
                                ],
                                "start_s": None,
                                "end_s": None,
                            })
                            # Apply high-confidence auto-correction
                            top = candidates[0]
                            if top["score"] >= 80.0:
                                corrected_text = corrected_text.replace(w, top["term"])
                                auto_corrections.append({
                                    "original": w,
                                    "corrected": top["term"],
                                })
            except Exception as exc:
                print(f"[correct] Hybrid matcher failed: {exc!r}")

    # ------------------------------------------------------------------
    # Stage 4: LLM Open Correction for remaining uncorrected flags
    # ------------------------------------------------------------------
    if USE_LLM:
        try:
            uncorrected = _find_uncorrected_spans(
                req.text, corrected_text, flags
            )
            if uncorrected:
                llm_corr = _get_llm_open_corrector()
                results = llm_corr.correct_batch(
                    uncorrected, context=req.text, timeout=60.0
                )
                for w, r in zip(uncorrected, results):
                    if r is None:
                        continue
                    already_flagged = any(
                        f.get("word", "") == w for f in flags
                    )
                    if not already_flagged:
                        flags.append({
                            "index": 0,
                            "word": w,
                            "reason": "llm_open_correction",
                            "candidates": [{
                                "term": r["term"],
                                "phonetic_similarity": r["score"] / 100.0,
                                "match_type": r.get("match_type", "llm_open"),
                                "confidence": r.get("confidence", 0.0),
                                "reason": r.get("reason", ""),
                            }],
                            "start_s": None,
                            "end_s": None,
                        })
                        # Apply to corrected_text if confident
                        if r["score"] >= 80.0:
                            corrected_text = corrected_text.replace(w, r["term"])
                            auto_corrections.append({
                                "original": w,
                                "corrected": r["term"],
                            })
        except Exception as exc:
            print(f"[correct] LLM open correction failed: {exc!r}")

    # ------------------------------------------------------------------
    # Optional LLM DETECT pass for additional flagging (existing behavior)
    # ------------------------------------------------------------------
    if USE_LLM:
        fake_words = []
        for tok in req.text.split():
            fake_words.append({
                "word": " " + tok,
                "start": 0.0,
                "end": 0.0,
                "probability": 1.0,
            })
        try:
            llm_spans = llm_detect.detect(fake_words)
            for span in llm_spans:
                text = span.get("text", "")
                if not text:
                    continue
                # Skip Arabic-script spans — the LLM detect is biased
                # toward flagging and will hallucinate English medical
                # terms for normal Arabic context words (e.g. 'لاحظنا'
                # -> 'losartan', 'السكر' -> 'saccharin').
                # Arabic medical transliterations are caught by earlier
                # deterministic stages.
                if _has_arabic(text):
                    continue
                already_flagged = any(
                    f.get("word", "").lower() == text.lower()
                    or text.lower() in f.get("word", "").lower()
                    or f.get("word", "").lower() in text.lower()
                    for f in flags
                )
                if not already_flagged:
                    flags.append({
                        "index": 0,
                        "word": text,
                        "reason": span.get("reason", "llm_flag"),
                        "candidates": [],
                        "start_s": None,
                        "end_s": None,
                    })
        except Exception as exc:
            print(f"[correct] LLM DETECT failed: {exc!r}")

    # ------------------------------------------------------------------
    # Build structured pipeline stages for the UI
    # ------------------------------------------------------------------
    tokens = re.split(r"\s+", req.text.strip())
    pipeline_scored_words = _build_scored_words(req.text, flags)
    pipeline_spans = _build_spans(flags, tokens)
    pipeline_candidates_list = _build_candidates_list(flags)
    pipeline_decisions = _build_decisions(flags, auto_corrections)

    pipeline = {
        "approaches": {
            "scoring": {
                "label": "deterministic",
                "description": "Lexicon-based fuzzy + phonetic matching",
                "status": "primary",
            },
            "flagging": {
                "label": "phonetic+skeleton",
                "description": "Phonetic similarity + Arabic consonant skeleton matching",
                "status": "primary",
            },
            "retrieval": {
                "label": "deterministic",
                "description": "Lexicon lookup + skeleton + embedding matcher",
                "status": "primary",
            },
            "decision": {
                "label": "auto_fix",
                "description": "High-confidence corrections auto-applied",
                "status": "primary",
            },
            "correction": {
                "label": "deterministic",
                "description": "Multi-stage pipeline correction",
                "status": "primary",
            },
        },
        "stages": [
            {"scored_words": pipeline_scored_words},
            {"spans": pipeline_spans},
            {"candidates_list": pipeline_candidates_list},
            {"decisions": pipeline_decisions},
            {
                "corrected_text": corrected_text,
                "original_text": req.text,
                "n_applied": len(auto_corrections),
            },
        ],
    }

    return {
        "raw_text": req.text,
        "corrected_text": corrected_text,
        "suspicious": suspicious,
        "flags": flags,
        "auto_corrections": auto_corrections,
        "note": "text-only correction (no audio) — multi-stage pipeline",
        "pipeline": pipeline,
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
    if asr_benchmark is None:
        return JSONResponse(
            {"error": "asr_benchmark is not available in this deployment."},
            status_code=501,
        )
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

from .services import alignment_v2 as _alignment, flag as _flag


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
    # Invalidate the cached corrector and hybrid matcher — this endpoint
    # adds terms to the lexicon just like /api/teach does.
    global _TEXT_CORRECTOR, _HYBRID_MATCHER
    _TEXT_CORRECTOR = None
    _HYBRID_MATCHER = None
    # Also invalidate the flag.py lexicon skeleton cache
    _clear_lexicon_skeleton_cache()

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
