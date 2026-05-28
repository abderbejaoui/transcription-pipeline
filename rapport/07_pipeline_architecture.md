# 07 — Runtime Pipeline Architecture

This chapter describes the **live system** that wraps the fine-tuned
ASR model. The model alone is not the product; the product is the
end-to-end pipeline that turns a microphone recording into a clinically
usable transcript.

## 7.1 High-level diagram

```
  ┌──────────────────────────┐
  │ Browser recorder UI      │   app/static/{index.html,app.js,styles.css}
  │ (4-tab: Corrected /      │   v=8
  │  Raw / Flags / Pipeline) │
  └─────────────┬────────────┘
                │ POST /api/transcribe_debug   (multipart audio)
                ▼
  ┌──────────────────────────┐
  │ FastAPI service          │   app/main.py
  │ Routes:                  │
  │   /api/transcribe        │
  │   /api/transcribe_debug  │
  │   /api/session_audio/:id │
  └─────────────┬────────────┘
                │
                ▼
  ┌──────────────────────────────────────────────┐
  │ Phase A — Acoustic decode                    │
  │ app/services/asr.py                          │
  │   • ffmpeg → 16 kHz mono wav                 │
  │   • qwen_asr wrapper (Qwen3-ASR + LoRA)      │
  │   • optional context biasing from            │
  │     medical_terms.txt                        │
  │   • Whisper fallback if Qwen unavailable     │
  └─────────────┬────────────────────────────────┘
                │  raw_transcript
                ▼
  ┌──────────────────────────────────────────────┐
  │ Phase B — Forced alignment (word timestamps) │
  │ app/services/alignment_v2.py                 │
  │   • MahmoudAshraf97/ctc-forced-aligner       │
  │     (MMS-1B based)                           │
  │   • Whisper-aligned v1 fallback              │
  │   produces: word + start_s + end_s           │
  └─────────────┬────────────────────────────────┘
                │  aligned_words
                ▼
  ┌──────────────────────────────────────────────┐
  │ Phase C — Phonetic flagging                  │
  │ app/services/flag.py                         │
  │   • Build phonetic candidates per word from  │
  │     medical_terms.txt (~196 terms)           │
  │   • bigram → trigram → single n-gram sweep   │
  │   • LCS-3 precision filter for 0.55-0.65     │
  │     similarity range                         │
  │   • drug-vs-disease tiebreaker, name lexicon │
  │   • output: flags[] each with span,          │
  │     candidate, score, evidence               │
  └─────────────┬────────────────────────────────┘
                │  flags
                ▼
  ┌──────────────────────────────────────────────┐
  │ Phase D — Optional LLM judge (dual-ASR)      │
  │ gated by USE_DUAL_ASR env flag                │
  │   • runs a second ASR (Whisper or Voxtral)   │
  │   • Calme LLM picks between the two outputs  │
  │     given the medical context                │
  └─────────────┬────────────────────────────────┘
                │  candidates
                ▼
  ┌──────────────────────────────────────────────┐
  │ Phase E — High-confidence auto-correction    │
  │ flag.apply_high_confidence_corrections()     │
  │   • Apply phonetic correction when           │
  │     similarity ≥ 0.85 AND term in lexicon    │
  │   • Otherwise leave as-is and surface as a   │
  │     flag for the doctor to confirm           │
  └─────────────┬────────────────────────────────┘
                │  corrected_transcript + flags + audio slices
                ▼
  ┌──────────────────────────────────────────────┐
  │ JSON response → UI                           │
  │ • Tab 1 Corrected                            │
  │ • Tab 2 Raw                                  │
  │ • Tab 3 Flags (audio playback per flagged    │
  │   word using MMS-aligned spans)              │
  │ • Tab 4 Pipeline (timing + provenance)       │
  └──────────────────────────────────────────────┘
```

## 7.2 Phase A — Acoustic decode (`app/services/asr.py`)

Responsibilities:
- Convert any incoming audio to 16 kHz mono PCM via ffmpeg.
- Load the Qwen3-ASR model and our LoRA adapter once at process start.
- Call the `qwen_asr` wrapper for inference.
- Optionally feed `medical_terms.txt` as context biasing (Qwen3-ASR
  supports a prompt-style hint).
- Fall back to Whisper if the Qwen model fails to load (e.g. version
  mismatch).

Dependencies pinned at the time of writing:
- `transformers == 4.57.6`
- `huggingface-hub == 0.36.2`
- newer versions break the `qwen_asr` wrapper used here.

## 7.3 Phase B — Forced alignment (`app/services/alignment_v2.py`)

Why we align: the UI needs to play back the audio for any individual
flagged word so the doctor can hear exactly what the patient said.
That requires word-level start/end timestamps.

Qwen3-ASR is generative and does not produce timestamps. We post-align
with **MMS-based CTC forced alignment**
(`MahmoudAshraf97/ctc-forced-aligner`, installed from GitHub — not the
PyPI variant which is an older project of the same name).

If MMS fails (rare on aarch64), v1 Whisper-based alignment is the
fallback. It is less precise (word boundaries within ~200 ms) but
keeps the UI functional.

Output:
```json
[
  {"word": "paracetamol", "start_s": 1.23, "end_s": 1.87},
  ...
]
```

## 7.4 Phase C — Phonetic flagging (`app/services/flag.py`)

Detailed in [08_correction_layer.md](08_correction_layer.md). One-line
summary: for each consecutive 1/2/3-word window in the transcript,
compute a phonetic-skeleton similarity against every term in the
medical lexicon and surface the best match if its score crosses a
calibrated threshold.

The flagger is intentionally **dumb and fast** — it does not call any
LLM. It is the deterministic safety net under the ASR.

## 7.5 Phase D — Optional dual-ASR with LLM judge

Gated by environment variable `USE_DUAL_ASR=1`. When enabled, a second
ASR backend (Whisper-large-v3 or Voxtral-Mini-3B) runs on the same
audio. Calme then picks between the two transcripts taking the
preceding clinical context into account.

This catches the case where Qwen3-ASR's LoRA adapter goes off the
rails because the audio is far enough from its training distribution,
but the second model — which has different inductive biases — gets it
right.

Off by default because it doubles compute. Turned on for batch
processing and dictation review, off for live real-time dictation.

## 7.6 Phase E — High-confidence auto-correction

The flagger returns candidates and scores. The system auto-applies a
correction only when **both** of these hold:

1. Phonetic similarity ≥ 0.85 to a known lexicon term.
2. Any LLM-suggested replacement (from Phase D) is also in the
   medical lexicon.

Otherwise the original word is kept and the flag is surfaced in the
UI for one-click acceptance by the doctor.

The rule is asymmetric on purpose: false positives (auto-replacing a
correct word with a wrong one) are clinically worse than missing a
correction (which the doctor still sees as a flag and can accept).

## 7.7 UI (`app/static/`)

Single-page recorder with four tabs:

| Tab | Shows |
|---|---|
| Corrected | Final transcript with flag indicators inline. One-click apply / reject per flag. |
| Raw       | Pure Qwen3-ASR output before any correction. |
| Flags     | List of flagged spans, each with phonetic alternatives and audio playback (uses the MMS-aligned span). |
| Pipeline  | Per-phase timing, model versions, decoded prompts — for debugging. |

Cached at `v=8` (the static assets carry a versioned query string so
browsers don't serve stale JS after a deploy).

## 7.8 What runs where

| Component | Process | Hardware |
|---|---|---|
| FastAPI app | uvicorn on port 8000 | DGX Spark |
| Qwen3-ASR + LoRA | in-process | DGX GPU |
| MMS aligner | in-process | DGX GPU |
| Whisper fallback | in-process | DGX GPU |
| Calme LLM (optional) | Ollama daemon on port 11434 | DGX GPU |
| VoxCPM2 TTS (training only) | Uvicorn on port 7900 | DGX GPU |

## 7.9 Endpoints

| Path | Verb | Purpose |
|---|---|---|
| `/api/transcribe` | POST (audio) | minimal: raw_transcript, corrected_transcript |
| `/api/transcribe_debug` | POST (audio) | full JSON for the 4-tab UI |
| `/api/session_audio/{id}` | GET | serve uploaded audio for flag playback |
| `/healthz` | GET | liveness |

## 7.10 Observability

Each request gets a request-id and a structured log line per phase:

```
[req=abc] phaseA asr.qwen3 dur=1.34s output_chars=84
[req=abc] phaseB align.mms words=14 dur=0.41s
[req=abc] phaseC flag.phonetic flags=2 dur=0.05s
[req=abc] phaseD dual_asr disabled
[req=abc] phaseE corrections_applied=1 dur=0.01s
```

These power the Pipeline tab in the UI and feed Prometheus counters
that the deployment dashboard reads.
