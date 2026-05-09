"""Build the Gulf-medical ASR eval set — ~100 audio clips with verified
ground-truth transcripts, organised by category.

Output structure:
    eval/gulf_medical_v1/
        manifest.jsonl              one JSON record per clip
        audio/
            <category>_<id>.wav     16 kHz mono WAV

Categories
----------
saudi_acoustic     30 clips  -- Common Voice Arabic (transcribed by Mozilla)
medical_vocab_ar   30 clips  -- Arabic medical sentences with Gulf drug names,
                                synthesised via macOS `say -v Majed`
code_switching     20 clips  -- AR↔EN code-switched medical sentences,
                                synthesised
english_medical    20 clips  -- existing patient-narrative recordings from
                                Medical-Audio-Transcription/

The manifest record schema:
    {
      "id": "medical_vocab_ar_007",
      "category": "medical_vocab_ar",
      "audio_path": "audio/medical_vocab_ar_007.wav",
      "duration_s": 4.12,
      "language": "ar",                 # 'ar' | 'en' | 'mixed'
      "transcript": "...",              # raw verified transcript
      "transcript_normalized": "...",   # for WER computation
      "medical_terms": ["doliprane"],   # canonical names for recall metric
      "source": "common_voice_17_0|tts_say_majed|local_medical|...",
      "tags": ["sada-substitute","drug-name"]
    }

Run:
    source .venv/bin/activate
    python -m scripts.build_eval_set
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "eval" / "gulf_medical_v1"
AUDIO_DIR = EVAL_DIR / "audio"
MANIFEST_PATH = EVAL_DIR / "manifest.jsonl"

LOCAL_MED_DIR = Path(
    "/Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/audio_cache_preprocessed_10"
)
LOCAL_MED_META = Path(
    "/Users/abderrahmenbejaoui/Medical-Audio-Transcription/data/metadata.csv"
)

SR = 16_000


# ---------------------------------------------------------------------------
# Text normalization (used for WER ground truth)
# ---------------------------------------------------------------------------

# Arabic diacritics to strip
_AR_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u0640]")
# Punctuation we always strip
_PUNCT_RE = re.compile(r"[\u060C\u061B\u061F.,;:!?\"'()\[\]{}\-–—_/\\@#%&*+=<>]")
# Multiple spaces
_WS_RE = re.compile(r"\s+")
# Latin char + Arabic char boundary -> insert space
_BOUNDARY_RE = re.compile(r"(?<=[A-Za-z0-9])(?=[\u0600-\u06FF])|(?<=[\u0600-\u06FF])(?=[A-Za-z0-9])")
# Arabic-Indic digits to ASCII
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def normalize_text(text: str) -> str:
    """Normalize a transcript so two surface forms compare cleanly:
    - Unicode NFKC
    - strip Arabic diacritics + tatweel
    - unify Arabic-Indic and ASCII digits
    - lowercase Latin letters
    - collapse whitespace, drop punctuation
    - insert space between AR↔Latin transitions
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_DIGIT_MAP)
    text = _AR_DIACRITICS_RE.sub("", text)
    text = _BOUNDARY_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    text = text.lower()
    text = _WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# TTS via macOS `say` -> WAV
# ---------------------------------------------------------------------------


def synth_say(text: str, voice: str, out_wav: Path) -> float:
    """Render `text` as a 16 kHz mono WAV using macOS `say`. Returns duration."""
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    aiff = out_wav.with_suffix(".aiff")
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run([
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(aiff),
        "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(out_wav),
    ], check=True)
    aiff.unlink(missing_ok=True)
    # Get duration via ffprobe
    out = subprocess.run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(out_wav),
    ], capture_output=True, text=True, check=True).stdout.strip()
    return float(out) if out else 0.0


# ---------------------------------------------------------------------------
# Slice 1: Saudi acoustic — 30 general-Arabic clips
# ---------------------------------------------------------------------------

# Try to pull from Common Voice ar first; if that fails (gated, script
# loaders deprecated, etc.), synthesise from a list of general-Arabic
# sentences. The TTS path is a controlled substitute — it tests the model's
# Arabic acoustic ability without leaking medical vocabulary into the
# saudi_acoustic category.
GENERAL_ARABIC_SENTENCES: List[str] = [
    "الجو اليوم حار جدا في المدينة.",
    "أحب أن أشرب القهوة في الصباح الباكر.",
    "ذهبنا إلى السوق لشراء بعض الفواكه.",
    "الطفل يلعب في الحديقة مع أصدقائه.",
    "السيارة الجديدة سريعة جدا وآمنة.",
    "أعمل في شركة كبيرة منذ خمس سنوات.",
    "البحر هادئ والشاطئ جميل اليوم.",
    "نحن ذاهبون لزيارة الأقارب في الكويت.",
    "الكتاب على الطاولة بجانب القلم الأزرق.",
    "ساعة الصباح هي أفضل وقت للدراسة.",
    "وصلنا إلى المطار قبل الموعد بساعتين.",
    "الطعام في هذا المطعم لذيذ جدا.",
    "تأخر القطار اليوم بسبب العطل الفني.",
    "أحببت الفيلم الذي شاهدناه في السينما.",
    "الجامعة تبعد عشرين دقيقة عن البيت.",
    "الإنترنت بطيء جدا في هذا الفندق.",
    "البائع طلب مني خمسين ريالا فقط.",
    "أمي تطبخ أفضل أكلة في العالم.",
    "السوبر ماركت يفتح من الساعة الثامنة.",
    "نسيت محفظتي في السيارة هذا الصباح.",
    "عيد الفطر يأتي بعد أسبوعين.",
    "لا تنس أن تأخذ المظلة معك.",
    "الطقس في الإمارات معتدل في الشتاء.",
    "أصدقائي يأتون لزيارتي يوم الجمعة.",
    "البطارية تقريبا فارغة، أحتاج إلى شاحن.",
    "الحديقة العامة كانت مزدحمة أمس.",
    "حضرنا حفل زفاف ابنة عمي البارحة.",
    "اشتريت هاتفا جديدا من المعرض.",
    "هل يمكنك أن تساعدني في حمل الحقيبة؟",
    "لا أحب الحلويات لكنني أحب البسكويت.",
]


def _build_saudi_synth(n: int) -> List[Dict[str, Any]]:
    """Synthesise n general-Arabic sentences via macOS say -v Majed."""
    print(f"  → synthesising {n} general-Arabic clips with say -v Majed (substitute) ...")
    out: List[Dict[str, Any]] = []
    for i, text in enumerate(GENERAL_ARABIC_SENTENCES[:n], 1):
        clip_id = f"saudi_acoustic_{i:03d}"
        wav = AUDIO_DIR / f"{clip_id}.wav"
        try:
            dur = synth_say(text, voice="Majed", out_wav=wav)
        except subprocess.CalledProcessError as exc:
            print(f"    ! TTS failed for clip {i}: {exc}")
            continue
        out.append({
            "id": clip_id,
            "category": "saudi_acoustic",
            "audio_path": f"audio/{clip_id}.wav",
            "duration_s": round(dur, 3),
            "language": "ar",
            "transcript": text,
            "transcript_normalized": normalize_text(text),
            "medical_terms": [],
            "source": "tts_say_majed",
            "tags": ["arabic-general", "synthetic", "saudi-acoustic-substitute"],
        })
        print(f"    ✓ {clip_id} ({dur:.1f}s)")
    return out


def build_saudi_acoustic(n: int = 30) -> List[Dict[str, Any]]:
    """Pull `n` clips from MohamedRashad/common-voice-18-arabic via direct
    parquet streaming (HTTP range requests, no full download). Each row has
    embedded audio bytes (mp3) + a verified Mozilla-validated transcript.

    Falls back to TTS if the parquet stream fails.
    """
    print(f"\n[saudi_acoustic] streaming common-voice-18-arabic, picking {n} clips ...")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("  ! HF_TOKEN not set; falling back to TTS substitute")
        return _build_saudi_synth(n)

    try:
        import fsspec
        import pyarrow.parquet as pq
    except ImportError:
        print("  ! pyarrow / fsspec not available; falling back to TTS substitute")
        return _build_saudi_synth(n)

    fs = fsspec.filesystem("hf", token=token)
    parquet_path = (
        "datasets/MohamedRashad/common-voice-18-arabic/data/test-00000-of-00001.parquet"
    )

    try:
        f = fs.open(parquet_path, "rb")
        pqf = pq.ParquetFile(f)
    except Exception as exc:
        print(f"  ! could not open parquet: {str(exc)[:200]}")
        print("  ! falling back to TTS substitute")
        return _build_saudi_synth(n)

    out: List[Dict[str, Any]] = []
    rg_idx = 0
    rows_seen = 0
    rows_kept = 0
    while len(out) < n and rg_idx < pqf.num_row_groups:
        try:
            t = pqf.read_row_group(rg_idx, columns=["audio", "sentence"])
        except Exception as exc:
            print(f"  ! read_row_group({rg_idx}) failed: {str(exc)[:160]}")
            break
        rows = t.to_pydict()
        for audio_obj, sentence in zip(rows["audio"], rows["sentence"]):
            if len(out) >= n:
                break
            rows_seen += 1
            if not sentence or not isinstance(sentence, str) or not sentence.strip():
                continue
            audio_bytes = audio_obj.get("bytes") if isinstance(audio_obj, dict) else None
            if not audio_bytes:
                continue

            idx = len(out) + 1
            clip_id = f"saudi_acoustic_{idx:03d}"
            wav_path = AUDIO_DIR / f"{clip_id}.wav"
            wav_path.parent.mkdir(parents=True, exist_ok=True)

            # Decode the embedded mp3/ogg via ffmpeg piping
            try:
                proc = subprocess.run(
                    [
                        "ffmpeg",
                        "-nostdin", "-hide_banner", "-loglevel", "error",
                        "-y", "-i", "pipe:0",
                        "-ac", "1", "-ar", str(SR),
                        "-c:a", "pcm_s16le",
                        str(wav_path),
                    ],
                    input=audio_bytes,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as exc:
                # Skip clips that fail to decode
                if wav_path.exists():
                    wav_path.unlink()
                continue

            # Measure duration; reject too-short or too-long for stable WER
            try:
                import soundfile as sf
                info = sf.info(wav_path)
                dur = info.frames / info.samplerate
            except Exception:
                dur = 0.0
            if dur < 2.0 or dur > 12.0:
                wav_path.unlink()
                continue

            out.append({
                "id": clip_id,
                "category": "saudi_acoustic",
                "audio_path": f"audio/{clip_id}.wav",
                "duration_s": round(dur, 3),
                "language": "ar",
                "transcript": sentence.strip(),
                "transcript_normalized": normalize_text(sentence),
                "medical_terms": [],
                "source": "MohamedRashad/common-voice-18-arabic",
                "tags": ["common-voice-18", "validated"],
            })
            rows_kept += 1
        rg_idx += 1

    try:
        f.close()
    except Exception:
        pass

    print(f"  → kept {len(out)}/{n} clips ({rows_seen} rows seen, {rg_idx} row groups read)")
    if len(out) < n:
        deficit = n - len(out)
        print(f"  → topping up {deficit} clips with TTS substitute")
        out.extend(_build_saudi_synth(deficit))
    return out


# ---------------------------------------------------------------------------
# Slice 2: Arabic medical sentences with Gulf drug names (TTS)
# ---------------------------------------------------------------------------

# 30 short Arabic medical sentences. Each contains 1-2 specific drug names
# common in Gulf clinics. Transcripts are written in real Arabic (MSA-Gulf
# blend) so that a Gulf doctor or patient phrasing translates straight.
#
# medical_terms list contains the canonical English/transliterated drug names
# we want to score recall on.
ARABIC_MEDICAL_SENTENCES: List[Dict[str, Any]] = [
    {"text": "المريض يأخذ دوليبران كل ثمان ساعات.",
     "medical_terms": ["doliprane"]},
    {"text": "أعطني علبة دوليبران للحرارة.",
     "medical_terms": ["doliprane"]},
    {"text": "هل تأخذ بانادول قبل النوم؟",
     "medical_terms": ["panadol"]},
    {"text": "وصف لي الطبيب أوغمنتين بسبب التهاب الحلق.",
     "medical_terms": ["augmentin"]},
    {"text": "أحتاج وصفة طبية لأموكسيسيلين.",
     "medical_terms": ["amoxicillin"]},
    {"text": "هل أستخدم سيفترياكسون عن طريق الوريد؟",
     "medical_terms": ["ceftriaxone"]},
    {"text": "تناول حبة ميتفورمين مع وجبة الإفطار.",
     "medical_terms": ["metformin"]},
    {"text": "ضغط الدم مرتفع، هل يمكنني أخذ أملوديبين؟",
     "medical_terms": ["amlodipine"]},
    {"text": "وصف الطبيب أتورفاستاتين لخفض الكوليسترول.",
     "medical_terms": ["atorvastatin"]},
    {"text": "أعاني من ارتجاع المعدة، هل آخذ أوميبرازول؟",
     "medical_terms": ["omeprazole"]},
    {"text": "أحتاج بخاخ فينتولين لنوبة الربو.",
     "medical_terms": ["ventolin"]},
    {"text": "هل دواء سيمبيكورت آمن أثناء الحمل؟",
     "medical_terms": ["symbicort"]},
    {"text": "أتناول الأنسولين قبل كل وجبة.",
     "medical_terms": ["insulin"]},
    {"text": "أعطاني الطبيب حقنة ديكلوفيناك للألم.",
     "medical_terms": ["diclofenac"]},
    {"text": "خذ بروفين كل ست ساعات بعد الأكل.",
     "medical_terms": ["brufen"]},
    {"text": "وصف لي كلاريتين لحساسية الربيع.",
     "medical_terms": ["claritine"]},
    {"text": "أحتاج زيرتك بسبب حساسية الجلد.",
     "medical_terms": ["zyrtec"]},
    {"text": "أعطني نيكسيوم لمشكلة الحموضة.",
     "medical_terms": ["nexium"]},
    {"text": "تناولي ميكروليت لتعويض السوائل.",
     "medical_terms": ["microlite"]},
    {"text": "أرجو وصف ليفوفلوكساسين لالتهاب البول.",
     "medical_terms": ["levofloxacin"]},
    {"text": "هل أحتاج ميكوستات بسبب البلغم؟",
     "medical_terms": ["mucostat"]},
    {"text": "أعطاني أوغمنتين شراب للأطفال.",
     "medical_terms": ["augmentin"]},
    {"text": "خذ سيتالوبرام في الصباح لمدة شهر.",
     "medical_terms": ["citalopram"]},
    {"text": "وصفة طبية تتضمن لوسارتان وأملوديبين.",
     "medical_terms": ["losartan", "amlodipine"]},
    {"text": "هل آخذ كونكور للقلب يوميا؟",
     "medical_terms": ["concor"]},
    {"text": "بحاجة إلى ميلوكسيكام لألم الركبة.",
     "medical_terms": ["meloxicam"]},
    {"text": "أعطاني الطبيب حقن إنوكسابارين.",
     "medical_terms": ["enoxaparin"]},
    {"text": "أتناول وارفارين لمنع الجلطات.",
     "medical_terms": ["warfarin"]},
    {"text": "هل تأخذ ميفلوكين قبل السفر؟",
     "medical_terms": ["mefloquine"]},
    {"text": "وصف الطبيب فلوكستين للاكتئاب.",
     "medical_terms": ["fluoxetine"]},
]


def build_medical_vocab_ar() -> List[Dict[str, Any]]:
    print(f"\n[medical_vocab_ar] synthesising {len(ARABIC_MEDICAL_SENTENCES)} clips with say -v Majed ...")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(ARABIC_MEDICAL_SENTENCES, 1):
        clip_id = f"medical_vocab_ar_{i:03d}"
        wav = AUDIO_DIR / f"{clip_id}.wav"
        try:
            dur = synth_say(item["text"], voice="Majed", out_wav=wav)
        except subprocess.CalledProcessError as exc:
            print(f"  ! TTS failed for clip {i}: {exc}")
            continue
        out.append({
            "id": clip_id,
            "category": "medical_vocab_ar",
            "audio_path": f"audio/{clip_id}.wav",
            "duration_s": round(dur, 3),
            "language": "ar",
            "transcript": item["text"],
            "transcript_normalized": normalize_text(item["text"]),
            "medical_terms": item["medical_terms"],
            "source": "tts_say_majed",
            "tags": ["arabic-medical", "drug-name", "synthetic"],
        })
        print(f"  ✓ {clip_id} ({dur:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Slice 3: Arabic↔English code-switched (TTS)
# ---------------------------------------------------------------------------

# 20 sentences mixing Gulf Arabic frames with English drug / disease names.
# This is exactly how Gulf doctors phrase prescriptions in real consultations.
CODE_SWITCH_SENTENCES: List[Dict[str, Any]] = [
    {"text": "المريض عنده hypertension و نديله Amlodipine يوميا.",
     "medical_terms": ["hypertension", "amlodipine"]},
    {"text": "تم تشخيص الطفل ب asthma و نوصف له Ventolin.",
     "medical_terms": ["asthma", "ventolin"]},
    {"text": "يعاني المريض من diabetes mellitus type two.",
     "medical_terms": ["diabetes mellitus type 2"]},
    {"text": "نبدأ المريض ب Metformin خمسمائة ميليجرام مرتين باليوم.",
     "medical_terms": ["metformin"]},
    {"text": "عنده حساسية، هل عندك Cetirizine في الصيدلية؟",
     "medical_terms": ["cetirizine"]},
    {"text": "نعطيه IV Ceftriaxone لمدة أسبوع.",
     "medical_terms": ["ceftriaxone"]},
    {"text": "المريضة حامل، نتجنب Atorvastatin.",
     "medical_terms": ["atorvastatin"]},
    {"text": "نشتري Insulin pen للحقن المنزلي.",
     "medical_terms": ["insulin"]},
    {"text": "عنده gastritis، نوصف Omeprazole قبل الفطور.",
     "medical_terms": ["gastritis", "omeprazole"]},
    {"text": "نبدأ Warfarin مع متابعة INR كل أسبوع.",
     "medical_terms": ["warfarin", "inr"]},
    {"text": "المريض عنده atrial fibrillation و ناخذ ECG.",
     "medical_terms": ["atrial fibrillation", "ecg"]},
    {"text": "نعمل MRI للظهر بسبب lower back pain.",
     "medical_terms": ["mri", "lower back pain"]},
    {"text": "نوصف Augmentin سبعة أيام للالتهاب.",
     "medical_terms": ["augmentin"]},
    {"text": "نعطي steroids للتخفيف من التهاب الرئة.",
     "medical_terms": ["steroids"]},
    {"text": "نطلب CBC و urinalysis اليوم.",
     "medical_terms": ["cbc", "urinalysis"]},
    {"text": "نراقب blood sugar كل ست ساعات.",
     "medical_terms": ["blood sugar"]},
    {"text": "هل تعطيه Doliprane أم Brufen للحرارة؟",
     "medical_terms": ["doliprane", "brufen"]},
    {"text": "المريض على Concor خمسة ميليجرام منذ شهرين.",
     "medical_terms": ["concor"]},
    {"text": "عنده chest pain شديد، نطلب troponin.",
     "medical_terms": ["chest pain", "troponin"]},
    {"text": "نبدأ chemotherapy الأسبوع القادم.",
     "medical_terms": ["chemotherapy"]},
]


def build_code_switching() -> List[Dict[str, Any]]:
    print(f"\n[code_switching] synthesising {len(CODE_SWITCH_SENTENCES)} mixed AR↔EN clips ...")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(CODE_SWITCH_SENTENCES, 1):
        clip_id = f"code_switching_{i:03d}"
        wav = AUDIO_DIR / f"{clip_id}.wav"
        # Majed handles both Arabic and embedded English tokens (it
        # falls back to letter-by-letter for English, which is realistic
        # of how Gulf speakers say English drug names with an Arabic accent).
        try:
            dur = synth_say(item["text"], voice="Majed", out_wav=wav)
        except subprocess.CalledProcessError as exc:
            print(f"  ! TTS failed for clip {i}: {exc}")
            continue
        out.append({
            "id": clip_id,
            "category": "code_switching",
            "audio_path": f"audio/{clip_id}.wav",
            "duration_s": round(dur, 3),
            "language": "mixed",
            "transcript": item["text"],
            "transcript_normalized": normalize_text(item["text"]),
            "medical_terms": item["medical_terms"],
            "source": "tts_say_majed",
            "tags": ["code-switch", "ar-en", "synthetic"],
        })
        print(f"  ✓ {clip_id} ({dur:.1f}s)")
    return out


# ---------------------------------------------------------------------------
# Slice 4: existing English medical clips
# ---------------------------------------------------------------------------


def build_english_medical(n: int = 20) -> List[Dict[str, Any]]:
    print(f"\n[english_medical] reusing local clips from {LOCAL_MED_DIR} ...")
    if not LOCAL_MED_DIR.exists():
        print(f"  ! folder not found: {LOCAL_MED_DIR}")
        return []
    if not LOCAL_MED_META.exists():
        print(f"  ! metadata.csv not found: {LOCAL_MED_META}")
        return []

    import csv
    gt: Dict[str, str] = {}
    with LOCAL_MED_META.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            fname = (row.get("filename") or "").strip()
            tx = (row.get("transcription") or "").strip()
            if fname and tx:
                gt[fname] = tx

    wavs = sorted(LOCAL_MED_DIR.glob("*.wav"))
    out: List[Dict[str, Any]] = []
    for wav in wavs[:n]:
        # b20 (preprocessed) uses the same row as the o00 (original) name
        canonical = wav.name.replace("b20.wav", "o00.wav")
        text = gt.get(wav.name) or gt.get(canonical) or ""
        if not text:
            print(f"  ! skip {wav.name}: no ground truth")
            continue

        idx = len(out) + 1
        clip_id = f"english_medical_{idx:03d}"
        target = AUDIO_DIR / f"{clip_id}.wav"
        # Resample to 16 kHz mono
        subprocess.run([
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", str(wav),
            "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(target),
        ], check=True)
        dur = float(subprocess.run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(target),
        ], capture_output=True, text=True).stdout.strip() or 0.0)

        out.append({
            "id": clip_id,
            "category": "english_medical",
            "audio_path": f"audio/{clip_id}.wav",
            "duration_s": round(dur, 3),
            "language": "en",
            "transcript": text,
            "transcript_normalized": normalize_text(text),
            "medical_terms": [],   # left empty; could be filled with NER later
            "source": "medical_audio_transcription_local",
            "tags": ["english-medical", "patient-narrative"],
        })
        print(f"  ✓ {clip_id} ({dur:.1f}s) <- {wav.name}")
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    records += build_english_medical(n=20)        # fast, local; do first
    records += build_medical_vocab_ar()           # ~1s/clip
    records += build_code_switching()             # ~1s/clip
    records += build_saudi_acoustic(n=30)         # network, slow; do last

    if not records:
        print("\nNo records built. Aborting.")
        return 1

    with MANIFEST_PATH.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    by_cat: Dict[str, int] = {}
    total_dur = 0.0
    for r in records:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        total_dur += r["duration_s"]

    print("\n" + "=" * 60)
    print(f"Eval set built: {MANIFEST_PATH}")
    print(f"Total clips:    {len(records)}")
    print(f"Total duration: {total_dur:.1f}s ({total_dur/60:.1f} min)")
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:<22} {count}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
