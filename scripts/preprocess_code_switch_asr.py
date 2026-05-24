"""Preprocess Arabic-English code-switched ASR manifests.

This script prepares supervised ASR rows for Saudi/Emirati fine-tuning:

* audio -> 16 kHz, mono, 16-bit PCM WAV
* energy-based leading/trailing silence trimming
* strict Arabic text normalization
* English lowercasing and Arabic/Latin boundary splitting
* digit + common medical-unit verbalization
* punctuation/control/HTML/URL cleanup
* duration/text sanity filters

Input is one or more JSONL manifests with an audio path and a transcript
field. The downloader scripts in this repo already emit this shape.

Example:
    python scripts/preprocess_code_switch_asr.py \
        --manifest data/dataset_samples/sada2022/manifest.jsonl \
        --manifest data/dataset_samples/mixat_emirati/manifest.jsonl \
        --out data/preprocessed/saudi_uae_asr

The most important public API is clean_asr_text(text). Use it for both the
Gulf base corpus and the medical layer so the fine-tune never sees two
different transcript formats.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16_000

TEXT_FIELDS = (
    "text",
    "transcript",
    "transcription",
    "processed_text",
    "ProcessedText",
    "Original_text",
    "sentence",
    "raw_transcription",
)
AUDIO_FIELDS = ("audio_path", "audio", "path", "file", "filename", "wav")

ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
LATIN_RE = re.compile(r"[A-Za-z]")
DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
TATWEEL = "\u0640"
CONTROL_RE = re.compile(r"[\u0000-\u001f\u007f-\u009f]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)

# Keep Arabic letters, Latin letters, numbers, and whitespace until number
# folding has happened. Everything else, including underscores, is a separator.
PUNCT_RE = re.compile(r"[^0-9A-Za-z\s\u0600-\u06ff]", re.UNICODE)
ARABIC_PUNCT_RE = re.compile(r"[،؛؟«»ـ]", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
NUMBER_RE = re.compile(r"(?<!\w)(\d+(?:[\.,]\d+)?)(?:\s*([A-Za-z%]+))?")

ARABIC_TRANSLATION = {
    ord("أ"): "ا",
    ord("إ"): "ا",
    ord("آ"): "ا",
    ord("ٱ"): "ا",
    ord("ى"): "ي",
    ord("ؤ"): "و",
    ord("ئ"): "ي",
    ord("ء"): "",
    ord("٠"): "0",
    ord("١"): "1",
    ord("٢"): "2",
    ord("٣"): "3",
    ord("٤"): "4",
    ord("٥"): "5",
    ord("٦"): "6",
    ord("٧"): "7",
    ord("٨"): "8",
    ord("٩"): "9",
    ord("۰"): "0",
    ord("۱"): "1",
    ord("۲"): "2",
    ord("۳"): "3",
    ord("۴"): "4",
    ord("۵"): "5",
    ord("۶"): "6",
    ord("۷"): "7",
    ord("۸"): "8",
    ord("۹"): "9",
}

PHRASE_NORMALIZATIONS = (
    (re.compile(r"\bان\s*شاء\s*الله\b"), "ان شاء الله"),
    (re.compile(r"\bانشالله\b"), "ان شاء الله"),
    (re.compile(r"\bما\s*شاء\s*الله\b"), "ما شاء الله"),
    (re.compile(r"\bماشاءالله\b"), "ما شاء الله"),
    (re.compile(r"\bاهلل\b"), "الله"),
    (re.compile(r"\bاللو\b"), "الله"),
)

UNIT_WORDS = {
    "mg": "مليغرام",
    "milligram": "مليغرام",
    "milligrams": "مليغرام",
    "mcg": "ميكروغرام",
    "ug": "ميكروغرام",
    "microgram": "ميكروغرام",
    "micrograms": "ميكروغرام",
    "g": "غرام",
    "gram": "غرام",
    "grams": "غرام",
    "kg": "كيلوغرام",
    "kilogram": "كيلوغرام",
    "kilograms": "كيلوغرام",
    "ml": "مليلتر",
    "milliliter": "مليلتر",
    "milliliters": "مليلتر",
    "l": "لتر",
    "liter": "لتر",
    "liters": "لتر",
    "mmhg": "مليمتر زئبق",
    "bpm": "نبضه في الدقيقه",
    "iu": "وحده دوليه",
    "%": "بالمئه",
}


def _np():
    import numpy as np

    return np


def _sf():
    import soundfile as sf

    return sf


@dataclass
class CleanOptions:
    ta_marbuta: str = "keep"  # keep | ha
    verbalize_numbers: bool = True
    remove_punctuation: bool = True
    lowercase_english: bool = True


def _num_to_arabic_words(raw: str) -> str:
    value = raw.replace(",", ".")
    try:
        from num2words import num2words
    except ImportError:
        return _num_to_arabic_words_fallback(value)
    try:
        if "." in value:
            return str(num2words(float(value), lang="ar"))
        return str(num2words(int(value), lang="ar"))
    except Exception:
        return raw


def _num_to_arabic_words_fallback(value: str) -> str:
    if "." in value:
        return value
    try:
        n = int(value)
    except ValueError:
        return value
    ones = {
        0: "صفر", 1: "واحد", 2: "اثنان", 3: "ثلاثه", 4: "اربعه", 5: "خمسه",
        6: "سته", 7: "سبعه", 8: "ثمانيه", 9: "تسعه", 10: "عشره",
        11: "احد عشر", 12: "اثنا عشر", 13: "ثلاثه عشر", 14: "اربعه عشر",
        15: "خمسه عشر", 16: "سته عشر", 17: "سبعه عشر", 18: "ثمانيه عشر",
        19: "تسعه عشر",
    }
    tens = {20: "عشرون", 30: "ثلاثون", 40: "اربعون", 50: "خمسون", 60: "ستون", 70: "سبعون", 80: "ثمانون", 90: "تسعون"}
    hundreds = {100: "مائه", 200: "مئتان", 300: "ثلاثمائه", 400: "اربعمائه", 500: "خمسمائه", 600: "ستمائه", 700: "سبعمائه", 800: "ثمانمائه", 900: "تسعمائه"}
    if n in ones:
        return ones[n]
    if n in tens:
        return tens[n]
    if n < 100:
        unit = n % 10
        ten = n - unit
        return f"{ones[unit]} و {tens[ten]}"
    if n in hundreds:
        return hundreds[n]
    if n < 1000:
        hundred = n - (n % 100)
        rest = n % 100
        return f"{hundreds[hundred]} و {_num_to_arabic_words_fallback(str(rest))}"
    if n == 1000:
        return "الف"
    if n < 10000:
        thousands = n // 1000
        rest = n % 1000
        prefix = "الف" if thousands == 1 else f"{_num_to_arabic_words_fallback(str(thousands))} الاف"
        return prefix if rest == 0 else f"{prefix} و {_num_to_arabic_words_fallback(str(rest))}"
    return value


def _verbalize_numbers(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        number = _num_to_arabic_words(match.group(1))
        unit = (match.group(2) or "").lower()
        if unit:
            return f" {number} {UNIT_WORDS.get(unit, unit)} "
        return f" {number} "

    return NUMBER_RE.sub(repl, text)


def _split_arabic_latin_boundaries(text: str) -> str:
    # Keep English medical words pure Latin for hotwording. Arabic clitics
    # attached to English words become separate tokens: "الdoctor" -> "ال doctor".
    text = re.sub(r"(?<=[\u0600-\u06ff])(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=[\u0600-\u06ff])", " ", text)
    return text


def _normalize_common_phrases(text: str) -> str:
    for pattern, replacement in PHRASE_NORMALIZATIONS:
        text = pattern.sub(replacement, text)
    text = text.replace("اهلل", "الله")
    text = text.replace("اللو", "الله")
    return text


def clean_asr_text(text: str, options: Optional[CleanOptions] = None) -> str:
    """Normalize Arabic-English code-switched ASR transcript text.

    The function is intentionally deterministic and dependency-light. It does
    not transliterate Arabic-script medical terms into English; those must be
    fixed upstream by lexicon/corpus preparation because guessing would create
    training noise.
    """
    opts = options or CleanOptions()
    text = html.unescape(str(text or ""))
    text = unicodedata.normalize("NFKC", text)
    text = URL_RE.sub(" ", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = CONTROL_RE.sub(" ", text)
    text = DIACRITICS_RE.sub("", text)
    text = text.replace(TATWEEL, "")
    text = text.translate(ARABIC_TRANSLATION)
    if opts.ta_marbuta == "ha":
        text = text.replace("ة", "ه")
    text = _split_arabic_latin_boundaries(text)
    if opts.lowercase_english:
        text = re.sub(r"[A-Za-z]+", lambda m: m.group(0).lower(), text)
    if opts.verbalize_numbers:
        text = _verbalize_numbers(text)
        # num2words may reintroduce hamza forms such as مائة; fold again so
        # generated number words obey the same Arabic normalization policy.
        text = text.translate(ARABIC_TRANSLATION)
    text = _normalize_common_phrases(text)
    if opts.remove_punctuation:
        text = ARABIC_PUNCT_RE.sub(" ", text)
        text = PUNCT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text


def _resolve_audio_path(row: Dict[str, Any], manifest_path: Path) -> Optional[Path]:
    for key in AUDIO_FIELDS:
        value = row.get(key)
        if not value:
            continue
        if isinstance(value, dict):
            value = value.get("path")
        if not isinstance(value, str):
            continue
        path = Path(value)
        if not path.is_absolute():
            path = manifest_path.parent / path
        if path.exists():
            return path
    return None


def _resolve_relative_path(value: Any, manifest_path: Path) -> Optional[Path]:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path if path.exists() else None


def _pick_text(row: Dict[str, Any]) -> str:
    for key in TEXT_FIELDS:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _ffmpeg_to_wav(src: Path, dst: Path) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for audio standardization")
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-i", str(src),
            "-ac", "1",
            "-ar", str(SAMPLE_RATE),
            "-sample_fmt", "s16",
            str(dst),
        ],
        check=True,
    )


def _trim_silence(wav, *, threshold_db: float = -45.0, pad_s: float = 0.10) -> Tuple[Any, float, float]:
    np = _np()
    if wav.size == 0:
        return wav, 0.0, 0.0
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    frame = max(1, int(0.025 * SAMPLE_RATE))
    hop = max(1, int(0.010 * SAMPLE_RATE))
    if wav.size < frame:
        return wav.astype(np.float32), 0.0, wav.size / SAMPLE_RATE
    rms_values: List[float] = []
    for start in range(0, wav.size - frame + 1, hop):
        chunk = wav[start:start + frame]
        rms_values.append(float(np.sqrt(np.mean(np.square(chunk))) + 1e-12))
    rms = np.asarray(rms_values, dtype=np.float32)
    threshold = 10 ** (threshold_db / 20.0)
    active = np.flatnonzero(rms >= threshold)
    if active.size == 0:
        return np.zeros(0, dtype=np.float32), 0.0, 0.0
    pad = int(pad_s * SAMPLE_RATE)
    start_sample = max(0, int(active[0] * hop) - pad)
    end_sample = min(wav.size, int(active[-1] * hop + frame) + pad)
    return wav[start_sample:end_sample].astype(np.float32), start_sample / SAMPLE_RATE, end_sample / SAMPLE_RATE


def _chars_per_second(text: str, duration_s: float) -> float:
    if duration_s <= 0:
        return math.inf
    return len(text.replace(" ", "")) / duration_s


def _bad_charset_tokens(text: str) -> List[str]:
    bad = []
    for token in text.split():
        has_allowed = ARABIC_RE.search(token) or LATIN_RE.search(token) or any(ch.isdigit() for ch in token)
        if not has_allowed:
            bad.append(token)
    return bad


def _load_manifest(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row["_manifest"] = str(path)
            row["_line_no"] = line_no
            yield row


def _load_segments(row: Dict[str, Any], manifest_path: Path) -> List[Dict[str, Any]]:
    path = _resolve_relative_path(row.get("segments_path"), manifest_path)
    if path is None:
        return []
    segments: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                segments.append(json.loads(line))
    return segments


def _segment_text(segment: Dict[str, Any]) -> str:
    for key in ("processed_text", "text", "ground_truth_text", "transcript", "transcription"):
        value = segment.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_vocab(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    vocab = sorted({tok for row in rows for tok in row["text"].split()})
    path.write_text("\n".join(vocab) + ("\n" if vocab else ""), encoding="utf-8")


def _unique_id(base_id: str, seen_ids: set[str]) -> str:
    sample_id = base_id
    suffix_i = 1
    while sample_id in seen_ids:
        suffix_i += 1
        sample_id = f"{base_id}_{suffix_i}"
    seen_ids.add(sample_id)
    return sample_id


def _reject(
    rejected: List[Dict[str, Any]],
    *,
    sample_id: str,
    reason: str,
    manifest: Path,
    source_audio: Optional[Path],
    raw_text: str,
    clean_text: str,
    duration_s: float,
    source_segment: Optional[Dict[str, Any]] = None,
) -> None:
    rejected.append({
        "id": sample_id,
        "reason": reason,
        "source_manifest": str(manifest),
        "source_audio": str(source_audio) if source_audio else None,
        "raw_text": raw_text,
        "clean_text": clean_text,
        "duration_s": round(duration_s, 3),
        "source_segment": source_segment,
    })


def _accept_or_reject_clip(
    *,
    wav,
    sample_id: str,
    raw_text: str,
    clean_text: str,
    out_dir: Path,
    manifest: Path,
    source_audio: Path,
    clean_rows: List[Dict[str, Any]],
    rejected: List[Dict[str, Any]],
    args: argparse.Namespace,
    source_segment: Optional[Dict[str, Any]] = None,
) -> None:
    sf = _sf()
    trim_start_s = 0.0
    trim_end_s = 0.0
    reject_reason = None
    if args.trim_silence:
        wav, trim_start_s, trim_end_s = _trim_silence(wav, threshold_db=args.silence_db, pad_s=args.silence_pad_s)
    duration_s = float(len(wav) / SAMPLE_RATE) if wav is not None else 0.0
    if duration_s < args.min_duration_s:
        reject_reason = "too_short"
    elif duration_s > args.max_duration_s:
        reject_reason = "too_long_needs_alignment_chunking"
    elif not clean_text:
        reject_reason = "empty_transcript"
    else:
        cps = _chars_per_second(clean_text, duration_s)
        if cps < args.min_chars_per_s:
            reject_reason = "text_too_short_for_duration"
        elif cps > args.max_chars_per_s:
            reject_reason = "text_too_long_for_duration"
        elif _bad_charset_tokens(clean_text):
            reject_reason = "bad_charset_tokens"
    if reject_reason is not None:
        _reject(
            rejected,
            sample_id=sample_id,
            reason=reject_reason,
            manifest=manifest,
            source_audio=source_audio,
            raw_text=raw_text,
            clean_text=clean_text,
            duration_s=duration_s,
            source_segment=source_segment,
        )
        return
    rel_audio = f"audio/{sample_id}.wav"
    (out_dir / rel_audio).parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / rel_audio), wav, SAMPLE_RATE, subtype="PCM_16")
    clean_rows.append({
        "id": sample_id,
        "audio_path": rel_audio,
        "text": clean_text,
        "duration_s": round(duration_s, 3),
        "source_manifest": str(manifest),
        "source_audio": str(source_audio),
        "raw_text": raw_text,
        "trim_start_s": round(trim_start_s, 3),
        "trim_end_s": round(trim_end_s, 3),
        "chars_per_s": round(_chars_per_second(clean_text, duration_s), 3),
        "source_segment": source_segment,
    })


def preprocess_manifests(args: argparse.Namespace) -> int:
    opts = CleanOptions(
        ta_marbuta=args.ta_marbuta,
        verbalize_numbers=not args.keep_digits,
        remove_punctuation=not args.keep_punctuation,
        lowercase_english=not args.keep_english_case,
    )
    out_dir: Path = args.out
    audio_out = out_dir / "audio"
    clean_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()

    for manifest in args.manifest:
        manifest = manifest.resolve()
        for row in _load_manifest(manifest):
            source_audio = _resolve_audio_path(row, manifest)
            source_id = str(row.get("id") or Path(str(row.get("audio_path") or row.get("audio") or "sample")).stem)
            base_id = re.sub(r"[^a-zA-Z0-9_\-]+", "_", source_id).strip("_") or "sample"
            segments = _load_segments(row, manifest)

            reject_reason = None
            if source_audio is None:
                reject_reason = "missing_audio"

            tmp_wav = None
            if reject_reason is None:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_wav = Path(tmp.name)
                    _ffmpeg_to_wav(source_audio, tmp_wav)
                    sf = _sf()
                    wav, sr = sf.read(str(tmp_wav), dtype="float32", always_2d=False)
                    if sr != SAMPLE_RATE:
                        raise RuntimeError(f"unexpected sample rate after ffmpeg: {sr}")
                    if segments:
                        for segment_index, segment in enumerate(segments):
                            raw_text = _segment_text(segment)
                            clean_text = clean_asr_text(raw_text, opts)
                            seg_start = max(0.0, float(segment.get("start_s") or 0.0))
                            seg_end = float(segment.get("end_s") or seg_start)
                            start_sample = max(0, int(seg_start * SAMPLE_RATE))
                            end_sample = min(len(wav), int(seg_end * SAMPLE_RATE))
                            segment_wav = wav[start_sample:end_sample]
                            seg_base = re.sub(r"[^a-zA-Z0-9_\-]+", "_", f"{base_id}_{segment_index:04d}").strip("_")
                            sample_id = _unique_id(seg_base, seen_ids)
                            _accept_or_reject_clip(
                                wav=segment_wav,
                                sample_id=sample_id,
                                raw_text=raw_text,
                                clean_text=clean_text,
                                out_dir=out_dir,
                                manifest=manifest,
                                source_audio=source_audio,
                                clean_rows=clean_rows,
                                rejected=rejected,
                                args=args,
                                source_segment=segment,
                            )
                    else:
                        raw_text = _pick_text(row)
                        clean_text = clean_asr_text(raw_text, opts)
                        sample_id = _unique_id(base_id, seen_ids)
                        _accept_or_reject_clip(
                            wav=wav,
                            sample_id=sample_id,
                            raw_text=raw_text,
                            clean_text=clean_text,
                            out_dir=out_dir,
                            manifest=manifest,
                            source_audio=source_audio,
                            clean_rows=clean_rows,
                            rejected=rejected,
                            args=args,
                        )
                except Exception as exc:
                    sample_id = _unique_id(base_id, seen_ids)
                    _reject(
                        rejected,
                        sample_id=sample_id,
                        reason=f"audio_error:{exc}",
                        manifest=manifest,
                        source_audio=source_audio,
                        raw_text=_pick_text(row),
                        clean_text=clean_asr_text(_pick_text(row), opts),
                        duration_s=0.0,
                    )
                finally:
                    if tmp_wav is not None:
                        tmp_wav.unlink(missing_ok=True)

            if reject_reason is not None:
                sample_id = _unique_id(base_id, seen_ids)
                raw_text = _pick_text(row)
                _reject(
                    rejected,
                    sample_id=sample_id,
                    reason=reject_reason,
                    manifest=manifest,
                    source_audio=source_audio,
                    raw_text=raw_text,
                    clean_text=clean_asr_text(raw_text, opts),
                    duration_s=0.0,
                )

    _write_jsonl(out_dir / "manifest.jsonl", clean_rows)
    _write_jsonl(out_dir / "rejected.jsonl", rejected)
    _write_vocab(out_dir / "vocab.txt", clean_rows)
    summary = {
        "kept": len(clean_rows),
        "rejected": len(rejected),
        "hours": round(sum(row["duration_s"] for row in clean_rows) / 3600.0, 4),
        "sample_rate": SAMPLE_RATE,
        "audio_format": "16-bit PCM mono WAV",
        "min_duration_s": args.min_duration_s,
        "max_duration_s": args.max_duration_s,
        "trim_silence": args.trim_silence,
        "ta_marbuta": args.ta_marbuta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"clean manifest: {out_dir / 'manifest.jsonl'}")
    print(f"reject log    : {out_dir / 'rejected.jsonl'}")
    print(f"vocab         : {out_dir / 'vocab.txt'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Preprocess Arabic-English code-switched ASR manifests.")
    parser.add_argument("--manifest", type=Path, action="append", required=True,
                        help="Input JSONL manifest. Can be passed multiple times.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory for clean audio + manifest.")
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--min-chars-per-s", type=float, default=1.0)
    parser.add_argument("--max-chars-per-s", type=float, default=25.0)
    parser.add_argument("--trim-silence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--silence-db", type=float, default=-45.0,
                        help="RMS threshold in dBFS for edge silence trimming.")
    parser.add_argument("--silence-pad-s", type=float, default=0.10)
    parser.add_argument("--ta-marbuta", choices=["keep", "ha"], default="keep",
                        help="Keep ة by default; use 'ha' if you want Gulf-pronunciation spelling.")
    parser.add_argument("--keep-digits", action="store_true",
                        help="Do not verbalize digits into Arabic words.")
    parser.add_argument("--keep-punctuation", action="store_true")
    parser.add_argument("--keep-english-case", action="store_true")
    args = parser.parse_args()
    return preprocess_manifests(args)


if __name__ == "__main__":
    sys.exit(main())
