"""Build a ~30-minute Gulf-Arabic bake-off test set.

Pulls labelled clips from:
  • WorldSpeech (`disco-eth/WorldSpeech`) Gulf country splits — ar_bh, ar_kw,
    ar_sa — these are free, transcribed, parliamentary recordings at 24 kHz.
  • SADA (`MohamedRashad/SADA22`) — optional, Saudi Arabic TV transcribed.
  • Any pre-existing local session recordings under `data/sessions/`
    that already have a paired transcript inside the lexicon or descriptions.

All clips are resampled to 16 kHz mono WAV and written to
`eval/<name>/audio/<id>.wav` with a manifest matching `bakeoff.py`'s schema:

    {
      "id": "...",
      "category": "gulf_acoustic" | "saudi_tv",
      "audio_path": "audio/<id>.wav",
      "duration_s": float,
      "language": "ar",
      "transcript": "...",
      "transcript_normalized": "...",
      "medical_terms": [],
      "source": "worldspeech_ar_bh" | "sada22" | ...,
      "tags": [...]
    }

By default writes to `eval/bakeoff_30min/`. Run:

    source .venv/bin/activate
    python -m scripts.build_bakeoff_testset --target-min 30

Then evaluate:

    python -m scripts.bakeoff --models qwen3 qwen3_uae --eval-dir eval/bakeoff_30min
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Text normalization — must match bakeoff.py
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")
_DIACRITICS_RE = re.compile(r"[\u064b-\u065f\u0670]")


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s)
    s = _DIACRITICS_RE.sub("", s)
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

SR = 16_000


def write_wav_16k_mono(arr: np.ndarray, sr: int, out_path: Path) -> float:
    """Write `arr` as a 16 kHz mono PCM16 WAV. Returns final duration in seconds."""
    import soundfile as sf

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        try:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=SR)
        except Exception:
            # Fall back to ffmpeg pipe if librosa is missing
            return _ffmpeg_resample(arr, sr, out_path)
        sr = SR
    sf.write(str(out_path), arr.astype(np.float32), sr, subtype="PCM_16")
    return len(arr) / sr


def _ffmpeg_resample(arr: np.ndarray, sr: int, out_path: Path) -> float:
    import soundfile as sf

    tmp = out_path.with_suffix(".tmp.wav")
    sf.write(str(tmp), arr.astype(np.float32), sr, subtype="PCM_16")
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error",
            "-i", str(tmp),
            "-ac", "1", "-ar", str(SR), str(out_path),
        ],
        check=True,
    )
    tmp.unlink(missing_ok=True)
    info = subprocess.check_output(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(out_path)]
    ).decode().strip()
    return float(info)


def _decode_audio_field(audio: Any) -> Tuple[Optional[np.ndarray], int]:
    """Return (waveform, sample_rate) from a HuggingFace audio field.

    Handles three cases the parquet streamer can yield on different
    `datasets` versions:
      1. `{array: np.ndarray, sampling_rate: int}` — already decoded.
      2. `{bytes: <opus/wav/mp3>, path: ..., sampling_rate?: int}` —
         raw encoded bytes; decode with soundfile or fall back to ffmpeg.
      3. `None` or unrecognised — return (None, 0).
    """
    if not isinstance(audio, dict):
        return None, 0
    arr = audio.get("array")
    sr_val = audio.get("sampling_rate")
    sr = int(sr_val) if sr_val else 0
    if arr is not None:
        return np.asarray(arr, dtype=np.float32), sr or 0
    data = audio.get("bytes")
    if not data:
        return None, 0
    # Try in-memory decode via soundfile (handles wav/flac/ogg/opus).
    try:
        import io
        import soundfile as sf
        wav, sr_dec = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        return wav.astype(np.float32), int(sr_dec)
    except Exception:
        pass
    # Fallback: shell out to ffmpeg for everything else.
    try:
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".bin") as tin, \
             tempfile.NamedTemporaryFile(suffix=".wav") as tout:
            tin.write(data); tin.flush()
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", tin.name,
                 "-ac", "1", "-ar", str(SR), tout.name],
                check=True,
            )
            import soundfile as sf
            wav, sr_dec = sf.read(tout.name, dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            return wav.astype(np.float32), int(sr_dec)
    except Exception:
        return None, 0


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def _stream_hf_dataset(repo_id: str, subset: Optional[str], split: str):
    """Stream a Hugging Face dataset by listing parquet shards directly.

    We do NOT fall back to `load_dataset(repo_id, ...)` because on
    some versions of `datasets` + `fsspec` (e.g. 2.21 on NGC) the
    legacy resolver crashes recursively walking the repo with a
    `PurePosixPath` AttributeError. Failing loudly is better.
    """
    return _stream_via_parquet_shards(repo_id, subset, split)


def _stream_via_parquet_shards(repo_id: str, subset: Optional[str], split: str):
    """List parquet shards on the HF Hub for `repo_id` and stream them.

    Bypasses the fsspec / HfFileSystem layer (which has known bugs on some
    datasets/fsspec/pathlib combos) by calling `list_repo_files` directly
    and constructing HTTPS resolve URLs from `hf_hub_url`.
    """
    import fnmatch
    from datasets import load_dataset
    from huggingface_hub import hf_hub_url, list_repo_files

    all_files = list_repo_files(repo_id, repo_type="dataset")
    # Build candidate glob patterns, most specific first. Patterns are
    # relative to the dataset repo root (no leading 'datasets/<repo>/').
    candidates = []
    if subset:
        candidates += [
            f"{subset}/{split}-*.parquet",
            f"{subset}/{split}/*.parquet",
            f"data/{subset}/{split}-*.parquet",
            f"data/{subset}/{split}/*.parquet",
            f"{subset}/*.parquet",
        ]
    else:
        candidates += [
            f"{split}-*.parquet",
            f"{split}/*.parquet",
            f"data/{split}-*.parquet",
            f"data/{split}/*.parquet",
            f"*.parquet",
        ]
    matches: list = []
    for pat in candidates:
        matches = sorted(f for f in all_files if fnmatch.fnmatch(f, pat))
        if matches:
            break
    if not matches:
        raise FileNotFoundError(
            f"no parquet shards under {repo_id} for subset={subset!r} "
            f"split={split!r}"
        )
    urls = [
        hf_hub_url(repo_id, filename=rel, repo_type="dataset")
        for rel in matches
    ]
    ds = load_dataset(
        "parquet",
        data_files={split: urls},
        split=split,
        streaming=True,
    )
    # Older `datasets` (e.g. 2.21 on NGC) won't auto-decode the audio
    # column when streaming parquet — rows come back with `array=None` and
    # only raw `bytes`. Casting to `Audio()` forces decode on iteration.
    try:
        from datasets import Audio
        if "audio" in (getattr(ds, "features", None) or {}):
            ds = ds.cast_column("audio", Audio())
    except Exception:
        try:
            from datasets import Audio
            ds = ds.cast_column("audio", Audio())
        except Exception:
            pass
    return ds


def iter_worldspeech_gulf(
    countries: Iterable[str],
    min_s: float = 3.0,
    max_s: float = 20.0,
    max_per_country: int = 30,
) -> Iterable[Tuple[str, np.ndarray, int, str, float, Dict[str, Any]]]:
    """Yield (id, waveform, sr, transcript, duration_s, meta) from WorldSpeech.

    Each Gulf country subset is a separate config (ar_bh, ar_kw, ar_sa).
    Streams + reservoir-samples up to `max_per_country` rows that fit the
    duration window. Logs per-reason rejection counts so we can see why
    clips get dropped if the kept count is 0.
    """
    for country in countries:
        seen = 0
        kept = 0
        rej = {"no_audio": 0, "no_sr": 0, "no_transcript": 0,
               "too_short": 0, "too_long": 0, "high_cer": 0}
        try:
            ds = _stream_hf_dataset("disco-eth/WorldSpeech", country, split="train")
        except Exception as exc:
            print(f"  [{country}] cannot stream: {exc!r}")
            continue
        for ex in ds:
            seen += 1
            if seen > max_per_country * 30:
                break
            arr, sr = _decode_audio_field(ex.get("audio"))
            transcript = (ex.get("human_transcript") or "").strip()
            dur_val = ex.get("duration")
            try:
                dur = float(dur_val) if dur_val is not None else (len(arr) / sr if arr is not None and sr else 0.0)
            except Exception:
                dur = 0.0
            if arr is None:
                rej["no_audio"] += 1; continue
            if sr <= 0:
                rej["no_sr"] += 1; continue
            if not transcript:
                rej["no_transcript"] += 1; continue
            if dur < min_s:
                rej["too_short"] += 1; continue
            if dur > max_s:
                rej["too_long"] += 1; continue
            cer = ex.get("cer")
            if cer is not None and cer > 0.25:
                rej["high_cer"] += 1; continue
            yield (
                f"worldspeech_{country}_{kept:03d}",
                np.asarray(arr, dtype=np.float32),
                sr,
                transcript,
                dur,
                {"source": f"worldspeech_{country}", "cer": cer},
            )
            kept += 1
            if kept >= max_per_country:
                break
        print(f"  [worldspeech/{country}] scanned {seen}, kept {kept}, rejected {rej}")


def iter_sada(
    min_s: float = 3.0,
    max_s: float = 20.0,
    max_clips: int = 60,
) -> Iterable[Tuple[str, np.ndarray, int, str, float, Dict[str, Any]]]:
    """Yield clips from the SADA Saudi corpus, with per-reason rejection logs."""
    seen = 0
    kept = 0
    rej = {"no_audio": 0, "no_sr": 0, "no_transcript": 0,
           "too_short": 0, "too_long": 0}
    try:
        ds = _stream_hf_dataset("MohamedRashad/SADA22", None, split="train")
    except Exception as exc:
        print(f"  [sada22] cannot stream: {exc!r}")
        return
    for ex in ds:
        seen += 1
        if seen > max_clips * 40:
            break
        arr, sr = _decode_audio_field(ex.get("audio"))
        text = (
            ex.get("cleaned_text")
            or ex.get("text")
            or ex.get("transcription")
            or ""
        ).strip()
        if arr is None:
            rej["no_audio"] += 1; continue
        if sr <= 0:
            rej["no_sr"] += 1; continue
        if len(text) < 3:
            rej["no_transcript"] += 1; continue
        dur = len(arr) / sr
        if dur < min_s:
            rej["too_short"] += 1; continue
        if dur > max_s:
            rej["too_long"] += 1; continue
        yield (
            f"sada_{kept:03d}",
            np.asarray(arr, dtype=np.float32),
            sr,
            text,
            dur,
            {
                "source": "sada22",
                "dialect": ex.get("speaker_dialect"),
                "gender": ex.get("speaker_gender"),
            },
        )
        kept += 1
        if kept >= max_clips:
            break
    print(f"  [sada22] scanned {seen}, kept {kept}, rejected {rej}")


def build(
    eval_dir: Path,
    target_min: float,
    seed: int,
    skip_sada: bool,
    min_s: float,
    max_s: float,
) -> None:
    random.seed(seed)

    audio_dir = eval_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_dir / "manifest.jsonl"

    target_s = target_min * 60.0
    print(f"target duration: {target_min:.1f} min  ({target_s:.0f} s)")

    # We pull a generous pool first, then pick a balanced subset.
    pool: List[Tuple[str, np.ndarray, int, str, float, Dict[str, Any]]] = []

    print("\n== streaming WorldSpeech Gulf country splits ==")
    for clip in iter_worldspeech_gulf(
        countries=["ar_bh", "ar_kw", "ar_sa"],
        min_s=min_s,
        max_s=max_s,
        max_per_country=40,
    ):
        pool.append(clip)

    if not skip_sada:
        print("\n== streaming SADA22 ==")
        for clip in iter_sada(min_s=min_s, max_s=max_s, max_clips=60):
            pool.append(clip)

    if not pool:
        print("\nNo clips pulled — check network / dataset access.")
        sys.exit(1)

    print(f"\npool size: {len(pool)} clips, total {sum(c[4] for c in pool):.1f} s")

    # Balance: pick from each source roughly proportionally until we reach target.
    by_source: Dict[str, List[Tuple]] = {}
    for clip in pool:
        by_source.setdefault(clip[5]["source"], []).append(clip)
    for src in by_source:
        random.shuffle(by_source[src])

    selected: List[Tuple] = []
    total_s = 0.0
    # Round-robin draw across sources to balance dialects.
    sources = list(by_source.keys())
    cursors = {s: 0 for s in sources}
    while total_s < target_s:
        progress = False
        for s in sources:
            if cursors[s] >= len(by_source[s]):
                continue
            clip = by_source[s][cursors[s]]
            cursors[s] += 1
            selected.append(clip)
            total_s += clip[4]
            progress = True
            if total_s >= target_s:
                break
        if not progress:
            break

    print(f"\nselected {len(selected)} clips, total {total_s:.1f} s "
          f"({total_s/60:.1f} min) across sources: "
          f"{', '.join(f'{s}={cursors[s]}' for s in sources)}")

    # Write WAVs + manifest.
    records: List[Dict[str, Any]] = []
    for clip_id, arr, sr, transcript, raw_dur, meta in selected:
        wav_path = audio_dir / f"{clip_id}.wav"
        try:
            dur_s = write_wav_16k_mono(arr, sr, wav_path)
        except Exception as exc:
            print(f"  ! skip {clip_id}: {exc!r}")
            continue
        source = meta.pop("source")
        tags = [source]
        category = "saudi_tv" if source == "sada22" else "gulf_acoustic"
        records.append({
            "id": clip_id,
            "category": category,
            "audio_path": f"audio/{wav_path.name}",
            "duration_s": round(dur_s, 3),
            "language": "ar",
            "transcript": transcript,
            "transcript_normalized": normalize_text(transcript),
            "medical_terms": [],
            "source": source,
            "tags": tags,
            **{k: v for k, v in meta.items() if v is not None},
        })

    with manifest_path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = sum(r["duration_s"] for r in records)
    print(f"\n✓ wrote {len(records)} clips ({total:.1f} s ≈ {total/60:.1f} min)")
    print(f"  manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"  audio:    {audio_dir.relative_to(PROJECT_ROOT)}/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out", default="eval/bakeoff_30min",
        help="Output eval directory (relative to project root).",
    )
    ap.add_argument("--target-min", type=float, default=30.0,
                    help="Target total duration in minutes.")
    ap.add_argument("--min-s", type=float, default=3.0,
                    help="Min clip duration in seconds.")
    ap.add_argument("--max-s", type=float, default=20.0,
                    help="Max clip duration in seconds.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sada", action="store_true",
                    help="Don't try to stream SADA22 (e.g. if not gated yet).")
    args = ap.parse_args()

    eval_dir = (PROJECT_ROOT / args.out).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    build(
        eval_dir=eval_dir,
        target_min=args.target_min,
        seed=args.seed,
        skip_sada=args.skip_sada,
        min_s=args.min_s,
        max_s=args.max_s,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
