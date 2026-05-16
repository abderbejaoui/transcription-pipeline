"""Build a ~30-minute UAE-Emirati ASR test set.

Source priority (best-available first):

  1. `vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k`
     The real UAE Emirati validation split (≈2.5k clips). HF-gated — you must
     `huggingface-cli login` and accept the dataset terms first. This is the
     dataset vadimbelsky reports 9.98% WER on, so it's the best apples-to-
     apples test set for UAE accent.

  2. `Nexdata/UAE_Arabic_Spontaneous_Speech_Data`
     Free 5-clip sample of a 749h commercial Emirati corpus. Tiny but
     authentic spontaneous UAE speech.

  3. WorldSpeech `ar_kw` (Kuwait) — closest free large-volume Gulf neighbor.
  4. WorldSpeech `ar_bh` (Bahrain) — second neighbor.

We pull a generous pool from each accessible source, then round-robin until
we reach the target duration.

Output schema matches `scripts/bakeoff.py`:

    eval/uae_30min/
      manifest.jsonl
      audio/<id>.wav             16 kHz mono PCM16

Run:
    source .venv/bin/activate
    python -m scripts.build_uae_testset --target-min 30

Then evaluate:
    python -m scripts.bakeoff \\
        --models qwen3 qwen3_uae \\
        --eval-dir eval/uae_30min
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Text normalization — same as bakeoff.py
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
# Audio helpers (same as build_bakeoff_testset)
# ---------------------------------------------------------------------------

SR = 16_000


def write_wav_16k_mono(arr: np.ndarray, sr: int, out_path: Path) -> float:
    import soundfile as sf

    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != SR:
        try:
            import librosa
            arr = librosa.resample(arr.astype(np.float32), orig_sr=sr, target_sr=SR)
            sr = SR
        except Exception:
            return _ffmpeg_resample(arr, sr, out_path)
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


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


Clip = Tuple[str, np.ndarray, int, str, float, Dict[str, Any]]


def _load_hf(repo_id: str, subset: Optional[str], split: str, streaming: bool):
    from datasets import load_dataset

    kwargs = {"split": split, "streaming": streaming, "trust_remote_code": False}
    if subset is None:
        return load_dataset(repo_id, **kwargs)
    return load_dataset(repo_id, subset, **kwargs)


def iter_vadimbelsky_uae(
    min_s: float, max_s: float, max_clips: int,
) -> Iterable[Clip]:
    """Stream vadimbelsky's UAE Emirati validation split (gated dataset)."""
    repo = "vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k"
    # The dataset card shows splits "train" and "validation". The 9.98% WER
    # is on the validation split; that's the one we want for parity.
    for split in ("validation", "test", "val", "dev"):
        try:
            ds = _load_hf(repo, None, split, streaming=True)
        except Exception:
            continue
        return _iter_uae_dataset(
            ds, source_name="vadimbelsky_uae_val",
            min_s=min_s, max_s=max_s, max_clips=max_clips,
        )
    print(f"  [{repo}] no validation split reachable (gated?)")
    return iter(())


def _iter_uae_dataset(
    ds,
    source_name: str,
    min_s: float,
    max_s: float,
    max_clips: int,
) -> Iterable[Clip]:
    seen = 0
    kept = 0
    for ex in ds:
        seen += 1
        if seen > max_clips * 40:
            break
        audio = ex.get("audio") or {}
        arr = audio.get("array")
        sr = int(audio.get("sampling_rate") or 0)
        text = (
            ex.get("text")
            or ex.get("transcription")
            or ex.get("transcript")
            or ex.get("sentence")
            or ex.get("normalized_text")
            or ""
        )
        text = text.strip() if isinstance(text, str) else ""
        if arr is None or sr <= 0 or len(text) < 3:
            continue
        dur = len(arr) / sr
        if dur < min_s or dur > max_s:
            continue
        yield (
            f"{source_name}_{kept:03d}",
            np.asarray(arr, dtype=np.float32),
            sr,
            text,
            dur,
            {"source": source_name},
        )
        kept += 1
        if kept >= max_clips:
            break
    print(f"  [{source_name}] scanned {seen}, kept {kept}")


def iter_nexdata_uae(
    min_s: float, max_s: float, max_clips: int,
) -> Iterable[Clip]:
    """Stream the free 5-clip sample from Nexdata's UAE corpus."""
    repo = "Nexdata/UAE_Arabic_Spontaneous_Speech_Data"
    for split in ("train", "test", "validation"):
        try:
            ds = _load_hf(repo, None, split, streaming=False)
        except Exception:
            continue
        return _iter_uae_dataset(
            ds, source_name="nexdata_uae_sample",
            min_s=min_s, max_s=max_s, max_clips=max_clips,
        )
    print(f"  [{repo}] no split reachable")
    return iter(())


def iter_worldspeech_uae_neighbors(
    countries: Iterable[str],
    min_s: float,
    max_s: float,
    max_per_country: int,
) -> Iterable[Clip]:
    """Stream WorldSpeech country splits closest to UAE (Kuwait, Bahrain)."""
    for country in countries:
        try:
            ds = _load_hf("disco-eth/WorldSpeech", country, "train", streaming=True)
        except Exception as exc:
            print(f"  [worldspeech/{country}] cannot stream: {exc!r}")
            continue
        seen = 0
        kept = 0
        for ex in ds:
            seen += 1
            if seen > max_per_country * 30:
                break
            audio = ex.get("audio") or {}
            arr = audio.get("array")
            sr = int(audio.get("sampling_rate") or 0)
            transcript = (ex.get("human_transcript") or "").strip()
            if arr is None or sr <= 0 or not transcript:
                continue
            dur = float(ex.get("duration") or (len(arr) / sr))
            if dur < min_s or dur > max_s:
                continue
            cer = ex.get("cer")
            if cer is not None and cer > 0.25:
                continue
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
        print(f"  [worldspeech/{country}] scanned {seen}, kept {kept}")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


SOURCE_PRIORITY = (
    # name, weight (how many to draw per round-robin tick), description
    ("vadimbelsky_uae_val",   3, "real UAE Emirati validation"),
    ("nexdata_uae_sample",    1, "real UAE Emirati spontaneous (sample)"),
    ("worldspeech_ar_kw",     2, "Gulf (Kuwait) free fallback"),
    ("worldspeech_ar_bh",     1, "Gulf (Bahrain) free fallback"),
)


def build(
    eval_dir: Path,
    target_min: float,
    seed: int,
    min_s: float,
    max_s: float,
    allow_neighbors: bool,
) -> None:
    random.seed(seed)
    audio_dir = eval_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = eval_dir / "manifest.jsonl"

    target_s = target_min * 60.0
    print(f"target: {target_min:.1f} min  ({target_s:.0f} s)")

    pool: Dict[str, List[Clip]] = {}

    print("\n== source 1: vadimbelsky UAE validation (gated) ==")
    pool["vadimbelsky_uae_val"] = list(
        iter_vadimbelsky_uae(min_s=min_s, max_s=max_s, max_clips=200)
    )

    print("\n== source 2: Nexdata UAE sample (free) ==")
    pool["nexdata_uae_sample"] = list(
        iter_nexdata_uae(min_s=min_s, max_s=max_s, max_clips=10)
    )

    if allow_neighbors:
        print("\n== source 3: WorldSpeech UAE neighbors (Kuwait, Bahrain) ==")
        ws_clips = list(iter_worldspeech_uae_neighbors(
            countries=["ar_kw", "ar_bh"],
            min_s=min_s, max_s=max_s, max_per_country=80,
        ))
        for clip in ws_clips:
            pool.setdefault(clip[5]["source"], []).append(clip)

    available = {k: len(v) for k, v in pool.items() if v}
    if not available:
        print("\nNo clips reachable. If vadimbelsky is gated, run "
              "`huggingface-cli login` and accept the dataset terms, "
              "or rerun with --allow-neighbors to use Kuwait/Bahrain.")
        sys.exit(1)

    print(f"\npool: {available}")

    # Shuffle each source for variety.
    for s in pool:
        random.shuffle(pool[s])

    # Greedy fill weighted by priority order. We accept neighbors only after
    # exhausting authentic UAE sources, so the testset is as UAE-pure as
    # available data allows.
    selected: List[Clip] = []
    total_s = 0.0
    cursors: Dict[str, int] = {s: 0 for s in pool}

    # Pass 1: fill primarily from authentic UAE sources.
    primary = ["vadimbelsky_uae_val", "nexdata_uae_sample"]
    for src in primary:
        if total_s >= target_s:
            break
        for _ in range(len(pool.get(src, []))):
            if total_s >= target_s:
                break
            if cursors[src] >= len(pool[src]):
                break
            clip = pool[src][cursors[src]]
            cursors[src] += 1
            selected.append(clip)
            total_s += clip[4]

    # Pass 2: top up with Gulf neighbors round-robin.
    if total_s < target_s and allow_neighbors:
        neighbor_keys = [k for k in pool.keys() if k.startswith("worldspeech_")]
        while total_s < target_s:
            progress = False
            for src in neighbor_keys:
                if total_s >= target_s:
                    break
                if cursors[src] >= len(pool[src]):
                    continue
                clip = pool[src][cursors[src]]
                cursors[src] += 1
                selected.append(clip)
                total_s += clip[4]
                progress = True
            if not progress:
                break

    print(f"\nselected {len(selected)} clips, total {total_s:.1f} s "
          f"({total_s/60:.1f} min)")
    by_src = {s: 0 for s in pool}
    for c in selected:
        by_src[c[5]["source"]] += 1
    for s, n in by_src.items():
        if n:
            print(f"  {s:35s}  {n:4d} clips")

    # Write audio + manifest.
    records: List[Dict[str, Any]] = []
    for clip_id, arr, sr, transcript, _raw_dur, meta in selected:
        wav_path = audio_dir / f"{clip_id}.wav"
        try:
            dur_s = write_wav_16k_mono(arr, sr, wav_path)
        except Exception as exc:
            print(f"  ! skip {clip_id}: {exc!r}")
            continue
        source = meta.pop("source")
        is_authentic_uae = source.startswith("vadimbelsky_uae") or source.startswith("nexdata_uae")
        category = "uae_real" if is_authentic_uae else "uae_neighbor"
        tags = [source, "uae" if is_authentic_uae else "gulf-neighbor"]
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
    real = sum(r["duration_s"] for r in records if r["category"] == "uae_real")
    neighbor = total - real
    print(f"\n✓ wrote {len(records)} clips ({total:.1f} s ≈ {total/60:.1f} min)")
    print(f"   authentic UAE: {real:.0f} s ({real/total*100:.0f}%)")
    print(f"   Gulf neighbor: {neighbor:.0f} s ({neighbor/total*100:.0f}%)")
    print(f"   manifest: {manifest_path.relative_to(PROJECT_ROOT)}")
    print(f"   audio:    {audio_dir.relative_to(PROJECT_ROOT)}/")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="eval/uae_30min",
                    help="Output eval directory (relative to project root).")
    ap.add_argument("--target-min", type=float, default=30.0)
    ap.add_argument("--min-s", type=float, default=3.0)
    ap.add_argument("--max-s", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--no-neighbors", action="store_true",
        help="Fail if authentic UAE data is insufficient; do not fall back "
             "to Kuwait/Bahrain WorldSpeech splits.",
    )
    args = ap.parse_args()

    eval_dir = (PROJECT_ROOT / args.out).resolve()
    eval_dir.mkdir(parents=True, exist_ok=True)
    build(
        eval_dir=eval_dir,
        target_min=args.target_min,
        seed=args.seed,
        min_s=args.min_s,
        max_s=args.max_s,
        allow_neighbors=not args.no_neighbors,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
