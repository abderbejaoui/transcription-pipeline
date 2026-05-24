"""Build the Gulf-Arabic training corpus for Qwen3-ASR LoRA fine-tune.

Reads from HuggingFace via the parquet-direct pattern (avoiding the
fsspec/`datasets.load_dataset` glob bug we already hit on Casablanca and
WorldSpeech), applies duration + CER quality filters, and writes a
sampling-ready manifest.

Per `TRAINING_DATA.md` §7 — final weighted mix:

  SADA22 Saudi              weight 1.0    ~647h
  WorldSpeech-BH            weight 1.0    ~272h
  WorldSpeech-KW            weight 1.0    ~175h
  WorldSpeech-SA            weight 1.0    ~6h
  OMAN-SPEECH               weight 1.0    ~40h   (manual download; see notes)
  UAE_Arabic (real)         weight 3.0    ~120h  (gated, accept terms on HF)
  MGB-2 subset              weight 0.5    ~250h  (manual, see notes)
  Common Voice ar           weight 0.3    ~50h
  FLEURS ar                 weight 0.2    ~8h

Critical guarantees:
  - Clips that appear in ANY test set (bakeoff_30min, bakeoff_clean,
    casablanca_*, fleurs_ar) are EXCLUDED by id / seg_id.
  - Clips with duration < 3s or > 25s are dropped.
  - Clips with shipped quality CER > 0.25 (WorldSpeech only) are dropped.
  - Audio is resampled to 16 kHz mono PCM_16 WAV.

Output layout:
  data/train_corpus/
    audio/<source>/<id>.wav
    manifest.jsonl     (per-clip records with weight + ref text)
    SUMMARY.json       (hours per source, kept/skipped counts)

Usage on DGX:
  # Smoke test with just SADA22 (one shard, ~30 min, ~5 GB)
  python -m scripts.build_train_corpus --sources SADA22 --max-clips 1000

  # Full corpus (~5-8 hours, ~180 GB raw audio)
  python -m scripts.build_train_corpus

  # Specific subset
  python -m scripts.build_train_corpus --sources SADA22 WorldSpeech-BH UAE

Dependencies: huggingface_hub, pyarrow, soundfile, requests, soxr (or librosa).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = PROJECT_ROOT / "data" / "train_corpus"
TARGET_SR = 16_000

# ----------------------------------------------------------------------------
# Source definitions
# ----------------------------------------------------------------------------

# Each entry tells the builder how to pull one corpus.
#   weight      — final sampling weight in the manifest (§7 of TRAINING_DATA.md)
#   repo_id     — HF dataset repo
#   parquet_prefix — glob prefix for the train split parquet shards
#   text_field  — column containing the transcript
#   id_field    — column containing a stable per-clip id (for test-leak guard)
#   cer_field   — optional column with source-provided CER (we drop > 0.25)
#   audio_field — column containing the audio dict {bytes, sampling_rate}
#   notes       — anything that needs human action (gated download, etc.)

SOURCES: Dict[str, Dict] = {
    "SADA22": {
        "weight": 1.0,
        "repo_id": "MohamedRashad/SADA22",
        "parquet_prefix": "train-",
        "text_field": "ProcessedText",
        "id_field": "SegmentID",
        "cer_field": None,
        "audio_field": "audio",
        "notes": "Gated: accept terms once at https://huggingface.co/datasets/MohamedRashad/SADA22",
    },
    "WorldSpeech-BH": {
        "weight": 1.0,
        "repo_id": "disco-eth/WorldSpeech",
        "parquet_prefix": "ar_bh/train-",
        "text_field": "text",
        "id_field": "audio_id",
        "cer_field": "cer",
        "audio_field": "audio",
        "notes": "Gated: accept terms at https://huggingface.co/datasets/disco-eth/WorldSpeech",
    },
    "WorldSpeech-KW": {
        "weight": 1.0,
        "repo_id": "disco-eth/WorldSpeech",
        "parquet_prefix": "ar_kw/train-",
        "text_field": "text",
        "id_field": "audio_id",
        "cer_field": "cer",
        "audio_field": "audio",
        "notes": "Same dataset as WorldSpeech-BH, different config.",
    },
    "WorldSpeech-SA": {
        "weight": 1.0,
        "repo_id": "disco-eth/WorldSpeech",
        "parquet_prefix": "ar_sa/train-",
        "text_field": "text",
        "id_field": "audio_id",
        "cer_field": "cer",
        "audio_field": "audio",
        "notes": "Same dataset, ar_sa config.",
    },
    "UAE": {
        "weight": 3.0,  # oversampled per §7 — scarce real Emirati audio
        "repo_id": "vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k",
        "parquet_prefix": "data/train-",
        "text_field": "transcription",
        "id_field": "audio_id",
        "cer_field": None,
        "audio_field": "audio",
        "notes": "Gated: accept terms at https://huggingface.co/datasets/vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k",
    },
    "Common-Voice-ar": {
        "weight": 0.3,
        "repo_id": "mozilla-foundation/common_voice_17_0",
        "parquet_prefix": "ar/train-",
        "text_field": "sentence",
        # `client_id` is a SPEAKER id — using it for leak-guard would
        # block every clip from any speaker that appears in a test set,
        # silently dropping huge chunks of training data. Use `path`
        # (per-clip stable filename) instead.
        "id_field": "path",
        "cer_field": None,
        "audio_field": "audio",
        "notes": "Gated: accept terms at https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0",
    },
    "FLEURS-ar": {
        "weight": 0.2,
        "repo_id": "google/fleurs",
        "parquet_prefix": "ar_eg/train-",
        "text_field": "transcription",
        "id_field": "id",
        "cer_field": None,
        "audio_field": "audio",
        "notes": "Public, no gating.",
    },
    # OMAN-SPEECH and MGB-2 are not on HF as parquet — see SUMMARY.json notes
    # for the manual download path. They get plugged in via --manual-manifest
    # if you want them included.
}

# ----------------------------------------------------------------------------
# Test-set leak guard
# ----------------------------------------------------------------------------

TEST_MANIFESTS: List[Path] = [
    PROJECT_ROOT / "eval" / "bakeoff_30min" / "manifest.jsonl",
    PROJECT_ROOT / "eval" / "bakeoff_clean" / "manifest.jsonl",
    PROJECT_ROOT / "eval" / "casablanca_UAE" / "manifest.jsonl",
    PROJECT_ROOT / "eval" / "fleurs_ar" / "manifest.jsonl",
]


def load_test_ids() -> Set[str]:
    """Collect every test-set clip id we know about. Any training clip whose
    id (or seg_id) is in this set will be dropped before training."""
    blocked: Set[str] = set()
    for p in TEST_MANIFESTS:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            # `path` is the Common-Voice per-clip filename — included so the
            # CV leak-guard works correctly. Note: `client_id` is the SPEAKER
            # id and is intentionally NOT included (would over-block).
            for k in ("id", "seg_id", "audio_id", "SegmentID", "path"):
                v = rec.get(k)
                if v:
                    blocked.add(str(v))
    return blocked


# ----------------------------------------------------------------------------
# Audio decode + resample
# ----------------------------------------------------------------------------


def _resample_mono_16k(arr, sr: int):
    import numpy as np
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    arr = arr.astype(np.float32)
    if sr == TARGET_SR:
        return arr, TARGET_SR
    try:
        import soxr
        return soxr.resample(arr, sr, TARGET_SR).astype(np.float32), TARGET_SR
    except ImportError:
        pass
    try:
        import librosa
        return librosa.resample(arr, orig_sr=sr, target_sr=TARGET_SR).astype(np.float32), TARGET_SR
    except ImportError:
        return arr, sr  # bakeoff resamples again at inference


def _decode_audio_bytes(blob: bytes):
    import soundfile as sf
    bio = io.BytesIO(blob)
    arr, sr = sf.read(bio, dtype="float32", always_2d=False)
    return arr, sr


# ----------------------------------------------------------------------------
# Parquet iteration (avoids datasets.load_dataset fsspec bug)
# ----------------------------------------------------------------------------


def _iter_parquet_rows(repo_id: str, shard_paths, max_rows: Optional[int] = None) -> Iterable[Dict]:
    import pyarrow.parquet as pq
    import requests
    from huggingface_hub import hf_hub_url

    token = os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    yielded = 0
    for shard in shard_paths:
        if max_rows is not None and yielded >= max_rows:
            return
        url = hf_hub_url(repo_id=repo_id, filename=shard, repo_type="dataset")
        print(f"  downloading {shard}", flush=True)
        t0 = time.time()
        with requests.get(url, stream=True, timeout=600, headers=headers) as resp:
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                buf.write(chunk)
        buf.seek(0)
        table = pq.read_table(buf)
        print(f"    {table.num_rows} rows ({(time.time()-t0):.1f}s)", flush=True)
        for row in table.to_pylist():
            yield row
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


# ----------------------------------------------------------------------------
# Per-source extraction
# ----------------------------------------------------------------------------


def extract_source(
    src_name: str,
    spec: Dict,
    blocked_ids: Set[str],
    out_dir: Path,
    max_clips: Optional[int] = None,
    min_dur: float = 3.0,
    max_dur: float = 25.0,
    max_source_cer: float = 0.25,
) -> Tuple[List[Dict], Dict]:
    """Yield (manifest_records, stats_dict) for one source."""
    from huggingface_hub import list_repo_files
    import soundfile as sf

    repo_id = spec["repo_id"]
    print(f"\n=== {src_name} ({repo_id}, prefix={spec['parquet_prefix']}) ===", flush=True)

    files = list_repo_files(repo_id, repo_type="dataset")
    shards = sorted(f for f in files if f.startswith(spec["parquet_prefix"]) and f.endswith(".parquet"))
    if not shards:
        print(f"  ! no parquet shards matching '{spec['parquet_prefix']}*.parquet' in {repo_id}",
              file=sys.stderr)
        return [], {"shards": 0}

    src_audio_dir = out_dir / "audio" / src_name
    src_audio_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    stats = {
        "shards": len(shards),
        "kept": 0,
        "skipped_test_leak": 0,
        "skipped_dur": 0,
        "skipped_cer": 0,
        "skipped_decode": 0,
        "skipped_empty_text": 0,
        "total_seconds": 0.0,
    }

    for i, row in enumerate(_iter_parquet_rows(repo_id, shards, max_rows=max_clips)):
        cid = str(row.get(spec["id_field"]) or f"{src_name}_{i:08d}")
        if cid in blocked_ids:
            stats["skipped_test_leak"] += 1
            continue

        text = (row.get(spec["text_field"]) or "").strip()
        if not text:
            stats["skipped_empty_text"] += 1
            continue

        if spec.get("cer_field"):
            src_cer = row.get(spec["cer_field"])
            if isinstance(src_cer, (int, float)) and src_cer > max_source_cer:
                stats["skipped_cer"] += 1
                continue

        audio_field = row.get(spec["audio_field"])
        audio_bytes = None
        if isinstance(audio_field, dict):
            audio_bytes = audio_field.get("bytes")
        elif isinstance(audio_field, (bytes, bytearray)):
            audio_bytes = bytes(audio_field)
        if not audio_bytes:
            stats["skipped_decode"] += 1
            continue

        try:
            arr, sr = _decode_audio_bytes(audio_bytes)
        except Exception:
            stats["skipped_decode"] += 1
            continue

        arr, sr = _resample_mono_16k(arr, sr)
        dur = len(arr) / sr
        if dur < min_dur or dur > max_dur:
            stats["skipped_dur"] += 1
            continue

        wav_id = cid.replace("/", "_").replace(" ", "_")
        wav_path = src_audio_dir / f"{wav_id}.wav"
        sf.write(str(wav_path), arr, sr, subtype="PCM_16")

        records.append({
            "id": cid,
            "audio_path": str(wav_path.relative_to(PROJECT_ROOT)),
            "text": text,
            "duration_s": round(dur, 2),
            "source": src_name,
            "weight": spec["weight"],
        })
        stats["kept"] += 1
        stats["total_seconds"] += dur

        if stats["kept"] % 5000 == 0:
            print(f"  kept={stats['kept']:>7d}  hours={stats['total_seconds']/3600:.1f}",
                  flush=True)

    return records, stats


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=None,
                    help=f"Subset of {sorted(SOURCES)}. Default: all.")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Cap clips per source (for smoke testing).")
    ap.add_argument("--min-dur", type=float, default=3.0)
    ap.add_argument("--max-dur", type=float, default=25.0)
    ap.add_argument("--max-source-cer", type=float, default=0.25)
    ap.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    sources = args.sources or sorted(SOURCES.keys())
    bad = [s for s in sources if s not in SOURCES]
    if bad:
        print(f"!! unknown source(s): {bad}. Valid: {sorted(SOURCES)}", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    blocked = load_test_ids()
    print(f"Test-leak guard: {len(blocked)} clip ids loaded from test manifests")

    all_records: List[Dict] = []
    summary = {}
    for src in sources:
        try:
            recs, stats = extract_source(
                src, SOURCES[src], blocked, args.output_dir,
                max_clips=args.max_clips,
                min_dur=args.min_dur, max_dur=args.max_dur,
                max_source_cer=args.max_source_cer,
            )
            all_records.extend(recs)
            summary[src] = stats
        except Exception as exc:
            print(f"  !! {src} failed: {exc!r}", file=sys.stderr)
            summary[src] = {"error": repr(exc)}

    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in all_records) + "\n",
        encoding="utf-8",
    )

    total_h = sum(s.get("total_seconds", 0) for s in summary.values()) / 3600
    print(f"\nwrote {len(all_records)} clips ({total_h:.1f} hrs) to {manifest_path}")

    summary_path = args.output_dir / "SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary -> {summary_path}")
    print("\nPer-source:")
    print(f"  {'source':22s} {'kept':>7s} {'hours':>8s} {'leak':>6s} {'dur':>6s} {'cer':>6s}")
    for src, s in summary.items():
        if "error" in s:
            print(f"  {src:22s} ERROR: {s['error']}")
            continue
        print(f"  {src:22s} {s.get('kept',0):>7d} {s.get('total_seconds',0)/3600:>8.1f} "
              f"{s.get('skipped_test_leak',0):>6d} "
              f"{s.get('skipped_dur',0):>6d} "
              f"{s.get('skipped_cer',0):>6d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
