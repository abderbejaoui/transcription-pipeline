"""Download Casablanca multi-dialect Arabic ASR test set and convert to
bakeoff manifest format.

Casablanca (Talafha et al., 2024, arXiv:2410.04527) is the gold-standard
public benchmark for dialectal Arabic ASR. Eight dialects, conversational
speech, human-verified transcripts.

  Algeria | Egypt | Jordan | Mauritania | Morocco | Palestine | UAE | Yemen

There is NO Saudi or Kuwait config. For Gulf coverage use `UAE` (closest to
our existing SADA22 / WorldSpeech Gulf clips).

License: **CC-BY-NC-ND-4.0** — research and evaluation only.

Implementation note
-------------------
We do NOT use `datasets.load_dataset()` because the parquet-glob pattern in
recent `datasets` versions hits a fsspec bug on Python 3.12:

    ValueError: Invalid pattern: '**' can only be an entire path component

Instead we read parquet shards directly via `huggingface_hub` + `pyarrow`,
which avoids the buggy glob resolver entirely. Same approach we use for
the WorldSpeech builder.

Output:
  eval/casablanca_{dialect}/
    manifest.jsonl
    audio/<id>.wav     (16 kHz mono PCM)
    README.md

Usage:
  pip install huggingface_hub pyarrow soundfile requests
  python -m scripts.build_casablanca_testset --dialect UAE
  python -m scripts.build_casablanca_testset --dialect UAE --max-clips 30
  python -m scripts.build_casablanca_testset --dialect UAE --dialect Yemen
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPO_ID = "UBC-NLP/Casablanca"
VALID_DIALECTS = {
    "Algeria", "Egypt", "Jordan", "Mauritania",
    "Morocco", "Palestine", "UAE", "Yemen",
}
TARGET_SR = 16_000


def _resample_mono_16k(arr, sr: int):
    """Resample to 16 kHz mono. Uses soxr if available, else librosa, else passes through."""
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
        # No resampler — keep original SR. bakeoff.py resamples again at inference.
        return arr, sr


def _decode_audio_bytes(blob: bytes):
    """Return (np.ndarray, sr) from a raw audio binary blob via soundfile."""
    import soundfile as sf
    bio = io.BytesIO(blob)
    arr, sr = sf.read(bio, dtype="float32", always_2d=False)
    return arr, sr


def _iter_parquet_rows(repo_id: str, shard_paths, max_rows: int | None):
    """Yield row-dicts from each parquet shard. Streams shards one at a time;
    each shard is downloaded fully into memory, but that's typically <600 MB
    for Casablanca and we discard it once iterated."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_url
    import requests

    yielded = 0
    for shard in shard_paths:
        if max_rows is not None and yielded >= max_rows:
            return
        url = hf_hub_url(repo_id=repo_id, filename=shard, repo_type="dataset")
        print(f"  downloading {shard} ...")
        # Stream into memory; parquet readers want a file-like object.
        with requests.get(url, stream=True, timeout=600) as resp:
            resp.raise_for_status()
            buf = io.BytesIO()
            for chunk in resp.iter_content(chunk_size=1 << 20):
                buf.write(chunk)
        buf.seek(0)
        table = pq.read_table(buf)
        print(f"    rows in shard: {table.num_rows}")
        for row in table.to_pylist():
            yield row
            yielded += 1
            if max_rows is not None and yielded >= max_rows:
                return


def build_one(dialect: str, split: str, max_clips: int | None) -> int:
    if dialect not in VALID_DIALECTS:
        print(f"!! unknown dialect '{dialect}'. Valid: {sorted(VALID_DIALECTS)}",
              file=sys.stderr)
        return 1

    try:
        import soundfile  # noqa: F401  (validate availability)
        from huggingface_hub import list_repo_files
        import pyarrow  # noqa: F401
        import requests  # noqa: F401
    except ImportError as exc:
        print(f"!! missing dependency: {exc}. "
              f"Run: pip install huggingface_hub pyarrow soundfile requests",
              file=sys.stderr)
        return 1

    out_dir = PROJECT_ROOT / "eval" / f"casablanca_{dialect}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)

    print(f"\nDiscovering parquet shards for {dialect}/{split} ...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"{dialect}/{split}-"
    shards = sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))
    if not shards:
        print(f"!! no parquet shards matching '{prefix}*.parquet' in {REPO_ID}",
              file=sys.stderr)
        return 1
    print(f"  found {len(shards)} shard(s): {shards}")

    manifest = []
    skipped_empty = 0
    skipped_decode = 0

    for i, row in enumerate(_iter_parquet_rows(REPO_ID, shards, max_clips)):
        transcript = (row.get("transcription") or "").strip()
        if not transcript:
            skipped_empty += 1
            continue

        audio_field = row.get("audio")
        audio_bytes = None
        if isinstance(audio_field, dict):
            audio_bytes = audio_field.get("bytes")
        elif isinstance(audio_field, (bytes, bytearray)):
            audio_bytes = bytes(audio_field)

        if not audio_bytes:
            skipped_decode += 1
            continue

        try:
            arr, sr = _decode_audio_bytes(audio_bytes)
        except Exception as exc:
            print(f"  !! decode failed for row {i}: {exc}")
            skipped_decode += 1
            continue

        arr, sr = _resample_mono_16k(arr, sr)

        cid = f"casablanca_{dialect.lower()}_{i:04d}"
        wav_path = out_dir / "audio" / f"{cid}.wav"
        _sf_write_safe(wav_path, arr, sr)

        manifest.append({
            "id": cid,
            "category": "casablanca",
            "language": "ar",
            "audio_path": f"audio/{cid}.wav",
            "duration_s": round(float(row.get("duration") or len(arr) / sr), 2),
            "transcript": transcript,
            "source": f"casablanca:{dialect}",
            "tags": ["casablanca", dialect.lower(), split],
            "medical_terms": [],
            "gender": row.get("gender"),
            "seg_id": row.get("seg_id"),
        })

        if (len(manifest)) % 25 == 0:
            print(f"  kept={len(manifest):4d}  last={cid} "
                  f"dur={manifest[-1]['duration_s']:.1f}s")

    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n",
        encoding="utf-8",
    )

    total_dur = sum(r["duration_s"] for r in manifest)
    readme = f"""# Casablanca-{dialect} test set (split={split})

- Clips: {len(manifest)}
- Total duration: {total_dur/60:.1f} min
- Skipped: {skipped_empty} empty-transcript, {skipped_decode} decode-failed
- Source: [UBC-NLP/Casablanca](https://huggingface.co/datasets/UBC-NLP/Casablanca/tree/main/{dialect}), config `{dialect}`
- License: **CC-BY-NC-ND-4.0** — non-commercial, no-derivatives. Research/eval only.
- Citation: Talafha et al. (2024), *Casablanca: Data and Models for Multidialectal Arabic Speech Recognition*. [arXiv:2410.04527](https://arxiv.org/abs/2410.04527)

## What this is

Casablanca is the current public benchmark for dialectal Arabic ASR.
Conversational speech, human-verified transcripts. The `{dialect}` config
covers the {dialect} dialect specifically.

## How to use

```bash
python -m scripts.bakeoff --eval-dir eval/casablanca_{dialect} --models qwen3 qwen3_uae
python3 scripts/eval_standard.py --testset eval/casablanca_{dialect}
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"\nwrote manifest with {len(manifest)} clips ({total_dur/60:.1f} min)")
    print(f"  manifest : {out_dir / 'manifest.jsonl'}")
    print(f"  audio    : {out_dir / 'audio'}")
    print(f"  readme   : {out_dir / 'README.md'}")
    if skipped_empty or skipped_decode:
        print(f"  skipped  : {skipped_empty} empty, {skipped_decode} decode-failed")
    return 0


def _sf_write_safe(path: Path, arr, sr: int) -> None:
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr, sr, subtype="PCM_16")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialect", action="append", default=None,
                    help=f"Casablanca config to materialise. Repeat for multiple. "
                         f"Default: UAE. Valid: {sorted(VALID_DIALECTS)}")
    ap.add_argument("--split", default="test", choices=["test", "validation"],
                    help="Which split to materialise (default: test).")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Optional cap per dialect (default: all).")
    args = ap.parse_args()

    dialects = args.dialect or ["UAE"]
    print(f"Dialects to build: {dialects}")
    print(f"Split            : {args.split}")
    if args.max_clips:
        print(f"Max clips        : {args.max_clips}")

    rc = 0
    for d in dialects:
        if build_one(d, args.split, args.max_clips) != 0:
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
