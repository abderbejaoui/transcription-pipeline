from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

try:
    from dataset_sample_utils import AUDIO_EXTS, add_common_args, output_dir, sample_kaggle_dataset, write_manifest, write_readme
except ImportError:
    from scripts.dataset_sample_utils import AUDIO_EXTS, add_common_args, output_dir, sample_kaggle_dataset, write_manifest, write_readme


DATASET_ID = "sdaiancai/sada2022"


def _download_kaggle_file(api, file_name: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    local_path = raw_dir / file_name
    if local_path.exists():
        return local_path
    api.dataset_download_file(DATASET_ID, file_name=file_name, path=str(raw_dir), force=False, quiet=False)
    direct = raw_dir / Path(file_name).name
    zip_path = raw_dir / f"{Path(file_name).name}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(raw_dir)
        zip_path.unlink(missing_ok=True)
    if direct.exists() and direct != local_path:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(direct), str(local_path))
    if local_path.exists():
        return local_path
    matches = list(raw_dir.rglob(Path(file_name).name))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Kaggle did not create expected file for {file_name}")


def _download_sada_csvs(api, raw_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name in ("train.csv", "valid.csv", "test.csv"):
        try:
            paths.append(_download_kaggle_file(api, name, raw_dir))
        except Exception as exc:
            print(f"[sada2022] metadata warning: could not download {name}: {exc}", file=sys.stderr)
    return paths


def _list_all_kaggle_files(api) -> list[str]:
    names: list[str] = []
    token = None
    while True:
        if token:
            response = api.dataset_list_files(DATASET_ID, page_size=200, page_token=token)
        else:
            response = api.dataset_list_files(DATASET_ID, page_size=200)
        names.extend(getattr(f, "name", str(f)) for f in response.files)
        token = response.next_page_token
        if not token:
            break
    return names


def _load_segments(csv_paths: list[Path], file_names: set[str]) -> dict[str, list[dict]]:
    by_file = {name: [] for name in file_names}
    for path in csv_paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("FileName") or ""
                if filename in by_file:
                    by_file[filename].append(row)
    return by_file


def _write_segments(out_dir: Path, sample_id: str, segments: list[dict]) -> str:
    rel = f"segments/{sample_id}.segments.jsonl"
    path = out_dir / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for segment in segments:
            handle.write(json.dumps({
                "segment_id": segment.get("SegmentID"),
                "start_s": float(segment.get("SegmentStart") or 0),
                "end_s": float(segment.get("SegmentEnd") or 0),
                "duration_s": float(segment.get("SegmentLength") or 0),
                "ground_truth_text": segment.get("GroundTruthText") or "",
                "processed_text": segment.get("ProcessedText") or "",
                "speaker": segment.get("Speaker"),
                "speaker_age": segment.get("SpeakerAge"),
                "speaker_gender": segment.get("SpeakerGender"),
                "speaker_dialect": segment.get("SpeakerDialect"),
                "environment": segment.get("Environment"),
                "category": segment.get("Category"),
            }, ensure_ascii=False) + "\n")
    return rel


def _sample_batch_one(limit: int, out_dir: Path, batch: str, all_batches: bool = False) -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import soundfile as sf
    except ImportError as exc:
        print(f"missing dependency: {exc}. Run: pip install kaggle soundfile", file=sys.stderr)
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        print(f"Kaggle authentication failed: {exc}", file=sys.stderr)
        print("Run: .venv/bin/kaggle auth login", file=sys.stderr)
        return 1

    try:
        files = _list_all_kaggle_files(api)
    except Exception as exc:
        print(f"failed to list SADA files: {exc}", file=sys.stderr)
        return 1

    prefix = f"{batch.strip('/')}/"
    audio_names = [
        name
        for name in files
        if (all_batches or name.startswith(prefix))
        and Path(name).suffix.lower() in AUDIO_EXTS
    ]
    audio_names = sorted(audio_names)
    if limit > 0:
        audio_names = audio_names[:limit]
    if not audio_names:
        print(f"No SADA audio files found under {prefix!r}", file=sys.stderr)
        return 2

    raw_dir = out_dir / "raw"
    csv_paths = _download_sada_csvs(api, raw_dir)
    segments_by_file = _load_segments(csv_paths, set(audio_names))
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, file_name in enumerate(audio_names):
        try:
            local = _download_kaggle_file(api, file_name, raw_dir)
            sample_id = f"sada2022_{idx:05d}"
            dest_rel = f"audio/{sample_id}.wav"
            dest = out_dir / dest_rel
            data, sr = sf.read(str(local), dtype="float32", always_2d=False)
            sf.write(str(dest), data, sr, subtype="PCM_16")
            duration_s = len(data) / sr if sr else 0.0
        except Exception as exc:
            print(f"[sada2022] skip {file_name}: {exc}", file=sys.stderr)
            continue
        segments = segments_by_file.get(file_name, [])
        texts = [(s.get("ProcessedText") or s.get("GroundTruthText") or "").strip() for s in segments]
        segments_rel = _write_segments(out_dir, sample_id, segments) if segments else None
        rows.append({
            "id": sample_id,
            "dataset": "sada2022",
            "source": f"kaggle:{DATASET_ID}",
            "batch": batch,
            "audio_path": dest_rel,
            "duration_s": round(float(duration_s), 3),
            "text": " ".join(t for t in texts if t),
            "text_column": "ProcessedText_joined_by_FileName" if segments else None,
            "segments_path": segments_rel,
            "segment_count": len(segments),
            "original_path": file_name,
        })
        target = "all" if limit <= 0 else str(limit)
        print(f"[sada2022] saved {len(rows)}/{target}: {dest_rel}")

    write_manifest(out_dir, rows)
    write_readme(
        out_dir,
        "SADA 2022 Saudi Audio Dataset",
        f"kaggle:{DATASET_ID}",
        rows,
        notes=f"Targeted per-file download from {batch}; segment transcripts are joined from train/valid/test CSV metadata.",
    )
    print(f"[sada2022] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10 sample clips from Kaggle SADA 2022.")
    add_common_args(parser, "sada2022")
    parser.add_argument("--batch", default="batch_1", help="Kaggle folder to sample from (default: batch_1).")
    parser.add_argument("--all-batches", action="store_true",
                        help="Download from all Kaggle batch folders instead of only --batch.")
    parser.add_argument("--allow-full-archive-download", action="store_true",
                        help="Download the full Kaggle archive before sampling. This can be very large.")
    args = parser.parse_args()
    if not args.allow_full_archive_download:
        return _sample_batch_one(args.limit, output_dir(args), args.batch, all_batches=args.all_batches)
    return sample_kaggle_dataset(
        slug=args.slug,
        kaggle_id=DATASET_ID,
        limit=args.limit,
        out_dir=output_dir(args),
        title="SADA 2022 Saudi Audio Dataset",
        notes="Requires Kaggle CLI credentials. The raw Kaggle archive is kept under raw/.",
        allow_full_download=args.allow_full_archive_download,
    )


if __name__ == "__main__":
    sys.exit(main())
