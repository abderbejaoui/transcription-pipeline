from __future__ import annotations

import argparse
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


def _sample_batch_one(limit: int, out_dir: Path, batch: str) -> int:
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
        files = api.dataset_list_files(DATASET_ID).files
    except Exception as exc:
        print(f"failed to list SADA files: {exc}", file=sys.stderr)
        return 1

    prefix = f"{batch.strip('/')}/"
    audio_names = [
        getattr(f, "name", str(f))
        for f in files
        if getattr(f, "name", str(f)).startswith(prefix)
        and Path(getattr(f, "name", str(f))).suffix.lower() in AUDIO_EXTS
    ]
    audio_names = sorted(audio_names)[:limit]
    if not audio_names:
        print(f"No SADA audio files found under {prefix!r}", file=sys.stderr)
        return 2

    raw_dir = out_dir / "raw"
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
        rows.append({
            "id": sample_id,
            "dataset": "sada2022",
            "source": f"kaggle:{DATASET_ID}",
            "batch": batch,
            "audio_path": dest_rel,
            "duration_s": round(float(duration_s), 3),
            "text": "",
            "original_path": file_name,
        })
        print(f"[sada2022] saved {len(rows)}/{limit}: {dest_rel}")

    write_manifest(out_dir, rows)
    write_readme(
        out_dir,
        "SADA 2022 Saudi Audio Dataset",
        f"kaggle:{DATASET_ID}",
        rows,
        notes=f"Targeted per-file download from {batch}; transcript metadata was not exposed in this Kaggle file listing.",
    )
    print(f"[sada2022] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10 sample clips from Kaggle SADA 2022.")
    add_common_args(parser, "sada2022")
    parser.add_argument("--batch", default="batch_1", help="Kaggle folder to sample from (default: batch_1).")
    parser.add_argument("--allow-full-archive-download", action="store_true",
                        help="Download the full Kaggle archive before sampling. This can be very large.")
    args = parser.parse_args()
    if not args.allow_full_archive_download:
        return _sample_batch_one(args.limit, output_dir(args), args.batch)
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
