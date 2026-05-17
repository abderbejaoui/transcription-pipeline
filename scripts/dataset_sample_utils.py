"""Small-sample dataset download helpers.

The scripts in ``scripts/download_*_samples.py`` use these helpers to pull a
quick inspection sample from one dataset at a time. Defaults are intentionally
small: 10 audio examples into ``data/dataset_samples/<dataset>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = PROJECT_ROOT / "data" / "dataset_samples"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm", ".aac"}
TEXT_COLUMNS = (
    "text",
    "sentence",
    "transcript",
    "transcription",
    "raw_transcription",
    "translation",
    "normalized_text",
    "arabic",
    "label",
)
AUDIO_COLUMNS = ("audio", "path", "file", "filename", "audio_path", "wav", "speech")


def add_common_args(parser: argparse.ArgumentParser, slug: str) -> None:
    parser.add_argument("--limit", type=int, default=10, help="Number of samples to save.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_ROOT,
                        help="Root folder for all dataset samples.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Exact output folder. Defaults to --out-root/<dataset>.")
    parser.add_argument("--split", default=None,
                        help="Dataset split to read. Script-specific default is used when omitted.")
    parser.add_argument("--seed", type=int, default=42, help="Reserved for future random sampling.")
    parser.set_defaults(slug=slug)


def output_dir(args: argparse.Namespace) -> Path:
    return args.out if args.out is not None else args.out_root / args.slug


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) <= 10 and all(isinstance(v, (str, int, float, bool, type(None))) for v in value):
            return list(value)
    return None


def _metadata_from_example(example: Dict[str, Any]) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    for key, value in example.items():
        if key in AUDIO_COLUMNS or key == "audio":
            continue
        safe = _safe_scalar(value)
        if safe is not None:
            meta[key] = safe
    return meta


def _pick_text(example: Dict[str, Any]) -> Tuple[str, Optional[str]]:
    for key in TEXT_COLUMNS:
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip(), key
    for key, value in example.items():
        if isinstance(value, str) and value.strip() and key not in AUDIO_COLUMNS:
            return value.strip(), key
    return "", None


def _audio_from_example(example: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[int], Optional[str]]:
    audio = example.get("audio")
    if isinstance(audio, dict):
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if array is not None and sampling_rate:
            return np.asarray(array, dtype=np.float32), int(sampling_rate), "audio"
        path = audio.get("path")
        if path:
            arr, sr = _read_audio_path(Path(path))
            if arr is not None and sr is not None:
                return arr, sr, "audio.path"
        data = audio.get("bytes")
        if data:
            try:
                import io

                arr, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
                return np.asarray(arr, dtype=np.float32), int(sr), "audio.bytes"
            except Exception:
                suffix = Path(str(audio.get("path") or "audio.bin")).suffix or ".bin"
                arr, sr = _decode_audio_bytes_with_ffmpeg(data, suffix)
                return arr, sr, "audio.bytes.ffmpeg"

    for key in AUDIO_COLUMNS:
        value = example.get(key)
        if isinstance(value, str):
            arr_sr = _read_audio_path(Path(value))
            if arr_sr[0] is not None:
                return arr_sr[0], arr_sr[1], key
    return None, None, None


def _read_audio_path(path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if not path.exists() or path.suffix.lower() not in AUDIO_EXTS:
        return None, None
    try:
        arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
        return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception:
        return None, None


def _decode_audio_bytes_with_ffmpeg(data: bytes, suffix: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    if shutil.which("ffmpeg") is None:
        return None, None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix) as src, tempfile.NamedTemporaryFile(suffix=".wav") as dst:
            src.write(data)
            src.flush()
            subprocess.run(
                [
                    "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
                    "-i", src.name,
                    "-ac", "1",
                    "-ar", "16000",
                    dst.name,
                ],
                check=True,
            )
            arr, sr = sf.read(dst.name, dtype="float32", always_2d=False)
            return np.asarray(arr, dtype=np.float32), int(sr)
    except Exception:
        return None, None


def _write_wav(path: Path, arr: np.ndarray, sr: int) -> float:
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), arr.astype(np.float32), sr, subtype="PCM_16")
    return float(len(arr) / sr) if sr else 0.0


def write_manifest(out_dir: Path, rows: Sequence[Dict[str, Any]]) -> None:
    manifest = out_dir / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_readme(out_dir: Path, title: str, source: str, rows: Sequence[Dict[str, Any]], notes: str = "") -> None:
    total_s = sum(float(row.get("duration_s") or 0.0) for row in rows)
    content = [
        f"# {title}",
        "",
        f"- Source: `{source}`",
        f"- Samples: {len(rows)}",
        f"- Total duration: {total_s / 60:.2f} min",
        "- Audio folder: `audio/`",
        "- Manifest: `manifest.jsonl`",
    ]
    if notes:
        content += ["", notes.strip()]
    content.append("")
    out_dir.joinpath("README.md").write_text("\n".join(content), encoding="utf-8")


def sample_hf_dataset(
    *,
    slug: str,
    dataset_id: str,
    config: Optional[str] = None,
    split: str = "train",
    limit: int = 10,
    out_dir: Path,
    streaming: bool = True,
    trust_remote_code: bool = False,
    token: Optional[str] = None,
    title: Optional[str] = None,
    notes: str = "",
) -> int:
    try:
        from datasets import Audio, load_dataset
    except ImportError as exc:
        print(f"Missing dependency: {exc}. Run: pip install datasets soundfile", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    kwargs: Dict[str, Any] = {
        "split": split,
        "streaming": streaming,
        "trust_remote_code": trust_remote_code,
    }
    hf_token = token or os.environ.get("HF_TOKEN")
    if hf_token:
        kwargs["token"] = hf_token

    print(f"[{slug}] loading Hugging Face dataset {dataset_id} config={config!r} split={split!r}")
    try:
        if config:
            dataset = load_dataset(dataset_id, config, **kwargs)
        else:
            dataset = load_dataset(dataset_id, **kwargs)
    except Exception as exc:
        print(f"[{slug}] failed to load dataset: {exc}", file=sys.stderr)
        print("If this is gated, run `huggingface-cli login` and accept the dataset terms first.", file=sys.stderr)
        return 1

    try:
        features = getattr(dataset, "features", {}) or {}
        if "audio" in features:
            dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        pass

    rows: List[Dict[str, Any]] = []
    for example in dataset:
        arr, sr, audio_column = _audio_from_example(example)
        if arr is None or not sr:
            continue
        text, text_column = _pick_text(example)
        sample_id = f"{slug}_{len(rows):05d}"
        wav_rel = f"audio/{sample_id}.wav"
        duration_s = _write_wav(out_dir / wav_rel, arr, sr)
        rows.append({
            "id": sample_id,
            "dataset": slug,
            "source": dataset_id,
            "config": config,
            "split": split,
            "audio_path": wav_rel,
            "duration_s": round(duration_s, 3),
            "text": text,
            "text_column": text_column,
            "audio_column": audio_column,
            "metadata": _metadata_from_example(example),
        })
        print(f"[{slug}] saved {len(rows)}/{limit}: {wav_rel} ({duration_s:.2f}s)")
        if len(rows) >= limit:
            break

    write_manifest(out_dir, rows)
    write_readme(out_dir, title or slug, dataset_id, rows, notes=notes)
    print(f"[{slug}] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        print(f"[{slug}] no decodable audio examples found; inspect the dataset schema.", file=sys.stderr)
        return 2
    return 0


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd) if cwd else None, check=True)


def _load_transcript_maps(root: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for csv_path in root.rglob("*.csv"):
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    text = ""
                    for key in TEXT_COLUMNS:
                        if row.get(key):
                            text = str(row[key]).strip()
                            break
                    name = row.get("name") or row.get("file") or row.get("filename") or row.get("path") or row.get("audio")
                    if name and text:
                        mapping[Path(str(name)).stem.lower()] = text
        except Exception:
            continue
    for jsonl_path in root.rglob("*.jsonl"):
        try:
            for line in jsonl_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text, _ = _pick_text(row)
                name = row.get("name") or row.get("file") or row.get("filename") or row.get("path") or row.get("audio")
                if name and text:
                    mapping[Path(str(name)).stem.lower()] = text
        except Exception:
            continue
    return mapping


def sample_kaggle_dataset(
    *,
    slug: str,
    kaggle_id: str,
    limit: int = 10,
    out_dir: Path,
    title: Optional[str] = None,
    notes: str = "",
    allow_full_download: bool = False,
) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    marker = raw_dir / ".downloaded"
    if not marker.exists():
        if not allow_full_download:
            print(
                f"[{slug}] refusing to download the full Kaggle archive for a {limit}-sample preview.",
                file=sys.stderr,
            )
            print(
                f"[{slug}] Kaggle does not provide streamable row-level sampling through this helper. "
                "Pass --allow-full-archive-download if you intentionally want the whole archive first.",
                file=sys.stderr,
            )
            return 3
        if shutil.which("kaggle") is None:
            print("Missing Kaggle CLI. Run: pip install kaggle", file=sys.stderr)
            print("Then configure ~/.kaggle/kaggle.json before running this script.", file=sys.stderr)
            return 1
        _run(["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(raw_dir), "--unzip"])
        for zip_path in raw_dir.glob("*.zip"):
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(raw_dir)
        marker.write_text(kaggle_id + "\n", encoding="utf-8")
    else:
        print(f"[{slug}] using existing Kaggle download in {raw_dir}")

    transcript_by_stem = _load_transcript_maps(raw_dir)
    audio_files = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in AUDIO_EXTS)

    rows: List[Dict[str, Any]] = []
    for source_path in audio_files[:limit]:
        sample_id = f"{slug}_{len(rows):05d}"
        dest_rel = f"audio/{sample_id}{source_path.suffix.lower()}"
        dest_path = out_dir / dest_rel
        shutil.copy2(source_path, dest_path)
        duration_s = 0.0
        try:
            info = sf.info(str(dest_path))
            duration_s = float(info.duration)
        except Exception:
            pass
        rows.append({
            "id": sample_id,
            "dataset": slug,
            "source": f"kaggle:{kaggle_id}",
            "audio_path": dest_rel,
            "duration_s": round(duration_s, 3),
            "text": transcript_by_stem.get(source_path.stem.lower(), ""),
            "original_path": str(source_path.relative_to(raw_dir)),
        })
        print(f"[{slug}] saved {len(rows)}/{limit}: {dest_rel}")

    write_manifest(out_dir, rows)
    write_readme(out_dir, title or slug, f"kaggle:{kaggle_id}", rows, notes=notes)
    print(f"[{slug}] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        print(f"[{slug}] no audio files found after Kaggle download.", file=sys.stderr)
        return 2
    return 0
