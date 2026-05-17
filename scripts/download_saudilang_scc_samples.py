from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dataset_sample_utils import add_common_args, output_dir, write_manifest, write_readme
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, write_manifest, write_readme


DATASET_ID = "SDAIANCAI/Saudilang-Code-Switch-Corpus"


def _pick_text(row: Dict[str, Any]) -> str:
    for key in ("ProcessedText", "Original_text", "text", "transcript", "transcription"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _float(row: Dict[str, Any], key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def _download_youtube_audio(url: str, out_path: Path) -> None:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is required for Saudilang audio. Run: pip install yt-dlp")
    subprocess.run(
        [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-f", "bestaudio/best",
            "-o", str(out_path),
            url,
        ],
        check=True,
    )


def _cut_segment(src: Path, dst: Path, start_s: float, end_s: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required for Saudilang segment cutting")
    dst.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.1, end_s - start_s)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-t", f"{duration:.3f}",
            "-i", str(src),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(dst),
        ],
        check=True,
    )


def _load_rows(split: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(f"missing dependency: {exc}. Run: pip install datasets") from exc
    return load_dataset(DATASET_ID, split=split)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download 10 Saudilang SCC samples from HF annotations and cut YouTube audio segments."
    )
    add_common_args(parser, "saudilang_scc")
    parser.add_argument("--download-audio", action="store_true",
                        help="Use yt-dlp + ffmpeg to cut YouTube audio. Without this, writes metadata only.")
    args = parser.parse_args()

    out_dir = output_dir(args)
    audio_dir = out_dir / "audio"
    raw_dir = out_dir / "raw_youtube"
    audio_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    split = args.split or "train"
    try:
        ds = _load_rows(split)
    except Exception as exc:
        print(f"[saudilang_scc] failed to load {DATASET_ID}: {exc}", file=sys.stderr)
        return 1

    rows: List[Dict[str, Any]] = []
    downloaded_by_link: Dict[str, Path] = {}
    errors: List[Dict[str, str]] = []

    with tempfile.TemporaryDirectory(prefix="saudilang_") as tmp_name:
        tmp_dir = Path(tmp_name)
        for raw in ds:
            if len(rows) >= args.limit:
                break
            link = raw.get("Link") or raw.get("link")
            if not isinstance(link, str) or not link.strip():
                continue
            text = _pick_text(raw)
            if not text:
                continue
            start_s = _float(raw, "SegmentStart")
            end_s = _float(raw, "SegmentEnd", start_s + _float(raw, "SegmentLength", 0.0))
            if end_s <= start_s:
                continue

            sample_id = f"saudilang_scc_{len(rows):05d}"
            wav_rel: Optional[str] = None
            duration_s = round(end_s - start_s, 3)
            if args.download_audio:
                try:
                    if link not in downloaded_by_link:
                        video_index = len(downloaded_by_link)
                        src_template = tmp_dir / f"video_{video_index:03d}.%(ext)s"
                        _download_youtube_audio(link, src_template)
                        matches = sorted(tmp_dir.glob(f"video_{video_index:03d}.*"))
                        if not matches:
                            raise RuntimeError("yt-dlp did not produce an audio file")
                        downloaded_by_link[link] = matches[0]
                    wav_rel = f"audio/{sample_id}.wav"
                    _cut_segment(downloaded_by_link[link], out_dir / wav_rel, start_s, end_s)
                except Exception as exc:
                    errors.append({"id": sample_id, "link": link, "error": str(exc)})
                    print(f"[saudilang_scc] skip {sample_id}: {exc}", file=sys.stderr)
                    continue

            rows.append({
                "id": sample_id,
                "dataset": "saudilang_scc",
                "source": DATASET_ID,
                "split": split,
                "audio_path": wav_rel,
                "duration_s": duration_s,
                "text": text,
                "youtube_url": link,
                "segment_start_s": start_s,
                "segment_end_s": end_s,
                "segment_id": raw.get("Segment_ID"),
                "metadata": {
                    "language": raw.get("Language"),
                    "environment": raw.get("Environment"),
                    "speaker": raw.get("Speaker"),
                    "speaker_gender": raw.get("SpeakerGender"),
                },
            })
            print(f"[saudilang_scc] saved {len(rows)}/{args.limit}: {wav_rel or 'metadata-only'}")

    write_manifest(out_dir, rows)
    write_readme(
        out_dir,
        "Saudilang Code-Switch Corpus",
        DATASET_ID,
        rows,
        notes=(
            "HF provides segment annotations with YouTube links. This script cuts "
            "the referenced audio segments with yt-dlp + ffmpeg unless --metadata-only is used."
        ),
    )
    if errors:
        (out_dir / "errors.jsonl").write_text(
            "\n".join(json.dumps(e, ensure_ascii=False) for e in errors) + "\n",
            encoding="utf-8",
        )
    print(f"[saudilang_scc] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
