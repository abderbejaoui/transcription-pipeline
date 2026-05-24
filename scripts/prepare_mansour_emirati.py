from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_PAGE_URL = "https://gulfarabicresources.com/mansour/"
TIMESTAMP_RE = re.compile(r"^\s*([0-9٠-٩۰-۹]{1,2})[:：]([0-9٠-٩۰-۹]{1,2})\s*$")
LEADING_COLON_TIMESTAMP_RE = re.compile(r"^\s*[:：]([0-9٠-٩۰-۹]{4})\s*$")
YOUTUBE_RE = re.compile(r"https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_\-]+")
PDF_RE = re.compile(r"href=[\"']([^\"']+\.pdf(?:\?[^\"']*)?)[\"']", re.I)
ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
SPEAKER_RE = re.compile(r"^\s*[^:：]{1,32}[:：]\s*")
ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _slug(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_\-]+", "_", text).strip("_").lower()
    return text or "mansour"


def _download_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    parsed = urllib.parse.urlsplit(url)
    safe_url = urllib.parse.urlunsplit((
        parsed.scheme,
        parsed.netloc,
        urllib.parse.quote(urllib.parse.unquote(parsed.path), safe="/%"),
        parsed.query,
        parsed.fragment,
    ))
    with urllib.request.urlopen(safe_url) as response:
        dest.write_bytes(response.read())
    return dest


def _pdf_links(page_url: str) -> List[str]:
    html = urllib.request.urlopen(page_url).read().decode("utf-8", "ignore")
    links = []
    for match in PDF_RE.finditer(html):
        links.append(urllib.parse.urljoin(page_url, match.group(1)))
    return list(dict.fromkeys(links))


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("pypdf is required. Run: pip install pypdf") from exc
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _timestamp_seconds(line: str) -> Optional[float]:
    line = line.translate(ARABIC_DIGITS)
    leading = LEADING_COLON_TIMESTAMP_RE.match(line)
    if leading:
        raw = leading.group(1)
        minutes = int(raw[:2])
        seconds = int(raw[2:][::-1])
        if seconds < 60:
            return minutes * 60.0 + seconds
    match = TIMESTAMP_RE.match(line)
    if not match:
        return None
    # pypdf extracts timestamps in visual RTL order: ٩١:١ is displayed for
    # 1:19, ٨٢:٠١ for 10:28, etc. Reverse each side to recover mm:ss.
    seconds = int(match.group(1)[::-1])
    minutes = int(match.group(2)[::-1])
    if seconds >= 60:
        return None
    return minutes * 60.0 + seconds


def _clean_line(line: str) -> Optional[str]:
    line = line.strip()
    if not line:
        return None
    if line.startswith("©") or "Gulf Arabic Resources" in line:
        return None
    if line.startswith("http") or "youtube.com" in line:
        return None
    if line.lower().endswith(".pdf"):
        return None
    if "=" in line:
        return None
    if not ARABIC_RE.search(line):
        return None
    if re.match(r"^[0-9٠-٩۰-۹]+\s", line):
        return None
    line = SPEAKER_RE.sub("", line)
    line = re.sub(r"[0-9٠-٩۰-۹]+", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line or None


def _parse_segments(text: str) -> Tuple[Optional[str], List[Dict]]:
    youtube_url = None
    url_match = YOUTUBE_RE.search(text)
    if url_match:
        youtube_url = url_match.group(0)
    segments: List[Dict] = []
    current_start: Optional[float] = None
    current_lines: List[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        timestamp = _timestamp_seconds(line)
        if timestamp is not None:
            if current_start is not None and current_lines:
                segments.append({"start_s": current_start, "end_s": timestamp, "text": " ".join(current_lines)})
            current_start = timestamp
            current_lines = []
            continue
        cleaned = _clean_line(line)
        if cleaned and current_start is not None:
            current_lines.append(cleaned)
    return youtube_url, segments


def _download_youtube_audio(url: str, out_template: Path) -> Path:
    if shutil.which("yt-dlp") is None:
        raise RuntimeError("yt-dlp is required. Run: pip install yt-dlp")
    before = set(out_template.parent.glob(out_template.name.replace("%(ext)s", "*")))
    subprocess.run(
        [
            "yt-dlp",
            "--quiet",
            "--no-warnings",
            "-f", "bestaudio/best",
            "-o", str(out_template),
            url,
        ],
        check=True,
    )
    after = set(out_template.parent.glob(out_template.name.replace("%(ext)s", "*")))
    new_files = sorted(after - before)
    if new_files:
        return new_files[0]
    existing = sorted(after)
    if existing:
        return existing[0]
    raise FileNotFoundError("yt-dlp did not produce an audio file")


def _cut_audio(src: Path, dest: Path, start_s: float, end_s: float) -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
            "-ss", f"{start_s:.3f}",
            "-t", f"{max(0.1, end_s - start_s):.3f}",
            "-i", str(src),
            "-ac", "1",
            "-ar", "16000",
            "-sample_fmt", "s16",
            str(dest),
        ],
        check=True,
    )


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare Mansour Emirati Arabic transcript/audio chunks from PDF transcripts.")
    parser.add_argument("--page-url", default=DEFAULT_PAGE_URL)
    parser.add_argument("--pdf-url", action="append", default=[], help="Specific Mansour PDF URL. Can be repeated.")
    parser.add_argument("--out", type=Path, default=Path("data/dataset_samples/mansour_emirati"))
    parser.add_argument("--limit-episodes", type=int, default=0, help="0 means all PDFs found.")
    parser.add_argument("--limit-segments", type=int, default=0, help="0 means all accepted segments.")
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--download-audio", action="store_true", help="Download YouTube audio and cut WAV chunks.")
    args = parser.parse_args()

    out_dir = args.out
    raw_dir = out_dir / "raw"
    audio_dir = out_dir / "audio"
    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    pdf_urls = list(args.pdf_url) or _pdf_links(args.page_url)
    if args.limit_episodes > 0:
        pdf_urls = pdf_urls[: args.limit_episodes]
    rows: List[Dict] = []
    rejected: List[Dict] = []
    for episode_index, pdf_url in enumerate(pdf_urls):
        pdf_name = urllib.parse.unquote(Path(urllib.parse.urlparse(pdf_url).path).name)
        episode_id = f"mansour_{episode_index:04d}_{_slug(pdf_name.rsplit('.', 1)[0])}"
        pdf_path = _download_url(pdf_url, raw_dir / f"{episode_id}.pdf")
        text = _extract_pdf_text(pdf_path)
        youtube_url, segments = _parse_segments(text)
        source_audio: Optional[Path] = None
        if args.download_audio and youtube_url:
            source_audio = _download_youtube_audio(youtube_url, raw_dir / f"{episode_id}.%(ext)s")
        for segment_index, segment in enumerate(segments):
            duration = float(segment["end_s"] - segment["start_s"])
            base_row = {
                "id": f"{episode_id}_{segment_index:04d}",
                "dataset": "mansour_emirati",
                "source": args.page_url,
                "episode_pdf": str(pdf_path.relative_to(out_dir)),
                "youtube_url": youtube_url,
                "start_s": round(float(segment["start_s"]), 3),
                "end_s": round(float(segment["end_s"]), 3),
                "duration_s": round(duration, 3),
                "text": segment["text"],
            }
            if duration < args.min_duration_s:
                rejected.append({**base_row, "reason": "too_short"})
                continue
            if duration > args.max_duration_s:
                rejected.append({**base_row, "reason": "too_long_needs_alignment_chunking"})
                continue
            row = dict(base_row)
            if source_audio is not None:
                audio_rel = f"audio/{row['id']}.wav"
                _cut_audio(source_audio, out_dir / audio_rel, segment["start_s"], segment["end_s"])
                row["audio_path"] = audio_rel
                row["source_audio"] = str(source_audio.relative_to(out_dir))
            rows.append(row)
            print(f"[mansour_emirati] accepted {len(rows)}: {row['id']} ({duration:.2f}s)")
            if args.limit_segments > 0 and len(rows) >= args.limit_segments:
                break
        if args.limit_segments > 0 and len(rows) >= args.limit_segments:
            break

    _write_jsonl(out_dir / "manifest.jsonl", rows)
    _write_jsonl(out_dir / "rejected.jsonl", rejected)
    (out_dir / "README.md").write_text(
        "# Mansour Emirati Arabic\n\n"
        f"- Source page: {args.page_url}\n"
        f"- Accepted segments: {len(rows)}\n"
        f"- Rejected segments: {len(rejected)}\n"
        "- Requires explicit permission for model training.\n"
        "- PDF timestamps are used for chunking; audio is downloaded only with `--download-audio`.\n",
        encoding="utf-8",
    )
    print(f"[mansour_emirati] wrote {len(rows)} rows -> {out_dir / 'manifest.jsonl'}")
    return 0 if rows else 2


if __name__ == "__main__":
    sys.exit(main())