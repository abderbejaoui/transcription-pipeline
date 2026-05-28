"""Extract a small set of UAE Emirati Arabic reference WAVs from the
existing training corpus, to be used as voice-cloning references by
VoxCPM2 during the medical-data synthesis pass.

Why
---
VoxCPM2's natural-language voice prompts ("Gulf Arabic doctor ...") do
NOT reliably steer to a specific Arabic dialect — the model card only
explicitly lists Chinese dialects as steerable. The DOCUMENTED way to
get a target accent out of VoxCPM2 is voice cloning from a short
reference clip ("Controllable Cloning" / "Ultimate Cloning" features).

This script samples ~10 clean reference clips from the vadimbelsky UAE
dataset (which is already in our 900h training data) and writes:

  data/tts_references/
    ref_001.wav          (4-10 s, 16 kHz mono)
    ref_001.txt          (exact transcript of ref_001.wav)
    ref_002.wav
    ref_002.txt
    ...
    references.jsonl     manifest with path + text + tags

Selection criteria
------------------
- Source: prefer `UAE` (vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k);
  fall back to `Mixat-Emirati` if no UAE clips are in the manifest.
- Duration: 4.0–10.0 seconds (VoxCPM2 docs say short clips work; 5-10s
  is the documented sweet spot).
- Audio loudness: skip clips with peak <-30 dBFS (too quiet to clone).
- Transcript: must be present and non-empty.
- Diversity: try to pick across distinct `audio_id` prefixes so the
  reference set spans multiple speakers, not 10 clips from one person.

Usage
-----
    python3 scripts/extract_uae_references.py \\
        --manifest data/dgx_full/preprocessed_audios/splits/train.jsonl \\
        --out data/tts_references \\
        --n 10
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional


def _row_source(row: Dict[str, Any]) -> str:
    """Best-effort dataset id from filename or source field."""
    for key in ("source", "dataset", "corpus"):
        if key in row and row[key]:
            return str(row[key]).lower()
    p = (
        row.get("audio_filepath") or row.get("audio_path")
        or row.get("audio") or ""
    )
    p = str(p).lower()
    # Order matters — more specific patterns first.
    if "nexdata_uae" in p or "nexdata-uae" in p:
        return "nexdata_uae"
    if "vadim" in p or "uae_bilingual" in p or "uae_arabic_english" in p:
        return "vadimbelsky_uae"
    if "mixat" in p:
        return "mixat"
    if "uae" in p:
        return "uae_other"
    if "sada" in p:
        return "sada"
    if "worldspeech" in p or "ar_bh" in p or "ar_kw" in p or "ar_sa" in p:
        return "worldspeech"
    if "oman" in p:
        return "oman"
    return "unknown"


def _row_audio(row: Dict[str, Any]) -> Optional[str]:
    for k in ("audio_filepath", "audio_path", "audio", "path"):
        v = row.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _resolve_audio_path(
    rel_or_abs: str, manifest_parent: Path, audio_root: Optional[Path]
) -> Optional[Path]:
    """Try several common roots to resolve the audio path.

    Manifest paths in this repo are usually relative to the manifest's
    grandparent (data/dgx_full/preprocessed_audios/). We try:
      1. as-is (absolute or relative to CWD)
      2. relative to --audio-root if given
      3. relative to manifest's parent (.../splits/)
      4. relative to manifest's grandparent (.../preprocessed_audios/)
    """
    candidates = []
    p = Path(rel_or_abs)
    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(Path.cwd() / p)
        if audio_root:
            candidates.append(audio_root / p)
        candidates.append(manifest_parent / p)
        candidates.append(manifest_parent.parent / p)
    for c in candidates:
        if c.exists() and c.is_file():
            return c
    return None


def _row_text(row: Dict[str, Any]) -> Optional[str]:
    for k in ("text", "transcription", "transcript", "sentence"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _row_duration(row: Dict[str, Any]) -> float:
    for k in ("duration_s", "duration"):
        v = row.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return 0.0


def _row_speaker_proxy(row: Dict[str, Any]) -> str:
    """Cheap speaker proxy — use the first 10 chars of the audio id."""
    aid = row.get("audio_id") or row.get("id") or ""
    aid = str(aid)
    if aid:
        return aid[:12]
    p = _row_audio(row) or ""
    return Path(p).stem[:12]


def _wav_peak_dbfs(path: Path) -> float:
    """Quick peak detection from a WAV file without loading numpy."""
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getsampwidth() != 2:
                return -80.0  # treat non-16-bit as too quiet
            frames = wf.readframes(min(wf.getnframes(), 16000 * 5))
            if not frames:
                return -80.0
            import struct
            samples = struct.unpack(f"<{len(frames) // 2}h", frames)
            peak = max(abs(s) for s in samples)
            if peak <= 0:
                return -80.0
            import math
            return 20 * math.log10(peak / 32767.0)
    except Exception:
        return -80.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="Source train manifest with UAE clips.")
    ap.add_argument("--out", default="data/tts_references",
                    help="Output directory.")
    ap.add_argument("--n", type=int, default=10,
                    help="Number of references to extract.")
    ap.add_argument("--min-dur", type=float, default=4.0)
    ap.add_argument("--max-dur", type=float, default=10.0)
    ap.add_argument("--peak-min-dbfs", type=float, default=-30.0,
                    help="Reject clips quieter than this (dBFS).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--preferred-sources", nargs="+",
                    default=["mixat", "vadimbelsky_uae",
                             "nexdata_uae", "uae_other"],
                    help="Preferred source datasets in priority order. "
                         "mixat is preferred for the medical project because "
                         "it is real spontaneous Emirati conversational speech "
                         "with code-switching — the exact target distribution. "
                         "nexdata_uae is mostly UAE parliamentary MSA which is "
                         "the wrong dialect for clinical use.")
    ap.add_argument("--audio-root",
                    help="Root directory for relative audio paths. "
                         "Auto-detected from manifest path if not given. "
                         "For this repo typically "
                         "data/dgx_full/preprocessed_audios/")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest_path = Path(args.manifest).resolve()
    manifest_parent = manifest_path.parent
    audio_root: Optional[Path] = (
        Path(args.audio_root).resolve() if args.audio_root else None
    )
    print(f"[ref] manifest:       {manifest_path}")
    print(f"[ref] manifest parent: {manifest_parent}")
    if audio_root:
        print(f"[ref] audio root:     {audio_root}")
    else:
        print(f"[ref] audio root:     (auto, will try several candidates)")

    # 1. Read manifest, bucket by source. Drop unresolvable paths early.
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    n_total = 0
    n_missing = 0
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            dur = _row_duration(row)
            if dur < args.min_dur or dur > args.max_dur:
                continue
            raw_audio = _row_audio(row)
            if not raw_audio or _row_text(row) is None:
                continue
            resolved = _resolve_audio_path(raw_audio, manifest_parent, audio_root)
            if resolved is None:
                n_missing += 1
                continue
            # Stash the resolved path back on the row so the rest of the
            # pipeline doesn't have to repeat the work.
            row["_resolved_audio"] = str(resolved)
            src = _row_source(row)
            by_source.setdefault(src, []).append(row)
            n_total += 1

    print(f"[ref] loaded {n_total} candidate clips from {len(by_source)} sources "
          f"(dropped {n_missing} with unresolvable paths)")
    for src, rows in sorted(by_source.items()):
        print(f"  {src:<20} {len(rows):>6}")

    # 2. Pick the source pool, in priority order.
    pool: List[Dict[str, Any]] = []
    for src in args.preferred_sources:
        if by_source.get(src):
            print(f"[ref] using source: {src} ({len(by_source[src])} clips)")
            pool = by_source[src]
            break
    if not pool:
        print(f"[ref] WARN no UAE-like source found. "
              f"Falling back to any Gulf source.")
        for src, rows in by_source.items():
            pool.extend(rows)

    if not pool:
        raise SystemExit("[ref] no usable clips in manifest.")

    # 3. Pick n with speaker diversity. We gather a generous superset
    # (5x target) and let the loudness filter trim further. If the
    # speaker proxy is collapsing too aggressively (small dataset where
    # all clips share a filename prefix) we relax the dedup so we still
    # get enough candidates.
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    seen_speakers: set[str] = set()
    chosen: List[Dict[str, Any]] = []
    target_superset = args.n * 5
    # First pass: strict speaker dedup.
    for row in pool:
        if len(chosen) >= target_superset:
            break
        spk = _row_speaker_proxy(row)
        if spk in seen_speakers:
            continue
        seen_speakers.add(spk)
        chosen.append(row)
    # Second pass: if we're starved, drop the dedup and just take more
    # clips so the loudness filter has more to chew through.
    if len(chosen) < args.n * 2:
        print(f"[ref] speaker dedup produced only {len(chosen)} candidates; "
              f"relaxing dedup to gather more")
        already = {id(r) for r in chosen}
        for row in pool:
            if len(chosen) >= target_superset:
                break
            if id(row) in already:
                continue
            chosen.append(row)
    print(f"[ref] candidate superset: {len(chosen)} clips "
          f"(target {target_superset})")

    # 4. Validate loudness + write refs. Track drop reasons so we can
    # tell what went wrong if we end up under target.
    refs_jsonl = out / "references.jsonl"
    fh = refs_jsonl.open("w", encoding="utf-8")
    written = 0
    n_missing = 0
    n_quiet = 0
    for idx, row in enumerate(chosen):
        if written >= args.n:
            break
        src_audio = Path(row.get("_resolved_audio") or _row_audio(row) or "")
        if not src_audio.exists():
            print(f"[ref]   skip (missing): {src_audio}")
            n_missing += 1
            continue
        peak = _wav_peak_dbfs(src_audio)
        if peak < args.peak_min_dbfs:
            print(f"[ref]   skip (quiet, {peak:.1f} dBFS): {src_audio.name}")
            n_quiet += 1
            continue
        dst_wav = out / f"ref_{written + 1:03d}.wav"
        dst_txt = out / f"ref_{written + 1:03d}.txt"
        shutil.copy2(src_audio, dst_wav)
        text = _row_text(row) or ""
        dst_txt.write_text(text, encoding="utf-8")
        ref_record = {
            "ref_id": f"ref_{written + 1:03d}",
            "audio_path": str(dst_wav.resolve()),
            "transcript": text,
            "duration_s": _row_duration(row),
            "source": _row_source(row),
            "original": str(src_audio),
            "peak_dbfs": round(peak, 1),
        }
        fh.write(json.dumps(ref_record, ensure_ascii=False) + "\n")
        written += 1
        print(f"[ref] {ref_record['ref_id']} "
              f"({ref_record['duration_s']:.1f}s, {peak:.1f} dBFS): "
              f"{text[:60]}...")

    fh.close()
    print(f"\n[ref] DONE")
    print(f"[ref]   written  : {written}")
    print(f"[ref]   missing  : {n_missing}")
    print(f"[ref]   too quiet: {n_quiet}")
    print(f"[ref]   manifest : {refs_jsonl}")
    if written < args.n:
        print(f"[ref] WARN only {written}/{args.n} references usable.")
        print(f"[ref] Suggestions:")
        print(f"[ref]   - Lower --peak-min-dbfs (default -30, try -40)")
        print(f"[ref]   - Add --preferred-sources mixat (more spontaneous)")
        print(f"[ref]   - Loosen --min-dur / --max-dur (default 4-10s)")


if __name__ == "__main__":
    main()
