#!/usr/bin/env python3
"""Download and prepare Gulf-Arabic ASR datasets into the repo's manifest format.

For each configured Hugging Face dataset this script:
  1. streams the split (so it never needs the whole set on disk at once),
  2. decodes each clip to 16 kHz mono WAV under ``data/preprocessed/<slug>/audio/``,
  3. writes a JSONL manifest consumed by ``scripts/finetune_qwen3_lora.py`` and
     ``scripts/test_asr.py``.

Manifest schema (one JSON object per line):
    {
      "audio_path": "audio/000123.wav",   # relative to the manifest file
      "text":       "النص العربي ...",
      "source":     "mixat",
      "dialect":    "emirati",
      "code_switch": true,                  # transcript contains Latin tokens
      "weight":     2.0,                     # sampler weight (Stage-2 up-weight)
      "stage":      2                        # 1 = base acoustic, 2 = CS/dialect
    }

HARD CONSTRAINT: real recorded audio only. Synthetic / TTS corpora are refused
(see ``SYNTHETIC_BLOCKLIST``).

Examples
--------
List the datasets this script knows about:
    python scripts/prepare_datasets.py --list

Prepare one dataset, capped at 200 clips for a smoke test:
    python scripts/prepare_datasets.py --dataset mixat --max-clips 200

Prepare every Stage-2 (code-switch) dataset:
    python scripts/prepare_datasets.py --stage 2

Prepare everything (will be large):
    python scripts/prepare_datasets.py --all
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "preprocessed"

# Datasets that are synthetic / TTS / spliced — NEVER ingest these.
SYNTHETIC_BLOCKLIST = {
    "vadimbelsky/uae_arabic_english_bilingual_dataset_40k",
}

# A Latin run of >=2 letters marks code-switch (English token inside Arabic).
_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_TASHKEEL = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0670]")


@dataclass
class DatasetSpec:
    slug: str                       # local folder + manifest name
    hf_id: str                      # Hugging Face dataset id
    dialect: str                    # emirati / saudi / gulf / arabic / msa
    stage: int                      # 1 base acoustic, 2 code-switch/dialect
    weight: float = 1.0             # base sampler weight for the dataset
    cs_weight: Optional[float] = None  # if set, weight for code-switch clips
    splits: List[str] = field(default_factory=lambda: ["train"])
    text_keys: List[str] = field(default_factory=list)   # override auto-detect
    audio_key: str = "audio"
    config: Optional[str] = None    # HF dataset config name
    notes: str = ""


# Only datasets that are openly loadable via `datasets.load_dataset` are wired
# here. Gated ones (ZAEBUC, Ramsa, ADI17) need a manual download first; add a
# spec with the local path once you have access.
REGISTRY: Dict[str, DatasetSpec] = {
    # --- Stage 2: code-switch -------------------------------------------------
    "mixat": DatasetSpec(
        slug="mixat", hf_id="sqrk/mixat-tri", dialect="emirati", stage=2,
        weight=2.0, cs_weight=3.0,
        text_keys=["transcript", "text"],
        notes="15h Emirati-English code-switch. CC-BY-NC-SA.",
    ),
    "scc22": DatasetSpec(
        slug="scc22", hf_id="MohamedRashad/SCC22", dialect="saudi", stage=2,
        weight=2.0, cs_weight=3.0,
        text_keys=["ProcessedText", "Original_text", "text", "transcript"],
        notes="Saudilang Code-Switch Corpus, ~5h. CC-BY-NC-SA, ungated.",
    ),
    # --- Stage 1: base Gulf / Arabic acoustic --------------------------------
    "sada22": DatasetSpec(
        slug="sada22", hf_id="MohamedRashad/SADA22", dialect="saudi", stage=1,
        weight=1.0, text_keys=["ProcessedText", "text", "transcript"],
        notes="668h Saudi Khaliji broadcast. CC-BY-NC-SA, ungated.",
    ),
    "emirati_shows": DatasetSpec(
        slug="emirati_shows",
        hf_id="eabayed/EmiratiDialictShowsAudioTranscription",
        dialect="emirati", stage=1, weight=1.5,
        text_keys=["text", "transcript", "transcription", "sentence"],
        notes="467 pure-Emirati clips. AFL-3.0, ungated.",
    ),
    "sawtarabi": DatasetSpec(
        slug="sawtarabi", hf_id="ArabicSpeech/sawtarabi", dialect="arabic",
        stage=1, weight=1.0,
        text_keys=["text", "transcript", "transcription", "sentence"],
        notes="~3.3k Arabic clips, small base pool.",
    ),
    "masc": DatasetSpec(
        slug="masc", hf_id="pain/MASC", dialect="arabic", stage=1, weight=1.0,
        text_keys=["text", "transcript", "transcription", "sentence"],
        notes="~1000h multi-dialect Arabic. CC-BY-4.0. Largest open base pool.",
    ),
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def is_code_switch(text: str) -> bool:
    return bool(_LATIN_RUN.search(text or ""))


def _normalize_ws(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    return re.sub(r"\s+", " ", text).strip()


def _pick_text(row: Dict[str, Any], keys: List[str]) -> str:
    candidates = keys or [
        "text", "transcript", "transcription", "sentence",
        "raw_transcription", "normalized_text", "arabic",
    ]
    for key in candidates:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_ws(val)
    return ""


def _save_wav(audio_obj: Any, dst: Path, target_sr: int = 16_000) -> bool:
    """Write a HF audio object (dict with array+sampling_rate, or a path)
    to a 16 kHz mono WAV. Returns True on success.

    Raises on failure so the caller can record *why* a clip was skipped
    (a silent ``return False`` once hid a 100%-skip bug for a whole run).
    """
    import numpy as np
    import soundfile as sf

    arr = None
    sr = None

    # Newer datasets (>=3.x) return a torchcodec AudioDecoder instead of a
    # {array, sampling_rate} dict. It is NOT a dict and has no .decode(); it
    # is subscriptable via __getitem__("array") / __getitem__("sampling_rate"),
    # which call get_all_samples() under the hood. Detect by class name so we
    # don't hard-depend on torchcodec being importable.
    if type(audio_obj).__name__ == "AudioDecoder" or (
        hasattr(audio_obj, "get_all_samples")
    ):
        try:
            samples = audio_obj.get_all_samples()
            data = samples.data
            # torch tensor -> numpy
            if hasattr(data, "cpu"):
                data = data.cpu().numpy()
            else:
                data = np.asarray(data)
            # torchcodec returns (channels, samples); average to mono.
            if data.ndim > 1:
                data = np.mean(data, axis=tuple(range(data.ndim - 1)))
            arr = data
            sr = int(samples.sample_rate)
        except Exception:
            # Fall back to the subscript API exposed by the datasets wrapper.
            try:
                arr = np.asarray(audio_obj["array"])
                sr = int(audio_obj["sampling_rate"])
            except Exception:
                pass  # fall through to the generic handlers / final raise

    if arr is None and isinstance(audio_obj, dict):
        arr = audio_obj.get("array")
        sr = audio_obj.get("sampling_rate")
        if arr is None and audio_obj.get("path"):
            audio_obj = audio_obj["path"]
        elif arr is None and audio_obj.get("bytes"):
            # Streaming datasets sometimes hand back undecoded bytes.
            import io
            arr, sr = sf.read(io.BytesIO(audio_obj["bytes"]),
                              dtype="float32", always_2d=False)

    if arr is None and isinstance(audio_obj, str):
        import librosa
        arr, sr = librosa.load(audio_obj, sr=target_sr, mono=True)
    if arr is None:
        raise ValueError(f"could not extract audio array from {type(audio_obj)!r}")

    arr = np.asarray(arr, dtype="float32")
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr and sr != target_sr:
        try:
            import soxr
            arr = soxr.resample(arr, sr, target_sr)
        except Exception:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
    if arr.size == 0:
        raise ValueError("decoded audio is empty")
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, arr, target_sr)
    return True


def prepare_one(
    spec: DatasetSpec,
    out_root: Path,
    max_clips: Optional[int],
    target_sr: int = 16_000,
) -> Path:
    from datasets import load_dataset

    if spec.hf_id.lower() in SYNTHETIC_BLOCKLIST:
        raise ValueError(
            f"Refusing to prepare {spec.hf_id}: it is on the synthetic blocklist."
        )

    out_dir = out_root / spec.slug
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"

    print(f"[prep] {spec.slug} <- {spec.hf_id} "
          f"(dialect={spec.dialect}, stage={spec.stage})")

    written = 0
    n_cs = 0
    skip_no_text = 0
    skip_no_audio = 0
    skip_decode = 0
    first_row_dumped = False
    first_decode_err_dumped = False
    with manifest_path.open("w", encoding="utf-8") as mf:
        for split in spec.splits:
            try:
                ds = load_dataset(
                    spec.hf_id, spec.config, split=split, streaming=True,
                )
            except Exception as exc:
                print(f"[prep]   split '{split}' unavailable: {exc!r}")
                continue
            # Force the audio column to decode to {array, sampling_rate}.
            # Without this, some streaming datasets hand back an undecoded
            # path/bytes dict and EVERY clip silently fails to save.
            try:
                from datasets import Audio
                if spec.audio_key in (ds.features or {}):
                    ds = ds.cast_column(
                        spec.audio_key, Audio(sampling_rate=target_sr)
                    )
            except Exception as exc:
                print(f"[prep]   (could not cast audio column: {exc!r})")

            for row in ds:
                if max_clips is not None and written >= max_clips:
                    break
                # Dump the schema of the very first row so a wrong text/audio
                # key name is obvious instead of producing a silent 0-clip run.
                if not first_row_dumped:
                    first_row_dumped = True
                    print(f"[prep]   first-row keys: {list(row.keys())}")

                text = _pick_text(row, spec.text_keys)
                if not text:
                    skip_no_text += 1
                    continue
                audio_obj = row.get(spec.audio_key) or row.get("audio")
                if audio_obj is None:
                    skip_no_audio += 1
                    continue
                rel = f"audio/{written:07d}.wav"
                try:
                    _save_wav(audio_obj, audio_dir / f"{written:07d}.wav", target_sr)
                except Exception as exc:
                    skip_decode += 1
                    if not first_decode_err_dumped:
                        first_decode_err_dumped = True
                        import traceback
                        print(f"[prep]   first decode failure (shown once): {exc!r}")
                        traceback.print_exc()
                    continue
                cs = is_code_switch(text)
                if cs:
                    n_cs += 1
                weight = spec.weight
                if cs and spec.cs_weight is not None:
                    weight = spec.cs_weight
                mf.write(json.dumps({
                    "audio_path": rel,
                    "text": text,
                    "source": spec.slug,
                    "dialect": spec.dialect,
                    "code_switch": cs,
                    "weight": weight,
                    "stage": spec.stage,
                }, ensure_ascii=False) + "\n")
                written += 1
                if written % 500 == 0:
                    print(f"[prep]   {written} clips ({n_cs} code-switch)...")
            if max_clips is not None and written >= max_clips:
                break

    skipped = skip_no_text + skip_no_audio + skip_decode
    summary = {
        "slug": spec.slug, "hf_id": spec.hf_id, "dialect": spec.dialect,
        "stage": spec.stage, "clips": written, "code_switch_clips": n_cs,
        "skipped": skipped, "skip_no_text": skip_no_text,
        "skip_no_audio": skip_no_audio, "skip_decode": skip_decode,
        "notes": spec.notes,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[prep] {spec.slug}: wrote {written} clips ({n_cs} CS), "
          f"skipped {skipped} "
          f"(no_text={skip_no_text}, no_audio={skip_no_audio}, "
          f"decode_fail={skip_decode}) -> {manifest_path}")
    if written == 0:
        print(f"[prep] WARNING: {spec.slug} produced 0 clips. "
              f"Check the first-row keys above against text_keys="
              f"{spec.text_keys} and audio_key='{spec.audio_key}'.")
    return manifest_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="Single dataset slug from the registry.")
    ap.add_argument("--stage", type=int, choices=[1, 2],
                    help="Prepare all datasets for this curriculum stage.")
    ap.add_argument("--all", action="store_true", help="Prepare every dataset.")
    ap.add_argument("--list", action="store_true", help="List datasets and exit.")
    ap.add_argument("--max-clips", type=int, default=None,
                    help="Cap clips per dataset (smoke test).")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    args = ap.parse_args()

    if args.list:
        print(f"{'slug':<16}{'stage':<7}{'dialect':<10}hf_id")
        for spec in REGISTRY.values():
            print(f"{spec.slug:<16}{spec.stage:<7}{spec.dialect:<10}{spec.hf_id}")
            if spec.notes:
                print(f"{'':<33}{spec.notes}")
        return 0

    if args.dataset:
        specs = [REGISTRY[args.dataset]] if args.dataset in REGISTRY else None
        if specs is None:
            print(f"Unknown dataset '{args.dataset}'. Use --list.", file=sys.stderr)
            return 2
    elif args.stage is not None:
        specs = [s for s in REGISTRY.values() if s.stage == args.stage]
    elif args.all:
        specs = list(REGISTRY.values())
    else:
        ap.error("Pass one of --dataset, --stage, --all, or --list.")
        return 2

    args.out_root.mkdir(parents=True, exist_ok=True)
    prepared: List[str] = []
    for spec in specs:
        try:
            prepare_one(spec, args.out_root, args.max_clips)
            prepared.append(spec.slug)
        except Exception as exc:
            print(f"[prep] {spec.slug} FAILED: {exc!r}", file=sys.stderr)

    print(f"[prep] done. prepared: {', '.join(prepared) if prepared else 'none'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
