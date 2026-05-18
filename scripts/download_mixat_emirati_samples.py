from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable, List

try:
    from dataset_sample_utils import (
        _audio_from_example,
        _metadata_from_example,
        _write_wav,
        add_common_args,
        output_dir,
        sample_hf_dataset,
        write_manifest,
        write_readme,
    )
except ImportError:
    from scripts.dataset_sample_utils import (
        _audio_from_example,
        _metadata_from_example,
        _write_wav,
        add_common_args,
        output_dir,
        sample_hf_dataset,
        write_manifest,
        write_readme,
    )


DATASET_ID = "sqrk/mixat-tri"
TITLE = "MixAT Emirati-English Code-Switched Speech"
NOTES = (
    "Uses the PolyWER-updated Hugging Face dataset. The training text is the "
    "`transcript` field; `transliteration` and `translation` are preserved in metadata only. "
    "License: CC BY-NC-SA 4.0."
)


def _iter_hf_split(split: str):
    from datasets import Audio, load_dataset

    dataset = load_dataset(DATASET_ID, split=split, streaming=True)
    try:
        dataset = dataset.cast_column("audio", Audio(decode=False))
    except Exception:
        pass
    return dataset


def _download_all_splits(args: argparse.Namespace) -> int:
    out_dir = output_dir(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    splits = ("train", "test")
    for split in splits:
        print(f"[mixat_emirati] loading {DATASET_ID} split={split!r}")
        try:
            dataset = _iter_hf_split(split)
        except Exception as exc:
            print(f"[mixat_emirati] failed to load split {split}: {exc}", file=sys.stderr)
            return 1
        for example in dataset:
            arr, sr, audio_column = _audio_from_example(example)
            if arr is None or not sr:
                continue
            text = str(example.get("transcript") or "").strip()
            if not text:
                continue
            sample_id = f"mixat_emirati_{split}_{len(rows):06d}"
            wav_rel = f"audio/{sample_id}.wav"
            duration_s = _write_wav(out_dir / wav_rel, arr, sr)
            rows.append({
                "id": sample_id,
                "dataset": "mixat_emirati",
                "source": DATASET_ID,
                "split": split,
                "audio_path": wav_rel,
                "duration_s": round(duration_s, 3),
                "text": text,
                "text_column": "transcript",
                "audio_column": audio_column,
                "metadata": _metadata_from_example(example),
            })
            print(f"[mixat_emirati] saved {len(rows)}/all: {wav_rel} ({duration_s:.2f}s)")
            if args.limit > 0 and len(rows) >= args.limit:
                break
        if args.limit > 0 and len(rows) >= args.limit:
            break
    write_manifest(out_dir, rows)
    write_readme(out_dir, TITLE, DATASET_ID, rows, notes=NOTES + " Downloaded splits: train,test.")
    print(f"[mixat_emirati] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Download sample clips from MixAT Emirati-English code-switched speech.")
    add_common_args(parser, "mixat_emirati")
    for action in parser._actions:
        if action.dest == "split":
            action.help = "Split to read: train, test, or all (train+test). Default: train."
    args = parser.parse_args()
    if (args.split or "train") == "all":
        return _download_all_splits(args)
    return sample_hf_dataset(
        slug=args.slug,
        dataset_id=DATASET_ID,
        split=args.split or "train",
        limit=args.limit,
        out_dir=output_dir(args),
        title=TITLE,
        notes=NOTES,
    )


if __name__ == "__main__":
    sys.exit(main())