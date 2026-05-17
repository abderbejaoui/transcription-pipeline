from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from dataset_sample_utils import add_common_args, output_dir, write_manifest, write_readme
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, write_manifest, write_readme


DATASET_ID = "Nexdata/UAE_Arabic_Spontaneous_Speech_Data"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download up to 10 WAV clips from Nexdata UAE Arabic sample.")
    add_common_args(parser, "nexdata_uae_sample")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import soundfile as sf
    except ImportError as exc:
        print(f"missing dependency: {exc}. Run: pip install huggingface_hub soundfile", file=sys.stderr)
        return 1

    out_dir = output_dir(args)
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    try:
        files = list_repo_files(DATASET_ID, repo_type="dataset")
    except Exception as exc:
        print(f"[nexdata_uae_sample] failed to list repo files: {exc}", file=sys.stderr)
        return 1

    wav_files = [f for f in files if f.lower().endswith(".wav")]
    rows = []
    for rel in wav_files[: args.limit]:
        sample_id = f"nexdata_uae_sample_{len(rows):05d}"
        try:
            local = Path(hf_hub_download(DATASET_ID, filename=rel, repo_type="dataset"))
            dest_rel = f"audio/{sample_id}.wav"
            dest = out_dir / dest_rel
            data, sr = sf.read(str(local), dtype="float32", always_2d=False)
            sf.write(str(dest), data, sr, subtype="PCM_16")
            duration_s = len(data) / sr if sr else 0.0
        except Exception as exc:
            print(f"[nexdata_uae_sample] skip {rel}: {exc}", file=sys.stderr)
            continue
        rows.append({
            "id": sample_id,
            "dataset": "nexdata_uae_sample",
            "source": DATASET_ID,
            "audio_path": dest_rel,
            "duration_s": round(float(duration_s), 3),
            "text": "",
            "original_path": rel,
        })
        print(f"[nexdata_uae_sample] saved {len(rows)}/{args.limit}: {dest_rel}")

    write_manifest(out_dir, rows)
    write_readme(
        out_dir,
        "Nexdata UAE Arabic Spontaneous Speech Sample",
        DATASET_ID,
        rows,
        notes="Direct HF file download of the tiny free WAV sample; transcripts are not bundled.",
    )
    print(f"[nexdata_uae_sample] done: {len(rows)} samples -> {out_dir}")
    if not rows:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
