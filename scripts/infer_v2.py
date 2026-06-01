"""Inference / A-B test for the v2 medical LoRA arms.

Loads the official Qwen3-ASR wrapper, applies a PEFT LoRA adapter onto the
underlying HF model (`wrapper.model`), and transcribes one or more audio clips.

This is for *qualitative* spot-checking (listen to a clip, see what each arm
transcribes). For quantitative WER scoring of cached predictions use
`scripts/eval_v2.py` instead.

Arms:
  A = stock base (`Qwen/Qwen3-ASR-1.7B`) + medical LoRA A
  B = Gulf-merged base (`runs/qwen3_gulf_merged_base`) + medical LoRA B

Audio source (pick ONE):
  --audio PATH                 transcribe a single file
  --manifest PATH [--limit N]  transcribe the first N rows of a manifest
                               (audio paths are resolved relative to the
                               manifest dir if not absolute)

Examples:
  python3 scripts/infer_v2.py --audio some/clip.wav
  python3 scripts/infer_v2.py --manifest eval/gulf_medical_v1/manifest.jsonl --limit 5
  python3 scripts/infer_v2.py --manifest eval/gulf_medical_v1/manifest.jsonl --arm B
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import torch
from peft import PeftModel
from qwen_asr import Qwen3ASRModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARMS = {
    "A": {
        "base": "Qwen/Qwen3-ASR-1.7B",
        "lora": "runs/qwen3_lora_v2_medical_A/final_adapter",
        "label": "ARM A (stock base + medical LoRA A)",
    },
    "B": {
        "base": "runs/qwen3_gulf_merged_base",
        "lora": "runs/qwen3_lora_v2_medical_B/final_adapter",
        "label": "ARM B (Gulf-merged base + medical LoRA B)",
    },
}


def resolve_audio(audio_path: str, manifest_dir: Optional[Path]) -> Path:
    """Resolve a possibly-relative audio path to an absolute one."""
    p = Path(audio_path)
    if p.is_absolute():
        return p
    if manifest_dir is not None:
        for cand in (manifest_dir / p, manifest_dir.parent / p,
                     manifest_dir.parent.parent / p):
            if cand.exists():
                return cand.resolve()
    return (PROJECT_ROOT / p).resolve()


def load_manifest_rows(manifest_path: Path, limit: int) -> List[Tuple[Path, str]]:
    mdir = manifest_path.parent.resolve()
    rows: List[Tuple[Path, str]] = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ap = row.get("audio_path") or row.get("audio") or row.get("path")
            if not ap:
                continue
            rows.append((resolve_audio(ap, mdir), row.get("text", "")))
            if len(rows) >= limit:
                break
    return rows


def transcribe(clips: List[Tuple[Path, str]], base_model: str,
               lora_path: Optional[str]) -> List[str]:
    """Load a (base + optional LoRA) once and transcribe every clip."""
    print(f"\n--- loading base: {base_model} ---")
    wrapper = Qwen3ASRModel.from_pretrained(base_model)

    if lora_path:
        print(f"--- applying LoRA: {lora_path} ---")
        wrapper.model = PeftModel.from_pretrained(wrapper.model, lora_path)

    wrapper.model.eval()
    wrapper.model.to("cuda")
    processor = wrapper.processor

    preds: List[str] = []
    for audio_path, ref in clips:
        if not audio_path.exists():
            print(f"  [skip] audio not found: {audio_path}")
            preds.append("<MISSING_AUDIO>")
            continue

        audio, _ = librosa.load(str(audio_path), sr=16000)
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        decoder_input_ids = processor.tokenizer(
            "<|ar|>", return_tensors="pt"
        ).input_ids.to("cuda")

        with torch.no_grad():
            out = wrapper.model.generate(
                **inputs,
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=256,
            )

        generated_ids = out[0][decoder_input_ids.shape[1]:]
        text = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        preds.append(text)

    # free GPU so the next arm can load
    del wrapper
    gc.collect()
    torch.cuda.empty_cache()
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--audio", help="single audio file to transcribe")
    src.add_argument("--manifest", help="manifest whose rows to transcribe")
    ap.add_argument("--limit", type=int, default=5,
                    help="max rows from --manifest (default 5)")
    ap.add_argument("--arm", choices=["A", "B", "both"], default="both",
                    help="which arm(s) to run (default both)")
    args = ap.parse_args()

    if args.audio:
        clips = [(resolve_audio(args.audio, None), "")]
    else:
        clips = load_manifest_rows(Path(args.manifest), args.limit)
        if not clips:
            raise SystemExit(f"No usable rows in {args.manifest}")

    arms = ["A", "B"] if args.arm == "both" else [args.arm]
    results = {}
    for arm in arms:
        cfg = ARMS[arm]
        print("\n" + "=" * 60)
        print(cfg["label"])
        print("=" * 60)
        results[arm] = transcribe(clips, cfg["base"], cfg["lora"])

    print("\n" + "#" * 60)
    print("RESULTS")
    print("#" * 60)
    for idx, (audio_path, ref) in enumerate(clips):
        print(f"\n[{idx + 1}] {audio_path.name}")
        if ref:
            print(f"    REF: {ref}")
        for arm in arms:
            print(f"    {arm}:   {results[arm][idx]}")


if __name__ == "__main__":
    main()
