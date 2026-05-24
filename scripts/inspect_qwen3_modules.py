"""Print the LoRA-eligible Linear modules in Qwen3-ASR.

Run this once on the DGX before fine-tuning to verify the LoRA target list.
It will:
  - Load Qwen/Qwen3-ASR-1.7B via the official wrapper (no GPU needed if
    you use --cpu-only).
  - Walk `model.named_modules()` and print every nn.Linear, grouped by
    sub-tree (audio_tower vs language_model).
  - Show how many would be picked up by the default suffix filter
    (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj) under
    language_model only.

Usage:
  python -m scripts.inspect_qwen3_modules
  python -m scripts.inspect_qwen3_modules --cpu-only
"""

from __future__ import annotations

import argparse
from collections import defaultdict


DEFAULT_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--cpu-only", action="store_true",
                    help="Force CPU load (slower but works without GPU).")
    ap.add_argument("--suffixes", nargs="+", default=DEFAULT_SUFFIXES)
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel

    print(f"[load] {args.model_path}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        device_map="cpu" if args.cpu_only else None,
    )
    model = asr_wrapper.model

    buckets = defaultdict(list)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            if "audio_tower" in name or "audio_encoder" in name:
                bucket = "audio_tower"
            elif "language_model" in name:
                bucket = "language_model"
            else:
                bucket = "other"
            buckets[bucket].append(name)

    for bucket in ("audio_tower", "language_model", "other"):
        names = buckets.get(bucket, [])
        print(f"\n=== {bucket}: {len(names)} Linear modules ===")
        for n in names[:8]:
            print(f"  {n}")
        if len(names) > 8:
            print(f"  ... ({len(names) - 8} more)")

    # Default-filter count
    matched = [
        n for n in buckets["language_model"]
        if n.rsplit(".", 1)[-1] in set(args.suffixes)
    ]
    print(f"\n[summary] LoRA targets (language_model & suffix in {args.suffixes}):")
    print(f"          {len(matched)} modules")
    for n in matched[:6]:
        print(f"  {n}")
    if len(matched) > 6:
        print(f"  ... ({len(matched) - 6} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
