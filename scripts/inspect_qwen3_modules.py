"""Print the LoRA-eligible Linear modules in Qwen3-ASR.

Run this once on the DGX before fine-tuning to verify the LoRA target list.
It will:
  - Load Qwen/Qwen3-ASR-1.7B via the official wrapper (no GPU needed if
    you use --cpu-only).
  - Walk `model.named_modules()` and print every nn.Linear, grouped by
    sub-tree (audio_tower vs decoder). In Qwen3-ASR-1.7B the LLM decoder
    lives at `thinker.model.layers.*` (there is NO `language_model` name),
    which is exactly what the trainer's _find_lora_target_modules targets.
  - Show how many would be picked up by the default suffix filter
    (q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj) under the
    decoder blocks only.

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
            # The LLM decoder lives at `thinker.model.layers.*` in this build
            # (NOT under a `language_model` sub-name). That is exactly what the
            # trainer's _find_lora_target_modules() targets, so bucket it as the
            # decoder when its name carries `model.layers`.
            elif "model.layers" in name:
                bucket = "decoder"
            else:
                bucket = "other"
            buckets[bucket].append(name)

    for bucket in ("audio_tower", "decoder", "other"):
        names = buckets.get(bucket, [])
        print(f"\n=== {bucket}: {len(names)} Linear modules ===")
        for n in names[:8]:
            print(f"  {n}")
        if len(names) > 8:
            print(f"  ... ({len(names) - 8} more)")

    # Default-filter count — mirror the trainer's selection: decoder Linear
    # modules under `model.layers` whose name ends in one of the suffixes.
    matched = [
        n for n in buckets["decoder"]
        if n.rsplit(".", 1)[-1] in set(args.suffixes)
    ]
    print(f"\n[summary] LoRA targets (decoder `model.layers` & suffix in {args.suffixes}):")
    print(f"          {len(matched)} modules")
    for n in matched[:6]:
        print(f"  {n}")
    if len(matched) > 6:
        print(f"  ... ({len(matched) - 6} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
