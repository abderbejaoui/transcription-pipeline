"""Weighted-average merge of two (or more) LoRA adapters — the final step
of ILT (Iterative LoRA Tuning).

Per the user's plan:
  - Train Round-1 LoRA on full corpus → adapter_r1/
  - Mine hard examples with `mine_hard_examples.py` → hard.jsonl
  - Train Round-2 LoRA on hard.jsonl → adapter_r2/
  - Merge with β=0.7 / 0.3 (Round-1 dominates):

      python -m scripts.merge_adapters \
        --adapters adapter_r1 adapter_r2 \
        --weights 0.7 0.3 \
        --output adapter_ilt

The output is itself a LoRA adapter usable by PEFT — not a merged-into-base
model — so it stays small (~50 MB) and you can keep iterating.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapters", nargs="+", required=True, help="Adapter dirs.")
    ap.add_argument("--weights", nargs="+", type=float, required=True,
                    help="Mixing weights, must match --adapters count and sum to 1.")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B")
    args = ap.parse_args()

    if len(args.adapters) != len(args.weights):
        print("!! --adapters and --weights must have same length", file=sys.stderr)
        return 1
    if abs(sum(args.weights) - 1.0) > 1e-3:
        print(f"!! --weights must sum to 1.0 (got {sum(args.weights):.4f})", file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForSpeechSeq2Seq

    print(f"[merge] loading base {args.base_model}")
    base = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # PEFT's add_weighted_adapter does exactly this.
    print(f"[merge] loading adapter 0: {args.adapters[0]} w={args.weights[0]}")
    model = PeftModel.from_pretrained(base, args.adapters[0], adapter_name="r0")
    for i, (ad, _w) in enumerate(zip(args.adapters[1:], args.weights[1:]), start=1):
        print(f"[merge] loading adapter {i}: {ad} w={_w}")
        model.load_adapter(ad, adapter_name=f"r{i}")

    adapter_names = [f"r{i}" for i in range(len(args.adapters))]
    print(f"[merge] add_weighted_adapter({adapter_names}, weights={args.weights})")
    model.add_weighted_adapter(
        adapters=adapter_names,
        weights=args.weights,
        adapter_name="merged",
        combination_type="linear",
    )
    model.set_adapter("merged")

    args.output.mkdir(parents=True, exist_ok=True)
    # Save only the merged adapter.
    model.save_pretrained(str(args.output), selected_adapters=["merged"])
    (args.output / "merge_info.json").write_text(
        json.dumps({
            "base_model": args.base_model,
            "adapters": [str(a) for a in args.adapters],
            "weights": list(args.weights),
            "combination_type": "linear",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] wrote merged adapter -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
