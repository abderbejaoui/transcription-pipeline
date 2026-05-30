"""Bake the v1 Gulf LoRA adapter permanently into the Qwen3-ASR base weights.

This is the Option-B prep step for the v2 medical fine-tune. Instead of throwing
away the 900h Gulf-Arabic LoRA (which is *better* than stock Qwen3 on Gulf), we
merge it into the base model so the dialect skill lives in the weights and can no
longer be forgotten. A FRESH medical LoRA is then trained on top of this merged
model (`scripts/finetune_qwen3_lora.py --model-path <this output dir>`).

Why this and not `scripts/merge_adapters.py`?
  - `merge_adapters.py` does a WEIGHTED AVERAGE of two *LoRA adapters* and writes a
    new *adapter* (small). Different purpose.
  - This script does PEFT `merge_and_unload()` of ONE adapter into the base and
    writes a FULL MODEL directory (large, ~3.4GB) that `Qwen3ASRModel.from_pretrained`
    can load directly as the new starting point.

Background (researched):
  - PEFT docs: `merge_and_unload()` is the standard sequential-fine-tuning pattern.
  - "LoRA vs Full Fine-tuning: An Illusion of Equivalence" (arXiv 2410.21228): a LoRA
    kept as a removable adapter introduces "intruder dimensions" and forgets prior
    knowledge when further trained; baking it into the base avoids that for the
    already-learned (Gulf) skill.

Usage (on the DGX, inside the venv):
  python -m scripts.merge_v1_into_base \
    --base-model Qwen/Qwen3-ASR-1.7B \
    --adapter runs/qwen3_lora_r6/final_adapter \
    --output runs/qwen3_gulf_merged_base
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen3-ASR-1.7B",
                    help="Stock Qwen3-ASR base (HF id or local dir).")
    ap.add_argument("--adapter", required=True,
                    help="Path to the trained v1 Gulf LoRA adapter dir "
                         "(the one containing adapter_config.json).")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output dir for the merged FULL model + processor.")
    args = ap.parse_args()

    adapter_dir = Path(args.adapter)
    if not (adapter_dir / "adapter_config.json").exists():
        print(f"!! no adapter_config.json in {adapter_dir} — is this a LoRA adapter dir?",
              file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel  # official wrapper, same one finetune uses

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print(f"[merge] loading base via Qwen3ASRModel.from_pretrained({args.base_model}) bf16={use_bf16}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.base_model,
        dtype=dtype,
        device_map=None,
    )
    base_model = asr_wrapper.model
    processor = asr_wrapper.processor

    print(f"[merge] attaching v1 adapter: {adapter_dir}")
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("[merge] merge_and_unload() — baking LoRA into base weights")
    merged = peft_model.merge_and_unload()

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged full model -> {args.output}")
    merged.save_pretrained(str(args.output))
    try:
        processor.save_pretrained(str(args.output))
    except Exception as e:  # noqa: BLE001 - processor save is best-effort
        print(f"[merge] WARN: processor.save_pretrained failed ({e}); "
              "copy the processor/tokenizer files from the base model dir manually.")

    (args.output / "merge_v1_info.json").write_text(
        json.dumps({
            "base_model": args.base_model,
            "adapter": str(adapter_dir),
            "method": "peft.merge_and_unload",
            "note": "Gulf v1 LoRA baked into base; use as --model-path for v2 medical LoRA.",
        }, indent=2),
        encoding="utf-8",
    )
    print(f"[merge] DONE. Point the v2 fine-tune at: --model-path {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
