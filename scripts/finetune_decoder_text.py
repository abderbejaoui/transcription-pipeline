"""Text-only LoRA adaptation of Qwen3-ASR's LLM decoder on Arabic medical text.

Purpose: shift the decoder's language-model prior toward Gulf-clinic
vocabulary (UAE drug names, Arabic diseases, clinical syntax) WITHOUT
touching the audio encoder or shipping any audio at all.

The decoder of Qwen3-ASR is a Qwen3 causal LM living at
`model.thinker.model.language_model`. We freeze the wrapper, freeze the
audio tower, and apply LoRA to the same Linears we trained in Phase 1
(q/k/v/o/gate/up/down) inside `language_model.*` only.

The output adapter is fully compatible with `scripts/merge_adapters.py`,
so you can blend it with the Phase-1 ASR adapter:

    python -m scripts.merge_adapters \\
        --adapters runs/qwen3_lora_ilt runs/qwen3_medical_text \\
        --weights 0.6 0.4 \\
        --output runs/qwen3_gulf_medical

Usage:
  python -m scripts.finetune_decoder_text \\
    --train-corpus data/medical_text/corpus.jsonl \\
    --output-dir runs/qwen3_medical_text \\
    --num-epochs 2

Notes:
  - This is a Causal-LM training loop, not a Seq2Seq one. The collator
    builds a single input_ids tensor per sentence and labels=input_ids
    (with -100 on pad).
  - There is NO audio path in this script. We bypass the Qwen3-ASR
    processor entirely and tokenize directly with `processor.tokenizer`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Manifest -> token-id list
# ---------------------------------------------------------------------------


def load_corpus(path: Path) -> List[Dict[str, Any]]:
    recs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        text = rec.get("text") or rec.get("sentence") or ""
        if not text:
            continue
        recs.append({
            "text": text,
            "source": rec.get("source", "unknown"),
            "weight": float(rec.get("weight", 1.0)),
        })
    return recs


# ---------------------------------------------------------------------------
# Pack short sentences together for training efficiency
# (standard causal-LM concat-and-chunk).
# ---------------------------------------------------------------------------


def tokenize_and_pack(
    records: List[Dict[str, Any]],
    tokenizer,
    block_size: int = 1024,
):
    eos = tokenizer.eos_token_id
    if eos is None:
        eos = tokenizer.pad_token_id or 0

    long_buffer: List[int] = []
    blocks: List[List[int]] = []
    for r in records:
        ids = tokenizer(r["text"], add_special_tokens=False)["input_ids"]
        long_buffer.extend(ids)
        long_buffer.append(eos)
        while len(long_buffer) >= block_size:
            blocks.append(long_buffer[:block_size])
            long_buffer = long_buffer[block_size:]
    # Drop the trailing partial block — keeps every train example exactly
    # the same length so we don't need attention-mask padding.
    return blocks


@dataclass
class CausalLMBlockDataset:
    """Tiny torch Dataset wrapper around pre-tokenised, equal-length blocks."""
    blocks: List[List[int]]

    def __len__(self) -> int:
        return len(self.blocks)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        import torch
        ids = torch.tensor(self.blocks[idx], dtype=torch.long)
        return {"input_ids": ids, "labels": ids.clone()}


@dataclass
class BlockCollator:
    """Stack pre-packed blocks into a batch tensor. No padding needed because
    every block is exactly `block_size` tokens long."""
    def __call__(self, batch):
        import torch
        input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"input_ids": input_ids, "labels": labels}


# ---------------------------------------------------------------------------
# LoRA on the LLM decoder of Qwen3-ASR
# ---------------------------------------------------------------------------


DEFAULT_LORA_TARGET_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def _find_language_model_targets(model, suffixes: Sequence[str]) -> List[str]:
    """Same logic as scripts/finetune_qwen3_lora.py — pick Linears under
    `language_model.*` only, skipping the audio tower entirely."""
    import torch.nn as nn
    targets: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if "audio_tower" in name or "audio_encoder" in name:
            continue
        if "language_model" not in name:
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail in suffixes:
            targets.append(name)
    return targets


def _get_language_model(asr_wrapper):
    """The actual causal-LM module to train. For Qwen3-ASR this is
    `wrapper.model.thinker.model.language_model` (or `wrapper.model.thinker`
    depending on version). We grab the wrapper.model so PEFT can attach
    adapters to all decoder layers consistently — same module the ASR
    fine-tune adapter targets, so merging works."""
    return asr_wrapper.model


def apply_lora(model, target_suffixes, r, alpha, dropout):
    from peft import LoraConfig, get_peft_model, TaskType
    for p in model.parameters():
        p.requires_grad = False
    for n, p in model.named_parameters():
        if "audio_tower" in n or "audio_encoder" in n:
            p.requires_grad = False
    targets = _find_language_model_targets(model, target_suffixes)
    if not targets:
        raise RuntimeError(
            "No LoRA target modules found under language_model. Run "
            "scripts.inspect_qwen3_modules and update the suffix list."
        )
    print(f"[lora] {len(targets)} target modules (first 3: {targets[:3]})")
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=targets,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Custom forward — bypass the multimodal wrapper. We need the thinker LM
# to take input_ids and labels only (no audio inputs).
# ---------------------------------------------------------------------------


def patch_text_only_forward(model):
    """Route the wrapper's forward to thinker(input_ids=..., labels=...)
    so a standard HF Trainer Causal-LM loop works."""
    cls = model.__class__
    if getattr(cls, "_text_only_patched", False):
        return

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=None,
            feature_attention_mask=None,
            labels=labels,
            **kwargs,
        )
    cls.forward = forward
    cls._text_only_patched = True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--train-corpus", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-suffixes", nargs="+",
                    default=DEFAULT_LORA_TARGET_SUFFIXES)
    ap.add_argument("--per-device-train-batch-size", type=int, default=8)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=8)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-epochs", type=float, default=2)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--save-steps", type=int, default=500)
    ap.add_argument("--save-total-limit", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import torch
    from qwen_asr import Qwen3ASRModel
    from transformers import Trainer, TrainingArguments

    args.output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"[load] {args.model_path} bf16={use_bf16}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = _get_language_model(asr_wrapper)
    processor = asr_wrapper.processor
    tokenizer = processor.tokenizer

    patch_text_only_forward(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = apply_lora(
        model,
        target_suffixes=args.lora_target_suffixes,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
    )

    print(f"[data] {args.train_corpus}")
    records = load_corpus(args.train_corpus)
    print(f"[data] {len(records)} sentences")

    print(f"[tokenize] packing into blocks of {args.block_size} tokens")
    blocks = tokenize_and_pack(records, tokenizer, block_size=args.block_size)
    print(f"[tokenize] {len(blocks)} blocks "
          f"({len(blocks)*args.block_size:,} tokens)")
    if len(blocks) < 100:
        print(
            "!! WARNING: very small corpus. Adaptation may have negligible "
            "effect. Aim for >= 10M tokens (>= 10k blocks at 1024 tokens).",
            file=sys.stderr,
        )

    ds = CausalLMBlockDataset(blocks=blocks)
    collator = BlockCollator()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type="linear",
        bf16=use_bf16,
        fp16=not use_bf16,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        optim="adamw_torch_fused",
    )

    class CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for k, v in list(inputs.items()):
                    if torch.is_tensor(v) and v.is_floating_point():
                        inputs[k] = v.to(dtype=model_dtype)
            return inputs

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    print("[train] starting")
    trainer.train()
    print("[train] done — saving final adapter")
    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    print(f"[saved] {final_dir}")
    print("\nNext step: merge with your Phase-1 ASR adapter, e.g.:")
    print(f"  python -m scripts.merge_adapters \\")
    print(f"      --adapters runs/qwen3_lora_ilt {final_dir} \\")
    print(f"      --weights 0.6 0.4 \\")
    print(f"      --output runs/qwen3_gulf_medical")
    return 0


if __name__ == "__main__":
    sys.exit(main())
