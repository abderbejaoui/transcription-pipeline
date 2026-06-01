"""LoRA fine-tuning of Qwen2.5-1.5B-Instruct from HITL user corrections.

Reads user corrections from ``data/user_corrections.jsonl`` (or a custom path)
and fine-tunes the local 4-bit LLM corrector to learn from those corrections.

Each correction record should have:
  - "original": the raw ASR text or span the user changed
  - "corrected": what the user changed it to
  - "context": optional surrounding text for context
  - "timestamp": optional when the correction was made

The script:
  1. Loads the base model in 4-bit (same config as llm_corrector.py)
  2. Applies LoRA adapters to the attention + FFN linears
  3. Trains on (original -> corrected) pairs
  4. Saves the adapter to ``output_dir/final_adapter``

Usage:
  python -m scripts.finetune_llm \\
    --data-path data/user_corrections.jsonl \\
    --output-dir runs/llm_finetune_r1 \\
    --num-epochs 3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_corrections(path: Path) -> List[Dict[str, str]]:
    """Load correction records from a JSONL file."""
    records: List[Dict[str, str]] = []
    if not path.exists():
        logger.warning("Correction data not found at %s", path)
        return records
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "original" not in rec or "corrected" not in rec:
                continue
            records.append(rec)
    return records


def format_correction_pairs(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format correction records into (input, output, system) triples."""
    system_prompt = (
        "You are a Gulf Arabic medical transcription corrector. "
        "Fix ASR errors in clinical dictation while preserving meaning."
    )
    pairs: List[Dict[str, str]] = []
    for rec in records:
        original = rec["original"].strip()
        corrected = rec["corrected"].strip()
        if not original or not corrected:
            continue
        context = rec.get("context", "").strip()
        if context:
            user_msg = f"Correct: {original}  (context: {context})"
        else:
            user_msg = f"Correct: {original}"
        pairs.append({"input": user_msg, "output": corrected, "system": system_prompt})
    return pairs


# ---------------------------------------------------------------------------
# Model loading (4-bit Qwen2.5-1.5B)
# ---------------------------------------------------------------------------


def load_model_and_tokenizer(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
) -> tuple:
    """Load the base model in 4-bit and its tokenizer."""
    logger.info("Loading model: %s (4-bit)", model_name)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded OK")
    return model, tokenizer


# ---------------------------------------------------------------------------
# LoRA application
# ---------------------------------------------------------------------------


def apply_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Apply LoRA adapters to attention + FFN linears."""
    for param in model.parameters():
        param.requires_grad = False
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]
    lora_config = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Label-masking tokenizer: mask system+user tokens so loss only
# flows through the assistant response.
# ---------------------------------------------------------------------------


def _tokenize_with_mask(
    examples: Dict[str, List[str]],
    tokenizer: Any,
    max_length: int,
) -> Dict[str, Any]:
    """Tokenize chat pairs, masking system+user tokens in labels.

    For each (input, output, system) triple:
      1. Tokenize the user portion (system + user message), keep the IDs.
      2. Tokenize the assistant response separately.
      3. Concatenate, with the user portion masked to -100 in labels.
    """
    inputs_list = examples["input"]
    outputs_list = examples["output"]
    systems = examples.get("system", [""] * len(inputs_list))

    all_input_ids: list[list[int]] = []
    all_labels: list[list[int]] = []

    for inp, out, sys_p in zip(inputs_list, outputs_list, systems):
        # User portion
        user_msgs = [
            {"role": "system", "content": sys_p},
            {"role": "user", "content": inp},
        ]
        user_text = tokenizer.apply_chat_template(
            user_msgs, tokenize=False, add_generation_prompt=True,
        )
        user_ids = tokenizer.encode(user_text, add_special_tokens=False)

        # Assistant portion
        assistant_msgs = [{"role": "assistant", "content": out}]
        assistant_text = tokenizer.apply_chat_template(
            assistant_msgs, tokenize=False, add_generation_prompt=False,
        )
        assistant_ids = tokenizer.encode(assistant_text, add_special_tokens=False)
        eos_id = tokenizer.eos_token_id
        if eos_id is not None:
            assistant_ids = assistant_ids + [eos_id]

        full_ids = user_ids + assistant_ids
        labels = ([-100] * len(user_ids)) + list(assistant_ids)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]
            labels = labels[:max_length]

        all_input_ids.append(full_ids)
        all_labels.append(labels)

    return {"input_ids": all_input_ids, "labels": all_labels}


# ---------------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------------


def _collate_fn(batch, tokenizer):
    """Pad to max length in batch, mask padding in labels."""
    input_ids = [torch.tensor(ex["input_ids"], dtype=torch.long) for ex in batch]
    labels = [torch.tensor(ex["labels"], dtype=torch.long) for ex in batch]
    padded_inputs = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id,
    )
    padded_labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=-100,
    )
    attention_mask = (padded_inputs != tokenizer.pad_token_id).long()
    return {"input_ids": padded_inputs, "labels": padded_labels, "attention_mask": attention_mask}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen2.5-1.5B from user corrections"
    )
    ap.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument(
        "--data-path", type=Path,
        default=PROJECT_ROOT / "data" / "user_corrections.jsonl",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--num-epochs", type=float, default=3)
    ap.add_argument("--per-device-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=4)
    ap.add_argument("--learning-rate", type=float, default=2e-4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-steps", type=int, default=10)
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=100)
    args = ap.parse_args()

    # --- Load corrections ---
    records = load_corrections(args.data_path)
    if not records:
        logger.error("No correction records found in %s.", args.data_path)
        return 1
    logger.info("Loaded %d correction records", len(records))

    pairs = format_correction_pairs(records)
    logger.info("Formatted %d training pairs", len(pairs))

    # --- Load model + LoRA ---
    model, tokenizer = load_model_and_tokenizer(args.model_name)
    model = apply_lora(model, r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # --- Tokenize dataset with label masking ---
    ds = Dataset.from_list(pairs)

    def _tokenize_fn(examples):
        return _tokenize_with_mask(examples, tokenizer, args.max_length)

    ds = ds.map(_tokenize_fn, batched=True, remove_columns=["input", "output", "system"])

    # --- Training arguments ---
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] < 8,
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        report_to=["tensorboard"],
        optim="adamw_torch_fused",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        data_collator=lambda batch: _collate_fn(batch, tokenizer),
        tokenizer=tokenizer,
    )

    # --- Train ---
    logger.info("Starting training (%d samples, %d epochs)", len(ds), args.num_epochs)
    trainer.train()
    logger.info("Training complete")

    # --- Save adapter ---
    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Adapter saved to %s", final_dir)

    # --- Sample predictions ---
    logger.info("Sample predictions after training:")
    model.eval()
    test_inputs = [
        "Correct: Patient has هستوري of دايابيتس",
        "Correct: hyperglacymia and wheezeng",
    ]
    for test in test_inputs:
        messages = [
            {"role": "system", "content": "You are a Gulf Arabic medical transcription corrector."},
            {"role": "user", "content": test},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        logger.info("  Input:  %s", test)
        logger.info("  Output: %s", result)

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    sys.exit(main())
