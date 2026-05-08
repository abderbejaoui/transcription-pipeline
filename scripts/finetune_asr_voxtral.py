from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, Trainer, TrainingArguments

from scripts.finetune_utils import (
    build_dataset_maps,
    build_datasets,
    evaluate_wer,
    infer_lora_targets,
    load_model_and_processor,
    make_ctc_collator,
    make_seq2seq_collator,
    save_metrics,
)

MODEL_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Voxtral-Mini-4B on Gulf Arabic WAVs")
    parser.add_argument("--train-root", type=Path, default=Path("data/finetuning/train"))
    parser.add_argument("--test-root", type=Path, default=Path("data/finetuning/test"))
    parser.add_argument("--csv-name", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/voxtral-asr"))
    parser.add_argument("--cache-dir", type=str, default=None)
    parser.add_argument("--per-device-train-batch", type=int, default=2)
    parser.add_argument("--per-device-eval-batch", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--lora", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--language", type=str, default="ar")
    parser.add_argument("--task", type=str, default="transcribe")
    parser.add_argument("--eval-max-samples", type=int, default=50)
    parser.add_argument("--eval-max-new-tokens", type=int, default=256)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    model, processor, model_type = load_model_and_processor(MODEL_ID, args.cache_dir)

    if args.lora:
        target_modules = infer_lora_targets(model)
        lora_task = TaskType.SEQ_2_SEQ_LM if model_type == "seq2seq" else TaskType.CTC
        lora_cfg = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=lora_task,
        )
        model = get_peft_model(model, lora_cfg)

    datasets = build_datasets(args.train_root, args.test_root, args.csv_name, args.sample_rate)
    train_raw = datasets.train
    eval_raw = datasets.test
    train_ds = build_dataset_maps(train_raw, processor, model_type)
    eval_ds = build_dataset_maps(eval_raw, processor, model_type)

    if model_type == "seq2seq":
        collator = make_seq2seq_collator(processor)
    else:
        collator = make_ctc_collator(processor)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pre_metrics = evaluate_wer(
        model,
        processor,
        eval_raw,
        model_type,
        device,
        args.per_device_eval_batch,
        args.eval_max_samples,
        args.eval_max_new_tokens,
        args.language,
        args.task,
    )
    print(f"[eval] pre-train WER={pre_metrics['wer']:.4f} over {pre_metrics['samples']} samples")

    if model_type == "seq2seq":
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.per_device_train_batch,
            per_device_eval_batch_size=args.per_device_eval_batch,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=25,
            predict_with_generate=True,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=processor,
        )
    else:
        training_args = TrainingArguments(
            output_dir=str(args.output_dir),
            per_device_train_batch_size=args.per_device_train_batch,
            per_device_eval_batch_size=args.per_device_eval_batch,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_steps=25,
            fp16=torch.cuda.is_available(),
            report_to=[],
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=processor,
        )

    trainer.train()

    post_metrics = evaluate_wer(
        model,
        processor,
        eval_raw,
        model_type,
        device,
        args.per_device_eval_batch,
        args.eval_max_samples,
        args.eval_max_new_tokens,
        args.language,
        args.task,
    )
    print(f"[eval] post-train WER={post_metrics['wer']:.4f} over {post_metrics['samples']} samples")

    metrics_payload = {
        "model_id": MODEL_ID,
        "mode": "lora" if args.lora else "full",
        "pre": pre_metrics,
        "post": post_metrics,
    }
    save_metrics(args.output_dir / "metrics.json", metrics_payload)

    trainer.save_model()
    try:
        processor.save_pretrained(args.output_dir)
    except AttributeError:
        pass


if __name__ == "__main__":
    main()
