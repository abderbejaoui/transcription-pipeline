from __future__ import annotations

import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Audio, Dataset
from transformers import AutoConfig, AutoModelForCTC, AutoModelForSpeechSeq2Seq, AutoProcessor


WORD_RE = re.compile(r"[\w']+", re.UNICODE)


@dataclass
class DatasetBundle:
    train: Dataset
    test: Dataset
    sample_rate: int


def _normalize(text: str) -> List[str]:
    words = WORD_RE.findall(text.lower())
    return [w for w in words if w.strip()]


def wer(reference: str, hypothesis: str) -> float:
    ref = _normalize(reference)
    hyp = _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    dp = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        dp[i][0] = i
    for j in range(len(hyp) + 1):
        dp[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[len(ref)][len(hyp)] / max(1, len(ref))


def _find_csv(batch_dir: Path, csv_name: Optional[str]) -> Optional[Path]:
    if csv_name:
        candidate = batch_dir / csv_name
        return candidate if candidate.exists() else None
    csvs = sorted(batch_dir.glob("*.csv"))
    return csvs[0] if csvs else None


def load_manifest(root: Path, csv_name: Optional[str] = None) -> List[Dict[str, str]]:
    if not root.exists():
        raise FileNotFoundError(f"Missing data root: {root}")
    records: List[Dict[str, str]] = []
    for batch_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        csv_path = _find_csv(batch_dir, csv_name)
        if not csv_path:
            print(f"[data] Skipping {batch_dir} (no CSV found)")
            continue
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                name = (row.get("name") or "").strip()
                transcript = (row.get("transcript") or "").strip()
                if not name or not transcript:
                    continue
                audio_path = batch_dir / name
                if audio_path.suffix.lower() != ".wav":
                    audio_path = audio_path.with_suffix(".wav")
                if not audio_path.exists():
                    print(f"[data] Missing audio: {audio_path}")
                    continue
                records.append(
                    {
                        "audio": str(audio_path),
                        "text": transcript,
                        "batch": batch_dir.name,
                    }
                )
    if not records:
        raise RuntimeError(f"No audio/transcript pairs found under {root}")
    return records


def build_datasets(
    train_root: Path,
    test_root: Path,
    csv_name: Optional[str],
    sample_rate: int,
) -> DatasetBundle:
    train_records = load_manifest(train_root, csv_name)
    test_records = load_manifest(test_root, csv_name)
    train_ds = Dataset.from_list(train_records)
    test_ds = Dataset.from_list(test_records)
    train_ds = train_ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    test_ds = test_ds.cast_column("audio", Audio(sampling_rate=sample_rate))
    return DatasetBundle(train=train_ds, test=test_ds, sample_rate=sample_rate)


def detect_model_type(config) -> str:
    if getattr(config, "is_encoder_decoder", False):
        return "seq2seq"
    return "ctc"


def load_model_and_processor(model_id: str, cache_dir: Optional[str]) -> Tuple[torch.nn.Module, Any, str]:
    config = AutoConfig.from_pretrained(model_id, cache_dir=cache_dir)
    model_type = detect_model_type(config)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir)
    if model_type == "seq2seq":
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, cache_dir=cache_dir)
    else:
        model = AutoModelForCTC.from_pretrained(model_id, cache_dir=cache_dir)
    return model, processor, model_type


def preprocess_batch(batch, processor, model_type: str):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"])
    if model_type == "seq2seq":
        batch["input_features"] = inputs["input_features"][0]
    else:
        batch["input_values"] = inputs["input_values"][0]
        if "attention_mask" in inputs:
            batch["attention_mask"] = inputs["attention_mask"][0]
    tokenizer = getattr(processor, "tokenizer", processor)
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def make_seq2seq_collator(processor, label_pad_token_id: int = -100):
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def collate(features: List[Dict]):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), label_pad_token_id
        )
        batch["labels"] = labels
        return batch

    return collate


def make_ctc_collator(processor, label_pad_token_id: int = -100):
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer

    def collate(features: List[Dict]):
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), label_pad_token_id
        )
        batch["labels"] = labels
        return batch

    return collate


def build_dataset_maps(dataset: Dataset, processor, model_type: str) -> Dataset:
    return dataset.map(
        lambda batch: preprocess_batch(batch, processor, model_type),
        remove_columns=dataset.column_names,
        num_proc=1,
    )


def infer_lora_targets(model: torch.nn.Module) -> List[str]:
    candidates = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "out_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "fc1",
        "fc2",
        "proj",
    }
    target_modules: set[str] = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            key = name.split(".")[-1]
            if key in candidates:
                target_modules.add(key)
    if not target_modules:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.add(name.split(".")[-1])
    return sorted(target_modules)


def maybe_get_forced_decoder_ids(processor, language: Optional[str], task: str):
    if not language:
        return None
    if hasattr(processor, "get_decoder_prompt_ids"):
        return processor.get_decoder_prompt_ids(language=language, task=task)
    return None


def evaluate_wer(
    model: torch.nn.Module,
    processor,
    dataset: Dataset,
    model_type: str,
    device: torch.device,
    batch_size: int,
    max_samples: Optional[int],
    max_new_tokens: int,
    language: Optional[str],
    task: str,
) -> Dict[str, float]:
    model.eval()
    indices = list(range(len(dataset)))
    if max_samples:
        indices = indices[:max_samples]
    total_wer = 0.0
    total_count = 0
    forced_decoder_ids = maybe_get_forced_decoder_ids(processor, language, task)
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        examples = [dataset[i] for i in batch_indices]
        audio_arrays = [ex["audio"]["array"] for ex in examples]
        sampling_rate = examples[0]["audio"]["sampling_rate"]
        inputs = processor(audio_arrays, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            if model_type == "seq2seq":
                gen_kwargs = {"max_new_tokens": max_new_tokens}
                if forced_decoder_ids:
                    gen_kwargs["forced_decoder_ids"] = forced_decoder_ids
                pred_ids = model.generate(**inputs, **gen_kwargs)
                pred_texts = processor.batch_decode(pred_ids, skip_special_tokens=True)
            else:
                logits = model(**inputs).logits
                pred_ids = torch.argmax(logits, dim=-1)
                pred_texts = processor.batch_decode(pred_ids)
        for ex, pred in zip(examples, pred_texts):
            total_wer += wer(ex["text"], pred)
            total_count += 1
    avg_wer = total_wer / max(1, total_count)
    return {"wer": avg_wer, "samples": total_count}


def save_metrics(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
