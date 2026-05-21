"""LoRA fine-tune of Qwen3-ASR-1.7B on Gulf Arabic.

This script mirrors the *official* Qwen3-ASR fine-tuning recipe
(`QwenLM/Qwen3-ASR/finetuning/qwen3_asr_sft.py`) and adds PEFT LoRA on top
of the LLM decoder ("thinker.model.language_model.*"), keeping the audio
tower frozen.

What's faithful to upstream:
  - `Qwen3ASRModel.from_pretrained(...)` to load the model + processor.
  - `patch_outer_forward(model)` to route forward() through
    `model.thinker.forward()` (without this, HF Trainer can't compute loss
    because the outer wrapper has no `labels` path).
  - Prefix-only label masking via the same two-pass processor call: the
    audio placeholders + system/user chat-template tokens are masked to
    -100 so the loss is computed only on the target transcript.
  - Sampling rate 16 kHz, librosa for loading.

What we add (carefully):
  - PEFT LoRA on the LLM decoder linears. Audio tower frozen via a
    positive `target_modules` regex that matches only the language model
    sub-tree, plus an explicit freeze of `audio_tower` params.
  - Per-source weighted sampler (UAE × 3, MGB-2 × 0.5, etc).
  - Custom callback that runs a held-out WER/CER eval with the
    Wang et al. 2024 Arabic normalizer every N steps.

Round 1 / Round 2 (ILT) both use this script with different
`--train-manifest` arguments. Adapters are merged with
`scripts.merge_adapters`.

Usage:
  python -m scripts.finetune_qwen3_lora \
    --train-manifest data/train_corpus/manifest.jsonl \
    --eval-manifests eval/casablanca_UAE/manifest.jsonl eval/bakeoff_30min/manifest.jsonl \
    --output-dir runs/qwen3_lora_r1 \
    --num-epochs 3
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Manifest -> upstream JSONL conversion
# ---------------------------------------------------------------------------


def _resolve_audio_path(audio_path: str, manifest_path: Path) -> str:
    """Resolve an audio path against multiple candidate roots.

    Manifests written by our preprocess pipeline store paths like
    ``audio/foo.wav`` which are relative to the preprocess output dir
    (the manifest's parent directory, or sometimes the directory
    containing the splits/ folder). We try several roots in order and
    return the first one that exists.

    If none exist, we still return the best-guess absolute path
    (manifest_parent / audio_path) so the downstream error message
    points at a sensible location.
    """
    p = Path(audio_path)
    if p.is_absolute():
        return str(p)

    manifest_dir = manifest_path.resolve().parent
    candidates = [
        manifest_dir / audio_path,                # splits/audio/foo.wav (unlikely)
        manifest_dir.parent / audio_path,         # preprocessed_audios_full/audio/foo.wav  <-- expected
        manifest_dir.parent.parent / audio_path,  # one level up
        PROJECT_ROOT / audio_path,                # repo root (legacy fallback)
    ]
    for cand in candidates:
        if cand.exists():
            return str(cand)
    # Fall back to the most likely location so the eventual error
    # message is informative.
    return str(manifest_dir.parent / audio_path)


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    """Load our manifest format and emit records ready for the upstream
    preprocess function: {audio: <abs path>, text: <full Qwen3-ASR target>,
    weight: <float>, source: <str>}.

    Qwen3-ASR expects text formatted with a language prefix:
        language Arabic<asr_text>الكلام...
    If our record's `text` already contains "<asr_text>" we leave it alone.
    Otherwise we prepend the Arabic language prefix (since this is a
    Gulf-Arabic adaptation run).
    """
    recs: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        audio_path = rec.get("audio_path") or rec.get("audio") or rec.get("path")
        text = rec.get("text") or rec.get("target") or ""
        if not audio_path or not text:
            continue
        audio_path = _resolve_audio_path(audio_path, path)
        if "<asr_text>" not in text:
            text = f"language Arabic<asr_text>{text}"
        recs.append({
            "audio": audio_path,
            "text": text,
            "weight": float(rec.get("weight", 1.0)),
            "source": rec.get("source", "unknown"),
            "prompt": rec.get("prompt", ""),
        })
    return recs


def write_upstream_jsonl(records: List[Dict[str, Any]], out_path: Path) -> Path:
    """Write a JSONL file in exactly the upstream format so `load_dataset`
    can consume it without further translation."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "audio": r["audio"],
                "text": r["text"],
                "prompt": r.get("prompt", ""),
            }, ensure_ascii=False) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# Upstream-compatible collator (copied/adapted from qwen3_asr_sft.py).
# Critical: Qwen3-ASR processor inserts audio placeholders INTO the text
# during processor(text=..., audio=...) — so we MUST call it with both
# together, not text alone.
# ---------------------------------------------------------------------------


def _load_audio_librosa(path: str, sr: int = 16_000):
    import librosa
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def _build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    """Build the per-example preprocess function. Same as upstream."""
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = _build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
    return _preprocess


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    """Exact copy of upstream collator. Two processor calls: one for the
    full sequence (prefix + target + eos), one for the prefix alone. The
    prefix tokens are masked to -100 so loss only flows through the target.
    """
    processor: Any
    sampling_rate: int = 16_000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        import torch

        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        audios = [_load_audio_librosa(p, sr=self.sampling_rate) for p in audio_paths]

        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ---------------------------------------------------------------------------
# patch_outer_forward — required by upstream so HF Trainer can hit the
# thinker (which is what owns the LM head + loss).
# ---------------------------------------------------------------------------


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Check that you loaded via Qwen3ASRModel.from_pretrained()."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )
    cls.forward = forward
    cls._forward_patched = True


# ---------------------------------------------------------------------------
# LoRA injection (decoder-only). Uses a POSITIVE target regex so we never
# touch the audio tower. Then we additionally freeze audio_tower params
# explicitly as a belt-and-suspenders guarantee.
# ---------------------------------------------------------------------------


# Module-name suffixes inside Qwen3-ASR's language model (Qwen3 architecture).
# Verified against
# qwen_asr/core/transformers_backend/modeling_qwen3_asr.py.
DEFAULT_LORA_TARGET_SUFFIXES = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # attention
    "gate_proj", "up_proj", "down_proj",     # FFN
]


def _find_lora_target_modules(model, suffixes: Sequence[str]) -> List[str]:
    """Walk model.named_modules() and return full module paths for every
    Linear whose name ends in one of `suffixes` AND lives under the LLM
    decoder transformer blocks (NOT under audio_tower).

    Qwen3-ASR-1.7B exposes its LLM decoder at `thinker.model.layers.*`
    (verified via scripts.inspect_qwen3_modules). The audio encoder lives
    at `thinker.audio_tower.layers.*`. We accept anything under
    `thinker.model.layers` and explicitly reject `audio_tower`/`audio_encoder`.
    """
    import torch.nn as nn
    targets: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        # Skip the audio tower entirely.
        if "audio_tower" in name or "audio_encoder" in name:
            continue
        # Must live in the LLM decoder transformer blocks.
        if "model.layers" not in name:
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail in suffixes:
            targets.append(name)
    return targets


def _patch_input_embeddings_shim(model) -> None:
    """Attach `get_input_embeddings` / `set_input_embeddings` to the outer
    Qwen3ASRForConditionalGeneration instance so that
    `transformers.PreTrainedModel.enable_input_require_grads()` (called by
    PEFT during gradient-checkpointing prep) can find the LLM embedding.

    The Qwen3-ASR-1.7B LLM embedding lives at `thinker.model.embed_tokens`
    (verified via scripts.inspect_qwen3_modules). The outer class does not
    auto-resolve it, so we provide an explicit shim.
    """
    import types

    # Walk known paths to locate the LLM input embedding.
    candidate_paths = (
        "thinker.model.embed_tokens",
        "thinker.model.language_model.embed_tokens",
        "thinker.language_model.model.embed_tokens",
    )
    emb_path = None
    for path in candidate_paths:
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            emb_path = path
            break
        except AttributeError:
            continue
    if emb_path is None:
        # Last resort: scan named_modules for the first nn.Embedding under thinker.
        import torch.nn as nn
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Embedding) and name.startswith("thinker."):
                emb_path = name
                break
    if emb_path is None:
        raise RuntimeError(
            "Could not locate LLM input embeddings on Qwen3-ASR model. "
            "Run `python -m scripts.inspect_qwen3_modules` to inspect."
        )

    parts = emb_path.split(".")
    parent_path, attr_name = parts[:-1], parts[-1]

    def _resolve_parent(self):
        obj = self
        for part in parent_path:
            obj = getattr(obj, part)
        return obj

    def _get_input_embeddings(self):
        return getattr(_resolve_parent(self), attr_name)

    def _set_input_embeddings(self, value):
        setattr(_resolve_parent(self), attr_name, value)

    model.get_input_embeddings = types.MethodType(_get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(_set_input_embeddings, model)
    print(f"[lora] patched get_input_embeddings -> {emb_path}")


def apply_lora(model, target_suffixes: Sequence[str], r: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model, TaskType

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    # Belt-and-suspenders: ensure audio tower is frozen and not touched.
    for n, p in model.named_parameters():
        if "audio_tower" in n or "audio_encoder" in n:
            p.requires_grad = False

    explicit_targets = _find_lora_target_modules(model, target_suffixes)
    if not explicit_targets:
        raise RuntimeError(
            "Could not find any LoRA target modules under thinker.model.layers. "
            "Run `python -m scripts.inspect_qwen3_modules` to print the "
            "actual module names from your installed Qwen3-ASR."
        )
    print(f"[lora] {len(explicit_targets)} target modules "
          f"(first 3: {explicit_targets[:3]})")

    # PEFT will call model.enable_input_require_grads() which uses
    # get_input_embeddings(); the outer Qwen3-ASR class doesn't implement
    # it, so install a shim that points to thinker.model.embed_tokens.
    _patch_input_embeddings_shim(model)

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=explicit_targets,  # explicit full paths
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


# ---------------------------------------------------------------------------
# Weighted sampler (per-source weight) — works on a HF Dataset via a
# dataloader-level injection.
# ---------------------------------------------------------------------------


def build_weighted_sampler(records: List[Dict[str, Any]]):
    import torch
    from torch.utils.data import WeightedRandomSampler
    weights = torch.tensor(
        [float(r.get("weight", 1.0)) for r in records], dtype=torch.float64,
    )
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ---------------------------------------------------------------------------
# Eval callback — Wang et al. 2024 Arabic normalizer, on held-out manifests.
# ---------------------------------------------------------------------------


def _run_eval(model, processor, manifest_path: Path) -> Tuple[float, float, int]:
    import jiwer
    import soundfile as sf
    import torch
    from scripts.eval_arabic import normalize_arabic_text

    refs, hyps = [], []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ap = rec.get("audio_path") or rec.get("path") or rec.get("audio")
        if not ap:
            continue
        ap = _resolve_audio_path(ap, manifest_path)
        arr, sr = sf.read(ap, dtype="float32", always_2d=False)
        if arr.ndim > 1:
            arr = arr.mean(axis=1)
        if sr != 16_000:
            try:
                import soxr
                arr = soxr.resample(arr, sr, 16_000)
            except ImportError:
                import librosa
                arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)

        # Build the Qwen3-ASR inference prompt (system + user-audio +
        # generation prompt), feed to processor + generate.
        prefix_msgs = _build_prefix_messages("", None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False,
        )[0]
        inputs = processor(
            text=[prefix_text], audio=[arr],
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **inputs, max_new_tokens=448,
                do_sample=False, num_beams=1,
            )
        # Strip prompt tokens then decode.
        out = gen[:, inputs["input_ids"].shape[1]:]
        hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
        # Strip the leading "language X<asr_text>" prefix if present.
        if "<asr_text>" in hyp:
            hyp = hyp.split("<asr_text>", 1)[1]
        ref = rec.get("text", "")
        if "<asr_text>" in ref:
            ref = ref.split("<asr_text>", 1)[1]
        refs.append(normalize_arabic_text(ref))
        hyps.append(normalize_arabic_text(hyp))
    if not refs:
        return float("nan"), float("nan"), 0
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps), len(refs)


def make_eval_callback(processor, eval_manifests: List[Path], every_steps: int):
    from transformers import TrainerCallback

    class GulfArabicEvalCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 0 or state.global_step % every_steps != 0:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            model.eval()
            for man in eval_manifests:
                try:
                    wer, cer, n = _run_eval(model, processor, man)
                    print(f"[eval-cb step={state.global_step}] {man.name}: "
                          f"WER={wer*100:.2f}%  CER={cer*100:.2f}%  n={n}",
                          flush=True)
                except Exception as exc:
                    print(f"[eval-cb step={state.global_step}] {man.name} FAILED: {exc!r}",
                          flush=True)
            model.train()
            return control

    return GulfArabicEvalCallback()


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--eval-manifests", type=Path, nargs="+", required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--lora-target-suffixes", nargs="+", default=DEFAULT_LORA_TARGET_SUFFIXES)
    ap.add_argument("--per-device-train-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-epochs", type=float, default=3)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every-steps", type=int, default=2000)
    ap.add_argument("--save-total-limit", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help=("Enable HF gradient checkpointing. OFF by default: with LoRA "
              "(~17M trainable params) we have plenty of VRAM, and enabling "
              "it causes a 'None of the inputs have requires_grad=True' "
              "silent-no-op when the outer model's forward is patched."),
    )
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from qwen_asr import Qwen3ASRModel  # official wrapper
    from transformers import (
        GenerationConfig, Trainer, TrainingArguments,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Translate our manifest into upstream JSONL format.
    print(f"[data] loading {args.train_manifest}")
    records = load_manifest(args.train_manifest)
    print(f"[data] {len(records)} clips after manifest conversion")
    upstream_jsonl = args.output_dir / "_upstream_train.jsonl"
    write_upstream_jsonl(records, upstream_jsonl)

    # 2. Load model + processor via the official wrapper.
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"[load] Qwen3ASRModel.from_pretrained({args.model_path}) bf16={use_bf16}")
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,  # let HF Trainer place it
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # 3. LoRA on the LLM decoder linears only.
    model = apply_lora(
        model,
        target_suffixes=args.lora_target_suffixes,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
    )

    # 4. Dataset via upstream preprocess.
    raw_ds = load_dataset("json", data_files={"train": str(upstream_jsonl)})
    preprocess = make_preprocess_fn_prefix_only(processor)
    ds = raw_ds.map(preprocess, num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    collator = DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=16_000)

    # 5. TrainingArguments (note: `eval_strategy`, not `evaluation_strategy`).
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
        gradient_checkpointing=args.gradient_checkpointing,
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.eval_every_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="no",  # we eval via callback with the Arabic normalizer
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=args.num_workers > 0,
        dataloader_prefetch_factor=2 if args.num_workers > 0 else None,
        optim="adamw_torch_fused",
    )

    eval_cb = make_eval_callback(processor, args.eval_manifests, args.eval_every_steps)

    sampler = build_weighted_sampler(records)

    class CastFloatInputsTrainer(Trainer):
        """Upstream cast — Qwen3-ASR expects all float inputs in model dtype."""
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for k, v in list(inputs.items()):
                    if torch.is_tensor(v) and v.is_floating_point():
                        inputs[k] = v.to(dtype=model_dtype)
            return inputs

        def _get_train_sampler(self, *args, **kwargs):
            # Signature changed across transformers versions:
            # older versions: (self)
            # newer versions: (self, train_dataset)
            # We accept either and return our pre-built weighted sampler.
            return sampler

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[eval_cb],
    )

    print("[train] starting")
    trainer.train()
    print("[train] done — saving final adapter")
    final_dir = args.output_dir / "final_adapter"
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)

    # 6. One last eval at full strength.
    print("[final-eval]")
    model.eval()
    for man in args.eval_manifests:
        try:
            wer, cer, n = _run_eval(model, processor, man)
            print(f"  {man.name}: WER={wer*100:.2f}%  CER={cer*100:.2f}%  n={n}")
        except Exception as exc:
            print(f"  {man.name} FAILED: {exc!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
