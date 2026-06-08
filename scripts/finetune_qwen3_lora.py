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

        # CRITICAL: force right-padding. The Qwen3ASRProcessor defaults to
        # left-padding (text_kwargs.padding_side="left"), which would put
        # the actual prefix tokens at the END of the padded sequence, not
        # the start. The subsequent `labels[i, :pl] = -100` line assumes
        # the prefix lives at positions [0, pl), so left-padding silently
        # produces a label tensor where loss is computed against the
        # prefix itself instead of the target. Training looks like it
        # runs but loss never decreases (sits at ~16 for vocab of ~152k,
        # i.e. WORSE than random because gradients pull the model toward
        # predicting prefix tokens it can't possibly know).
        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
            padding_side="right",
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
            padding_side="right",
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # One-time sanity checks on the FIRST batch only (cheap, prints
        # to stdout). Catches the silent flat-loss failure mode early.
        if not getattr(self.__class__, "_sanity_checked", False):
            self.__class__._sanity_checked = True
            self._sanity_check_labels(
                full_inputs, prefix_inputs, prefix_lens, labels,
            )

        full_inputs["labels"] = labels
        return full_inputs

    def _sanity_check_labels(self, full_inputs, prefix_inputs, prefix_lens, labels):
        """First-batch sanity prints. Doesn't raise — just logs so a human
        can see in the first 10 seconds of training whether the labels
        look right.
        """
        import torch
        n_active = (labels != -100).sum().item()
        n_total = labels.numel()
        n_per_sample = ((labels != -100).sum(dim=1)).tolist()
        # For sample 0: check that input_ids[:prefix_lens[0]] of full_inputs
        # equals input_ids[:prefix_lens[0]] of prefix_inputs. If not, the
        # prefix doesn't live at the start of full_inputs and the label
        # masking is wrong.
        full_prefix_slice = full_inputs["input_ids"][0, :prefix_lens[0]]
        ref_prefix_slice = prefix_inputs["input_ids"][0, :prefix_lens[0]]
        prefix_match = bool(torch.equal(full_prefix_slice, ref_prefix_slice))
        print(
            f"[collator-sanity] active labels = {n_active}/{n_total} "
            f"({100*n_active/n_total:.1f}%), per-sample = {n_per_sample}, "
            f"prefix tokens match between full and prefix_inputs: {prefix_match}",
            flush=True,
        )
        if not prefix_match:
            print(
                "[collator-sanity] *** WARNING *** prefix tokens do NOT match "
                "between full_inputs and prefix_inputs. Label masking is "
                "almost certainly wrong. Check padding_side handling.",
                flush=True,
            )
        if n_active < 5 * len(prefix_lens):
            print(
                "[collator-sanity] *** WARNING *** less than 5 active labels "
                "per sample on average. Loss signal will be very weak.",
                flush=True,
            )


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


def _audio_tower_layer_count(model) -> int:
    """Return the number of transformer blocks in the audio encoder, i.e.
    the highest index N appearing in ``audio_tower(.layers|.blocks).<N>``.
    Returns 0 if no audio-tower layers are found.
    """
    import re
    max_idx = -1
    pat = re.compile(r"audio_(?:tower|encoder)\..*?(?:layers|blocks)\.(\d+)\.")
    for name, _ in model.named_modules():
        m = pat.search(name + ".")
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx + 1


def _find_lora_target_modules(
    model,
    suffixes: Sequence[str],
    unfreeze_encoder_layers: int = 0,
) -> List[str]:
    """Walk model.named_modules() and return full module paths for every
    Linear whose name ends in one of `suffixes` AND lives under the LLM
    decoder transformer blocks (NOT under audio_tower).

    Qwen3-ASR-1.7B exposes its LLM decoder at `thinker.model.layers.*`
    (verified via scripts.inspect_qwen3_modules). The audio encoder lives
    at `thinker.audio_tower.layers.*`. We accept anything under
    `thinker.model.layers` and explicitly reject `audio_tower`/`audio_encoder`.

    When ``unfreeze_encoder_layers > 0`` (teacher's *Change 4 — encoder
    unfreezing*), we ALSO add LoRA to the matching linears in the LAST N
    transformer blocks of the audio encoder. Only the highest-index N
    blocks are touched (the layers nearest the decoder, which carry the
    most accent/dialect-specific acoustic information). The convolutional
    front-end and the lower encoder blocks stay frozen.
    """
    import re
    import torch.nn as nn

    enc_lo = None
    if unfreeze_encoder_layers and unfreeze_encoder_layers > 0:
        n_enc = _audio_tower_layer_count(model)
        if n_enc <= 0:
            raise RuntimeError(
                "--unfreeze-encoder-layers was set but no audio_tower layers "
                "were found. Run `python -m scripts.inspect_qwen3_modules` to "
                "inspect the actual encoder module names."
            )
        enc_lo = max(0, n_enc - unfreeze_encoder_layers)

    enc_pat = re.compile(r"audio_(?:tower|encoder)\..*?(?:layers|blocks)\.(\d+)\.")

    targets: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        tail = name.rsplit(".", 1)[-1]
        if tail not in suffixes:
            continue
        is_encoder = ("audio_tower" in name) or ("audio_encoder" in name)
        if is_encoder:
            # Only the last N encoder blocks, and only when explicitly opted in.
            if enc_lo is None:
                continue
            m = enc_pat.search(name + ".")
            if m is None or int(m.group(1)) < enc_lo:
                continue
            targets.append(name)
            continue
        # Decoder path: must live in the LLM decoder transformer blocks.
        if "model.layers" not in name:
            continue
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


def apply_lora(
    model,
    target_suffixes: Sequence[str],
    r: int,
    alpha: int,
    dropout: float,
    use_rslora: bool = False,
    use_dora: bool = False,
    unfreeze_encoder_layers: int = 0,
):
    """Inject LoRA adapters into the LLM decoder linears.

    When ``use_rslora=True``, PEFT uses the rank-stabilized scaling
    ``lora_alpha/sqrt(r)`` (Kalajdzievski 2023, arXiv:2312.03732). The
    default LoRA scaling ``lora_alpha/r`` is known to slow learning at
    higher ranks, causing r>32 to plateau LOWER than smaller ranks.
    rsLoRA unlocks the benefit of larger ranks (r=64, 128, 256+).
    Use it whenever ``r > 32``.

    When ``use_dora=True`` (teacher's *Change 1 — DoRA*, Liu et al. 2024,
    arXiv:2402.09353), PEFT decomposes each weight into magnitude and
    direction and learns them separately. DoRA consistently beats plain
    LoRA, especially at low rank, with NO inference overhead after merge
    (the magnitude/direction are folded back into W). Training is ~+39%
    slower. ``use_dora`` and ``use_rslora`` are independent and may be
    combined.

    When ``unfreeze_encoder_layers > 0`` (teacher's *Change 4*), LoRA is
    additionally placed on the last N audio-encoder blocks. The encoder
    weights themselves stay frozen — only their LoRA adapters train — so
    instability is bounded. Use a LOWER LR on those adapters via
    ``--encoder-lora-lr`` (handled in main()).
    """
    from peft import LoraConfig, get_peft_model, TaskType

    # Freeze everything first.
    for p in model.parameters():
        p.requires_grad = False

    explicit_targets = _find_lora_target_modules(
        model, target_suffixes, unfreeze_encoder_layers=unfreeze_encoder_layers,
    )
    if not explicit_targets:
        raise RuntimeError(
            "Could not find any LoRA target modules under thinker.model.layers. "
            "Run `python -m scripts.inspect_qwen3_modules` to print the "
            "actual module names from your installed Qwen3-ASR."
        )

    # Belt-and-suspenders: ensure the BASE audio-tower weights stay frozen.
    # We never set their requires_grad to True; PEFT only trains the LoRA
    # adapters we explicitly target (including any encoder targets above).
    target_set = set(explicit_targets)
    n_encoder_targets = sum(
        1 for t in explicit_targets
        if ("audio_tower" in t) or ("audio_encoder" in t)
    )
    for n, p in model.named_parameters():
        if "audio_tower" in n or "audio_encoder" in n:
            p.requires_grad = False

    print(f"[lora] {len(explicit_targets)} target modules "
          f"({n_encoder_targets} on encoder; first 3: {explicit_targets[:3]})")
    print(f"[lora] r={r}  alpha={alpha}  dropout={dropout}  "
          f"use_rslora={use_rslora}  use_dora={use_dora}  "
          f"unfreeze_encoder_layers={unfreeze_encoder_layers}")

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
        use_rslora=use_rslora,
        use_dora=use_dora,
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


def _run_eval(
    model,
    processor,
    manifest_path: Path,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> Tuple[float, float, int]:
    """Run WER/CER on a held-out manifest.

    When ``max_samples`` is set, randomly subsample that many lines
    (deterministic via ``seed``). This is critical because the full
    validation split (~17k clips) would block training for ~14 hours
    per eval round if evaluated end-to-end with greedy generate().
    A 500-sample WER tracks the full-set WER within ~±1% — fine for
    progress monitoring during training. The post-training final-eval
    block (see end of main()) deliberately passes ``max_samples=None``
    to report the real number.
    """
    import random
    import jiwer
    import soundfile as sf
    import torch
    # Robust import: when launched as `python scripts/finetune_qwen3_lora.py`,
    # the `scripts.` package isn't on sys.path. Add the repo root then import.
    try:
        from scripts.eval_arabic import normalize_arabic_text
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from scripts.eval_arabic import normalize_arabic_text

    # One-time full traceback on the first per-sample failure, so a broken
    # generate/decode path can't hide behind the skip counter again.
    if not hasattr(_run_eval, "_traceback_shown"):
        _run_eval._traceback_shown = False

    lines = [ln for ln in manifest_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_samples is not None and 0 < max_samples < len(lines):
        rng = random.Random(seed)
        lines = rng.sample(lines, max_samples)

    refs, hyps = [], []
    n_skipped = 0
    for line in lines:
        rec = json.loads(line)
        ap = rec.get("audio_path") or rec.get("path") or rec.get("audio")
        if not ap:
            continue
        ap = _resolve_audio_path(ap, manifest_path)
        # Robustly load audio: a single missing/corrupt file must NOT abort
        # the whole eval (that bug left us training blind for 2 full epochs
        # when one worldspeech clip was missing). Skip the bad sample and
        # keep going; report the skip count at the end.
        try:
            arr, sr = sf.read(ap, dtype="float32", always_2d=False)
        except Exception:
            n_skipped += 1
            continue
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
        # Cast floating-point tensors (audio features) to the model's dtype.
        # The model is loaded in bfloat16 but the processor returns float32,
        # which causes "Input type (float) and bias type (c10::BFloat16)
        # should be the same" inside the audio encoder. Training side
        # handles this via CastFloatInputsTrainer._prepare_inputs; we do
        # the same here for eval.
        model_dtype = getattr(model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        # A single failed generate (e.g. transient OOM) must not abort the
        # whole eval round. Skip and continue. The decode/slice steps are
        # inside the SAME try because `model.generate(..., return_dict_in_
        # generate=...)` (or certain transformers versions) can return a
        # `GenerateDecoderOnlyOutput`/tuple rather than a plain LongTensor.
        # Doing `gen[:, X:]` on a tuple raises
        # "tuple indices must be integers or slices, not tuple" and that
        # was silently nuking the WHOLE eval round (every sample -> nan).
        try:
            with torch.no_grad():
                gen = model.generate(
                    **inputs, max_new_tokens=448,
                    do_sample=False, num_beams=1,
                )
            # Normalize generate() output to a LongTensor of shape [B, T].
            if not torch.is_tensor(gen):
                # GenerateOutput dataclass exposes `.sequences`; a bare
                # tuple puts sequences first. (Avoid `a or b` here: numpy/
                # torch tensors raise on bool() of multi-element tensors.)
                seq = getattr(gen, "sequences", None)
                gen = seq if seq is not None else gen[0]
            # Strip prompt tokens then decode.
            out = gen[:, inputs["input_ids"].shape[1]:]
            hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
        except Exception:
            n_skipped += 1
            if not _run_eval._traceback_shown:
                import traceback
                _run_eval._traceback_shown = True
                print(f"[eval] {manifest_path.name}: first sample failure "
                      f"traceback (shown once):", flush=True)
                traceback.print_exc()
            continue
        # Strip the leading "language X<asr_text>" prefix if present.
        if "<asr_text>" in hyp:
            hyp = hyp.split("<asr_text>", 1)[1]
        ref = rec.get("text", "")
        if "<asr_text>" in ref:
            ref = ref.split("<asr_text>", 1)[1]
        refs.append(normalize_arabic_text(ref))
        hyps.append(normalize_arabic_text(hyp))
    if n_skipped:
        print(f"[eval] {manifest_path.name}: skipped {n_skipped} "
              f"unreadable/failed sample(s)", flush=True)
    if not refs:
        return float("nan"), float("nan"), 0
    return jiwer.wer(refs, hyps), jiwer.cer(refs, hyps), len(refs)


def make_eval_callback(
    processor,
    eval_manifests: List[Path],
    every_steps: int,
    max_samples: Optional[int] = None,
    early_stopping_patience: int = 0,
    early_stopping_metric: str = "wer",
    early_stopping_threshold: float = 0.001,
    output_dir: Optional[Path] = None,
    eval_at_start: bool = False,
):
    """Build the held-out eval callback.

    Args:
        processor: Qwen3-ASR processor (for tokenization/decoding).
        eval_manifests: List of manifests to evaluate. The FIRST manifest is
            treated as the "primary" — early stopping decisions are based on it.
        every_steps: Run eval every this many optimizer steps.
        max_samples: Cap each eval manifest to this many random samples
            (saves time during training; None means use the full set).
        early_stopping_patience: If > 0, stop training when the primary
            manifest's metric has not improved by ``early_stopping_threshold``
            for this many consecutive evals. 0 disables early stopping.
        early_stopping_metric: "wer" or "cer". Lower is better.
        early_stopping_threshold: Minimum improvement to count as "better".
            Default 0.001 = 0.1 percentage points absolute (WER goes from
            25.30% to 25.19% counts as improvement; 25.30% to 25.25% does not).
        output_dir: Where to save the best-adapter checkpoint. If None, no
            best-adapter saving (the trainer still saves periodic checkpoints).
    """
    from transformers import TrainerCallback

    class GulfArabicEvalCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            # Per-manifest list of (step, wer, cer) tuples — useful for plotting.
            self.history: Dict[str, List[Tuple[int, float, float]]] = {
                m.name: [] for m in eval_manifests
            }
            self.best_metric: float = float("inf")
            self.best_step: int = -1
            self.evals_without_improvement: int = 0

        def _do_eval(self, model, step: int) -> Optional[float]:
            """Run every eval manifest at the given step; return the primary
            metric (first manifest) or None if it failed. Used both by the
            periodic on_step_end hook and the optional eval-at-start hook."""
            model.eval()
            primary_metric_value: Optional[float] = None
            for idx, man in enumerate(eval_manifests):
                try:
                    wer, cer, n = _run_eval(
                        model, processor, man, max_samples=max_samples,
                    )
                    print(f"[eval-cb step={step}] {man.name}: "
                          f"WER={wer*100:.2f}%  CER={cer*100:.2f}%  n={n}",
                          flush=True)
                    self.history[man.name].append((step, wer, cer))
                    if idx == 0:  # primary manifest
                        primary_metric_value = wer if early_stopping_metric == "wer" else cer
                except Exception as exc:
                    print(f"[eval-cb step={step}] {man.name} FAILED: "
                          f"{exc!r}", flush=True)
            model.train()
            return primary_metric_value

        def on_train_begin(self, args, state, control, **kwargs):
            # Prove the validation path works BEFORE training (catches a
            # broken eval up-front instead of hours in). No early-stop
            # bookkeeping here — this is a baseline sanity check only.
            if not eval_at_start:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            print("[eval-cb] eval-at-start: baseline held-out eval (step 0)",
                  flush=True)
            self._do_eval(model, state.global_step)
            return control

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 0 or state.global_step % every_steps != 0:
                return control
            model = kwargs.get("model")
            if model is None:
                return control
            primary_metric_value = self._do_eval(model, state.global_step)

            # Early-stopping bookkeeping (only if enabled and primary eval succeeded).
            if early_stopping_patience > 0 and primary_metric_value is not None:
                improved = (self.best_metric - primary_metric_value) > early_stopping_threshold
                if improved:
                    prev = self.best_metric
                    self.best_metric = primary_metric_value
                    self.best_step = state.global_step
                    self.evals_without_improvement = 0
                    print(f"[early-stop] new best {early_stopping_metric}="
                          f"{primary_metric_value*100:.2f}% (was {prev*100:.2f}%) "
                          f"at step {state.global_step}", flush=True)
                    # Save the best adapter immediately so we don't lose it
                    # if training crashes later.
                    if output_dir is not None:
                        best_dir = output_dir / "best_adapter"
                        try:
                            model.save_pretrained(best_dir)
                            processor.save_pretrained(best_dir)
                            print(f"[early-stop] saved best adapter -> {best_dir}",
                                  flush=True)
                        except Exception as exc:
                            print(f"[early-stop] failed to save best adapter: {exc!r}",
                                  flush=True)
                else:
                    self.evals_without_improvement += 1
                    print(f"[early-stop] no improvement ({self.evals_without_improvement}/"
                          f"{early_stopping_patience}) — best {early_stopping_metric}="
                          f"{self.best_metric*100:.2f}% at step {self.best_step}",
                          flush=True)
                    if self.evals_without_improvement >= early_stopping_patience:
                        print(f"[early-stop] STOPPING — no improvement for "
                              f"{early_stopping_patience} consecutive evals. "
                              f"Best {early_stopping_metric}="
                              f"{self.best_metric*100:.2f}% at step {self.best_step}.",
                              flush=True)
                        control.should_training_stop = True
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
    ap.add_argument(
        "--use-rslora",
        action="store_true",
        help=("Use rank-stabilized LoRA scaling (lora_alpha/sqrt(r)) per "
              "Kalajdzievski 2023. Strongly recommended when r > 32, since "
              "the default lora_alpha/r scaling causes high-rank adapters "
              "to learn slower and plateau lower than low-rank ones."),
    )
    ap.add_argument(
        "--use-dora",
        action="store_true",
        help=("Use DoRA (weight-decomposed LoRA; Liu et al. 2024, "
              "arXiv:2402.09353). Consistently beats plain LoRA, especially "
              "at low rank, with NO inference overhead after merge. Training "
              "is ~+39%% slower. Independent of --use-rslora."),
    )
    ap.add_argument(
        "--unfreeze-encoder-layers",
        type=int,
        default=0,
        help=("Also place LoRA adapters on the LAST N transformer blocks of "
              "the audio encoder (teacher's Change 4). The encoder base "
              "weights stay frozen; only their LoRA adapters train. 0 "
              "(default) keeps the encoder fully frozen. Try 2-4 for accent/"
              "dialect adaptation. Pair with a lower --encoder-lora-lr."),
    )
    ap.add_argument(
        "--encoder-lora-lr",
        type=float,
        default=None,
        help=("Separate (lower) learning rate for the encoder LoRA adapters "
              "when --unfreeze-encoder-layers > 0. Defaults to 0.1 x "
              "--learning-rate. Ignored if no encoder layers are unfrozen."),
    )
    ap.add_argument(
        "--lr-scheduler-type",
        default="linear",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                 "constant", "constant_with_warmup"],
        help="LR schedule (passed to TrainingArguments). Default 'linear'.",
    )
    ap.add_argument(
        "--warmup-steps",
        type=int,
        default=0,
        help=("Absolute warmup steps. If > 0, OVERRIDES --warmup-ratio "
              "(HuggingFace precedence). Default 0 = use --warmup-ratio."),
    )
    ap.add_argument("--lora-target-suffixes", nargs="+", default=DEFAULT_LORA_TARGET_SUFFIXES)
    ap.add_argument("--per-device-train-batch-size", type=int, default=4)
    ap.add_argument("--gradient-accumulation-steps", type=int, default=16)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--num-epochs", type=float, default=3)
    ap.add_argument("--warmup-ratio", type=float, default=0.02)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--eval-every-steps", type=int, default=2000)
    ap.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help=("Hard cap on optimizer steps. -1 (default) trains for "
              "--num-epochs. Set a small value (e.g. 10) for a SMOKE TEST "
              "that exercises the full train+eval+save path quickly before "
              "committing to a multi-hour run."),
    )
    ap.add_argument(
        "--eval-at-start",
        action="store_true",
        help=("Run a held-out eval ONCE before training begins (step 0). "
              "Use this to prove the validation path works up-front instead "
              "of discovering a broken eval hours into training."),
    )
    ap.add_argument(
        "--eval-max-samples",
        type=int,
        default=500,
        help=("During training, cap each eval manifest to this many random "
              "samples (default 500). Pass 0 to evaluate the full set. "
              "The post-training final eval ALWAYS uses the full set."),
    )
    ap.add_argument(
        "--early-stopping-patience",
        type=int,
        default=3,
        help=("Stop training when the FIRST eval manifest's metric has not "
              "improved by --early-stopping-threshold for this many consecutive "
              "evals. Default 3. Pass 0 to disable early stopping."),
    )
    ap.add_argument(
        "--early-stopping-metric",
        choices=["wer", "cer"],
        default="wer",
        help="Metric to monitor for early stopping. Lower is better.",
    )
    ap.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.001,
        help=("Minimum absolute improvement to count as 'better'. "
              "0.001 = 0.1 percentage points (e.g. 25.30%% -> 25.19%% "
              "counts; 25.30%% -> 25.25%% does not)."),
    )
    ap.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help=("Resume training from a saved HF Trainer checkpoint directory "
              "(e.g. runs/phase1/checkpoint-8000). Restores model weights, "
              "optimizer, scheduler, RNG state, and step count, so training "
              "picks up exactly where it left off. NOTE: this needs a FULL "
              "Trainer checkpoint (optimizer.pt/scheduler.pt/trainer_state.json), "
              "not a bare adapter dir. To START a NEW run (fresh optimizer) "
              "from a previously trained adapter, use --init-adapter instead."),
    )
    ap.add_argument(
        "--init-adapter",
        type=Path,
        default=None,
        help=("Initialise the LoRA weights from a previously trained adapter "
              "directory (e.g. runs/phase1/best_adapter) and start a FRESH "
              "training run (new optimizer/scheduler/step count). This is how "
              "Phase 2 continues from Phase 1: the Phase-1 adapter is loaded as "
              "the starting point, then trained further on the Phase-2 mix at a "
              "lower LR. The adapter's LoRA shape (r/alpha/targets) must match "
              "this run's --lora-* flags. Mutually exclusive with "
              "--resume-from-checkpoint."),
    )
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

    # 3. LoRA on the LLM decoder linears (+ optionally last N encoder blocks).
    model = apply_lora(
        model,
        target_suffixes=args.lora_target_suffixes,
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        use_rslora=args.use_rslora,
        use_dora=args.use_dora,
        unfreeze_encoder_layers=args.unfreeze_encoder_layers,
    )

    # 3b. Phase-2 warm start: load a previously trained adapter as the LoRA
    # initialisation (FRESH optimizer/scheduler — NOT a Trainer resume). The
    # fresh LoRA layers from apply_lora() are overwritten with the saved
    # Phase-1 weights, then training continues on the Phase-2 mix.
    if getattr(args, "init_adapter", None):
        if getattr(args, "resume_from_checkpoint", None):
            raise SystemExit(
                "[init-adapter] --init-adapter and --resume-from-checkpoint are "
                "mutually exclusive. Use --init-adapter to start a fresh run "
                "from a prior adapter; use --resume-from-checkpoint only with a "
                "full HF Trainer checkpoint dir."
            )
        from safetensors.torch import load_file as _load_safetensors
        from peft import set_peft_model_state_dict
        adapter_dir = Path(args.init_adapter)
        weights_file = adapter_dir / "adapter_model.safetensors"
        bin_file = adapter_dir / "adapter_model.bin"
        if weights_file.exists():
            sd = _load_safetensors(str(weights_file))
        elif bin_file.exists():
            sd = torch.load(str(bin_file), map_location="cpu")
        else:
            raise SystemExit(
                f"[init-adapter] no adapter weights found in {adapter_dir} "
                f"(looked for adapter_model.safetensors / .bin)."
            )
        load_res = set_peft_model_state_dict(model, sd)
        missing = getattr(load_res, "missing_keys", []) or []
        unexpected = getattr(load_res, "unexpected_keys", []) or []
        print(f"[init-adapter] loaded {len(sd)} tensors from {adapter_dir} "
              f"(missing={len(missing)} unexpected={len(unexpected)})")
        if unexpected:
            raise SystemExit(
                f"[init-adapter] {len(unexpected)} unexpected keys — the saved "
                f"adapter's LoRA shape does not match this run's --lora-* flags. "
                f"First few: {unexpected[:5]}"
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
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        lr_scheduler_type=args.lr_scheduler_type,
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

    # During training, cap eval to a random subsample. The final eval after
    # training completes uses the full set (see end of main()).
    eval_max = args.eval_max_samples if args.eval_max_samples > 0 else None
    eval_cb = make_eval_callback(
        processor,
        args.eval_manifests,
        args.eval_every_steps,
        max_samples=eval_max,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_threshold=args.early_stopping_threshold,
        output_dir=args.output_dir,
        eval_at_start=args.eval_at_start,
    )

    sampler = build_weighted_sampler(records)

    # When encoder layers are unfrozen, give their LoRA adapters a lower LR.
    encoder_lora_lr = None
    if args.unfreeze_encoder_layers and args.unfreeze_encoder_layers > 0:
        encoder_lora_lr = (
            args.encoder_lora_lr
            if args.encoder_lora_lr is not None
            else args.learning_rate * 0.1
        )
        print(f"[opt] encoder LoRA LR = {encoder_lora_lr:g} "
              f"(decoder LoRA LR = {args.learning_rate:g})")

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

        def create_optimizer(self):
            # Default path when not unfreezing the encoder: keep upstream behaviour.
            if encoder_lora_lr is None:
                return super().create_optimizer()
            if self.optimizer is not None:
                return self.optimizer
            # Two parameter groups: encoder LoRA (lower LR) vs everything else.
            decay = self.args.weight_decay
            enc_params, base_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if ("audio_tower" in n) or ("audio_encoder" in n):
                    enc_params.append(p)
                else:
                    base_params.append(p)
            # `get_optimizer_cls_and_kwargs` signature drifts across
            # transformers versions: static(args) vs static(args, model) vs
            # instance(self.args). Try the known forms in order.
            optimizer_cls = optimizer_kwargs = None
            for call in (
                lambda: Trainer.get_optimizer_cls_and_kwargs(self.args, self.model),
                lambda: Trainer.get_optimizer_cls_and_kwargs(self.args),
                lambda: self.get_optimizer_cls_and_kwargs(self.args),
            ):
                try:
                    optimizer_cls, optimizer_kwargs = call()
                    break
                except TypeError:
                    continue
            if optimizer_cls is None:
                # Last resort: let the base build a single-group optimizer,
                # then override the encoder group's LR in-place below.
                opt = super().create_optimizer()
                for group in opt.param_groups:
                    if any(
                        id(p) in {id(e) for e in enc_params} for p in group["params"]
                    ):
                        group["lr"] = encoder_lora_lr
                self.optimizer = opt
                return self.optimizer
            optimizer_kwargs.pop("lr", None)
            groups = [
                {"params": base_params, "lr": self.args.learning_rate,
                 "weight_decay": decay},
            ]
            if enc_params:
                groups.append(
                    {"params": enc_params, "lr": encoder_lora_lr,
                     "weight_decay": decay}
                )
            self.optimizer = optimizer_cls(groups, **optimizer_kwargs)
            return self.optimizer

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[eval_cb],
    )

    print("[train] starting")
    resume_ckpt = getattr(args, "resume_from_checkpoint", None)
    if resume_ckpt:
        print(f"[train] resuming from checkpoint: {resume_ckpt}")
        trainer.train(resume_from_checkpoint=str(resume_ckpt))
    else:
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
