#!/usr/bin/env python3
"""Robust held-out evaluation for Qwen3-ASR (base or LoRA fine-tuned).

This shares its inference recipe with training: it imports ``_run_eval``,
``_build_prefix_messages`` and ``_resolve_audio_path`` from
``scripts/finetune_qwen3_lora.py`` so the prompt, dtype casting and
generate() handling are byte-for-byte identical to what the training-time
eval callback uses. That guarantees the WER you see here matches the WER the
trainer reported.

What it computes
----------------
* Overall WER / CER (jiwer, Arabic-normalized).
* Per-source, per-dialect, and code-switch vs non-code-switch breakdowns
  (driven by the ``source`` / ``dialect`` / ``code_switch`` manifest fields
  written by ``prepare_datasets.py`` and ``mine_code_switch.py``).
* Medical-term recall on a curated Gulf medical term list.

Robustness guarantees (mirrors the training eval):
* A single unreadable/corrupt clip is SKIPPED, never aborts the run.
* generate() output is normalized (tuple / GenerateOutput / tensor).
* Held-out only: this script never trains and never touches the train set,
  so there is no leakage. Pass the *validation*/*test* manifest(s).

Examples
--------
Evaluate the base model:
    python scripts/test_asr.py \
        --model-path Qwen/Qwen3-ASR-1.7B \
        --manifest eval/gulf_medical_v1/wavs/manifest.jsonl \
        --out eval_results/base.json

Evaluate a fine-tuned LoRA adapter:
    python scripts/test_asr.py \
        --model-path Qwen/Qwen3-ASR-1.7B \
        --adapter runs/qwen3_lora/best_adapter \
        --manifest eval/gulf_medical_v1/wavs/manifest.jsonl \
        --breakdown --out eval_results/ft.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import the EXACT inference path used by training so numbers are comparable.
from scripts.finetune_qwen3_lora import (  # noqa: E402
    _run_eval,
    _resolve_audio_path,
    _build_prefix_messages,
)
from scripts.eval_arabic import normalize_arabic_text  # noqa: E402

# Curated Gulf medical terms for recall tracking (English + transliterations).
MEDICAL_TERMS = [
    "brufen", "voltaren", "cipro", "panadol", "ventolin", "zyrtec",
    "seretide", "nexium", "augmentin", "zithromax", "claritin", "gaviscon",
    "flagyl", "amoxil", "uti", "tonsillitis", "migraine", "hypertension",
    "diabetes", "asthma", "bronchitis", "sinusitis", "gastritis",
    "dehydration", "anxiety", "covid", "blood pressure", "blood sugar",
    "chest pain", "headache", "sore throat", "fever", "cough", "nausea",
    "vomiting", "acid reflux", "back pain", "ear infection",
]


def load_model(model_path: str, adapter: Optional[str]):
    """Load Qwen3-ASR via the official wrapper, then attach a LoRA adapter
    if one is given. Mirrors the loading used in training."""
    import torch
    from qwen_asr import Qwen3ASRModel

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    print(f"[test] loading {model_path} bf16={use_bf16}")
    wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = wrapper.model
    processor = wrapper.processor

    if adapter:
        from peft import PeftModel
        print(f"[test] attaching adapter {adapter}")
        model = PeftModel.from_pretrained(model, adapter)
        # Merge for faster inference; falls back gracefully if unsupported
        # (e.g. DoRA on some peft versions).
        try:
            model = model.merge_and_unload()
            print("[test] adapter merged into base for inference")
        except Exception as exc:
            print(f"[test] merge_and_unload skipped ({exc!r}); running with PEFT wrapper")

    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    return model, processor


def medical_term_recall(ref: str, hyp: str) -> Tuple[int, int]:
    """Return (found_in_hyp, present_in_ref) for medical terms."""
    ref_l = ref.lower()
    hyp_l = hyp.lower()
    present = 0
    found = 0
    for term in MEDICAL_TERMS:
        if term in ref_l:
            present += 1
            if term in hyp_l:
                found += 1
    return found, present


def _transcribe(model, processor, audio_path: str) -> str:
    """Single-clip transcription using the same prompt/dtype/generate path
    as the training eval (kept in sync with ``_run_eval``)."""
    import soundfile as sf
    import torch

    arr, sr = sf.read(audio_path, dtype="float32", always_2d=False)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    if sr != 16_000:
        try:
            import soxr
            arr = soxr.resample(arr, sr, 16_000)
        except ImportError:
            import librosa
            arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)

    prefix_msgs = _build_prefix_messages("", None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs], add_generation_prompt=True, tokenize=False,
    )[0]
    inputs = processor(
        text=[prefix_text], audio=[arr], return_tensors="pt", padding=True,
    ).to(model.device)
    model_dtype = getattr(model, "dtype", None)
    if model_dtype is not None:
        for k, v in list(inputs.items()):
            if torch.is_tensor(v) and v.is_floating_point():
                inputs[k] = v.to(dtype=model_dtype)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=448, do_sample=False, num_beams=1)
    if not torch.is_tensor(gen):
        seq = getattr(gen, "sequences", None)
        gen = seq if seq is not None else gen[0]
    out = gen[:, inputs["input_ids"].shape[1]:]
    hyp = processor.batch_decode(out, skip_special_tokens=True)[0]
    if "<asr_text>" in hyp:
        hyp = hyp.split("<asr_text>", 1)[1]
    return hyp


def evaluate(
    model, processor, manifest: Path, out_path: Optional[Path],
    breakdown: bool, max_samples: Optional[int],
) -> Dict:
    """Per-sample evaluation with optional source/dialect/CS breakdown."""
    import jiwer

    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if max_samples is not None and 0 < max_samples < len(lines):
        lines = lines[:max_samples]

    refs: List[str] = []
    hyps: List[str] = []
    per_sample: List[Dict] = []
    skipped = 0
    med_found = 0
    med_present = 0

    # Buckets: key -> (refs, hyps)
    buckets: Dict[str, Tuple[List[str], List[str]]] = defaultdict(lambda: ([], []))

    t0 = time.time()
    for i, line in enumerate(lines):
        rec = json.loads(line)
        ap = rec.get("audio_path") or rec.get("path") or rec.get("audio")
        if not ap:
            skipped += 1
            continue
        ap = _resolve_audio_path(ap, manifest)
        ref = rec.get("text") or rec.get("target") or ""
        if "<asr_text>" in ref:
            ref = ref.split("<asr_text>", 1)[1]
        try:
            hyp = _transcribe(model, processor, ap)
        except Exception as exc:
            skipped += 1
            if skipped <= 1:
                print(f"[test] first failure on {ap}: {exc!r}")
            continue

        rn = normalize_arabic_text(ref)
        hn = normalize_arabic_text(hyp)
        refs.append(rn)
        hyps.append(hn)

        f, p = medical_term_recall(ref, hyp)
        med_found += f
        med_present += p

        if breakdown:
            for key in (
                f"source:{rec.get('source', 'unknown')}",
                f"dialect:{rec.get('dialect', 'unknown')}",
                f"code_switch:{bool(rec.get('code_switch', False))}",
            ):
                buckets[key][0].append(rn)
                buckets[key][1].append(hn)

        if out_path is not None:
            per_sample.append({
                "audio_path": ap, "ref": ref, "hyp": hyp,
                "wer": round(jiwer.wer([rn], [hn]), 4),
                "source": rec.get("source"), "dialect": rec.get("dialect"),
                "code_switch": rec.get("code_switch"),
            })

        if (i + 1) % 20 == 0:
            cur = jiwer.wer(refs, hyps) if refs else float("nan")
            print(f"[test] {i+1}/{len(lines)} running WER={cur*100:.2f}% "
                  f"({time.time()-t0:.0f}s)")

    if not refs:
        print("[test] no usable samples!", file=sys.stderr)
        return {"manifest": str(manifest), "n": 0}

    overall = {
        "manifest": str(manifest),
        "n": len(refs),
        "skipped": skipped,
        "wer": round(jiwer.wer(refs, hyps), 4),
        "cer": round(jiwer.cer(refs, hyps), 4),
        "medical_term_recall": round(med_found / med_present, 4) if med_present else None,
        "medical_terms_present": med_present,
        "elapsed_s": round(time.time() - t0, 1),
    }

    if breakdown:
        bd = {}
        for key, (r, h) in sorted(buckets.items()):
            if not r:
                continue
            bd[key] = {
                "n": len(r),
                "wer": round(jiwer.wer(r, h), 4),
                "cer": round(jiwer.cer(r, h), 4),
            }
        overall["breakdown"] = bd

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(overall)
        payload["results"] = per_sample
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[test] wrote per-sample results -> {out_path}")

    return overall


def _print_summary(model_path: str, adapter: Optional[str], res: Dict) -> None:
    print("\n" + "=" * 60)
    print(f"model:   {model_path}")
    if adapter:
        print(f"adapter: {adapter}")
    print(f"samples: {res.get('n', 0)} (skipped {res.get('skipped', 0)})")
    if res.get("n"):
        print(f"WER:     {res['wer']*100:.2f}%")
        print(f"CER:     {res['cer']*100:.2f}%")
        mtr = res.get("medical_term_recall")
        if mtr is not None:
            print(f"med recall: {mtr*100:.2f}% "
                  f"({res.get('medical_terms_present', 0)} terms in refs)")
        if res.get("breakdown"):
            print("-" * 60)
            print(f"{'bucket':<32}{'n':>6}{'WER':>10}{'CER':>10}")
            for key, b in res["breakdown"].items():
                print(f"{key:<32}{b['n']:>6}{b['wer']*100:>9.2f}%{b['cer']*100:>9.2f}%")
    print("=" * 60)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-path", default="Qwen/Qwen3-ASR-1.7B",
                    help="Base model HF id or local path.")
    ap.add_argument("--adapter", default=None,
                    help="Optional LoRA adapter dir (best_adapter/final_adapter).")
    ap.add_argument("--manifest", type=Path, nargs="+", required=True,
                    help="Held-out validation/test manifest(s).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Write per-sample JSON results (single-manifest only).")
    ap.add_argument("--breakdown", action="store_true",
                    help="Report per-source / per-dialect / code-switch WER.")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Cap clips per manifest (smoke test). Default: full set.")
    ap.add_argument("--fast", action="store_true",
                    help="Use the training _run_eval path (overall WER/CER only, "
                         "no breakdown/medical-recall) for a quick comparable number.")
    args = ap.parse_args()

    model, processor = load_model(args.model_path, args.adapter)

    if args.fast:
        for man in args.manifest:
            wer, cer, n = _run_eval(model, processor, man, max_samples=args.max_samples)
            print(f"[fast] {man.name}: WER={wer*100:.2f}%  CER={cer*100:.2f}%  n={n}")
        return 0

    if args.out is not None and len(args.manifest) > 1:
        print("[test] --out only supported with a single manifest; ignoring --out.",
              file=sys.stderr)
        args.out = None

    for man in args.manifest:
        res = evaluate(
            model, processor, man, args.out, args.breakdown, args.max_samples,
        )
        _print_summary(args.model_path, args.adapter, res)
    return 0


if __name__ == "__main__":
    sys.exit(main())
