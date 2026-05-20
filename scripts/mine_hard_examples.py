"""ILT Round 2 prep: mine high-WER training clips after Round 1.

After Round-1 LoRA finishes, run inference over the training corpus with the
adapter loaded, score each clip with the Wang et al. 2024 Arabic normalizer,
and write a new manifest containing only the top-K hardest clips (per-clip
WER >= threshold). Round 2 trains a second LoRA on these. The two adapters
are merged with `scripts.merge_adapters` (weighted average β=0.7/0.3).

This mirrors the official Qwen3-ASR loader (`Qwen3ASRModel.from_pretrained`)
plus PEFT adapter loading on `model.thinker.model.language_model`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    ap.add_argument("--adapter", type=Path, required=True,
                    help="Path to Round-1 LoRA adapter directory.")
    ap.add_argument("--train-manifest", type=Path, required=True)
    ap.add_argument("--output-manifest", type=Path, required=True)
    ap.add_argument("--min-wer", type=float, default=0.30,
                    help="Keep clips with per-clip WER >= this.")
    ap.add_argument("--max-keep", type=int, default=200_000,
                    help="Cap total kept clips after sorting by WER desc.")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import torch
    import soundfile as sf
    import jiwer
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel
    from scripts.eval_arabic import normalize_arabic_text

    print(f"[load] base={args.model}  adapter={args.adapter}")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    base_model = asr_wrapper.model
    processor = asr_wrapper.processor

    # Attach PEFT adapter to the same module the trainer trained
    # (the outer wrapper, with patch_outer_forward routing into thinker).
    model = PeftModel.from_pretrained(base_model, str(args.adapter))
    model.eval()

    # Build the inference prefix once.
    prefix_msgs = _build_prefix_messages("", None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs], add_generation_prompt=True, tokenize=False,
    )[0]

    records: List[Dict[str, Any]] = []
    for line in args.train_manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    print(f"[mine] scoring {len(records)} clips at batch={args.batch_size}")

    scored: List[Dict[str, Any]] = []
    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        audios = []
        for r in batch:
            ap_path = r.get("audio_path") or r.get("audio")
            if not ap_path.startswith("/"):
                ap_path = str(PROJECT_ROOT / ap_path)
            arr, sr = sf.read(ap_path, dtype="float32", always_2d=False)
            if arr.ndim > 1:
                arr = arr.mean(axis=1)
            if sr != 16_000:
                try:
                    import soxr
                    arr = soxr.resample(arr, sr, 16_000)
                except ImportError:
                    import librosa
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=16_000)
            audios.append(arr)
        inputs = processor(
            text=[prefix_text] * len(audios), audio=audios,
            return_tensors="pt", padding=True,
        ).to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=448,
                                 do_sample=False, num_beams=1)
        out = gen[:, inputs["input_ids"].shape[1]:]
        hyps = processor.batch_decode(out, skip_special_tokens=True)
        for r, h in zip(batch, hyps):
            if "<asr_text>" in h:
                h = h.split("<asr_text>", 1)[1]
            ref = r["text"]
            if "<asr_text>" in ref:
                ref = ref.split("<asr_text>", 1)[1]
            ref_n = normalize_arabic_text(ref)
            hyp_n = normalize_arabic_text(h)
            if not ref_n:
                continue
            w = jiwer.wer([ref_n], [hyp_n])
            if w >= args.min_wer:
                r2 = dict(r)
                r2["round1_wer"] = round(w, 4)
                r2["round1_hyp"] = h
                # Boost weight for very-hard clips so Round 2 sees them more often.
                r2["weight"] = float(r2.get("weight", 1.0)) * (1.0 + min(w, 1.0))
                scored.append(r2)
        if (i // args.batch_size) % 50 == 0:
            print(f"  scored {i+len(batch)}/{len(records)} kept_so_far={len(scored)}")

    scored.sort(key=lambda r: -r["round1_wer"])
    scored = scored[: args.max_keep]
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in scored) + "\n",
        encoding="utf-8",
    )
    print(f"[mine] wrote {len(scored)} hard clips -> {args.output_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
