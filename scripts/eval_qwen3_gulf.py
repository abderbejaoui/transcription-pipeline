"""
Evaluate Qwen3-ASR (base or fine-tuned) on the Gulf medical test set.

Reads manifest.jsonl, transcribes each WAV, computes WER per sample
and overall, plus medical term recall.

Usage (on DGX):
    # Test vadimbelsky's KSA model (baseline before fine-tuning)
    python scripts/eval_qwen3_gulf.py \
        --model vadimbelsky/qwen3-asr-arabic-ksa \
        --manifest eval/gulf_medical_v1/wavs/manifest.jsonl \
        --out eval/gulf_medical_v1/qwen3_ksa_baseline.json

    # Test base Qwen3-ASR (no Gulf fine-tuning)
    python scripts/eval_qwen3_gulf.py \
        --model Qwen/Qwen3-ASR-1.7B \
        --manifest eval/gulf_medical_v1/wavs/manifest.jsonl \
        --out eval/gulf_medical_v1/qwen3_base_baseline.json

    # Test YOUR fine-tuned model after training
    python scripts/eval_qwen3_gulf.py \
        --model data/training/checkpoints/gulf-medical-final \
        --manifest eval/gulf_medical_v1/wavs/manifest.jsonl \
        --out eval/gulf_medical_v1/qwen3_gulf_medical.json
"""
import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

# Medical terms to track recall on
MEDICAL_TERMS = [
    "brufen", "voltaren", "cipro", "panadol", "ventolin", "zyrtec",
    "seretide", "nexium", "augmentin", "zithromax", "claritin",
    "gaviscon", "flagyl", "amoxil", "uti", "tonsillitis", "migraine",
    "hypertension", "diabetes", "asthma", "bronchitis", "sinusitis",
    "gastritis", "dehydration", "anxiety", "covid",
    "blood pressure", "blood sugar", "chest pain", "headache",
    "sore throat", "fever", "cough", "nausea", "vomiting",
    "burning urination", "frequent urination", "skin infection",
    "acid reflux", "back pain", "ear infection",
]


def normalize_text(text: str) -> str:
    """Normalize for WER comparison."""
    text = unicodedata.normalize("NFKC", text)
    # Remove diacritics (Arabic tashkeel)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    # Normalize alef variants
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    # Normalize teh marbuta
    text = text.replace("ة", "ه")
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def wer(ref: str, hyp: str) -> float:
    """Word Error Rate."""
    ref_words = ref.split()
    hyp_words = hyp.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def medical_term_recall(ref: str, hyp: str) -> Dict:
    """Check which medical terms from ref appear in hyp."""
    ref_lower = ref.lower()
    hyp_lower = hyp.lower()
    found_in_ref = []
    found_in_hyp = []
    missed = []

    for term in MEDICAL_TERMS:
        if term in ref_lower:
            found_in_ref.append(term)
            if term in hyp_lower:
                found_in_hyp.append(term)
            else:
                missed.append(term)

    if not found_in_ref:
        return {"recall": None, "found": [], "missed": [], "total": 0}
    return {
        "recall": len(found_in_hyp) / len(found_in_ref),
        "found": found_in_hyp,
        "missed": missed,
        "total": len(found_in_ref),
    }


def load_model(model_name: str):
    """Load Qwen3-ASR model."""
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch

    print(f"[eval] Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()
    print(f"[eval] Model loaded on {next(model.parameters()).device}")
    return model, processor


def transcribe_one(model, processor, audio_path: str) -> str:
    """Transcribe a single WAV file with Qwen3-ASR."""
    import torch
    import librosa

    # Load audio at 16kHz
    audio, sr = librosa.load(audio_path, sr=16000)

    # Build the chat messages (Qwen3-ASR format)
    messages = [
        {"role": "system", "content": "You are a speech recognition model. Transcribe the audio."},
        {"role": "user", "content": [
            {"type": "audio", "audio": audio_path},
        ]},
    ]

    # Try the qwen_asr interface first, fall back to raw transformers
    try:
        from qwen_asr import Qwen3ASRModel
        if not hasattr(transcribe_one, "_qwen_model"):
            transcribe_one._qwen_model = Qwen3ASRModel.from_pretrained(
                model.config._name_or_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        result = transcribe_one._qwen_model.transcribe(audio_path, language="Arabic")
        return result if isinstance(result, str) else str(result)
    except (ImportError, Exception):
        pass

    # Fallback: raw transformers pipeline
    inputs = processor(
        audios=[audio],
        sampling_rate=16000,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.0,
            do_sample=False,
        )
    # Decode only new tokens
    input_len = inputs.get("input_ids", torch.tensor([])).shape[-1] if "input_ids" in inputs else 0
    transcript = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
    )[0]
    return transcript.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model name or local path")
    parser.add_argument("--manifest", required=True, help="manifest.jsonl with audio+text pairs")
    parser.add_argument("--out", required=True, help="Output JSON results file")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit samples (for quick test)")
    args = parser.parse_args()

    # Load manifest
    samples = []
    with open(args.manifest) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    if args.max_samples:
        samples = samples[:args.max_samples]
    print(f"[eval] {len(samples)} samples from {args.manifest}")

    # Load model
    model, processor = load_model(args.model)

    # Evaluate
    results = []
    total_wer = 0.0
    total_med_recall = []
    t0 = time.time()

    for i, sample in enumerate(samples):
        audio_path = sample["audio"]
        ref_text = sample["text"]

        try:
            t1 = time.time()
            hyp_text = transcribe_one(model, processor, audio_path)
            inference_s = time.time() - t1
        except Exception as e:
            print(f"  [{i+1}] ERROR: {e}")
            results.append({"audio": audio_path, "error": str(e)})
            continue

        ref_norm = normalize_text(ref_text)
        hyp_norm = normalize_text(hyp_text)
        sample_wer = wer(ref_norm, hyp_norm)
        med = medical_term_recall(ref_text, hyp_text)

        total_wer += sample_wer
        if med["recall"] is not None:
            total_med_recall.append(med["recall"])

        results.append({
            "audio": audio_path,
            "ref": ref_text,
            "hyp": hyp_text,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "wer": round(sample_wer, 4),
            "medical_recall": med,
            "inference_s": round(inference_s, 2),
            "speaker": sample.get("speaker", ""),
        })

        if (i + 1) % 20 == 0 or i < 3:
            elapsed = time.time() - t0
            avg_wer = total_wer / (i + 1)
            print(f"  [{i+1}/{len(samples)}] WER={sample_wer:.2%} avg={avg_wer:.2%} ({elapsed:.0f}s)")

    # Summary
    elapsed = time.time() - t0
    n = len([r for r in results if "wer" in r])
    avg_wer = total_wer / n if n else 0
    avg_med = sum(total_med_recall) / len(total_med_recall) if total_med_recall else None

    summary = {
        "model": args.model,
        "manifest": args.manifest,
        "n_samples": n,
        "overall_wer": round(avg_wer, 4),
        "medical_term_recall": round(avg_med, 4) if avg_med is not None else None,
        "total_time_s": round(elapsed, 1),
        "results": results,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"Samples: {n}")
    print(f"Overall WER: {avg_wer:.2%}")
    print(f"Medical Term Recall: {avg_med:.2%}" if avg_med else "Medical Term Recall: N/A")
    print(f"Time: {elapsed:.0f}s")
    print(f"Results: {args.out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
