"""Train the n-gram language model on medical text and save as pickle.

Usage:
    python scripts/train_lm.py
    python scripts/train_lm.py --data data/lm_training_corpus.txt --order 4 --discount 0.75
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(PROJECT_ROOT / "data" / "lm_training_corpus.txt"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "app" / "services" / "medical_lm.pkl"))
    ap.add_argument("--order", type=int, default=4)
    ap.add_argument("--discount", type=float, default=0.75)
    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[train_lm] ERROR: training data not found at {data_path}")
        print(f"[train_lm] Run `python scripts/generate_lm_training_data.py` first.")
        return 1

    sys.path.insert(0, str(PROJECT_ROOT))
    from app.services.ngram_lm import NGramLM, tokenize

    # Load and tokenize
    print(f"[train_lm] Loading data from {data_path}...")
    lines = data_path.read_text(encoding="utf-8").splitlines()
    sentences = [tokenize(line) for line in lines if line.strip()]
    print(f"[train_lm] Loaded {len(sentences)} sentences")

    # Count tokens
    total_tokens = sum(len(s) for s in sentences)
    print(f"[train_lm] Total tokens: {total_tokens}")

    # Train
    print(f"[train_lm] Training {args.order}-gram LM (discount={args.discount})...")
    t0 = time.time()
    lm = NGramLM(order=args.order, discount=args.discount)
    lm.train(sentences)
    elapsed = time.time() - t0
    print(f"[train_lm] Trained in {elapsed:.2f}s")
    print(f"[train_lm] Vocabulary size: {lm.get_vocab_size()}")
    print(f"[train_lm] Unique n-grams: {lm.get_total_ngrams()}")

    # Quick sanity check — word is a str, context is list[str]
    test_cases = [
        ("pain", ["chest", "severe"]),
        ("pain", ["chest"]),
        ("x-ray", ["chest"]),
        ("elephant", ["chest"]),  # should be very unlikely
        ("الدم", ["نسبة"]),
        ("asdfgh", ["the"]),  # OOV
    ]
    print(f"\n[train_lm] Sanity checks (word perplexity = -log10 prob; HIGHER = more surprising):")
    for word, context in test_cases:
        ppl = lm.word_perplexity(word, context)
        print(f"  PPL({word!r} | {' '.join(context)}) = {ppl:.4f}")

    # Save
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lm.save(str(out_path))
    print(f"\n[train_lm] Model saved to {out_path}")
    print(f"[train_lm] File size: {out_path.stat().st_size / 1024:.1f} KB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
