"""Embed a list of medical terms into pronunciation embeddings.

Usage:
    python run.py                               # uses medical_terms.txt
    python run.py --terms my_terms.txt          # custom list
    python run.py --save-audio out/wavs         # also save synthesized .wav files
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pipeline import (
    SoundEmbedder,
    cosine_similarity_matrix,
    load_terms,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--terms", default="medical_terms.txt",
                        help="Path to a text file with one term per line.")
    parser.add_argument("--out", default="embeddings.npz",
                        help="Output .npz file (keys: terms, embeddings).")
    parser.add_argument("--save-audio", default=None,
                        help="If set, also save each synthesized utterance "
                             "as a .wav file in this directory.")
    args = parser.parse_args()

    terms = load_terms(args.terms)
    print(f"Loaded {len(terms)} terms from {args.terms}")

    embedder = SoundEmbedder.load()

    print("\nSynthesizing + embedding...")
    emb = embedder.embed_terms(terms, save_audio_dir=args.save_audio)
    print(f"\nembeddings matrix shape: {emb.shape}")

    np.savez(args.out, terms=np.array(terms, dtype=object), embeddings=emb)
    print(f"Saved -> {os.path.abspath(args.out)}")

    print("\nSanity check: most similar pairs in the set")
    sims = cosine_similarity_matrix(emb)
    np.fill_diagonal(sims, -np.inf)
    flat = [(sims[i, j], terms[i], terms[j])
            for i in range(len(terms))
            for j in range(i + 1, len(terms))]
    flat.sort(key=lambda x: -x[0])
    for score, a, b in flat[:10]:
        print(f"  {score:6.3f}  {a:35s}  <->  {b}")


if __name__ == "__main__":
    main()
