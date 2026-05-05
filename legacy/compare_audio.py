"""Compare a real audio recording against the text-derived embeddings.

Example:
    # First build the index:
    python run.py

    # Then match a recording against every term:
    python compare_audio.py path/to/recording.wav --top 5
"""

from __future__ import annotations

import argparse

import numpy as np

from pipeline import SoundEmbedder, cosine_similarity


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", help="Audio file (wav/flac/mp3/...) of a spoken term.")
    parser.add_argument("--index", default="embeddings.npz",
                        help="Saved embeddings file produced by run.py.")
    parser.add_argument("--top", type=int, default=5,
                        help="How many best matches to display.")
    args = parser.parse_args()

    data = np.load(args.index, allow_pickle=True)
    terms = list(data["terms"])
    emb = data["embeddings"]  # (N, 768), already L2-normalized
    print(f"Loaded index: {len(terms)} terms, dim={emb.shape[1]}")

    embedder = SoundEmbedder.load()
    query = embedder.embed_audio_file(args.audio)  # (768,), L2-normalized

    sims = emb @ query  # cosine since both sides are L2-normalized
    order = np.argsort(-sims)

    print(f"\nTop {args.top} matches for {args.audio!r}:")
    for rank, idx in enumerate(order[: args.top], 1):
        print(f"  {rank}. {sims[idx]:6.3f}  {terms[idx]}")


if __name__ == "__main__":
    main()
