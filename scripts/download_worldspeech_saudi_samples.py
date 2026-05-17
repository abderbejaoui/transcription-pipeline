from __future__ import annotations

import argparse
import sys

try:
    from dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10 sample clips from WorldSpeech Saudi Arabic.")
    add_common_args(parser, "worldspeech_saudi")
    args = parser.parse_args()
    return sample_hf_dataset(
        slug=args.slug,
        dataset_id="disco-eth/WorldSpeech",
        config="ar_sa",
        split=args.split or "train",
        limit=args.limit,
        out_dir=output_dir(args),
        title="WorldSpeech Arabic Saudi Arabia",
        notes="Free/gated on Hugging Face; accept terms and login if required.",
    )


if __name__ == "__main__":
    sys.exit(main())
