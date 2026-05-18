from __future__ import annotations

import argparse
import sys

try:
    from dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download sample clips from WorldSpeech Bahrain Arabic.")
    add_common_args(parser, "worldspeech_bahrain")
    args = parser.parse_args()
    return sample_hf_dataset(
        slug=args.slug,
        dataset_id="disco-eth/WorldSpeech",
        config="ar_bh",
        split=args.split or "train",
        limit=args.limit,
        out_dir=output_dir(args),
        title="WorldSpeech Arabic Bahrain",
        notes="Neighbor-Gulf augmentation. Formal/parliamentary domain; use as auxiliary data, not Emirati replacement.",
    )


if __name__ == "__main__":
    sys.exit(main())