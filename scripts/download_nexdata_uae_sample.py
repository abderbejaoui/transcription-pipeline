from __future__ import annotations

import argparse
import sys

try:
    from dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download up to 10 sample clips from Nexdata UAE Arabic sample.")
    add_common_args(parser, "nexdata_uae_sample")
    args = parser.parse_args()
    return sample_hf_dataset(
        slug=args.slug,
        dataset_id="Nexdata/UAE_Arabic_Spontaneous_Speech_Data",
        split=args.split or "train",
        limit=args.limit,
        out_dir=output_dir(args),
        streaming=False,
        title="Nexdata UAE Arabic Spontaneous Speech Sample",
        notes="Tiny free sample of a commercial UAE dataset; useful only for inspection.",
    )


if __name__ == "__main__":
    sys.exit(main())
