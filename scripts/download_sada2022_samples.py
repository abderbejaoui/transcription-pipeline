from __future__ import annotations

import argparse
import sys

try:
    from dataset_sample_utils import add_common_args, output_dir, sample_kaggle_dataset
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, sample_kaggle_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10 sample clips from Kaggle SADA 2022.")
    add_common_args(parser, "sada2022")
    args = parser.parse_args()
    return sample_kaggle_dataset(
        slug=args.slug,
        kaggle_id="sdaiancai/sada2022",
        limit=args.limit,
        out_dir=output_dir(args),
        title="SADA 2022 Saudi Audio Dataset",
        notes="Requires Kaggle CLI credentials. The raw Kaggle archive is kept under raw/.",
    )


if __name__ == "__main__":
    sys.exit(main())
