from __future__ import annotations

import argparse
import sys

try:
    from dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset
except ImportError:
    from scripts.dataset_sample_utils import add_common_args, output_dir, sample_hf_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10 sample clips from UAE Arabic-English bilingual dataset.")
    add_common_args(parser, "uae_bilingual")
    args = parser.parse_args()
    return sample_hf_dataset(
        slug=args.slug,
        dataset_id="vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k",
        split=args.split or "train",
        limit=args.limit,
        out_dir=output_dir(args),
        title="UAE Arabic-English Bilingual Dataset 40k",
        notes="Free/gated on Hugging Face; accept terms and login if required.",
    )


if __name__ == "__main__":
    sys.exit(main())
