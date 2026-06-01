#!/usr/bin/env python3
"""Extract drug names from data/drugs.json into data/drugs_cleaned.json.

The source file is a large JSON object with a top-level "results" array.
This script streams that array to avoid loading the full file into memory.
"""

import argparse
import json
import sys
from pathlib import Path

NAME_FIELDS = ("brand_name", "generic_name", "substance_name")


def iter_results(path: Path):
    """Yield objects from the top-level results array in a large JSON file."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buf = ""
        # Find the start of the "results" array.
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                raise RuntimeError("results array not found")
            buf += chunk
            key_idx = buf.find('"results"')
            if key_idx != -1:
                arr_idx = buf.find("[", key_idx)
                if arr_idx != -1:
                    buf = buf[arr_idx + 1 :]
                    break
            if len(buf) > 200:
                buf = buf[-200:]
        # Stream items from the array.
        while True:
            buf = buf.lstrip()
            if buf.startswith("]"):
                return
            try:
                obj, end = decoder.raw_decode(buf)
            except json.JSONDecodeError:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    raise
                buf += chunk
                continue
            yield obj
            buf = buf[end:]
            buf = buf.lstrip()
            if buf.startswith(","):
                buf = buf[1:]


def extract_names(record):
    openfda = record.get("openfda") or {}
    for field in NAME_FIELDS:
        value = openfda.get(field)
        if not value:
            continue
        if isinstance(value, list):
            for item in value:
                yield item
        elif isinstance(value, str):
            yield value


def clean_name(name: str):
    if not isinstance(name, str):
        return None
    cleaned = " ".join(name.split())
    return cleaned or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract drug names from data/drugs.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/drugs.json"),
        help="Path to the source drugs.json file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/drugs_cleaned.json"),
        help="Path to write the cleaned names JSON",
    )
    parser.add_argument(
        "--progress",
        type=int,
        default=5000,
        help="Print progress every N records (0 to disable)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    names_by_key = {}
    total = 0
    for total, record in enumerate(iter_results(args.input), start=1):
        for raw_name in extract_names(record):
            cleaned = clean_name(raw_name)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key not in names_by_key:
                names_by_key[key] = cleaned
        if args.progress and total % args.progress == 0:
            print(f"Processed {total} records...", file=sys.stderr)

    names = sorted(names_by_key.values(), key=str.lower)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(names, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print(
        f"Wrote {len(names)} unique names to {args.output}", file=sys.stderr
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
