#!/usr/bin/env python3
"""Extract medicine names from data/Medicine_Details.csv.

The first column combines name + dosage + form. This script keeps only the
name by cutting off at the first dosage or form token and de-duplicating.

Output format:
    [
        {"name": "<cleaned name>", "line": <row index>},
        ...
    ]

`line` is a 1-based row index in the CSV data (header excluded).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

UNITS = {
    "mg",
    "mcg",
    "ug",
    "g",
    "kg",
    "ml",
    "l",
    "iu",
    "u",
    "unit",
    "units",
    "meq",
    "%",
}

FORM_TOKENS = {
    "tablet",
    "tab",
    "capsule",
    "cap",
    "injection",
    "inj",
    "syrup",
    "suspension",
    "solution",
    "drop",
    "drops",
    "eye",
    "nasal",
    "oral",
    "topical",
    "cream",
    "gel",
    "ointment",
    "lotion",
    "shampoo",
    "powder",
    "dusting",
    "spray",
    "patch",
    "inhaler",
    "respules",
    "granules",
    "sachet",
    "lozenge",
    "chewable",
    "mouth",
    "gargle",
    "elixir",
    "emulsion",
    "vial",
    "ampoule",
    "amp",
    "syringe",
    "prefilled",
    "pen",
    "dispersible",
    "od",
    "dt",
    "sr",
    "er",
    "xr",
    "xl",
    "cr",
    "pr",
    "mr",
    "dr",
    "ir",
    "retard",
}

DOSAGE_NUM_RE = re.compile(r"^\d+(?:\.\d+)?$")
DOSAGE_FRACTION_RE = re.compile(r"^\d+(?:\.\d+)?/\d+(?:\.\d+)?$")


def normalize(text: str) -> str:
    return " ".join(text.split())


def is_dosage_token(token: str) -> bool:
    cleaned = token.strip(" ,;()[]{}")
    if not cleaned or not cleaned[0].isdigit():
        return False
    lower = cleaned.lower()
    if any(unit in lower for unit in UNITS):
        return True
    if DOSAGE_NUM_RE.fullmatch(lower):
        return True
    if DOSAGE_FRACTION_RE.fullmatch(lower):
        return True
    return False


def is_form_token(token: str) -> bool:
    cleaned = token.strip(" ,;()[]{}").lower()
    return cleaned in FORM_TOKENS


def extract_name(raw: str) -> str | None:
    if not raw:
        return None
    text = normalize(raw)
    if not text:
        return None
    tokens = text.split(" ")
    cut_idx = None
    for idx, token in enumerate(tokens):
        if is_dosage_token(token) or is_form_token(token):
            cut_idx = idx
            break
    if cut_idx is None:
        return text
    if cut_idx == 0:
        return text
    name = " ".join(tokens[:cut_idx]).strip()
    return name or text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract medicine names from Medicine_Details.csv"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/Medicine_Details.csv"),
        help="Path to the source CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/medicine_details_cleaned.json"),
        help="Path to write the cleaned names JSON",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Input file not found: {args.input}", file=sys.stderr)
        return 2

    names_by_key = {}
    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header and header[0].strip().lower() != "medicine name":
            # Header was not a standard one; treat it as data.
            name = extract_name(header[0])
            if name:
                names_by_key[name.lower()] = {"name": name, "line": 1}
        for line_no, row in enumerate(reader, start=1):
            if not row:
                continue
            name = extract_name(row[0])
            if not name:
                continue
            key = name.lower()
            if key not in names_by_key:
                names_by_key[key] = {"name": name, "line": line_no}

    names = sorted(names_by_key.values(), key=lambda d: d["name"].lower())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(names, handle, ensure_ascii=True, indent=2)
        handle.write("\n")

    print(f"Wrote {len(names)} unique names to {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
