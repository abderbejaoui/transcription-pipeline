#!/usr/bin/env python3
"""Mine code-switch clips out of existing Gulf-Arabic manifests.

Scans one or more JSONL manifests for transcripts that mix Arabic script
with Latin/English tokens (e.g. "المريض عنده blood pressure مرتفع") and emits
a Stage-2 manifest containing only those clips, up-weighted so the weighted
sampler in ``scripts/finetune_qwen3_lora.py`` over-samples them during the
code-switch specialization stage.

A clip counts as code-switch when its transcript contains at least
``--min-latin-tokens`` Latin runs of length >= ``--min-latin-len`` that are
NOT in the ignore list (pure digits, bare units, etc. are excluded).

Output rows keep the original ``audio_path`` (resolved to absolute so the
mined manifest can live anywhere) and add ``code_switch=true``, ``stage=2``,
and an up-weighted ``weight``.

Examples
--------
    python scripts/mine_code_switch.py \
        --in data/preprocessed/saudi_uae_asr/manifest.jsonl \
        --out data/preprocessed/mined_cs/manifest.jsonl \
        --weight 3.0

    # Scan several manifests at once and report stats only:
    python scripts/mine_code_switch.py --in data/preprocessed/*/manifest.jsonl --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]

_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_ARABIC = re.compile(r"[\u0600-\u06FF]")

# Latin tokens that are NOT meaningful code-switch on their own.
_IGNORE_LATIN = {
    "a", "i", "o", "x", "ml", "mg", "kg", "cm", "mm", "km", "tv", "ok", "uae",
}


def _resolve_audio(audio_path: str, manifest: Path) -> str:
    p = Path(audio_path)
    if p.is_absolute():
        return str(p)
    for base in (manifest.parent, manifest.parent.parent, PROJECT_ROOT):
        cand = (base / audio_path)
        if cand.exists():
            return str(cand.resolve())
    return str((manifest.parent / audio_path).resolve())


def _strip_tashkeel(text: str) -> str:
    return re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text or "")


def count_latin_tokens(text: str, min_len: int) -> int:
    n = 0
    for m in _LATIN_RUN.finditer(text or ""):
        tok = m.group(0)
        if len(tok) < min_len:
            continue
        if tok.lower() in _IGNORE_LATIN:
            continue
        n += 1
    return n


def is_code_switch(text: str, min_tokens: int, min_len: int) -> bool:
    text = unicodedata.normalize("NFKC", text or "")
    if not _ARABIC.search(text):
        # No Arabic at all -> not Arabic<->English code-switch.
        return False
    return count_latin_tokens(text, min_len) >= min_tokens


def _read_text(rec: Dict[str, Any]) -> str:
    text = rec.get("text") or rec.get("target") or ""
    if "<asr_text>" in text:
        text = text.split("<asr_text>", 1)[1]
    return text


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True,
                    help="Input manifest JSONL file(s).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output mined manifest JSONL (omit with --dry-run).")
    ap.add_argument("--weight", type=float, default=3.0,
                    help="Up-weight applied to every mined clip (default 3.0).")
    ap.add_argument("--min-latin-tokens", type=int, default=1,
                    help="Min qualifying Latin tokens to count as code-switch.")
    ap.add_argument("--min-latin-len", type=int, default=2,
                    help="Min length of a Latin token to count (default 2).")
    ap.add_argument("--stage", type=int, default=2,
                    help="Stage tag written to mined rows (default 2).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report stats only; do not write output.")
    args = ap.parse_args()

    if not args.dry_run and args.out is None:
        ap.error("--out is required unless --dry-run is given.")

    total = 0
    mined = 0
    out_rows: List[str] = []
    examples: List[str] = []

    for manifest in args.inputs:
        if not manifest.exists():
            print(f"[mine] skip missing {manifest}", file=sys.stderr)
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            total += 1
            rec = json.loads(line)
            text = _read_text(rec)
            if not text:
                continue
            if not is_code_switch(text, args.min_latin_tokens, args.min_latin_len):
                continue
            mined += 1
            if len(examples) < 8:
                examples.append(text)
            audio_path = rec.get("audio_path") or rec.get("audio") or rec.get("path")
            if not audio_path:
                continue
            out_rows.append(json.dumps({
                "audio_path": _resolve_audio(audio_path, manifest),
                "text": text,
                "source": f"mined_cs:{rec.get('source', manifest.stem)}",
                "dialect": rec.get("dialect", "gulf"),
                "code_switch": True,
                "weight": args.weight,
                "stage": args.stage,
            }, ensure_ascii=False))

    pct = (100.0 * mined / total) if total else 0.0
    print(f"[mine] scanned {total} clips, found {mined} code-switch "
          f"({pct:.1f}%)")
    if examples:
        print("[mine] examples:")
        for ex in examples:
            print(f"    {ex[:120]}")

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(out_rows) + ("\n" if out_rows else ""),
                        encoding="utf-8")
    print(f"[mine] wrote {len(out_rows)} mined clips -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
