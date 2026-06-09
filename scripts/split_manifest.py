#!/usr/bin/env python3
"""Deterministically split a manifest into disjoint train / validation sets.

Guards against the single most common fine-tuning flaw: train/val leakage.
The split is:
  * deterministic (seeded shuffle), so reruns reproduce the exact partition;
  * disjoint by construction (each line goes to exactly one side);
  * optionally stratified by a field (e.g. ``source`` or ``dialect``) so every
    bucket is represented in both train and val;
  * optionally deduplicated by transcript text BEFORE splitting, so the same
    sentence cannot land in both sides (near-duplicate leakage).

It writes ``<out-prefix>.train.jsonl`` and ``<out-prefix>.val.jsonl`` and prints
counts plus a sanity assertion that the two sides share zero lines.

Examples
--------
    python scripts/split_manifest.py \
        --in data/preprocessed/sada22/manifest.jsonl \
        --out-prefix data/preprocessed/sada22/sada22 \
        --val-frac 0.05 --stratify-by source

    # Combine several prepared manifests, then split:
    python scripts/split_manifest.py \
        --in data/preprocessed/*/manifest.jsonl \
        --out-prefix data/splits/gulf --val-frac 0.03 \
        --stratify-by dialect --dedup-text

    # Phase-1 split AND carve ~100h of train into a Phase-2 rehearsal pool
    # (the carved clips are REMOVED from <prefix>.train.jsonl):
    python scripts/split_manifest.py \
        --in data/preprocessed/*/manifest.jsonl \
        --out-prefix data/splits/phase1 --val-frac 0.02 \
        --stratify-by source --dedup-text \
        --carve-hours 100 --carve-out data/splits/phase2_rehearsal.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List


def _norm_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def _read_text(rec: Dict) -> str:
    t = rec.get("text") or rec.get("target") or ""
    return t.split("<asr_text>", 1)[1] if "<asr_text>" in t else t


_DURATION_KEYS = ("duration", "duration_s", "duration_sec", "dur",
                  "length", "seconds")

# Aliases so a manifest written with one schema (e.g. ``source_manifest``)
# can still be stratified by the canonical field name (``source``).
_FIELD_ALIASES = {
    "source": ("source", "source_manifest", "dataset", "corpus", "origin"),
    "dialect": ("dialect", "lang", "language"),
}


def _norm_source(val) -> str:
    """Reduce a path-like source value to a short dataset name."""
    if not isinstance(val, str):
        return str(val)
    s = val.replace("\\", "/").strip("/")
    if "/" not in s:
        return s
    parts = s.split("/")
    if parts[-1].endswith((".jsonl", ".json", ".tsv", ".csv")):
        parts = parts[:-1]
    return parts[-1] if parts else s


def _field(rec: Dict, name: str, default: str = "unknown") -> str:
    """Resolve a stratify field across schema aliases, normalizing paths."""
    for k in _FIELD_ALIASES.get(name, (name,)):
        v = rec.get(k)
        if v not in (None, ""):
            return _norm_source(v) if name == "source" else str(v)
    return default


def _clip_sec(rec: Dict, default_sec: float) -> float:
    """Best-effort clip duration in seconds for hour accounting."""
    for k in _DURATION_KEYS:
        d = rec.get(k)
        try:
            d = float(d)
            if d > 0:
                return d
        except (TypeError, ValueError):
            continue
    return default_sec


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--in", dest="inputs", type=Path, nargs="+", required=True)
    ap.add_argument("--out-prefix", type=Path, required=True,
                    help="Writes <prefix>.train.jsonl and <prefix>.val.jsonl.")
    ap.add_argument("--val-frac", type=float, default=0.05,
                    help="Fraction of clips for validation (default 0.05).")
    ap.add_argument("--stratify-by", default=None,
                    help="Manifest field to stratify on (e.g. source, dialect).")
    ap.add_argument("--dedup-text", action="store_true",
                    help="Drop duplicate transcripts before splitting.")
    ap.add_argument("--carve-hours", type=float, default=0.0,
                    help="Move ~N hours of TRAIN clips into a separate Phase-2 "
                         "rehearsal manifest (removed from the train set). "
                         "0 (default) disables carving.")
    ap.add_argument("--carve-out", type=Path, default=None,
                    help="Path for the carved Phase-2 rehearsal manifest. "
                         "Required when --carve-hours > 0.")
    ap.add_argument("--default-clip-sec", type=float, default=8.0,
                    help="Assumed seconds/clip when a row has no 'duration' "
                         "field, used for hour accounting (default 8.0).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (0.0 < args.val_frac < 1.0):
        ap.error("--val-frac must be in (0, 1).")
    if args.carve_hours < 0.0:
        ap.error("--carve-hours must be >= 0.")
    if args.carve_hours > 0.0 and args.carve_out is None:
        ap.error("--carve-out is required when --carve-hours > 0.")

    rows: List[Dict] = []
    for man in args.inputs:
        if not man.exists():
            print(f"[split] skip missing {man}", file=sys.stderr)
            continue
        # Each source manifest stores audio_path RELATIVE to its own directory
        # (e.g. "audio/foo.wav" lives next to the manifest). The written split
        # lands in data/splits/, so a relative path would no longer resolve.
        # Rewrite to an absolute path anchored at the source manifest's dir so
        # every split row is self-contained regardless of where it is consumed.
        man_dir = man.resolve().parent
        for line in man.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ap_key = "audio_path" if "audio_path" in rec else (
                "audio" if "audio" in rec else ("path" if "path" in rec else None))
            if ap_key:
                raw = str(rec.get(ap_key) or "")
                if raw and not os.path.isabs(raw):
                    rec[ap_key] = str(man_dir / raw)
            rows.append(rec)
    if not rows:
        print("[split] no rows read.", file=sys.stderr)
        return 1

    # Held-out benchmark rows (tagged eval_only by prepare_datasets.py) must
    # NEVER enter the train/val split — they are a separate test set. Drop them
    # here so a stray eval-only manifest in the glob cannot leak into training.
    n_eval_only = sum(1 for r in rows if r.get("eval_only"))
    if n_eval_only:
        rows = [r for r in rows if not r.get("eval_only")]
        print(f"[split] excluded {n_eval_only} eval_only (held-out benchmark) "
              f"row(s) from the train/val split")
    if not rows:
        print("[split] no trainable rows after eval_only filter.", file=sys.stderr)
        return 1

    if args.dedup_text:
        seen = set()
        deduped = []
        for r in rows:
            k = _norm_key(_read_text(r))
            if k and k in seen:
                continue
            seen.add(k)
            deduped.append(r)
        print(f"[split] dedup-text: {len(rows)} -> {len(deduped)} rows")
        rows = deduped

    rng = random.Random(args.seed)

    if args.stratify_by:
        groups: Dict[str, List[Dict]] = defaultdict(list)
        for r in rows:
            groups[_field(r, args.stratify_by)].append(r)
        train, val = [], []
        for key, items in groups.items():
            rng.shuffle(items)
            n_val = max(1, round(len(items) * args.val_frac)) if len(items) > 1 else 0
            val.extend(items[:n_val])
            train.extend(items[n_val:])
    else:
        rng.shuffle(rows)
        n_val = max(1, round(len(rows) * args.val_frac))
        val, train = rows[:n_val], rows[n_val:]

    # Sanity: the two sides must be disjoint at the object level.
    def _sig(r: Dict) -> str:
        return (r.get("audio_path") or r.get("audio") or r.get("path") or "") + "\x00" + _read_text(r)

    train_sigs = {_sig(r) for r in train}
    overlap = sum(1 for r in val if _sig(r) in train_sigs)
    if overlap:
        print(f"[split] FATAL: {overlap} val clips also appear in train "
              f"(leakage). Aborting.", file=sys.stderr)
        return 2

    # ------------------------------------------------------------------ carve
    # Optionally pull ~N hours OUT of the train set into a separate Phase-2
    # rehearsal manifest. The carved clips are REMOVED from train so the same
    # audio is never seen in both phases. We carve proportionally across the
    # stratify field (or `source` as a fallback) so the rehearsal pool stays as
    # diverse as the full corpus rather than draining one dataset.
    carved: List[Dict] = []
    if args.carve_hours > 0.0:
        target_sec = args.carve_hours * 3600.0
        carve_key = args.stratify_by or "source"
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for r in train:
            buckets[_field(r, carve_key)].append(r)
        total_sec = sum(_clip_sec(r, args.default_clip_sec) for r in train)
        if total_sec <= 0:
            print("[split] FATAL: train has zero total duration; cannot carve.",
                  file=sys.stderr)
            return 2
        if target_sec >= total_sec:
            print(f"[split] FATAL: --carve-hours {args.carve_hours:.1f}h >= "
                  f"available train ({total_sec/3600:.1f}h). Aborting.",
                  file=sys.stderr)
            return 2
        frac = target_sec / total_sec
        carve_rng = random.Random(args.seed + 1)
        carved_sigs: set = set()
        for key, items in buckets.items():
            carve_rng.shuffle(items)
            want_sec = sum(_clip_sec(r, args.default_clip_sec)
                           for r in items) * frac
            acc = 0.0
            for r in items:
                if acc >= want_sec:
                    break
                carved.append(r)
                carved_sigs.add(_sig(r))
                acc += _clip_sec(r, args.default_clip_sec)
        # Drop carved rows from train and re-tag them for Phase 2 rehearsal.
        train = [r for r in train if _sig(r) not in carved_sigs]
        for r in carved:
            r["rehearsal"] = True
            r["stage"] = 2
        carved_sec = sum(_clip_sec(r, args.default_clip_sec) for r in carved)
        args.carve_out.parent.mkdir(parents=True, exist_ok=True)
        args.carve_out.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in carved) + "\n",
            encoding="utf-8")
        print(f"[split] carved {len(carved)} clips (~{carved_sec/3600:.1f}h, "
              f"target {args.carve_hours:.1f}h) into Phase-2 rehearsal -> "
              f"{args.carve_out}")

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    train_path = args.out_prefix.parent / f"{args.out_prefix.name}.train.jsonl"
    val_path = args.out_prefix.parent / f"{args.out_prefix.name}.val.jsonl"
    train_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n",
        encoding="utf-8")
    val_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n",
        encoding="utf-8")

    vf = len(val) / (len(train) + len(val)) if (train or val) else 0.0
    carved_note = f"  carved={len(carved)}" if carved else ""
    print(f"[split] train={len(train)}  val={len(val)}{carved_note}  "
          f"(val_frac={vf:.3f} of train+val, leakage=0)")
    print(f"[split] -> {train_path}")
    print(f"[split] -> {val_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
