from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK_DIR = PROJECT_ROOT / "data" / "dgx_full"

AVAILABLE_AUDIO_DATASETS = (
    "sada2022",
    "worldspeech_saudi",
    "nexdata_uae_sample",
)


def _run(cmd: Sequence[str]) -> None:
    print("$ " + " ".join(str(c) for c in cmd), flush=True)
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    subprocess.run([str(c) for c in cmd], cwd=str(PROJECT_ROOT), env=env, check=True)


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _dataset_dirs(work_dir: Path) -> Dict[str, Path]:
    raw_dir = work_dir / "raw_datasets"
    return {
        "sada2022": raw_dir / "sada2022",
        "worldspeech_saudi": raw_dir / "worldspeech_saudi",
        "nexdata_uae_sample": raw_dir / "nexdata_uae_sample",
    }


def download_available_audio_datasets(work_dir: Path) -> List[Path]:
    dirs = _dataset_dirs(work_dir)
    commands = [
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "download_sada2022_samples.py",
            "--limit", "0",
            "--all-batches",
            "--out", dirs["sada2022"],
        ],
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "download_worldspeech_saudi_samples.py",
            "--limit", "0",
            "--out", dirs["worldspeech_saudi"],
        ],
        [
            sys.executable,
            PROJECT_ROOT / "scripts" / "download_nexdata_uae_sample.py",
            "--limit", "0",
            "--out", dirs["nexdata_uae_sample"],
        ],
    ]
    for cmd in commands:
        _run(cmd)
    manifests = [dirs[name] / "manifest.jsonl" for name in AVAILABLE_AUDIO_DATASETS]
    missing = [str(path) for path in manifests if not path.exists()]
    if missing:
        raise RuntimeError("missing downloaded manifests: " + ", ".join(missing))
    return manifests


def preprocess(manifests: Sequence[Path], out_dir: Path, *, min_duration: float, max_duration: float) -> None:
    cmd: List[object] = [
        sys.executable,
        PROJECT_ROOT / "scripts" / "preprocess_code_switch_asr.py",
        "--out", out_dir,
        "--min-duration-s", str(min_duration),
        "--max-duration-s", str(max_duration),
    ]
    for manifest in manifests:
        cmd.extend(["--manifest", manifest])
    _run(cmd)


def _split_group_key(row: Dict) -> str:
    # Keep all segments from one source recording together. This avoids train
    # vs eval leakage from the same 10-minute SADA/Nexdata recording.
    source_audio = row.get("source_audio") or row.get("audio_path") or row.get("id")
    source_manifest = row.get("source_manifest") or ""
    return f"{Path(source_manifest).parent.name}:{source_audio}"


def _source_name(row: Dict) -> str:
    return Path(row.get("source_manifest", "")).parent.name or "unknown"


def _assign_groups_by_duration(groups: List[List[Dict]], ratios: Dict[str, float], seed: int) -> Dict[str, List[Dict]]:
    rng = random.Random(seed)
    rng.shuffle(groups)
    total_duration = sum(sum(float(row.get("duration_s") or 0.0) for row in group) for group in groups)
    targets = {name: total_duration * ratio for name, ratio in ratios.items()}
    splits = {name: [] for name in ratios}
    durations = {name: 0.0 for name in ratios}
    ordered_names = list(ratios.keys())

    # If possible, seed non-train splits with at least one recording group.
    # This matters for small dry runs and still has negligible effect on full
    # corpora. We continue to preserve recording-level grouping.
    seeded_names = [name for name in ordered_names if name != "train" and ratios[name] > 0.0]
    remaining_groups = list(groups)
    if len(remaining_groups) > len(seeded_names):
        for name in seeded_names:
            group = remaining_groups.pop(0)
            group_duration = sum(float(row.get("duration_s") or 0.0) for row in group)
            splits[name].extend(group)
            durations[name] += group_duration

    for group in remaining_groups:
        group_duration = sum(float(row.get("duration_s") or 0.0) for row in group)
        # Greedy duration balancing against target ratios, preserving groups.
        # Choose the split with the largest remaining duration deficit.
        best_name = max(
            ordered_names,
            key=lambda name: (targets[name] - durations[name], ratios[name]),
        )
        splits[best_name].extend(group)
        durations[best_name] += group_duration
    return splits


def write_train_validation_test_splits(
    preprocessed_dir: Path,
    *,
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict:
    rows = _read_jsonl(preprocessed_dir / "manifest.jsonl")
    if not rows:
        raise RuntimeError(f"no rows in {preprocessed_dir / 'manifest.jsonl'}")
    ratio_sum = train_ratio + validation_ratio + test_ratio
    if ratio_sum <= 0:
        raise ValueError("split ratios must sum to > 0")
    ratios = {
        "train": train_ratio / ratio_sum,
        "validation": validation_ratio / ratio_sum,
        "test": test_ratio / ratio_sum,
    }

    by_source_and_group: Dict[str, Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_source_and_group[_source_name(row)][_split_group_key(row)].append(row)

    final_splits = {"train": [], "validation": [], "test": []}
    for source, grouped in sorted(by_source_and_group.items()):
        source_groups = list(grouped.values())
        source_seed = seed + sum((idx + 1) * ord(ch) for idx, ch in enumerate(source))
        source_splits = _assign_groups_by_duration(source_groups, ratios, source_seed)
        for split_name, split_rows in source_splits.items():
            final_splits[split_name].extend(split_rows)

    split_dir = preprocessed_dir / "splits"
    for split_name, split_rows in final_splits.items():
        split_rows = sorted(split_rows, key=lambda row: row["id"])
        _write_jsonl(split_dir / f"{split_name}.jsonl", split_rows)

    summary = {
        "total_clips": len(rows),
        "total_hours": round(sum(float(row.get("duration_s") or 0.0) for row in rows) / 3600.0, 6),
        "ratios": ratios,
        "seed": seed,
        "splits": {},
    }
    for split_name, split_rows in final_splits.items():
        by_source = defaultdict(lambda: {"clips": 0, "seconds": 0.0})
        for row in split_rows:
            src = _source_name(row)
            by_source[src]["clips"] += 1
            by_source[src]["seconds"] += float(row.get("duration_s") or 0.0)
        summary["splits"][split_name] = {
            "clips": len(split_rows),
            "hours": round(sum(float(row.get("duration_s") or 0.0) for row in split_rows) / 3600.0, 6),
            "by_source": {
                src: {"clips": data["clips"], "minutes": round(data["seconds"] / 60.0, 4)}
                for src, data in sorted(by_source.items())
            },
        }
    (split_dir / "split_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "DGX full-data pipeline: download available Saudi/UAE audio datasets, "
            "preprocess aligned ASR clips, and write train/validation/test splits."
        )
    )
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--preprocessed-dir", type=Path, default=None,
                        help="Defaults to <work-dir>/preprocessed_audios.")
    parser.add_argument("--train-ratio", type=float, default=0.90)
    parser.add_argument("--validation-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-duration-s", type=float, default=1.0)
    parser.add_argument("--max-duration-s", type=float, default=30.0)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-preprocess", action="store_true")
    parser.add_argument("--confirm-full-download", action="store_true",
                        help="Required unless --skip-download is used. Full SADA is hundreds of GB.")
    args = parser.parse_args()

    work_dir = args.work_dir.resolve()
    preprocessed_dir = (args.preprocessed_dir or (work_dir / "preprocessed_audios")).resolve()

    if not args.skip_download and not args.confirm_full_download:
        print(
            "Refusing to start full downloads without --confirm-full-download. "
            "This will download the full available SADA/WorldSpeech/Nexdata audio corpora.",
            file=sys.stderr,
        )
        return 2

    work_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_download:
        dirs = _dataset_dirs(work_dir)
        manifests = [dirs[name] / "manifest.jsonl" for name in AVAILABLE_AUDIO_DATASETS]
    else:
        manifests = download_available_audio_datasets(work_dir)

    if not args.skip_preprocess:
        if preprocessed_dir.exists():
            shutil.rmtree(preprocessed_dir)
        preprocess(manifests, preprocessed_dir, min_duration=args.min_duration_s, max_duration=args.max_duration_s)

    summary = write_train_validation_test_splits(
        preprocessed_dir,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"preprocessed: {preprocessed_dir}")
    print(f"splits      : {preprocessed_dir / 'splits'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())