from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "dataset_samples"


@dataclass(frozen=True)
class DatasetJob:
    slug: str
    script: str
    dialect: str
    hours: Optional[float]
    source: str
    notes: str = ""


DATASETS: List[DatasetJob] = [
    DatasetJob(
        slug="sada2022",
        script="download_sada2022_samples.py",
        dialect="saudi",
        hours=668.0,
        source="kaggle:sdaiancai/sada2022",
        notes="Kaggle archive can be very large; skipped unless --allow-full-kaggle-archives is used.",
    ),
    DatasetJob(
        slug="saudilang_scc",
        script="download_saudilang_scc_samples.py",
        dialect="saudi_code_switch",
        hours=4.45,
        source="hf:SDAIANCAI/Saudilang-Code-Switch-Corpus + YouTube segments",
        notes="HF annotations are small; YouTube audio is skipped unless --download-saudilang-audio is used.",
    ),
    DatasetJob(
        slug="worldspeech_saudi",
        script="download_worldspeech_saudi_samples.py",
        dialect="saudi",
        hours=6.1,
        source="hf:disco-eth/WorldSpeech config ar_sa",
        notes="May require HF login and accepted terms.",
    ),
    DatasetJob(
        slug="uae_bilingual",
        script="download_uae_bilingual_samples.py",
        dialect="emirati_uae",
        hours=120.0,
        source="hf:vadimbelsky/UAE_Arabic_English_Bilingual_Dataset_40k",
        notes="May require HF login and accepted terms. Hours are estimated.",
    ),
    DatasetJob(
        slug="nexdata_uae_sample",
        script="download_nexdata_uae_sample.py",
        dialect="emirati_uae",
        hours=1.0,
        source="hf:Nexdata/UAE_Arabic_Spontaneous_Speech_Data",
        notes="Free sample is under 1 hour; counted as 1.0 in percentage upper-bound math.",
    ),
]


def _load_manifest(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _distribution() -> List[Dict]:
    total = sum(job.hours or 0.0 for job in DATASETS)
    rows = []
    for job in DATASETS:
        hours = job.hours or 0.0
        rows.append({
            "dataset": job.slug,
            "dialect": job.dialect,
            "planned_hours": hours,
            "planned_share_pct": round((hours / total) * 100.0, 2) if total else 0.0,
            "source": job.source,
            "notes": job.notes,
        })
    return rows


def _run_job(job: DatasetJob, args: argparse.Namespace) -> Dict:
    script_path = PROJECT_ROOT / "scripts" / job.script
    out_dir = args.out_root / job.slug
    cmd = [
        sys.executable,
        str(script_path),
        "--limit", str(args.limit),
        "--out", str(out_dir),
    ]
    if args.split:
        cmd += ["--split", args.split]
    if job.slug == "sada2022" and args.allow_full_kaggle_archives:
        cmd.append("--allow-full-archive-download")
    if job.slug == "saudilang_scc" and args.download_saudilang_audio:
        cmd.append("--download-audio")
    print("\n==> " + job.slug)
    print("$ " + " ".join(cmd))
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    rows = _load_manifest(out_dir / "manifest.jsonl")
    return {
        "dataset": job.slug,
        "dialect": job.dialect,
        "script": job.script,
        "out_dir": str(out_dir),
        "exit_code": result.returncode,
        "samples": len(rows),
        "sample_duration_s": round(sum(float(row.get("duration_s") or 0.0) for row in rows), 3),
        "planned_hours": job.hours,
        "source": job.source,
        "notes": job.notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download 10-sample previews for all Saudi/UAE target datasets.")
    parser.add_argument("--limit", type=int, default=10, help="Samples per dataset.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--only", action="append", choices=[job.slug for job in DATASETS],
                        help="Run only one dataset slug. Can be passed multiple times.")
    parser.add_argument("--split", default=None, help="Optional split override passed to all scripts.")
    parser.add_argument("--continue-on-error", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-full-kaggle-archives", action="store_true",
                        help="Allow Kaggle scripts to download full archives before sampling. Can be very large.")
    parser.add_argument("--download-saudilang-audio", action="store_true",
                        help="Use yt-dlp to download/cut Saudilang YouTube audio. Default writes metadata only.")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    selected = [job for job in DATASETS if not args.only or job.slug in set(args.only)]
    summaries: List[Dict] = []

    for job in selected:
        summary = _run_job(job, args)
        summaries.append(summary)
        if summary["exit_code"] != 0 and not args.continue_on_error:
            break

    combined: List[Dict] = []
    for summary in summaries:
        manifest_path = Path(summary["out_dir"]) / "manifest.jsonl"
        for row in _load_manifest(manifest_path):
            row = dict(row)
            row["dataset"] = summary["dataset"]
            row["dialect"] = summary["dialect"]
            row["dataset_dir"] = summary["out_dir"]
            combined.append(row)

    _write_jsonl(args.out_root / "combined_manifest.jsonl", combined)
    payload = {
        "limit_per_dataset": args.limit,
        "out_root": str(args.out_root),
        "datasets": summaries,
        "distribution": _distribution(),
        "combined_samples": len(combined),
        "combined_sample_duration_s": round(sum(float(row.get("duration_s") or 0.0) for row in combined), 3),
    }
    (args.out_root / "download_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nWrote:")
    print(f"  {args.out_root / 'download_summary.json'}")
    print(f"  {args.out_root / 'combined_manifest.jsonl'}")
    failed = [s for s in summaries if s["exit_code"] != 0]
    if failed:
        print("\nSome datasets failed or were unavailable:")
        for item in failed:
            print(f"  - {item['dataset']} exit={item['exit_code']} out={item['out_dir']}")
        return 1 if not args.continue_on_error else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())