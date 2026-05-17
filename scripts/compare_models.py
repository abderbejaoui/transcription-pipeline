"""Side-by-side bake-off of multi-dialectal Arabic ASR models on a single
test set, using the Open Universal Arabic ASR Leaderboard normalizer.

This is the canonical Arabic-ASR comparison harness. It:

  1. Runs `scripts.bakeoff` once per requested model on `--eval-dir`
     (skipping clips that already have a cached prediction).
  2. Re-scores all cached predictions with `scripts.eval_arabic.py`
     (Wang et al. 2024 normalizer, jiwer corpus-level WER + CER).
  3. Prints a single leaderboard table side-by-side.

Default model list reflects the public Arabic ASR landscape today:
  - omniASR        Meta omnilingual-asr LLM-7B (top of Open Universal LB)
  - qwen3          Qwen3-ASR-1.7B base (Apache-2.0, strong + fast)
  - qwen3_uae      vadimbelsky/qwen3-asr-arabic-uae (Gulf fine-tune)
  - voxtral_mini   mistralai/Voxtral-Mini-3B-2507
  - vibevoice      microsoft/VibeVoice-ASR (code-switch friendly)
  - whisper        openai/whisper-large-v3-turbo (familiarity baseline)

Usage:
  # On DGX (one full run, ~30-60 min for 813 Casablanca UAE clips):
  python -m scripts.build_casablanca_testset --dialect UAE
  python -m scripts.compare_models --eval-dir eval/casablanca_UAE

  # Smoke test (30 clips, ~10 min):
  python -m scripts.build_casablanca_testset --dialect UAE --max-clips 30
  python -m scripts.compare_models --eval-dir eval/casablanca_UAE \\
      --models qwen3 qwen3_uae omniASR voxtral_mini

  # Score-only (re-aggregate without re-running inference):
  python -m scripts.compare_models --eval-dir eval/casablanca_UAE --score-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Default 4-model comparison. omniASR is heavy, so we put qwen3 first
# (fast model first → fail fast if something is wrong).
DEFAULT_MODELS: List[str] = [
    "qwen3",
    "qwen3_uae",
    "omniASR",
    "voxtral_mini",
    "vibevoice",
]


def run_bakeoff(eval_dir: Path, models: List[str], skip_existing: bool) -> int:
    """Spawn `python -m scripts.bakeoff` with --eval-dir + --models.
    Skips clips that already have a cached prediction (so re-running this
    script is cheap after a partial run)."""
    cmd = [
        sys.executable, "-m", "scripts.bakeoff",
        "--eval-dir", str(eval_dir),
        "--models", *models,
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(f"\n$ {' '.join(cmd)}", flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    print(f"  bake-off finished in {(time.time()-t0)/60:.1f} min "
          f"(rc={proc.returncode})", flush=True)
    return proc.returncode


def run_scoring(eval_dir: Path) -> int:
    """Spawn `python scripts/eval_arabic.py` against the same test set."""
    cmd = [
        sys.executable, "scripts/eval_arabic.py",
        "--testset", str(eval_dir),
    ]
    print(f"\n$ {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--eval-dir", type=Path, required=True,
        help="Test set directory (must contain manifest.jsonl). "
             "E.g. eval/casablanca_UAE")
    ap.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help=f"bakeoff.py model keys to compare. Default: {DEFAULT_MODELS}")
    ap.add_argument(
        "--no-skip-existing", action="store_true",
        help="Re-run inference even for clips that already have predictions.")
    ap.add_argument(
        "--score-only", action="store_true",
        help="Skip inference entirely; only re-score cached predictions.")
    args = ap.parse_args()

    eval_dir = args.eval_dir if args.eval_dir.is_absolute() \
        else (PROJECT_ROOT / args.eval_dir).resolve()
    if not (eval_dir / "manifest.jsonl").exists():
        print(f"!! no manifest.jsonl in {eval_dir}", file=sys.stderr)
        print(f"   Build one first: e.g. python -m scripts.build_casablanca_testset --dialect UAE",
              file=sys.stderr)
        return 2

    print(f"Eval dir : {eval_dir.relative_to(PROJECT_ROOT)}")
    print(f"Models   : {', '.join(args.models)}")

    if not args.score_only:
        rc = run_bakeoff(eval_dir, args.models,
                         skip_existing=not args.no_skip_existing)
        if rc != 0:
            print(f"!! bake-off failed with rc={rc}", file=sys.stderr)
            return rc

    return run_scoring(eval_dir)


if __name__ == "__main__":
    sys.exit(main())
