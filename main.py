"""CLI entry point for the medical transcript correction pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from app.pipeline.runner import run_pipeline


# -- Load .env ---------------------------------------------------------

ENV_PATH = Path(__file__).resolve().parent / ".env"


def load_env() -> None:
    if not ENV_PATH.exists():
        return
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and v:
            os.environ.setdefault(k, v)


def main() -> None:
    load_env()
    parser = argparse.ArgumentParser(description="Correct a medical transcript")
    parser.add_argument("--transcript", required=True, help="Wrong transcript string")
    parser.add_argument("--no-interactive", action="store_true", help="Skip HITL prompts")
    args = parser.parse_args()
    result = run_pipeline(args.transcript, interactive=not args.no_interactive)
    print(result.corrected_text)
    print()
    print("--- Provider Summary ---")
    for stage_name, info in result.report.get("approaches", {}).items():
        mode = info.get("mode", "?")
        status = info.get("status", "?")
        label = info.get("label", "?")
        print(f"  [Stage] {stage_name}: {mode} ({status}) — {label}")
    print()
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()