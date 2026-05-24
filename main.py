"""CLI entry point for the medical transcript correction pipeline."""

from __future__ import annotations

import argparse
import json

from app.pipeline.runner import run_pipeline
import importlib


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct a medical transcript")
    parser.add_argument("--transcript", required=True, help="Wrong transcript string")
    parser.add_argument("--no-interactive", action="store_true", help="Skip HITL prompts")
    args = parser.parse_args()
    try:
        result = run_pipeline(args.transcript, interactive=not args.no_interactive)
    except RuntimeError as exc:
        msg = str(exc)
        if "Unable to load local BART checkpoint" in msg:
            print("[warning] BART checkpoint not available — using lightweight fallback scorer for CLI run.")
            scorer_mod = importlib.import_module("app.pipeline.scorer")
            from app.pipeline.models import ScoredWord

            def _fallback_score_transcript(transcript: str):
                toks = scorer_mod.tokenize_transcript(transcript)
                scored = []
                for i, (t, s, e) in enumerate(toks):
                    in_lex = scorer_mod._lexicon_entry(t)
                    is_stop = scorer_mod._is_stop_word(t)
                    suspicion = 0.0 if is_stop or in_lex else 0.8
                    scored.append(ScoredWord(index=i, text=t, suspicion=suspicion, in_lexicon=in_lex, start=s, end=e))
                return scored

            scorer_mod.score_transcript = _fallback_score_transcript
            # runner.py imported score_transcript at module import time; patch it too.
            runner_mod = importlib.import_module("app.pipeline.runner")
            setattr(runner_mod, "score_transcript", _fallback_score_transcript)
            result = run_pipeline(args.transcript, interactive=not args.no_interactive)
        else:
            raise
    print(result.corrected_text)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()