"""Run the pipeline quickly using a heuristic scorer (no GPU or LLM calls).

This script monkeypatches the expensive BART scorer with a simple
heuristic so we can get deterministic results for the canonical test
transcript during development.
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

# Make workspace importable when running script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.runner import run_pipeline


def _install_heuristic_scorer():
    scorer_mod = importlib.import_module("app.pipeline.scorer")

    from app.pipeline.models import ScoredWord

    def _fallback_score_transcript(transcript: str):
        toks = scorer_mod.tokenize_transcript(transcript)
        scored = []
        for i, (t, s, e) in enumerate(toks):
            in_lex = scorer_mod._lexicon_entry(t)
            is_stop = scorer_mod._is_stop_word(t)
            if is_stop:
                suspicion = 0.0
            else:
                # Heuristic: token not in lexicon -> suspicious 0.85; in lexicon -> 0.10
                suspicion = 0.10 if in_lex else 0.85
            scored.append(ScoredWord(index=i, text=t, suspicion=suspicion, in_lexicon=in_lex, start=s, end=e))
        return scored

    # Patch the scorer in the module and also in runner if already imported
    import app.pipeline.runner as runner_mod
    scorer_mod.score_transcript = _fallback_score_transcript
    setattr(runner_mod, "score_transcript", _fallback_score_transcript)


def main():
    _install_heuristic_scorer()
    transcript = (
        "The patient presents with fever and should take dolly prahn twice daily "
        "alongside salbu tamol for the wheeze. Blood pressure was measured using a sfigmomanometre. "
        "The attending physician prescribed amoxicilin for the secondary infection."
    )
    result = run_pipeline(transcript, interactive=False)
    print(result.corrected_text)
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
