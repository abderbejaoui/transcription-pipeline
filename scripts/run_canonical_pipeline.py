"""Run the pipeline for the canonical test transcript with exact Stage 1 scores.

This forces Stage 1 to return the suspicion values specified in
DESIRED_PIPELINE.md so that downstream stages produce the canonical
corrected transcript deterministically.
"""
from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.pipeline.runner import run_pipeline
from app.pipeline.models import ScoredWord


def _canonical_score_transcript(transcript: str):
    # Tokenize using the project's tokenizer to get start/end offsets
    from app.pipeline.scorer import tokenize_transcript

    toks = tokenize_transcript(transcript)
    # Expected suspicion map from DESIRED_PIPELINE.md (defaults for others)
    suspicion_by_index = {
        1: 0.05,
        2: 0.08,
        4: 0.06,
        6: 0.03,
        7: 0.04,
        8: 0.87,
        9: 0.92,
        10: 0.04,
        11: 0.04,
        13: 0.84,
        14: 0.81,
        17: 0.09,
        19: 0.05,
        21: 0.04,
        24: 0.96,
        26: 0.04,
        27: 0.06,
        28: 0.05,
        29: 0.71,
        32: 0.04,
        33: 0.07,
    }

    # Set in_lexicon according to DESIRED: fever, wheeze, infection True; others False
    in_lexicon_indices = {4, 17, 33}

    scored = []
    for i, (token, s, e) in enumerate(toks):
        suspicion = float(suspicion_by_index.get(i, 0.0))
        in_lex = i in in_lexicon_indices
        scored.append(ScoredWord(index=i, text=token, suspicion=suspicion, in_lexicon=in_lex, start=s, end=e))
    return scored


def main():
    transcript = (
        "The patient presents with fever and should take dolly prahn twice daily "
        "alongside salbu tamol for the wheeze. Blood pressure was measured using a sfigmomanometre. "
        "The attending physician prescribed amoxicilin for the secondary infection."
    )
    # Monkeypatch the runner's scorer
    import app.pipeline.runner as runner_mod
    runner_mod.score_transcript = _canonical_score_transcript

    result = run_pipeline(transcript, interactive=False)
    print(result.corrected_text)
    import json
    print(json.dumps(result.report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
