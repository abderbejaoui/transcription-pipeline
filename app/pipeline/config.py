"""Named constants for the correction pipeline."""

SUSPICION_THRESHOLD = 0.5
TOP_K = 5
USER_AUTO_FIX_THRESHOLD = 0.90
LLM_MIN_CONFIDENCE = 0.60
PHONETIC_FLAG_THRESHOLD = 0.55  # phonetic similarity (0-1) to lexicon to trigger flagging

# -- Scorer model selection ---------------------------------------------

SCORER_MODEL = "qwen"
"""Which contextual language model to use for Stage 1 scoring.

- ``"qwen"`` (default, recommended): Qwen2.5-1.5B-Instruct — causal LM
  log-probability scoring with a medical-reviewer system prompt prefix.
  Single forward pass, ~1.5B params on GPU.
- ``"bart"``: facebook/bart-large — fill-mask MLM, 406M params, one
  forward pass per flagged word.

Fallback chain: Qwen → BART → heuristic. Setting this to ``"qwen"``
will try Qwen first and fall back to BART if Qwen is unavailable.

To switch::

    import app.pipeline.config as cfg
    cfg.SCORER_MODEL = "bart"  # or "qwen" (default)
"""