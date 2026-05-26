"""Named constants for the correction pipeline."""

SUSPICION_THRESHOLD = 0.5
TOP_K = 5
USER_AUTO_FIX_THRESHOLD = 0.90
LLM_MIN_CONFIDENCE = 0.60
PHONETIC_FLAG_THRESHOLD = 0.55  # phonetic similarity (0-1) to lexicon to trigger flagging

# -- Scorer model selection ---------------------------------------------

SCORER_MODEL = "modernbert"
"""Which contextual language model to use for Stage 1 scoring.

- ``"modernbert"`` (default, recommended): answerdotai/ModernBERT-large —
  bidirectional encoder fill-mask MLM, 395M params, ~1.5 GB VRAM.
  One forward pass per flagged word.

Fallback chain: ModernBERT → heuristic.

To switch::

    import app.pipeline.config as cfg
    cfg.SCORER_MODEL = "modernbert"  # default
"""