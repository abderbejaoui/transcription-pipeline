"""Central configuration for the correction pipeline.

All feature flags are controllable via environment variables.
Defaults are set for local-first operation with GPU (8GB+ VRAM recommended).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class PipelineConfig:
    # ── LLM corrector (local 4-bit model) ──────────────────────────────
    use_llm_corrector: bool = True
    llm_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    llm_confidence_threshold: float = 0.85
    llm_max_new_tokens: int = 512
    llm_device: str = "auto"  # "auto", "cuda", "cpu"

    # ── API fallback (OpenRouter) ──────────────────────────────────────
    use_api_fallback: bool = True
    api_model: str = "qwen/qwen-2.5-72b-instruct"
    api_timeout: float = 15.0

    # ── Vector lexicon ─────────────────────────────────────────────────
    vector_lexicon_enabled: bool = True
    # Backend: "ngram" (fast, deterministic) or "transformer" (semantic)
    vector_backend: str = "ngram"
    embedding_model_name: str = "distilbert-base-multilingual-cased"
    vector_similarity_threshold: float = 0.15

    # ── Fallback to rule-based ─────────────────────────────────────────
    fallback_to_rules: bool = True
    use_skeleton_matching: bool = True  # keep as fallback even with vector

    # ── Arabic filler / normalcy ───────────────────────────────────────
    use_reduced_filler: bool = True  # use minimal filler set

    # ── HITL / feedback ────────────────────────────────────────────────
    feedback_data_path: str = "data/user_corrections.jsonl"

    # ── LM perplexity ──────────────────────────────────────────────────
    use_perplexity_scorer: bool = True

    # ── Debug ──────────────────────────────────────────────────────────
    verbose: bool = False


def load_config() -> PipelineConfig:
    """Load configuration from environment variables (overrides defaults)."""
    cfg = PipelineConfig()

    # LLM corrector
    if os.environ.get("USE_LLM_CORRECTOR", "").lower() in ("0", "false", "no"):
        cfg.use_llm_corrector = False
    if os.environ.get("LLM_MODEL_NAME"):
        cfg.llm_model_name = os.environ["LLM_MODEL_NAME"]
    if os.environ.get("LLM_CONFIDENCE_THRESHOLD"):
        cfg.llm_confidence_threshold = float(os.environ["LLM_CONFIDENCE_THRESHOLD"])

    # API fallback
    if os.environ.get("USE_API_FALLBACK", "").lower() in ("0", "false", "no"):
        cfg.use_api_fallback = False

    # Vector lexicon
    if os.environ.get("VECTOR_LEXICON_ENABLED", "").lower() in ("0", "false", "no"):
        cfg.vector_lexicon_enabled = False
    if os.environ.get("VECTOR_BACKEND"):
        cfg.vector_backend = os.environ["VECTOR_BACKEND"]

    # Fallback
    if os.environ.get("FALLBACK_TO_RULES", "").lower() in ("0", "false", "no"):
        cfg.fallback_to_rules = False

    # Filler
    if os.environ.get("USE_REDUCED_FILLER", "").lower() in ("0", "false", "no"):
        cfg.use_reduced_filler = False

    return cfg


# Singleton
_CONFIG: PipelineConfig | None = None


def get_config() -> PipelineConfig:
    global _CONFIG
    if _CONFIG is None:
        _CONFIG = load_config()
    return _CONFIG
