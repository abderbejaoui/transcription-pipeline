"""Tests for app/services/config.py — pipeline configuration."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from app.services import config


class TestPipelineConfigDefaults:
    """Verify default values of the PipelineConfig dataclass."""

    def test_default_use_llm_corrector(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.use_llm_corrector is True

    def test_default_model_name(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.llm_model_name == "Qwen/Qwen2.5-1.5B-Instruct"

    def test_default_confidence_threshold(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.llm_confidence_threshold == 0.85

    def test_default_vector_lexicon_enabled(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.vector_lexicon_enabled is True

    def test_default_vector_backend(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.vector_backend == "ngram"

    def test_default_fallback_to_rules(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.fallback_to_rules is True

    def test_default_use_reduced_filler(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.use_reduced_filler is True

    def test_default_feedback_data_path(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.feedback_data_path == "data/user_corrections.jsonl"

    def test_default_use_api_fallback(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.use_api_fallback is True

    def test_default_api_model(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.api_model == "qwen/qwen-2.5-72b-instruct"

    def test_default_use_perplexity_scorer(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.use_perplexity_scorer is True

    def test_default_verbose_false(self) -> None:
        cfg = config.PipelineConfig()
        assert cfg.verbose is False


class TestLoadConfigFromEnv:
    """load_config() must respect environment variable overrides."""

    def test_env_disables_llm_corrector(self) -> None:
        with patch.dict(os.environ, {"USE_LLM_CORRECTOR": "false"}, clear=True):
            cfg = config.load_config()
            assert cfg.use_llm_corrector is False

    def test_env_disables_llm_corrector_zero(self) -> None:
        with patch.dict(os.environ, {"USE_LLM_CORRECTOR": "0"}, clear=True):
            cfg = config.load_config()
            assert cfg.use_llm_corrector is False

    def test_env_enabled_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            cfg = config.load_config()
            assert cfg.use_llm_corrector is True

    def test_env_overrides_model_name(self) -> None:
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "test/model"}, clear=True):
            cfg = config.load_config()
            assert cfg.llm_model_name == "test/model"

    def test_env_overrides_confidence_threshold(self) -> None:
        with patch.dict(os.environ, {"LLM_CONFIDENCE_THRESHOLD": "0.5"}, clear=True):
            cfg = config.load_config()
            assert cfg.llm_confidence_threshold == 0.5

    def test_env_disables_api_fallback(self) -> None:
        with patch.dict(os.environ, {"USE_API_FALLBACK": "false"}, clear=True):
            cfg = config.load_config()
            assert cfg.use_api_fallback is False

    def test_env_disables_vector_lexicon(self) -> None:
        with patch.dict(os.environ, {"VECTOR_LEXICON_ENABLED": "false"}, clear=True):
            cfg = config.load_config()
            assert cfg.vector_lexicon_enabled is False

    def test_env_overrides_vector_backend(self) -> None:
        with patch.dict(os.environ, {"VECTOR_BACKEND": "transformer"}, clear=True):
            cfg = config.load_config()
            assert cfg.vector_backend == "transformer"

    def test_env_disables_fallback_to_rules(self) -> None:
        with patch.dict(os.environ, {"FALLBACK_TO_RULES": "false"}, clear=True):
            cfg = config.load_config()
            assert cfg.fallback_to_rules is False

    def test_env_disables_reduced_filler(self) -> None:
        with patch.dict(os.environ, {"USE_REDUCED_FILLER": "false"}, clear=True):
            cfg = config.load_config()
            assert cfg.use_reduced_filler is False

    def test_env_case_insensitive(self) -> None:
        with patch.dict(os.environ, {"USE_LLM_CORRECTOR": "FALSE"}, clear=True):
            cfg = config.load_config()
            assert cfg.use_llm_corrector is False


class TestGetConfigSingleton:
    """get_config() returns a singleton, reloadable via load_config()."""

    def test_returns_same_instance(self) -> None:
        config._CONFIG = None  # reset
        a = config.get_config()
        b = config.get_config()
        assert a is b

    def test_load_config_creates_fresh(self) -> None:
        config._CONFIG = None
        a = config.load_config()
        b = config.load_config()
        # load_config creates new instances each call
        assert a is not b
        assert a.use_llm_corrector == b.use_llm_corrector

    def test_get_config_after_load_config(self) -> None:
        config._CONFIG = None
        # load_config creates a fresh instance (does NOT set _CONFIG)
        fresh = config.load_config()
        # get_config calls load_config() internally and caches it
        sing = config.get_config()
        # get_config creates its own instance — different from fresh
        assert sing is not fresh
        # But calling get_config twice returns the same instance
        sing2 = config.get_config()
        assert sing is sing2

    def test_singleton_holds_values(self) -> None:
        config._CONFIG = None
        with patch.dict(os.environ, {"LLM_MODEL_NAME": "custom/model"}, clear=True):
            cfg = config.get_config()
            assert cfg.llm_model_name == "custom/model"
