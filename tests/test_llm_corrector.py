"""Tests for app/services/llm_corrector.py — JSON parsing, prompt building, result class.

These tests do NOT load the local model (no GPU required). They verify:
  - _parse_json_response: correct JSON, markdown fences, malformed output
  - SYSTEM_PROMPT and USER_PROMPT_TEMPLATE are well-formed
  - LLMCorrectorResult construction and defaults
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from app.services import llm_corrector


# ===========================================================================
# _parse_json_response
# ===========================================================================


class TestParseJsonResponse:
    """_parse_json_response extracts JSON from LLM output, handling various formats."""

    def test_plain_json(self) -> None:
        raw = '{"corrected": "test", "corrections": [], "confidence": 0.95}'
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "test"
        assert result["confidence"] == 0.95

    def test_markdown_fence_json(self) -> None:
        raw = "```json\n{\"corrected\": \"patient history\", \"corrections\": [], \"confidence\": 0.9}\n```"
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "patient history"

    def test_markdown_fence_no_lang(self) -> None:
        raw = "```\n{\"corrected\": \"result\", \"corrections\": [], \"confidence\": 1.0}\n```"
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "result"

    def test_json_with_leading_text(self) -> None:
        raw = "Here is the corrected transcript:\n\n{\"corrected\": \"final\", \"corrections\": [], \"confidence\": 0.88}"
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "final"

    def test_json_with_trailing_text(self) -> None:
        raw = '{"corrected": "test", "corrections": [], "confidence": 0.9}\n\nHope this helps!'
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "test"

    def test_empty_string_returns_none(self) -> None:
        result = llm_corrector._parse_json_response("")
        assert result is None

    def test_non_json_returns_none(self) -> None:
        result = llm_corrector._parse_json_response("This is not JSON at all")
        assert result is None

    def test_malformed_json_returns_none(self) -> None:
        result = llm_corrector._parse_json_response('{"corrected": "test", broken}')
        assert result is None

    def test_json_with_arabic_text(self) -> None:
        raw = json.dumps({
            "corrected": "Patient has history of diabetes",
            "corrections": [
                {"original": "هستوري", "corrected": "history", "type": "transliteration"}
            ],
            "confidence": 0.95,
        }, ensure_ascii=False)
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["corrected"] == "Patient has history of diabetes"
        assert len(result["corrections"]) == 1
        assert result["corrections"][0]["original"] == "هستوري"

    def test_multiple_corrections(self) -> None:
        raw = json.dumps({
            "corrected": "Patient has diabetes and hypertension",
            "corrections": [
                {"original": "دايابيتس", "corrected": "diabetes", "type": "transliteration"},
                {"original": "هايبرتنشن", "corrected": "hypertension", "type": "transliteration"},
            ],
            "confidence": 0.93,
        }, ensure_ascii=False)
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert len(result["corrections"]) == 2

    def test_realistic_output(self) -> None:
        """Simulate what the LLM might actually return for a Gulf Arabic transcript."""
        raw = """{
            "corrected": "The patient has a history of diabetes and hypertension.",
            "corrections": [
                {"original": "هستوري", "corrected": "history", "type": "transliteration"},
                {"original": "دايابيتس", "corrected": "diabetes", "type": "transliteration"},
                {"original": "هايبرتنشن", "corrected": "hypertension", "type": "transliteration"}
            ],
            "confidence": 0.94
        }"""
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert "history" in result["corrected"]
        assert "diabetes" in result["corrected"]

    def test_confidence_zero(self) -> None:
        raw = '{"corrected": "no change", "corrections": [], "confidence": 0.0}'
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["confidence"] == 0.0

    def test_confidence_one(self) -> None:
        raw = '{"corrected": "same", "corrections": [], "confidence": 1.0}'
        result = llm_corrector._parse_json_response(raw)
        assert result is not None
        assert result["confidence"] == 1.0


# ===========================================================================
# Prompt templates
# ===========================================================================


class TestPromptTemplates:
    """SYSTEM_PROMPT and USER_PROMPT_TEMPLATE should be well-formed."""

    def test_system_prompt_has_correction_rules(self) -> None:
        assert "transliteration" in llm_corrector.SYSTEM_PROMPT.lower()
        assert "corrected" in llm_corrector.SYSTEM_PROMPT.lower()

    def test_system_prompt_has_arabic_examples(self) -> None:
        assert "هستوري" in llm_corrector.SYSTEM_PROMPT
        assert "دايابيتس" in llm_corrector.SYSTEM_PROMPT
        assert "بلاد شوجر" in llm_corrector.SYSTEM_PROMPT

    def test_system_prompt_has_json_format(self) -> None:
        assert "corrected" in llm_corrector.SYSTEM_PROMPT
        assert "corrections" in llm_corrector.SYSTEM_PROMPT
        assert "confidence" in llm_corrector.SYSTEM_PROMPT

    def test_user_prompt_formats_correctly(self) -> None:
        result = llm_corrector.USER_PROMPT_TEMPLATE.format(
            transcript="Patient has هستوري of دايابيتس"
        )
        assert "Patient has هستوري of دايابيتس" in result


# ===========================================================================
# LLMCorrectorResult
# ===========================================================================


class TestLLMCorrectorResult:
    """LLMCorrectorResult dataclass construction and defaults."""

    def test_default_empty(self) -> None:
        result = llm_corrector.LLMCorrectorResult()
        assert result.corrected_text == ""
        assert result.corrections == []
        assert result.confidence == 0.0
        assert result.source is None
        assert result.success is False

    def test_success_result(self) -> None:
        result = llm_corrector.LLMCorrectorResult(
            corrected_text="Patient has diabetes",
            corrections=[{"original": "دايابيتس", "corrected": "diabetes", "type": "transliteration"}],
            confidence=0.95,
            source="local",
            success=True,
        )
        assert result.corrected_text == "Patient has diabetes"
        assert len(result.corrections) == 1
        assert result.confidence == 0.95
        assert result.source == "local"
        assert result.success is True

    def test_empty_transcript(self) -> None:
        result = llm_corrector.correct_transcript("")
        assert result.success is False
        assert result.corrected_text == ""

    def test_whitespace_transcript(self) -> None:
        result = llm_corrector.correct_transcript("   ")
        assert result.success is False
        assert result.corrected_text == "   "

    def test_config_disabled_returns_immediately(self) -> None:
        """When use_llm_corrector is False, correct_transcript returns with success=False.

        This test patches get_config() to return a config with use_llm_corrector=False
        and verifies the function short-circuits without calling any model code.
        """
        from app.services import config as cfg_mod
        disabled_cfg = cfg_mod.PipelineConfig(use_llm_corrector=False)
        with patch.object(cfg_mod, "get_config", return_value=disabled_cfg):
            with patch.object(llm_corrector, "_local_correct") as mock_local:
                with patch.object(llm_corrector, "_api_correct") as mock_api:
                    result = llm_corrector.correct_transcript(
                        "Patient has هستوري", use_api_fallback=True
                    )
                    assert result.success is False
                    assert result.corrected_text == "Patient has هستوري"
                    # Neither local nor API should be called
                    mock_local.assert_not_called()
                    mock_api.assert_not_called()


# ===========================================================================
# Edge cases for correct_transcript (without loading actual model)
# ===========================================================================


class TestCorrectTranscriptEdgeCases:
    """correct_transcript edge cases — tested by mocking _local_correct."""

    def test_local_correct_success(self) -> None:
        """When _local_correct succeeds, the result should be used."""
        mock_result = {
            "corrected": "Patient has diabetes",
            "corrections": [{"original": "دايابيتس", "corrected": "diabetes", "type": "transliteration"}],
            "confidence": 0.95,
        }
        with patch.object(llm_corrector, "_local_correct", return_value=mock_result):
            result = llm_corrector.correct_transcript("Patient has دايابيتس", use_api_fallback=False)
            assert result.success is True
            assert result.source == "local"
            assert result.corrected_text == "Patient has diabetes"
            assert result.confidence == 0.95
            assert len(result.corrections) == 1

    def test_local_fails_api_fallback_disabled(self) -> None:
        """When local fails and API fallback is disabled, return failure."""
        with patch.object(llm_corrector, "_local_correct", return_value=None):
            result = llm_corrector.correct_transcript(
                "Patient has هستوري", use_api_fallback=False
            )
            assert result.success is False
            assert result.source is None
            assert result.corrected_text == "Patient has هستوري"  # unchanged

    def test_local_fails_api_fallback_tried(self) -> None:
        """When local fails and API is enabled, try API."""
        mock_api = {
            "corrected": "Patient has history",
            "corrections": [{"original": "هستوري", "corrected": "history", "type": "transliteration"}],
            "confidence": 0.90,
        }
        with patch.object(llm_corrector, "_local_correct", return_value=None):
            with patch.object(llm_corrector, "_api_correct", return_value=mock_api):
                result = llm_corrector.correct_transcript(
                    "Patient has هستوري", use_api_fallback=True
                )
                assert result.success is True
                assert result.source == "api"
                assert result.corrected_text == "Patient has history"

    def test_both_fail_return_unchanged(self) -> None:
        """When both local and API fail, return unchanged text with success=False."""
        with patch.object(llm_corrector, "_local_correct", return_value=None):
            with patch.object(llm_corrector, "_api_correct", return_value=None):
                result = llm_corrector.correct_transcript(
                    "Patient has هستوري", use_api_fallback=True
                )
                assert result.success is False
                assert result.corrected_text == "Patient has هستوري"

    def test_empty_corrections_list(self) -> None:
        """When LLM returns empty corrections list, still use the corrected text."""
        mock_result = {
            "corrected": "Patient is stable",
            "corrections": [],
            "confidence": 0.99,
        }
        with patch.object(llm_corrector, "_local_correct", return_value=mock_result):
            result = llm_corrector.correct_transcript("Patient is stable", use_api_fallback=False)
            assert result.success is True
            assert result.corrected_text == "Patient is stable"
            assert result.corrections == []

    def test_missing_corrected_field(self) -> None:
        """When LLM returns a JSON without 'corrected', fall back to original text."""
        mock_result = {"corrections": [], "confidence": 0.5}
        with patch.object(llm_corrector, "_local_correct", return_value=mock_result):
            result = llm_corrector.correct_transcript("Patient has هستوري", use_api_fallback=False)
            # success=True because the mock returned something, but corrected_text
            # falls back to original since "corrected" is missing
            assert result.corrected_text == "Patient has هستوري"

    def test_non_list_corrections(self) -> None:
        """When corrections is not a list, treat as empty list."""
        mock_result = {
            "corrected": "Patient has history",
            "corrections": "invalid",
            "confidence": 0.9,
        }
        with patch.object(llm_corrector, "_local_correct", return_value=mock_result):
            result = llm_corrector.correct_transcript("Patient has هستوري", use_api_fallback=False)
            assert result.success is True
            assert result.corrections == []


# ===========================================================================
# _JSON_RE pattern
# ===========================================================================


class TestJsonRegex:
    """_JSON_RE regex pattern must handle various fence formats."""

    def test_no_fence(self) -> None:
        text = '{"key": "value"}'
        m = llm_corrector._JSON_RE.search(text)
        assert m is not None
        assert m.group(1) == '{"key": "value"}'

    def test_triple_backtick_json(self) -> None:
        text = "```json\n{\"key\": \"value\"}\n```"
        m = llm_corrector._JSON_RE.search(text)
        assert m is not None
        assert '"key"' in m.group(1)

    def test_triple_backtick_plain(self) -> None:
        text = "```\n{\"key\": \"value\"}\n```"
        m = llm_corrector._JSON_RE.search(text)
        assert m is not None

    def test_nested_braces(self) -> None:
        text = '{"outer": {"inner": "value"}}'
        m = llm_corrector._JSON_RE.search(text)
        assert m is not None
        assert "outer" in m.group(1)
        assert "inner" in m.group(1)
