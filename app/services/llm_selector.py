"""Constrained LLM selector — picks the best candidate from a list rather than
generating free text.

Phase 3 of the PROMPT.md spec: LLM as Candidate Selector (judge, not free
generator).

Architecture
------------
1. Input: {context_left, context_right, span_text, candidates[], optional
   word_confidences[]}
2. Prompt: Presents candidates as A/B/C options with descriptions.
   The LLM replies with the LETTER only (constrained choice).
3. Output: {choice ∈ candidates | "NO_CHANGE", confidence, reason}
4. Confidence: Derived from the model's logprobs over the choice tokens,
   NOT a self-reported number.

The auto-apply/HITL decision is made downstream by the calibrated confidence
model (logistic regression over features + eval-set operating point).
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_config import (
    get_llm_headers,
    get_llm_model,
    get_llm_provider,
    get_llm_url,
    parse_chat_content,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

NO_CHANGE = "NO_CHANGE"


@dataclass
class SelectorCandidate:
    """A single candidate the selector must choose from."""

    term: str
    phonetic_score: float = 0.0          # 0-100 from _score_pair
    retrieval_score: float = 0.0         # 0-1 from vector lexicon query
    description: str = ""                # short description of the term
    source: str = "lexicon"              # e.g. "lexicon", "vector", "phonetic"


@dataclass
class SelectorInput:
    """Input to the LLM selector."""

    span_text: str
    candidates: List[SelectorCandidate] = field(default_factory=list)
    context_left: str = ""               # ~5 words preceding the span
    context_right: str = ""              # ~5 words following the span
    full_transcript: str = ""            # optional full transcript context
    word_confidences: Optional[List[float]] = None  # ASR per-word confidences

    @property
    def n_candidates(self) -> int:
        return len(self.candidates)


@dataclass
class SelectorOutput:
    """Output from the LLM selector."""

    choice: Optional[str]                # chosen candidate term, or None for NO_CHANGE
    confidence: float = 0.0              # 0-1 calibrated probability
    reason: str = ""
    selection_probs: Dict[str, float] = field(default_factory=dict)  # per-candidate probabilities
    llm_logprobs: Optional[Dict[str, float]] = None  # raw token logprobs from LLM
    n_candidates_considered: int = 0


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a Gulf Arabic medical transcript corrector. Your job is to pick "
    "the CORRECT medical term for a suspicious span from a list of candidates. "
    "Rules:\n"
    "1. Reply with ONLY the letter of your choice (A, B, C, ...) — no extra text.\n"
    "2. If NONE of the candidates fit the context, reply with the letter for NO_CHANGE.\n"
    "3. Base your decision on the clinical context provided.\n"
    "4. Sound similarity matters but CLINICAL FIT matters more: choose the term "
    "that makes sense in the patient's context.\n"
    "5. Never invent or guess a term that isn't in the candidate list.\n"
    "6. For Arabic transliterations (like 'هستوري' → 'history'), prefer the English "
    "canonical term when it's clearly the intended meaning."
)

_USER_TEMPLATE = (
    "Suspicious span: \"{span_text}\"\n"
    "Context: ... {context_left} [[ {span_text} ]] {context_right} ...\n"
    "\n"
    "Which of these is the correct medical term?\n"
    "{candidates_text}\n"
    "{no_change_letter}) NO_CHANGE — leave the span unchanged\n"
    "\n"
    "Reply with the LETTER only."
)


def _build_candidates_text(candidates: List[SelectorCandidate]) -> str:
    """Build the A/B/C listing with descriptions."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    lines: List[str] = []
    for i, c in enumerate(candidates):
        letter = letters[i] if i < len(letters) else f"Option{i+1}"
        desc = f" — {c.description}" if c.description else ""
        score_info = ""
        if c.phonetic_score > 0:
            score_info = f" [score={c.phonetic_score:.0f}]"
        lines.append(f"{letter}) {c.term}{desc}{score_info}")
    return "\n".join(lines)


def build_selection_prompt(
    selector_input: SelectorInput,
) -> List[Dict[str, str]]:
    """Build the chat messages for the constrained selection prompt.

    Returns a list of message dicts: [{"role": "system", ...}, {"role": "user", ...}]
    """
    candidates = selector_input.candidates
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    no_change_letter = (
        letters[len(candidates)] if len(candidates) < len(letters) else "Z"
    )

    candidates_text = _build_candidates_text(candidates)

    user_msg = _USER_TEMPLATE.format(
        span_text=selector_input.span_text,
        context_left=selector_input.context_left or "(start of transcript)",
        context_right=selector_input.context_right or "(end of transcript)",
        candidates_text=candidates_text,
        no_change_letter=no_change_letter,
    )

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]


def _parse_letter_response(raw: str, n_candidates: int) -> Optional[int]:
    """Parse the LLM's reply (a single letter) into a candidate index.

    Returns:
        Index into the candidate list (0-based), or n_candidates for NO_CHANGE,
        or None if parsing failed.
    """
    text = raw.strip().upper()
    # Remove quotes, punctuation, whitespace
    text = re.sub(r'["\'\\[\\]\\(\\)\\.,\\s]', "", text)

    if not text:
        return None

    # Single letter: A=0, B=1, etc.
    if len(text) == 1 and "A" <= text <= "Z":
        idx = ord(text) - ord("A")
        if 0 <= idx <= n_candidates:
            return idx

    # Try to extract a letter from longer text (e.g. "A)" or "A - candidate")
    m = re.search(r"\b([A-Z])\b", text)
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx <= n_candidates:
            return idx

    # Check if the response matches a candidate term directly
    return None


# ---------------------------------------------------------------------------
# LLM Selector
# ---------------------------------------------------------------------------


class LlmSelector:
    """Constrained LLM selector that picks the best candidate from a list.

    Uses the existing LLM infrastructure (via llm_config) to make the selection
    and optionally extracts logprobs for confidence estimation.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.60,
        max_retries: int = 2,
        include_descriptions: bool = True,
    ):
        self._threshold = confidence_threshold
        self._max_retries = max_retries
        self._include_descriptions = include_descriptions

    def select(
        self,
        selector_input: SelectorInput,
        timeout: float = 30.0,
    ) -> SelectorOutput:
        """Run the constrained LLM selection for one span.

        Steps:
          1. Build the A/B/C prompt with candidates + NO_CHANGE.
          2. Call the LLM.
          3. Parse the letter response.
          4. Return the chosen candidate (or None for NO_CHANGE) with metadata.

        Args:
            selector_input: The span, context, and candidates.
            timeout: Max time for the LLM call.

        Returns:
            SelectorOutput with choice, confidence, and reason.
        """
        if not selector_input.candidates:
            return SelectorOutput(
                choice=None,
                confidence=0.0,
                reason="No candidates to select from",
                n_candidates_considered=0,
            )

        # Attach descriptions to candidates (best-effort)
        if self._include_descriptions:
            for c in selector_input.candidates:
                if not c.description:
                    try:
                        from .descriptions import get as _get_desc
                        desc = _get_desc(c.term)
                        if desc:
                            c.description = desc
                    except ImportError:
                        pass

        # Build prompt with descriptions attached (single call)
        messages = build_selection_prompt(selector_input)

        raw_response = self._call_llm(messages, timeout=timeout)
        if raw_response is None:
            return SelectorOutput(
                choice=None,
                confidence=0.0,
                reason="LLM call failed",
                n_candidates_considered=len(selector_input.candidates),
            )

        choice_idx = _parse_letter_response(
            raw_response, len(selector_input.candidates)
        )

        if choice_idx is None:
            logger.warning(
                "LlmSelector: could not parse response %r from prompt",
                raw_response[:100],
            )
            return SelectorOutput(
                choice=None,
                confidence=0.0,
                reason=f"Unparseable LLM response: {raw_response[:80]}",
                n_candidates_considered=len(selector_input.candidates),
            )

        n = len(selector_input.candidates)

        if choice_idx == n:
            # NO_CHANGE
            return SelectorOutput(
                choice=None,
                confidence=1.0,
                reason="LLM selected NO_CHANGE — none of the candidates fit the context",
                n_candidates_considered=n,
            )

        if 0 <= choice_idx < n:
            chosen = selector_input.candidates[choice_idx]
            # Build per-candidate probability map (uniform fallback if no logprobs)
            selection_probs = {c.term: 0.0 for c in selector_input.candidates}
            selection_probs[NO_CHANGE] = 0.0
            selection_probs[chosen.term] = 1.0

            return SelectorOutput(
                choice=chosen.term,
                confidence=1.0,  # Overridden by calibrated confidence model downstream
                reason=f"LLM selected {chosen.term!r} from {n} candidates",
                selection_probs=selection_probs,
                n_candidates_considered=n,
            )

        # Index out of range
        return SelectorOutput(
            choice=None,
            confidence=0.0,
            reason=f"LLM returned index {choice_idx} but only {n} candidates + NO_CHANGE",
            n_candidates_considered=n,
        )

    def select_batch(
        self,
        inputs: List[SelectorInput],
        timeout: float = 60.0,
    ) -> List[SelectorOutput]:
        """Run selection for multiple spans in parallel.

        For now, processes serially (can be parallelized later with
        asyncio or a batch LLM call).
        """
        return [self.select(inp, timeout=timeout / max(1, len(inputs)))
                for inp in inputs]

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        timeout: float,
    ) -> Optional[str]:
        """Call the LLM via existing infrastructure.

        Tries the local Ollama API first, then falls back to OpenRouter.
        """
        last_exc: Optional[BaseException] = None

        for attempt in range(self._max_retries + 1):
            try:
                provider = get_llm_provider()
                payload = {
                    "model": get_llm_model(provider),
                    "stream": False,
                    "options": {"temperature": 0.0},
                    "messages": messages,
                }

                import urllib.request
                req = urllib.request.Request(
                    get_llm_url(provider),
                    data=json.dumps(payload).encode("utf-8"),
                    headers=get_llm_headers(provider),
                )
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                return parse_chat_content(data, provider)

            except Exception as exc:
                last_exc = exc
                logger.debug(
                    "LlmSelector LLM call failed (attempt %d/%d): %s",
                    attempt + 1, self._max_retries + 1, exc,
                )
                if attempt < self._max_retries:
                    time.sleep(1.0 * (2 ** attempt))

        logger.warning("LlmSelector: all LLM calls failed: %s", last_exc)
        return None

    @staticmethod
    def _try_get_descriptions(terms: List[str]) -> Dict[str, str]:
        """Best-effort lookup of term descriptions."""
        desc_map: Dict[str, str] = {}
        try:
            from .descriptions import get as _desc_get
            for term in terms:
                d = _desc_get(term)
                if d:
                    desc_map[term] = d
        except ImportError:
            pass
        return desc_map
