"""Stage 5: human-in-the-loop correction and logging."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from app.services import lexicon
from app.services.phonetics import text_to_ipa


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HITL_LOG_PATH = PROJECT_ROOT / "data" / "hitl_log.jsonl"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def log_hitl(wrong_form: str, correct_term: str, sentence_context: str, session_id: Optional[str] = None) -> None:
    HITL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "timestamp": _now_iso(),
        "wrong_form": wrong_form,
        "correct_term": correct_term,
        "sentence_context": sentence_context,
        "session_id": session_id,
    }
    with HITL_LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def apply_human_correction(
    transcript: str,
    span_text: str,
    correct_term: str,
    *,
    term_type: str = "unknown",
    description: str = "",
    session_id: Optional[str] = None,
) -> Tuple[str, lexicon.LexiconEntry]:
    if not correct_term.strip():
        raise ValueError("correct_term must be non-empty")
    existing = lexicon.find_by_alias(correct_term)
    if existing is not None:
        lexicon.add_entry(
            lexicon.LexiconEntry(
                term=existing.term,
                canonical_form=existing.canonical_form,
                term_type=existing.term_type,
                aliases=(span_text,),
                ipa=existing.ipa,
                description=existing.description,
                source=existing.source,
                added_at=existing.added_at,
            )
        )
        log_hitl(span_text, existing.term, transcript, session_id=session_id)
        return transcript.replace(span_text, existing.term, 1), existing

    entry = lexicon.LexiconEntry(
        term=correct_term.strip(),
        canonical_form=correct_term.strip().lower(),
        term_type=term_type,
        aliases=(span_text,),
        ipa=text_to_ipa(correct_term),
        description=description,
        source="user",
        added_at=_now_iso(),
    )
    saved = lexicon.add_entry(entry)
    log_hitl(span_text, saved.term, transcript, session_id=session_id)
    return transcript.replace(span_text, saved.term, 1), saved


def prompt_for_human_correction(sentence: str, span_text: str, best_guess: Optional[str] = None) -> Optional[str]:
    print("[NEEDS YOUR INPUT]")
    print(f'Sentence  : "{sentence}"')
    print(f'Flagged   : "{span_text}"')
    if best_guess:
        print(f"Best guess: {best_guess}")
    response = input("Enter the correct term (or press Enter to leave unchanged): ").strip()
    return response or None


def suggestion_to_lexicon(
    wrong_form: str,
    correct_term: str,
    *,
    term_type: str = "unknown",
    description: str = "",
    session_id: Optional[str] = None,
) -> lexicon.LexiconEntry:
    """Save a human correction suggestion to the lexicon.

    Adds ``wrong_form`` as an alias for ``correct_term``.
    This is the lightweight version used by the HITL review UI.
    """
    if not correct_term.strip():
        raise ValueError("correct_term must be non-empty")

    # Check if the correct term already exists
    existing = lexicon.find_by_alias(correct_term)
    if existing is not None:
        # Just add the wrong_form as a new alias
        lexicon.add_entry(
            lexicon.LexiconEntry(
                term=existing.term,
                canonical_form=existing.canonical_form,
                term_type=existing.term_type,
                aliases=tuple(set(list(existing.aliases) + [wrong_form])),
                ipa=existing.ipa,
                description=existing.description,
                source=existing.source,
                added_at=existing.added_at,
            )
        )
        log_hitl(wrong_form, existing.term, "(HITL review)", session_id=session_id)
        return existing

    # New term — create it
    from app.services.phonetics import text_to_ipa
    entry = lexicon.LexiconEntry(
        term=correct_term.strip(),
        canonical_form=correct_term.strip().lower(),
        term_type=term_type,
        aliases=(wrong_form,),
        ipa=text_to_ipa(correct_term),
        description=description,
        source="user",
        added_at=_now_iso(),
    )
    saved = lexicon.add_entry(entry)
    log_hitl(wrong_form, saved.term, "(HITL review)", session_id=session_id)
    return saved