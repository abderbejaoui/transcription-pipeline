"""Seed the medical lexicon with IPA values for entries that are missing them.

This script reads `data/medical_lexicon.jsonl`, computes IPA with the local
`app.services.phonetics.text_to_ipa()` helper for any entry that does not
already have an `ipa` field, and writes the file back in place.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import sys

# Make workspace importable when running script directly
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
import json as _json

from app.services.phonetics import text_to_ipa
try:
    # reuse the Gemini helper for safe API calls
    from app.services import llm as _llm
    HAVE_GEMINI = True
except Exception:
    _llm = None
    HAVE_GEMINI = False

# Keep a stable reference (do not reassign _llm inside function scope – that
# would make Python treat it as a local variable and cause UnboundLocalError).
LLMMODULE = _llm if HAVE_GEMINI else None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEXICON_PATH = PROJECT_ROOT / "data" / "medical_lexicon.jsonl"


def _load_rows(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def seed_lexicon(path: Path = LEXICON_PATH) -> int:
    rows = _load_rows(path)
    updated = 0
    total = len(rows)
    for row in rows:
        ipa = str(row.get("ipa") or "").strip()
        term = str(row.get("term") or "").strip()
        if not term:
            continue
        did_update = False
        llm_mod = LLMMODULE

        # If ipa equals the raw term text, compute IPA and overwrite.
        if ipa.lower() == term.lower():
            # Only attempt IPA if phonemizer is available; text_to_ipa may
            # fallback silently — detect phonemizer presence first.
            try:
                import phonemizer  # type: ignore
                have_phonemizer = True
            except Exception:
                have_phonemizer = False

            if not have_phonemizer:
                print(f"Warning: phonemizer/espeak-ng not available; skipping IPA for: {term}")
            else:
                try:
                    new_ipa = text_to_ipa(term)
                except Exception:
                    print(f"Warning: text_to_ipa failed for: {term}; skipping")
                    new_ipa = ""

                # If text_to_ipa returned something non-empty and different
                # from the raw term, accept it.
                if new_ipa and new_ipa.strip().lower() != term.lower():
                    row["ipa"] = new_ipa
                    did_update = True
                else:
                    print(f"Warning: phonemizer did not produce IPA for: {term}; skipping")

        # Ensure description is present: call Gemini if empty and key available
        desc = str(row.get("description") or "").strip()
        if not desc:
            # If we don't already have a GEMINI llm module reference, try loading
            # .env and importing it locally into `llm_mod` so we don't mutate the
            # module-level _llm (which would make it a local variable).
            if llm_mod is None:
                env_path = Path(__file__).resolve().parents[1] / ".env"
                if env_path.exists():
                    try:
                        print(f"Loading .env from {env_path}")
                        for ln in env_path.read_text(encoding="utf-8").splitlines():
                            if "=" in ln and not ln.strip().startswith("#"):
                                k, v = ln.split("=", 1)
                                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
                        from app.services import llm as _llm_local

                        llm_mod = _llm_local
                        print(".env loaded; GEMINI key present:", bool(os.environ.get("GEMINI_API_KEY")))
                    except Exception as exc:
                        llm_mod = None
                        print("Failed to load .env for GEMINI key:", exc)

            if llm_mod is not None:
                try:
                    # Use the llm helper's gemini_describe convenience function
                    if hasattr(llm_mod, "gemini_describe"):
                        description = llm_mod.gemini_describe(term)
                    else:
                        description = None
                    if description:
                        row["description"] = description
                        did_update = True
                        print(f"Fetched description for {term}: {description}")
                    else:
                        print(f"Warning: empty Gemini description for: {term}")
                except Exception as exc:
                    print(f"Warning: Gemini description fetch failed for: {term}; {exc}")
            else:
                print(f"Warning: GEMINI_API_KEY not available; skipping description for: {term}")

        if did_update:
            updated += 1
            print(f"Updated {updated}/{total}: {term}")

    if updated:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_json.dumps(row, ensure_ascii=False) + "\n")
    return updated


def main() -> None:
    updated = seed_lexicon()
    print(f"Updated {updated} lexicon entries with missing IPA values.")


if __name__ == "__main__":
    main()