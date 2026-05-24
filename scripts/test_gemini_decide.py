"""Test Gemini DECIDE on the four canonical spans using GEMINI_API_KEY from .env.

This script loads .env, builds minimal candidate lists, calls
app.services.llm.llm_decide and prints the model's raw choice.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import json


def load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        k, v = line.split('=', 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        os.environ.setdefault(k, v)


class C:
    def __init__(self, term, term_type='drug', description='', phonetic_score=0.0):
        self.term = term
        self.term_type = term_type
        self.description = description
        self.phonetic_score = phonetic_score


def main():
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / '.env')

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Force the newer model requested in DESIRED_PIPELINE.md unless the env
    # explicitly overrides it.
    os.environ.setdefault('GEMINI_MODEL', 'gemini-2.0-flash')

    from app.services.llm import llm_decide, _build_prompt, _post_json, _extract_text, _parse_choice, _candidate_lookup, _normalize_choice, NO_CHANGE

    # Candidate lists matching DESIRED_PIPELINE.md expectations
    tests = [
        ("The patient ... take dolly prahn twice daily ...", "dolly prahn", [
            C("Doliprane", term_type="drug_brand", description="Paracetamol brand", phonetic_score=0.82),
            C("Diprivan", term_type="drug_brand", phonetic_score=0.48),
            C("Paracetamol", term_type="drug_generic", phonetic_score=0.31),
        ]),
        ("... wheeze ... salbu tamol ...", "salbu tamol", [
            C("Salbutamol", term_type="drug_generic", phonetic_score=0.89),
            C("Salmeterol", term_type="drug_generic", phonetic_score=0.61),
            C("Paracetamol", term_type="drug_generic", phonetic_score=0.29),
        ]),
        ("Blood pressure was measured using a sfigmomanometre.", "sfigmomanometre", [
            C("sphygmomanometer", term_type="device", phonetic_score=0.74),
            C("dynamometer", term_type="device", phonetic_score=0.41),
        ]),
        ("... prescribed amoxicilin for the secondary infection.", "amoxicilin", [
            C("amoxicillin", term_type="drug_generic", phonetic_score=0.94),
            C("ampicillin", term_type="drug_generic", phonetic_score=0.67),
        ]),
    ]

    results = []
    for sentence, span, candidates in tests:
        prompt = _build_prompt(sentence, span, candidates)
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": (
                            "You are a constrained medical transcript correction reranker. "
                            "You must choose exactly one provided candidate term string or NO_CHANGE. "
                            "Do not invent new terms."
                        )
                    }
                ]
            },
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "topP": 1.0,
                "maxOutputTokens": 64,
                "responseMimeType": "application/json",
            },
        }
        print("\n=== SPAN ===")
        print(span)
        print("--- PROMPT ---")
        print(prompt)
        try:
            raw = _post_json(payload, timeout=30)
            print("--- RAW JSON ---")
            print(json.dumps(raw, indent=2, ensure_ascii=False))
            extracted = _extract_text(raw)
            print("--- EXTRACTED TEXT ---")
            print(extracted)
            parsed = _parse_choice(extracted)
            print("--- PARSED CHOICE ---")
            print(parsed)
            lookup = _candidate_lookup(candidates)
            normalized = _normalize_choice(parsed)
            if normalized == _normalize_choice(NO_CHANGE):
                final = NO_CHANGE
            else:
                final = lookup.get(normalized, NO_CHANGE)
            print("--- FINAL VALIDATED ---")
            print(final)
            results.append({"span": span, "choice": final})
        except Exception as exc:
            print("--- ERROR ---")
            print(type(exc).__name__, str(exc))
            # Try to surface the response body if this is an HTTPError
            try:
                import urllib.error
                if isinstance(exc, urllib.error.HTTPError):
                    body = exc.read().decode('utf-8', errors='replace')
                    print("--- ERROR BODY ---")
                    print(body)
            except Exception:
                pass
            results.append({"span": span, "choice": None, "error": type(exc).__name__})

    print('\nSummary:')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
