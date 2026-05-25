import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print(f"LLM_PROVIDER: {os.environ.get('LLM_PROVIDER', 'not_set')}")
print(f"GEMINI_MODEL: {os.environ.get('GEMINI_MODEL', 'not_set')}")
api_key = os.environ.get('GEMINI_API_KEY', '')
print(f"GEMINI_API_KEY: SET ({len(api_key)} chars)" if api_key else "GEMINI_API_KEY: NOT SET")
print()

from app.services.llm import llm_decide, NO_CHANGE
from app.pipeline.models import Candidate

sentence = "Patient with myokardial infarction"
span = "myokardial"
candidates = [
    Candidate(term="myocardial", ipa="", description="relating to the heart muscle", phonetic_score=0.85, source="lexicon", term_type="medical", match_type="phonetic"),
    Candidate(term="myocardium", ipa="", description="heart muscle tissue", phonetic_score=0.60, source="lexicon", term_type="medical", match_type="phonetic"),
]

print("=== Testing llm_decide with GEMINI ===")
print()

try:
    result = llm_decide(sentence, span, candidates)
    print(f"Raw result: {repr(result)}")
    print(f"NO_CHANGE constant: {repr(NO_CHANGE)}")
    if result == NO_CHANGE:
        print("RESULT: LLM returned NO_CHANGE (Gemini didn't pick a candidate)")
    elif result:
        print(f"RESULT: Gemini chose: {result}")
    else:
        print("RESULT: Empty result")
except Exception as e:
    print(f"ERROR: {type(e).__name__}: {e}")
