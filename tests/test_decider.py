from app.pipeline.decider import decide_span
from app.pipeline.models import Candidate, SpanWithCandidates, SuspiciousSpan


def test_invalid_llm_response_falls_back_to_top_candidate():
    span = SuspiciousSpan(start=0, end=1, text="dolly prahn", suspicion=0.8, reason="both")
    candidates = [Candidate(term="Doliprane", ipa="", term_type="drug_brand", description="", phonetic_score=0.8, source="seed")]
    item = SpanWithCandidates(span=span, candidates=candidates)
    decision = decide_span("The patient should take dolly prahn.", item, llm=lambda s, p, c: "made up")
    # Invalid LLM response falls back to top candidate (score 0.8 >= 0.60 threshold)
    assert decision.chosen == "Doliprane"
    assert decision.path == "top_fallback"
    assert decision.confidence == 0.8


def test_no_change_from_llm_with_below_threshold_candidate_escalates():
    span = SuspiciousSpan(start=0, end=1, text="some weird word", suspicion=0.7, reason="both")
    candidates = [Candidate(term="guess", ipa="", term_type="drug", description="", phonetic_score=0.4, source="seed")]
    item = SpanWithCandidates(span=span, candidates=candidates)
    decision = decide_span("Some weird word in context.", item, llm=lambda s, p, c: "NO_CHANGE")
    # LLM says NO_CHANGE but top candidate is below threshold — escalate
    assert decision.chosen is None
    assert decision.path == "hitl_escalate"


def test_user_source_candidate_auto_fixes_without_llm():
    span = SuspiciousSpan(start=0, end=1, text="salbu tamol", suspicion=0.95, reason="both")
    candidates = [Candidate(term="Salbutamol", ipa="", term_type="drug_generic", description="", phonetic_score=0.90, source="user")]
    item = SpanWithCandidates(span=span, candidates=candidates)
    called = {"count": 0}

    def fake_llm(*args, **kwargs):
        called["count"] += 1
        return "NO_CHANGE"

    decision = decide_span("The patient should take salbu tamol.", item, llm=fake_llm)
    assert decision.chosen == "Salbutamol"
    assert decision.path == "auto_fix"
    assert called["count"] == 0