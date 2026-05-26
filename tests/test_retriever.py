from app.pipeline.models import SuspiciousSpan
from app.pipeline.retriever import retrieve_candidates
from app.services import lexicon
from app.services.phonetics import text_to_ipa


def test_exact_alias_match_returns_score_one(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=["dolly prahn"], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    monkeypatch.setattr("app.pipeline.retriever.text_to_ipa", lambda text: (_ for _ in ()).throw(AssertionError("phonetics should not run")))
    span = SuspiciousSpan(start=0, end=1, text="dolly prahn", suspicion=0.9, reason="both")
    item = retrieve_candidates(span)
    assert item.candidates[0].term == "Doliprane"
    assert item.candidates[0].phonetic_score == 1.0


# ── Open-world spell-check (Path C) tests ───────────────────────────


def test_spell_correct_finds_dehydration(tmp_path, monkeypatch):
    """Path C must find 'dehydration' from misspelling 'dehidration'."""
    path = tmp_path / "medical_lexicon.jsonl"
    # Seed an unrelated term so the lexicon is non-empty but doesn't contain
    # "dehydration". This forces Path C to fire.
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    from app.pipeline.retriever import _spell_correct
    candidates = _spell_correct("dehidration")
    terms = [c.lower() for c in candidates]
    assert "dehydration" in terms, f"Expected 'dehydration' in {terms}"


def test_spell_correct_finds_vomiting(tmp_path, monkeypatch):
    """Path C must find 'vomiting' from misspelling 'vommiting'."""
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    from app.pipeline.retriever import _spell_correct
    candidates = _spell_correct("vommiting")
    terms = [c.lower() for c in candidates]
    assert "vomiting" in terms, f"Expected 'vomiting' in {terms}"


def test_spell_correct_does_not_fire_on_correct_word(tmp_path, monkeypatch):
    """Path C must return empty for correctly-spelled 'nebulization'."""
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    from app.pipeline.retriever import _spell_correct
    candidates = _spell_correct("nebulization")
    assert len(candidates) == 0, f"Expected empty, got {candidates}"


# ── Integration test: Path C fires in retrieve_candidates ───────────────

def test_retrieve_candidates_fires_spell_correct_when_paths_a_and_b_fail(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    # Monkeypatch IPA to always return empty string so Path B produces no candidates > 0.60
    monkeypatch.setattr("app.pipeline.retriever.text_to_ipa", lambda text: "")
    monkeypatch.setattr("app.pipeline.retriever.fallback_text_to_ipa", lambda text: "")
    span = SuspiciousSpan(start=0, end=0, text="dehidration", suspicion=0.9, reason="both")
    result = retrieve_candidates(span)
    spell_correct_hits = [c for c in result.candidates if c.match_type == "spell_correct"]
    terms = [c.term.lower() for c in spell_correct_hits]
    assert "dehydration" in terms, f"Expected 'dehydration' in spell_correct candidates, got {terms}"


def test_dolly_prahn_is_ranked_within_top_three(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    lexicon.add_term("Diprivan", type_="drug_brand", aliases=[], path=path)
    lexicon.add_term("Paracetamol", type_="drug_generic", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)

    mapping = {
        "dolly prahn": "dɒlipreɪn",
        "Doliprane": "dɒlipreɪn",
        "Diprivan": "dɪprɪvən",
        "Paracetamol": "pærəsɛtəmɒl",
    }
    monkeypatch.setattr("app.pipeline.retriever.text_to_ipa", lambda text: mapping.get(text, text.lower()))
    span = SuspiciousSpan(start=0, end=1, text="dolly prahn", suspicion=0.9, reason="both")
    item = retrieve_candidates(span)
    assert item.candidates[0].term == "Doliprane"
    assert any(candidate.term == "Doliprane" for candidate in item.candidates[:3])