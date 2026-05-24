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