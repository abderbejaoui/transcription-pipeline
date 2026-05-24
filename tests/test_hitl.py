from app.pipeline.hitl import apply_human_correction
from app.services import lexicon


def test_new_term_is_saved_and_reloadable(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    corrected, entry = apply_human_correction("The patient should take dolly prahn.", "dolly prahn", "Doliprane", term_type="drug_brand")
    assert "Doliprane" in corrected
    entries = lexicon.load_lexicon(path)
    assert any(item.term == "Doliprane" for item in entries)
    assert entry.term == "Doliprane"


def test_existing_term_appends_alias_without_duplicate_entry(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=["dolly prahn"], path=path)
    apply_human_correction("The patient should take dolly prahn.", "dolly prahn", "Doliprane", term_type="drug_brand")
    entries = lexicon.load_lexicon(path)
    assert len(entries) == 1
    assert "dolly prahn" in {alias.lower() for alias in entries[0].aliases}