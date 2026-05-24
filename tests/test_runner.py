from app.pipeline import runner
from app.pipeline.models import ScoredWord
from app.services import lexicon


def test_full_pipeline_corrects_two_spans(tmp_path, monkeypatch):
    path = tmp_path / "medical_lexicon.jsonl"
    lexicon.add_term("Doliprane", type_="drug_brand", aliases=[], path=path)
    lexicon.add_term("Salbutamol", type_="drug_generic", aliases=[], path=path)
    monkeypatch.setattr(lexicon, "DEFAULT_LEXICON_PATH", path)
    monkeypatch.setattr("app.pipeline.retriever.text_to_ipa", lambda text: text.lower().replace(" ", ""))

    transcript = "The patient should take dolly prahn twice daily alongside salbu tamol."
    scored = [
        ScoredWord(0, "The", 0.0, True, 0, 3),
        ScoredWord(1, "patient", 0.0, True, 4, 11),
        ScoredWord(2, "should", 0.0, True, 12, 18),
        ScoredWord(3, "take", 0.0, True, 19, 23),
        ScoredWord(4, "dolly", 0.9, False, 24, 29),
        ScoredWord(5, "prahn", 0.92, False, 30, 35),
        ScoredWord(6, "twice", 0.0, True, 36, 41),
        ScoredWord(7, "daily", 0.0, True, 42, 47),
        ScoredWord(8, "alongside", 0.0, True, 48, 57),
        ScoredWord(9, "salbu", 0.88, False, 58, 63),
        ScoredWord(10, "tamol", 0.9, False, 64, 69),
    ]
    monkeypatch.setattr(runner, "score_transcript", lambda text: scored)
    monkeypatch.setattr(runner, "decide_spans", lambda sentence, items: [
        runner.Decision(span=items[0].span, chosen="Doliprane", confidence=0.95, path="llm"),
        runner.Decision(span=items[1].span, chosen="Salbutamol", confidence=0.90, path="auto_fix"),
    ])

    result = runner.run_pipeline(transcript, interactive=False)
    assert "Doliprane" in result.corrected_text
    assert "Salbutamol" in result.corrected_text