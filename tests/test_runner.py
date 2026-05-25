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
        ScoredWord(index=0, text="The", suspicion=0.0, in_lexicon=True, start=0, end=3),
        ScoredWord(index=1, text="patient", suspicion=0.0, in_lexicon=True, start=4, end=11),
        ScoredWord(index=2, text="should", suspicion=0.0, in_lexicon=True, start=12, end=18),
        ScoredWord(index=3, text="take", suspicion=0.0, in_lexicon=True, start=19, end=23),
        ScoredWord(index=4, text="dolly", suspicion=0.9, in_lexicon=False, start=24, end=29),
        ScoredWord(index=5, text="prahn", suspicion=0.92, in_lexicon=False, start=30, end=35),
        ScoredWord(index=6, text="twice", suspicion=0.0, in_lexicon=True, start=36, end=41),
        ScoredWord(index=7, text="daily", suspicion=0.0, in_lexicon=True, start=42, end=47),
        ScoredWord(index=8, text="alongside", suspicion=0.0, in_lexicon=True, start=48, end=57),
        ScoredWord(index=9, text="salbu", suspicion=0.88, in_lexicon=False, start=58, end=63),
        ScoredWord(index=10, text="tamol", suspicion=0.9, in_lexicon=False, start=64, end=69),
    ]
    monkeypatch.setattr(runner, "score_transcript", lambda text: scored)
    monkeypatch.setattr(runner, "decide_spans", lambda sentence, items: [
        runner.Decision(span=items[0].span, chosen="Doliprane", confidence=0.95, path="llm"),
        runner.Decision(span=items[1].span, chosen="Salbutamol", confidence=0.90, path="auto_fix"),
    ])

    result = runner.run_pipeline(transcript, interactive=False)
    assert "Doliprane" in result.corrected_text
    assert "Salbutamol" in result.corrected_text