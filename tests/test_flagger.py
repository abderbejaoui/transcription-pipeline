from app.pipeline.flagger import flag_suspicious_spans
from app.pipeline.models import ScoredWord


def test_flagger_merges_adjacent_suspicious_words():
    scored = [
        ScoredWord(index=0, text="dolly", suspicion=0.90, in_lexicon=False, start=0, end=5),
        ScoredWord(index=1, text="prahn", suspicion=0.92, in_lexicon=False, start=6, end=11),
        ScoredWord(index=2, text="twice", suspicion=0.0, in_lexicon=True, start=12, end=17),
    ]

    spans = flag_suspicious_spans(scored)

    assert len(spans) == 1
    assert spans[0].text == "dolly prahn"
    assert spans[0].start == 0
    assert spans[0].end == 1
    assert spans[0].suspicion == 0.92


def test_flagger_does_not_merge_across_function_word():
    """Function words ("and", "with", "of", etc.) are span boundaries.
    Two suspicious words separated by a function word become separate spans."""
    scored = [
        ScoredWord(index=0, text="dolly", suspicion=0.90, in_lexicon=False, start=0, end=5),
        ScoredWord(index=1, text="and", suspicion=0.0, in_lexicon=True, start=6, end=9),
        ScoredWord(index=2, text="prahn", suspicion=0.93, in_lexicon=False, start=10, end=15),
    ]

    spans = flag_suspicious_spans(scored)

    assert len(spans) == 2
    assert spans[0].text == "dolly"
    assert spans[0].start == 0
    assert spans[0].end == 0
    assert spans[0].suspicion == 0.90
    assert spans[1].text == "prahn"
    assert spans[1].start == 2
    assert spans[1].end == 2
    assert spans[1].suspicion == 0.93


def test_flagger_canonical_transcript_spans_are_exact():
    scored = [
        ScoredWord(index=0, text="The", suspicion=0.0, in_lexicon=False, start=0, end=3),
        ScoredWord(index=1, text="patient", suspicion=0.05, in_lexicon=False, start=4, end=11),
        ScoredWord(index=2, text="presents", suspicion=0.08, in_lexicon=False, start=12, end=20),
        ScoredWord(index=3, text="with", suspicion=0.0, in_lexicon=False, start=21, end=25),
        ScoredWord(index=4, text="fever", suspicion=0.06, in_lexicon=True, start=26, end=31),
        ScoredWord(index=5, text="and", suspicion=0.0, in_lexicon=False, start=32, end=35),
        ScoredWord(index=6, text="should", suspicion=0.03, in_lexicon=False, start=36, end=42),
        ScoredWord(index=7, text="take", suspicion=0.04, in_lexicon=False, start=43, end=47),
        ScoredWord(index=8, text="dolly", suspicion=0.87, in_lexicon=False, start=48, end=53),
        ScoredWord(index=9, text="prahn", suspicion=0.92, in_lexicon=False, start=54, end=59),
        ScoredWord(index=10, text="twice", suspicion=0.04, in_lexicon=False, start=60, end=65),
        ScoredWord(index=11, text="daily", suspicion=0.04, in_lexicon=False, start=66, end=71),
        ScoredWord(index=12, text="alongside", suspicion=0.0, in_lexicon=False, start=72, end=81),
        ScoredWord(index=13, text="salbu", suspicion=0.84, in_lexicon=False, start=82, end=87),
        ScoredWord(index=14, text="tamol", suspicion=0.81, in_lexicon=False, start=88, end=93),
        ScoredWord(index=15, text="for", suspicion=0.0, in_lexicon=False, start=94, end=97),
        ScoredWord(index=16, text="the", suspicion=0.0, in_lexicon=False, start=98, end=101),
        ScoredWord(index=17, text="wheeze", suspicion=0.09, in_lexicon=True, start=102, end=108),
        ScoredWord(index=18, text="Blood", suspicion=0.0, in_lexicon=False, start=110, end=115),
        ScoredWord(index=19, text="pressure", suspicion=0.05, in_lexicon=False, start=116, end=124),
        ScoredWord(index=20, text="was", suspicion=0.0, in_lexicon=False, start=125, end=128),
        ScoredWord(index=21, text="measured", suspicion=0.04, in_lexicon=False, start=129, end=137),
        ScoredWord(index=22, text="using", suspicion=0.0, in_lexicon=False, start=138, end=143),
        ScoredWord(index=23, text="a", suspicion=0.0, in_lexicon=False, start=144, end=145),
        ScoredWord(index=24, text="sfigmomanometre", suspicion=0.96, in_lexicon=False, start=146, end=161),
        ScoredWord(index=25, text="The", suspicion=0.0, in_lexicon=False, start=163, end=166),
        ScoredWord(index=26, text="attending", suspicion=0.04, in_lexicon=False, start=167, end=176),
        ScoredWord(index=27, text="physician", suspicion=0.06, in_lexicon=False, start=177, end=186),
        ScoredWord(index=28, text="prescribed", suspicion=0.05, in_lexicon=False, start=187, end=197),
        ScoredWord(index=29, text="amoxicilin", suspicion=0.71, in_lexicon=False, start=198, end=208),
        ScoredWord(index=30, text="for", suspicion=0.0, in_lexicon=False, start=209, end=212),
        ScoredWord(index=31, text="the", suspicion=0.0, in_lexicon=False, start=213, end=216),
        ScoredWord(index=32, text="secondary", suspicion=0.04, in_lexicon=False, start=217, end=226),
        ScoredWord(index=33, text="infection", suspicion=0.07, in_lexicon=True, start=227, end=236),
    ]

    spans = flag_suspicious_spans(scored)

    assert [span.start for span in spans] == [8, 13, 24, 29]
    assert [span.end for span in spans] == [9, 14, 24, 29]
    assert [span.text for span in spans] == ["dolly prahn", "salbu tamol", "sfigmomanometre", "amoxicilin"]
    assert [span.suspicion for span in spans] == [0.92, 0.84, 0.96, 0.71]