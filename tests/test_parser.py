from src.models.schema import Note
from src.parsing.parser import parse_measures


def test_parse_measures_with_bars():
    notes = [
        Note(time=0.1, string_index=0, fret=1, confidence=0.9),
        Note(time=0.6, string_index=1, fret=2, confidence=0.9),
        Note(time=1.2, string_index=2, fret=3, confidence=0.9),
    ]
    bar_times = [1.0]

    measures = parse_measures(notes, bar_times)

    assert len(measures) == 2
    assert measures[0].start_time == 0.1
    assert measures[0].end_time == 1.0
    assert len(measures[0].notes) == 2
    assert measures[1].start_time == 1.0
    assert measures[1].end_time == 1.2
    assert len(measures[1].notes) == 1


def test_parse_measures_empty_notes():
    assert parse_measures([], [0.5, 1.0]) == []
