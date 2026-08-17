"""The sheet metric's note-to-bar assignment.

Worth testing on its own: if a note on a bar line were claimed by both bars it
would score as a hit twice while the bar it left still read as empty, which is
the exact failure the metric was built to expose.
"""
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from measure_truth import SHEET_EDGE_TOLERANCE_S, notes_by_measure


@dataclass
class _Note:
    time: float
    string_index: int
    fret: int


MEASURES = [
    {"index": 0, "t0": 0.0, "t1": 1.0, "notes": []},
    {"index": 1, "t0": 1.0, "t1": 2.0, "notes": []},
]


def test_a_note_inside_a_bar_belongs_to_it():
    out = notes_by_measure([_Note(0.5, 3, 1)], MEASURES)
    assert out[0] == [(3, 1)]
    assert out[1] == []


def test_a_note_on_the_bar_line_is_claimed_once():
    out = notes_by_measure([_Note(1.0, 2, 4)], MEASURES)
    claimed = out[0] + out[1]
    assert claimed == [(2, 4)], "a seam note must land in exactly one bar"


def test_a_note_just_early_is_pulled_back_not_duplicated():
    out = notes_by_measure([_Note(1.0 - SHEET_EDGE_TOLERANCE_S / 2, 1, 2)], MEASURES)
    assert out[0] + out[1] == [(1, 2)]


def test_a_note_a_whole_bar_out_stays_in_the_wrong_bar():
    # The metric must not quietly rescue this: printing a note in the next bar
    # is the fault it exists to count.
    out = notes_by_measure([_Note(1.5, 3, 1)], MEASURES)
    assert out[0] == []
    assert out[1] == [(3, 1)]


def test_a_note_outside_every_bar_is_ignored():
    out = notes_by_measure([_Note(9.0, 3, 1)], MEASURES)
    assert out[0] == [] and out[1] == []


def test_notes_keep_their_order_within_a_bar():
    notes = [_Note(0.8, 0, 2), _Note(0.2, 3, 1), _Note(0.5, 1, 0)]
    assert notes_by_measure(notes, MEASURES)[0] == [(3, 1), (1, 0), (0, 2)]
