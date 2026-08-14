import importlib.util
import os
import sys

import pytest

from src.app.config import Config
from src.parsing.paged_tab import _playhead_times, _time_from_playhead
from src.vision.paged import MeasureSpan, Page


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "timing_truth", os.path.join(ROOT, "scripts", "timing_truth.py"))
timing_truth = importlib.util.module_from_spec(_spec)
sys.modules["timing_truth"] = timing_truth
_spec.loader.exec_module(timing_truth)


def _scan(heads, fps=10.0):
    return {"heads": heads, "fps": fps}


def _page(n):
    return Page(index=0, first_frame=0, last_frame=n - 1, t0=0.0, t1=n / 10.0)


def test_a_resting_cursor_is_dated_by_when_it_arrived():
    """The cursor sits on a pixel for several frames. np.interp needs its x values
    strictly increasing, and fed duplicates it reads off the last — dating every
    note to when the cursor left it rather than arrived."""
    config = Config()
    # ten positions, each held five frames
    heads = [x for x in range(10, 110, 10) for _ in range(5)]
    samples = _playhead_times(_scan(heads), _page(len(heads)), config)
    assert samples[:3] == [(10, 0.0), (20, 0.5), (30, 1.0)]
    assert len(samples) == 10
    xs = [x for x, _ in samples]
    assert xs == sorted(set(xs))             # strictly increasing, as interp requires


def test_finer_sampling_does_not_push_notes_later():
    """The bug grew with the sample rate: more frames per pixel, later times."""
    config = Config()
    positions = list(range(10, 110, 10))
    coarse_heads = [x for x in positions for _ in range(2)]
    fine_heads = [x for x in positions for _ in range(8)]
    coarse = _playhead_times(_scan(coarse_heads, fps=4.0), _page(len(coarse_heads)), config)
    fine = _playhead_times(_scan(fine_heads, fps=16.0), _page(len(fine_heads)), config)
    assert _time_from_playhead(20, coarse) == pytest.approx(
        _time_from_playhead(20, fine), abs=1e-6)


def test_a_page_the_cursor_barely_crossed_is_not_trusted():
    """Forty samples all at one x are no evidence at all, so the threshold counts
    distinct cursor positions rather than frames."""
    config = Config()
    assert _playhead_times(_scan([5] * 40), _page(40), config) is None


def test_merged_bars_are_split_back_apart():
    """A tracked measure at twice the median is two bars the tracker ran together;
    left merged, every note in it is measured against twice its true span."""
    bars = [MeasureSpan(0, 1, 0.0, 2.0), MeasureSpan(0, 1, 2.0, 4.0),
            MeasureSpan(0, 1, 4.0, 8.0), MeasureSpan(0, 1, 8.0, 10.0)]
    spans = timing_truth._bar_spans(bars)
    assert len(spans) == 5
    assert spans[2] == (4.0, 6.0) and spans[3] == (6.0, 8.0)


def test_the_grid_is_searched_not_assumed():
    """Six notes to a bar scored against a grid of eight look random however
    perfect the timing."""
    bars = [MeasureSpan(0, 1, 0.0, 3.0)]
    sixes = [i * 0.5 for i in range(6)]
    score, divisions = timing_truth.metrical(sixes, bars)
    assert divisions in (3, 6)
    assert score < 0.02


def test_perfect_timing_scores_near_zero_and_noise_scores_near_random():
    bars = [MeasureSpan(0, 1, 0.0, 4.0)]
    on_beat, _ = timing_truth.metrical([0.0, 1.0, 2.0, 3.0], bars)
    assert on_beat < 0.01
    scattered, _ = timing_truth.metrical([0.13, 0.61, 1.37, 2.11, 2.83, 3.44], bars)
    assert scattered > on_beat


def test_matching_pairs_notes_by_identity_not_by_order():
    """A dropped note must be reported as unmatched, not shift every later pairing
    and turn one fault into a whole clip of apparent timing error."""
    from src.models.schema import Note
    truth = [{"time": 1.0, "string": 3, "fret": 1},
             {"time": 2.0, "string": 3, "fret": 5},
             {"time": 3.0, "string": 3, "fret": 7}]
    read = [Note(1.02, 3, 1, 1.0), Note(3.05, 3, 7, 1.0)]   # the middle one is missing
    errors, unmatched = timing_truth._match(truth, read)
    assert unmatched == 1
    assert len(errors) == 2
    assert max(errors) < 0.06


def test_a_note_too_far_away_is_not_claimed_as_a_match():
    from src.models.schema import Note
    truth = [{"time": 1.0, "string": 0, "fret": 3}]
    read = [Note(1.0 + timing_truth.MATCH_WINDOW_S + 0.5, 0, 3, 1.0)]
    errors, unmatched = timing_truth._match(truth, read)
    assert errors == [] and unmatched == 1
