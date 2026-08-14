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


def _bar(n, fps=10.0, x0=0, x1=1000):
    """A measure spanning n samples, wide enough that any cursor sweep covers it."""
    return MeasureSpan(x0=x0, x1=x1, t0=0.0, t1=n / fps)


def test_a_resting_cursor_is_dated_by_when_it_arrived():
    """The cursor sits on a pixel for several frames. np.interp needs its x values
    strictly increasing, and fed duplicates it reads off the last — dating every
    note to when the cursor left it rather than arrived."""
    config = Config()
    # ten positions, each held five frames
    heads = [x for x in range(10, 110, 10) for _ in range(5)]
    # bar edges sit exactly on the cursor's travel, so no anchors are added
    samples = _playhead_times(_scan(heads), _bar(len(heads), x0=10, x1=100), config)
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
    coarse = _playhead_times(_scan(coarse_heads, fps=4.0), _bar(len(coarse_heads), fps=4.0, x1=110), config)
    fine = _playhead_times(_scan(fine_heads, fps=16.0), _bar(len(fine_heads), fps=16.0, x1=110), config)
    assert _time_from_playhead(20, coarse) == pytest.approx(
        _time_from_playhead(20, fine), abs=1e-6)


def test_a_page_the_cursor_barely_crossed_is_not_trusted():
    """Forty samples all at one x are no evidence at all, so the threshold counts
    distinct cursor positions rather than frames."""
    config = Config()
    assert _playhead_times(_scan([5] * 40), _bar(40, x1=10), config) is None


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


def test_sparse_cursor_evidence_is_still_used():
    """Timing a page from a handful of cursor positions beats falling back to
    where notes sit in the bar — measured at 16ms against 61ms on a clip whose
    pages turn too fast to gather many. The gate must not be stricter than the
    evidence warrants."""
    config = Config()
    heads = [x for x in range(10, 70, 10) for _ in range(3)]   # six positions
    assert _playhead_times(_scan(heads), _bar(len(heads), x0=10, x1=70), config) is not None


def test_a_bar_the_cursor_barely_crossed_is_timed_by_position_instead():
    """Half a bar of cursor evidence cannot speak for the whole bar. Timing the
    covered notes from the cursor and the rest from their position mixed two
    models inside one bar and put the notes out of order with each other."""
    config = Config()
    heads = [x for x in range(10, 70, 10) for _ in range(3)]
    # the cursor crossed 60px of a 400px bar
    assert _playhead_times(_scan(heads), _bar(len(heads), x0=0, x1=400), config) is None


def test_two_sweeps_in_one_window_cannot_fold_the_times():
    """A page holding two bars holds two cursor sweeps. Sorting those by x
    interleaves them, and np.interp handed times that fall as x rises returns
    nonsense — which put notes far apart in the bar at nearly the same instant."""
    config = Config()
    sweep = [x for x in range(100, 500, 20)]
    heads = sweep + sweep                    # the cursor crosses, snaps back, crosses again
    # scoped to one bar's own window, only the first sweep is in view
    bar = MeasureSpan(x0=100, x1=480, t0=0.0, t1=len(sweep) / 10.0)
    samples = _playhead_times(_scan(heads), bar, config)
    times = [t for _, t in samples]
    assert times == sorted(times)            # rises with x, as interp requires


def test_notes_past_the_cursor_do_not_pile_onto_one_instant():
    """np.interp pins anything beyond its last sample to that sample's time, so
    every note past where the cursor was seen came out at the same instant —
    distinct notes reading as one. The bar's own edges extend the mapping."""
    config = Config()
    heads = [x for x in range(100, 300, 10) for _ in range(2)]   # cursor stops at 290
    bar = MeasureSpan(x0=100, x1=700, t0=0.0, t1=4.05)
    samples = _playhead_times(_scan(heads), bar, config)
    late = [_time_from_playhead(x, samples) for x in (400, 550, 700)]
    assert late == sorted(late)
    assert len(set(late)) == 3          # three notes, three different instants


def test_time_past_the_cursor_carries_on_at_the_cursor_s_own_pace():
    """The bar's far edge cannot anchor the mapping: the highlight often leaves
    before the cursor reaches it, which crams the tail of the bar into whatever
    milliseconds remain and compresses notes half a bar apart onto one beat."""
    config = Config()
    heads = [x for x in range(100, 300, 10) for _ in range(2)]   # 100..290 over 4s
    bar = MeasureSpan(x0=40, x1=800, t0=0.0, t1=4.05)            # ends just after
    samples = _playhead_times(_scan(heads), bar, config)
    speed = (290 - 100) / (samples[-1][1] - samples[0][1])
    beyond = _time_from_playhead(490, samples)
    assert beyond == pytest.approx(samples[-1][1] + 200 / speed)
    assert beyond > bar.t1        # honest about running past the highlight


def test_notes_far_apart_are_never_squeezed_onto_one_beat():
    """Distinct notes 500px apart must not come out 20ms apart, whether by being
    pinned to the cursor's last sighting or crammed against the bar's edge."""
    config = Config()
    heads = [x for x in range(100, 300, 10) for _ in range(2)]
    bar = MeasureSpan(x0=100, x1=700, t0=0.0, t1=4.05)
    samples = _playhead_times(_scan(heads), bar, config)
    times = [_time_from_playhead(x, samples) for x in (350, 520, 690)]
    assert times == sorted(times)
    assert all(b - a > 0.3 for a, b in zip(times, times[1:]))
