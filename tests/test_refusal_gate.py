"""The gate that refuses a video this reader cannot time.

Thresholds come from measurement, not taste: the four labelled clips hold 5 to
43 distinct highlight positions and travel several times the highlight's own
width, while a warm background matching the same mask holds one position and
travels none. These tests pin both ends of that gap.
"""
from src.app.config import Config
from src.vision import paged


def _scan(spans):
    return {"spans": spans}


def test_a_highlight_that_steps_between_measures_reads():
    # Eight measures across the page, each held for a few frames.
    spans = []
    for measure in range(8):
        spans.extend([(100 * measure, 100 * measure + 80)] * 5)
    assert paged.highlight_diagnosis(_scan(spans), Config()) is None


def test_no_highlight_at_all_is_refused():
    reason = paged.highlight_diagnosis(_scan([None] * 500), Config())
    assert reason is not None
    assert "no measure highlight" in reason


def test_a_highlight_seen_in_a_handful_of_frames_is_refused():
    spans = [None] * 1000 + [(10, 90)] * 3
    reason = paged.highlight_diagnosis(_scan(spans), Config())
    assert reason is not None
    assert "no measure highlight" in reason


def test_a_warm_background_matching_every_frame_is_refused():
    # What a cream wall behind the player's hands looks like: the mask fires on
    # every frame, from the same place, for the whole video.
    reason = paged.highlight_diagnosis(_scan([(0, 1400)] * 1200), Config())
    assert reason is not None
    assert "never moves" in reason


def test_a_highlight_that_barely_shifts_is_refused():
    # Jitter around a fixed position must not read as stepping between measures.
    spans = [(200 + (i % 3), 280 + (i % 3)) for i in range(600)]
    reason = paged.highlight_diagnosis(_scan(spans), Config())
    assert reason is not None
    assert "never moves" in reason


def test_the_refusal_says_which_of_the_two_faults_it_was():
    absent = paged.highlight_diagnosis(_scan([None] * 500), Config())
    static = paged.highlight_diagnosis(_scan([(0, 1400)] * 500), Config())
    assert absent != static, "the two faults need different fixes and must read differently"
