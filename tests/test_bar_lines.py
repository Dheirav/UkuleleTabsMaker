"""Bar lines for a sheet timed by its soundtrack.

The highlight used to supply bar times, and a video timed from audio has no
highlight. Without them a whole song parses as one measure and prints as a tab
with no divisions in it.
"""
import cv2
import numpy as np
import pytest

from src.app.config import Config
from src.parsing.audio_timing import bar_times_from_x
from src.vision.page_digits import find_bar_lines

WIDTH, HEIGHT = 1200, 400
STAFF = [150, 180, 210, 240]


class Digit:
    def __init__(self, x):
        self.x_center = x


def page(draw=()):
    """A tab staff on white, plus whatever else the caller wants drawn."""
    img = np.full((HEIGHT, WIDTH, 3), 245, np.uint8)
    for y in STAFF:
        img[y, :] = 60
    for item in draw:
        item(img)
    return img


def bar_line(x, colour=(40, 40, 40), top=STAFF[0], bottom=STAFF[-1]):
    def draw(img):
        cv2.line(img, (x, top), (x, bottom), colour, 2)
    return draw


def test_bar_lines_are_found_where_they_are_drawn():
    config = Config()
    img = page([bar_line(400), bar_line(700), bar_line(1000)])
    assert find_bar_lines(img, STAFF, config) == pytest.approx([400, 700, 1000], abs=3)


def test_a_staff_with_no_bar_lines_yields_none():
    assert find_bar_lines(page(), STAFF, Config()) == []


def test_a_rhythm_stem_below_the_staff_is_not_a_bar_line():
    """These pages hang rhythm stems under the tab. A stem is easily as tall as a
    bar line and clips the bottom line on its way down; counting those gave one
    video 36 bars of under half a second."""
    config = Config()
    stem = bar_line(400, top=STAFF[-1], bottom=STAFF[-1] + 70)
    assert find_bar_lines(page([stem]), STAFF, config) == []


def test_a_coloured_playback_marker_is_not_a_bar_line():
    """One player steps a saturated orange marker through the music, exactly as
    tall and exactly as vertical as the bar line beside it."""
    config = Config()
    orange = bar_line(600, colour=(30, 120, 240))     # BGR: saturated orange
    assert find_bar_lines(page([orange]), STAFF, config) == []


def test_the_edges_of_the_system_are_not_bars_within_it():
    """A staff is closed off at both ends by a rule of its own, and the page
    already begins where it begins."""
    config = Config()
    img = page([bar_line(4), bar_line(600), bar_line(WIDTH - 5)])
    assert find_bar_lines(img, STAFF, config) == pytest.approx([600], abs=3)


def test_a_bar_line_must_span_the_staff_not_merely_cross_it():
    config = Config()
    half = bar_line(500, top=STAFF[0], bottom=STAFF[1])
    assert find_bar_lines(page([half]), STAFF, config) == []


def test_without_staff_lines_nothing_is_claimed():
    assert find_bar_lines(page(), [], Config()) == []


# --- turning positions into times ------------------------------------------

def test_a_bar_line_is_dated_by_the_notes_around_it():
    placed = [(Digit(100), 10.0), (Digit(300), 12.0)]
    assert bar_times_from_x([200], placed, 9.0, 13.0) == pytest.approx([11.0])


def test_a_bar_line_past_the_last_note_is_pinned_to_the_page():
    """A system's closing bar sits beyond every note in it, and a speed guessed
    from two notes would throw it a long way out."""
    placed = [(Digit(100), 10.0), (Digit(300), 12.0)]
    assert bar_times_from_x([1800], placed, 9.0, 13.0) == pytest.approx([13.0])
    assert bar_times_from_x([5], placed, 9.0, 13.0) == pytest.approx([9.0])


def test_no_notes_means_no_bar_times():
    assert bar_times_from_x([200], [], 9.0, 13.0) == []
