"""Finding the tab when it is drawn over a video of someone playing.

Two thirds of tab videos on YouTube are shaped this way. find_content_rows only
strips letterboxing, so it hands the reader the player as well as the tab, and
the page signature then tracks a pair of moving hands instead of the notation.
"""
import cv2
import numpy as np
import pytest

from src.app.config import Config
from src.vision.paged import find_content_rows, find_overlay_band, find_tab_rows

WIDTH, HEIGHT, FRAMES = 320, 240, 60


def write_video(path, frame_for):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20,
                             (WIDTH, HEIGHT))
    assert writer.isOpened()
    for i in range(FRAMES):
        writer.write(frame_for(i))
    writer.release()
    return str(path)


def paper(rows):
    """A slab of tab: pale, with a dark rule across it so it is not blank."""
    band = np.full((rows, WIDTH, 3), 240, np.uint8)
    band[rows // 2, :] = 20
    return band


def noise(rows, seed):
    """Stands in for the video of a player: every pixel changes every frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, (rows, WIDTH, 3), dtype=np.uint8)


@pytest.fixture
def overlay_top(tmp_path):
    """Tab across the top, player underneath — the common layout."""
    def frame(i):
        return np.vstack([paper(90), noise(HEIGHT - 90, i)])
    return write_video(tmp_path / "overlay_top.mp4", frame)


@pytest.fixture
def overlay_bottom(tmp_path):
    """The same video the other way up. Both layouts occur in the wild."""
    def frame(i):
        return np.vstack([noise(HEIGHT - 90, i), paper(90)])
    return write_video(tmp_path / "overlay_bottom.mp4", frame)


def test_the_tab_band_is_found_above_a_moving_player(overlay_top):
    band = find_overlay_band(overlay_top, Config())
    assert band is not None
    y0, y1 = band
    assert y0 == 0 and 80 <= y1 <= 100


def test_the_tab_band_is_found_below_a_moving_player(overlay_bottom):
    band = find_overlay_band(overlay_bottom, Config())
    assert band is not None
    y0, y1 = band
    assert y1 == HEIGHT and 140 <= y0 <= 160


def test_a_screencast_keeps_the_whole_frame(tmp_path):
    """Nothing moves but a thin cursor. There is no player to cut away, and
    cropping to some arbitrary still band would only lose notation."""
    def frame(i):
        f = np.full((HEIGHT, WIDTH, 3), 240, np.uint8)
        f[40:200, :] = 250
        f[40:200, (i * 4) % WIDTH:(i * 4) % WIDTH + 2] = 0   # the cursor
        return f
    path = write_video(tmp_path / "screencast.mp4", frame)
    assert find_overlay_band(path, Config()) is None


def test_a_player_that_scrolls_is_not_mistaken_for_an_overlay(tmp_path):
    """A scrolling tab player animates a strip and leaves stillness both above
    and below it — notation at the top, a fretboard diagram at the bottom.
    Cropping to either half would throw away the notation or read the diagram as
    if it were notation, so this must be declined rather than guessed at."""
    def frame(i):
        return np.vstack([paper(80), noise(80, i), paper(80)])
    path = write_video(tmp_path / "scrolling.mp4", frame)
    assert find_overlay_band(path, Config()) is None


def test_letterboxing_is_not_mistaken_for_a_tab(tmp_path):
    """Black bars are every bit as still as a tab and, being the larger run,
    would win outright — cropping the reader onto nothing."""
    def frame(i):
        return np.vstack([
            np.zeros((60, WIDTH, 3), np.uint8),
            paper(60),
            noise(60, i),
            np.zeros((60, WIDTH, 3), np.uint8),
        ])
    path = write_video(tmp_path / "letterboxed.mp4", frame)
    band = find_overlay_band(path, Config())
    assert band is not None
    y0, y1 = band
    assert 50 <= y0 <= 70 and 110 <= y1 <= 130


def test_find_tab_rows_falls_back_where_there_is_no_overlay(tmp_path):
    """A screencast must go on being read exactly as it was."""
    def frame(i):
        f = np.full((HEIGHT, WIDTH, 3), 240, np.uint8)
        f[:30] = 0
        f[210:] = 0
        f[100, :] = 20
        return f
    path = write_video(tmp_path / "plain.mp4", frame)
    config = Config()
    assert find_overlay_band(path, config) is None
    assert find_tab_rows(path, config) == find_content_rows(path, config)


def test_find_tab_rows_prefers_the_band_where_there_is_one(overlay_top):
    config = Config()
    assert find_tab_rows(overlay_top, config) == find_overlay_band(overlay_top, config)
