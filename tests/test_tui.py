import io

import pytest

from src.app.progress import STAGES, Progress
from src.app.tui import Screen, _State, _bar, _compose, _run_dir, normalise_target
from src.video.downloader import _progress_hook


class _Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, stage, value, detail=""):
        self.calls.append((stage, value, detail))


def test_progress_bands_cover_the_whole_range_and_stay_ordered():
    rec = _Recorder()
    progress = Progress(rec, [key for key, _ in STAGES])
    for key, _ in STAGES:
        progress.stage(key)
        progress.tick(1.0)
    progress.done()
    values = [value for _, value, _ in rec.calls]
    assert values == sorted(values)
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)


def test_skipping_a_stage_rescales_the_remaining_ones():
    rec = _Recorder()
    stages = [key for key, _ in STAGES if key != "download"]
    progress = Progress(rec, stages)
    progress.stage("scan")
    progress.tick(1.0)
    # Without download taking 30% of the bar, finishing the scan is well past
    # halfway rather than the third of the way it would be with it.
    assert rec.calls[-1][1] > 0.5


def test_counter_reports_each_item_as_a_fraction_of_the_stage():
    rec = _Recorder()
    progress = Progress(rec, ["read"])
    progress.stage("read")
    report = progress.counter(4)
    report(2)
    assert rec.calls[-1][1] == pytest.approx(0.5)
    assert rec.calls[-1][2] == "2 of 4"


def test_counter_never_divides_by_zero():
    progress = Progress(None, ["read"])
    progress.stage("read")
    progress.counter(0)(1)  # must not raise


@pytest.mark.parametrize("raw,expected", [
    ("https://www.youtube.com/watch?v=abc", ("url", "https://www.youtube.com/watch?v=abc")),
    ("youtu.be/abc", ("url", "https://youtu.be/abc")),
    ("  https://youtu.be/abc  ", ("url", "https://youtu.be/abc")),
    ('"https://youtu.be/abc"', ("url", "https://youtu.be/abc")),
    ("", ("", "")),
])
def test_urls_are_recognised_however_they_are_pasted(raw, expected):
    assert normalise_target(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    # The path shapes Windows hands over when a file is dragged in.
    (r"\\wsl.localhost\Ubuntu\home\me\clip.mp4", "/home/me/clip.mp4"),
    (r"\\wsl$\Ubuntu\home\me\clip.mp4", "/home/me/clip.mp4"),
    (r"C:\Users\me\clip.mp4", "/mnt/c/Users/me/clip.mp4"),
    (r"outputs\run\video.mp4", "outputs/run/video.mp4"),
    ("benchmark/videos/clip.mp4", "benchmark/videos/clip.mp4"),
])
def test_windows_paths_are_understood_at_the_prompt(raw, expected):
    assert normalise_target(raw) == ("path", expected)


def test_each_source_gets_its_own_output_directory():
    a = _run_dir("url", "https://youtu.be/aaa", "/out")
    b = _run_dir("url", "https://youtu.be/bbb", "/out")
    assert a != b
    assert _run_dir("path", "/videos/My Clip.mp4", "/out") == "/out/My-Clip"


def test_bar_fills_in_proportion():
    assert _bar(0.0, 10) == "░" * 10
    assert _bar(1.0, 10) == "█" * 10
    assert _bar(0.5, 10).count("█") == 5


def test_non_tty_output_is_never_painted_over():
    screen = Screen(io.StringIO())
    assert not screen.tty
    screen.paint(["one", "two"], force=True)
    assert screen.stream.getvalue() == ""  # no escape codes into a pipe or file


def test_display_marks_finished_stages_and_shows_the_current_detail():
    screen = Screen(io.StringIO())
    state = _State(stages=["scan", "pages", "read"], stage="pages",
                   overall=0.5, details={"scan": "frame 10 of 10", "pages": "9 pages"})
    text = "\n".join(_compose(screen, state, "clip.mp4", 0.0))
    assert "✓ scan frames" in text
    assert "▸ build pages" in text and "9 pages" in text
    assert "  read notation" in text


def test_long_details_are_clipped_rather_than_wrapping():
    screen = Screen(io.StringIO())
    state = _State(stages=["scan"], stage="scan", details={"scan": "x" * 500})
    for line in _compose(screen, state, "clip.mp4", 0.0):
        assert len(line) < 200


def test_download_hook_reports_a_fraction_and_a_readable_size():
    seen = []
    hook = _progress_hook(lambda frac, detail: seen.append((frac, detail)))
    hook({"status": "downloading", "downloaded_bytes": 5_000_000,
          "total_bytes": 10_000_000, "speed": 2_000_000})
    hook({"status": "finished"})
    assert len(seen) == 1
    fraction, detail = seen[0]
    assert fraction == pytest.approx(0.5)
    assert "5.0 MB of 10.0 MB" in detail and "2.0 MB/s" in detail


def test_download_hook_survives_an_unknown_total():
    seen = []
    hook = _progress_hook(lambda frac, detail: seen.append((frac, detail)))
    hook({"status": "downloading", "downloaded_bytes": 1_000_000})
    assert seen[0][0] == 0.0
