import io
import os

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


def test_the_format_ladder_exhausts_h264_before_anything_else():
    """A codec this build cannot decode is worse than a smaller picture: OpenCV
    reads AV1 as zero frames, which yielded a confidently empty tab sheet."""
    from src.video.downloader import DECODABLE_RUNGS, FORMAT_LADDER

    labels = [label for label, _ in FORMAT_LADDER]
    selectors = [sel for _, sel in FORMAT_LADDER]
    assert all("avc1" in s for s in selectors[:DECODABLE_RUNGS])
    assert not any("avc1" in s for s in selectors[DECODABLE_RUNGS:])
    assert labels[0].startswith("H.264")


def test_a_refused_format_is_retried_before_being_abandoned():
    """YouTube answers 403 to a good format under load. Dropping a rung on one
    refusal trades a decodable video for an undecodable one over a hiccup."""
    from src.video import downloader

    assert downloader.ATTEMPTS_PER_FORMAT >= 2


def test_a_folder_is_named_after_the_song():
    from src.app.library import slug

    assert slug("Coco - Remember Me (Easy Ukulele Tabs Tutorial)") == \
        "Coco - Remember Me (Easy Ukulele Tabs Tutorial)"
    # only what a filesystem refuses is removed; spaces and brackets are what
    # make a title legible at a glance
    assert slug('A/B: "C" <D>|E?') == "AB C DE"
    assert slug("   ") == "untitled"
    assert len(slug("x" * 200)) <= 80


def test_a_song_is_found_again_under_whatever_name_it_took(tmp_path):
    """The title is not known until the video is fetched, so a run starts in a
    working folder and is renamed after. A second run has to find the first."""
    from src.app.library import existing_sheet, finished_dir, working_dir

    root = str(tmp_path)
    url = "https://youtu.be/abc"
    start = working_dir(url, root)
    os.makedirs(start, exist_ok=True)
    open(os.path.join(start, "tabs.json"), "w").close()
    final = finished_dir(url, root, start, "My Song")

    assert os.path.basename(final) == "My Song"
    assert working_dir(url, root) == final          # found again by name
    assert existing_sheet(url, root) == final


def test_two_songs_with_one_name_do_not_overwrite_each_other(tmp_path):
    from src.app.library import finished_dir, working_dir

    root = str(tmp_path)
    for url in ("https://youtu.be/one", "https://youtu.be/two"):
        start = working_dir(url, root)
        os.makedirs(start, exist_ok=True)
        open(os.path.join(start, "tabs.json"), "w").close()
        finished_dir(url, root, start, "Same Title")
    names = sorted(n for n in os.listdir(root) if not n.startswith("."))
    assert names == ["Same Title", "Same Title (2)"]


def test_a_song_read_before_the_index_existed_is_still_found(tmp_path):
    """Each sheet records where it came from, so the folders can be searched when
    the index cannot answer. Without this the same song is fetched again and
    lands beside itself under a numbered name."""
    import json as _json
    from src.app.library import working_dir

    root = str(tmp_path)
    url = "https://youtu.be/abc"
    folder = os.path.join(root, "Some Song")
    os.makedirs(folder)
    with open(os.path.join(folder, "tabs.json"), "w") as fh:
        _json.dump({"notes": [], "metadata": {"source_url": url}}, fh)

    assert working_dir(url, root) == folder          # found without an index
    assert working_dir("https://youtu.be/other", root) != folder
