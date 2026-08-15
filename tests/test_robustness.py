import os

import cv2
import numpy as np
import pytest

from src.app.config import Config
from src.models.schema import Note
from src.parsing.parser import parse_measures
from src.parsing.reconstruct import _cluster_times, _dedup_notes, _estimate_speed
from src.video.sampler import sample_frames
from src.vision.digits import _build_templates, _match_template
from src.vision.lines import detect_bar_lines, detect_string_lines


def _frame(ts, bars):
    return {"timestamp": ts, "roi": None, "bar_lines": bars, "digits": []}


def test_estimate_speed_rejects_outlier_jump():
    frames = [_frame(i, [1000 - i * 20]) for i in range(10)]
    frames.append(_frame(10.0, [1000]))
    speed = _estimate_speed(frames)
    assert speed is not None
    assert abs(speed - (-20.0)) < 1.0


def test_dedup_merges_nearby_notes_keeping_best_confidence():
    notes = [
        Note(time=1.0, string_index=0, fret=3, confidence=0.5),
        Note(time=1.1, string_index=0, fret=3, confidence=0.9),
        Note(time=2.5, string_index=0, fret=3, confidence=0.8),
    ]
    result = _dedup_notes(notes, tolerance=0.2)
    assert len(result) == 2
    assert result[0].time == 1.1
    assert result[0].confidence == 0.9
    assert result[1].time == 2.5


def test_dedup_keeps_distinct_notes():
    notes = [
        Note(time=1.0, string_index=0, fret=3, confidence=0.9),
        Note(time=1.05, string_index=0, fret=5, confidence=0.9),
        Note(time=1.1, string_index=1, fret=3, confidence=0.9),
    ]
    result = _dedup_notes(notes, tolerance=0.2)
    assert len(result) == 3


def test_cluster_times_merges_nearby():
    result = _cluster_times([1.0, 1.1, 1.05, 5.0, 5.02], tolerance=0.2)
    assert len(result) == 2
    assert abs(result[0] - 1.05) < 0.01
    assert abs(result[1] - 5.0) < 0.02


def test_parse_measures_ignores_bars_before_first_note():
    notes = [Note(time=5.0, string_index=0, fret=1, confidence=0.9)]
    measures = parse_measures(notes, [1.0, 3.0])
    assert len(measures) == 1
    assert measures[0].start_time == 5.0
    assert len(measures[0].notes) == 1


def _digit_image(digit, white_on_black):
    img = np.zeros((28, 28), dtype=np.uint8)
    cv2.putText(img, str(digit), (2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
    if not white_on_black:
        img = 255 - img
    return img


def test_match_template_polarity_invariant():
    templates = _build_templates()
    for digit in (0, 2, 3, 5, 6, 8, 9):
        pred_light, score_light = _match_template(_digit_image(digit, True), templates)
        pred_dark, score_dark = _match_template(_digit_image(digit, False), templates)
        assert pred_light == digit, (digit, pred_light)
        assert pred_dark == digit, (digit, pred_dark)
        assert score_dark >= 0.3


def test_string_lines_ignore_short_strokes():
    roi = np.full((200, 400), 255, dtype=np.uint8)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for y in (40, 80, 120, 160):
        cv2.line(roi, (10, y), (390, y), 0, 3)
    cv2.line(roi, (10, 190), (60, 190), 0, 3)
    config = Config()
    lines = detect_string_lines(roi, config)
    assert len(lines) == 4
    for got, expected in zip(lines, (40, 80, 120, 160)):
        assert abs(got - expected) <= 6


def test_bar_lines_ignore_short_strokes():
    roi = np.full((200, 400), 255, dtype=np.uint8)
    roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
    for x in (50, 150, 250, 350):
        cv2.line(roi, (x, 20), (x, 180), 0, 3)
    cv2.line(roi, (390, 20), (390, 40), 0, 3)
    config = Config()
    lines = detect_bar_lines(roi, config)
    assert len(lines) == 4


def test_sample_frames_raises_on_missing_video():
    config = Config()
    with pytest.raises(FileNotFoundError):
        sample_frames("/nonexistent/video.mp4", config)


class _UndecodableCapture:
    """A container that opens and reports frames but decodes none.

    This is how an unsupported codec fails: read() never succeeds, but nothing
    raises, so the sampler would otherwise return an empty list that looks
    exactly like a blank video.
    """

    def __init__(self, path):
        self.path = path

    def isOpened(self):
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 89.0
        return 0.0

    def read(self):
        return False, None

    def release(self):
        pass


def test_sample_frames_raises_when_no_frame_decodes(monkeypatch):
    monkeypatch.setattr(cv2, "VideoCapture", _UndecodableCapture)
    config = Config()
    with pytest.raises(RuntimeError, match="Decoded 0 of ~89 frames"):
        sample_frames("/some/av1_clip.mp4", config)


AV1_FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "av1_sample.mp4")


def test_opencv_build_decodes_av1():
    """Guards the opencv-python pin in requirements-opencv.txt.

    Every release before 4.14.0.94 opens this file and reports 10 frames, then
    decodes none -- including 5.0.0.93, which has a higher version number but
    was cut before the dav1d fix landed. If this fails, OpenCV was upgraded or
    floated onto a build without the fix, and AV1 sources decode to nothing.
    """
    cap = cv2.VideoCapture(AV1_FIXTURE)
    assert cap.isOpened(), f"could not open {AV1_FIXTURE}"
    decoded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        assert frame is not None and frame.size > 0
        decoded += 1
    cap.release()
    assert decoded == 10, (
        f"decoded {decoded}/10 AV1 frames with OpenCV {cv2.__version__}; "
        "this build lacks the dav1d fix (needs opencv-python>=4.14.0.94)"
    )


def test_sample_frames_reads_an_av1_source():
    config = Config()
    sample_frames(AV1_FIXTURE, config)
