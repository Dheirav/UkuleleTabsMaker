from src.app.config import Config
from src.models.schema import DigitDetection
from src.vision.digits import _merge_digits


def _det(value, x, y, w, h, string=0, conf=0.9):
    return DigitDetection(
        value=value,
        bbox=(x, y, w, h),
        confidence=conf,
        string_index=string,
        x_center=x + w // 2,
    )


def test_merge_chains_three_digit_fret():
    config = Config()
    dets = [_det(1, 10, 10, 8, 14), _det(2, 20, 12, 8, 14), _det(3, 30, 10, 8, 16)]
    merged = _merge_digits(dets, config)
    assert len(merged) == 2
    assert merged[0].value == 12
    assert merged[1].value == 3


def test_merge_caps_at_max_fret():
    config = Config()
    dets = [_det(1, 10, 10, 8, 14), _det(0, 20, 12, 8, 14), _det(1, 30, 10, 8, 16)]
    merged = _merge_digits(dets, config)
    assert len(merged) == 2
    assert merged[0].value == 10
    assert merged[1].value == 1


def test_merge_bbox_height_spans_both_digits():
    config = Config()
    dets = [_det(1, 10, 10, 8, 20), _det(2, 20, 15, 8, 20)]
    merged = _merge_digits(dets, config)
    assert merged[0].bbox == (10, 10, 18, 25)


def test_merge_keeps_digits_on_different_strings():
    config = Config()
    dets = [_det(1, 10, 10, 8, 14, string=0), _det(2, 20, 12, 8, 14, string=1)]
    merged = _merge_digits(dets, config)
    assert len(merged) == 2


def test_merge_does_not_merge_wide_gap():
    config = Config()
    dets = [_det(1, 10, 10, 8, 14), _det(2, 30, 12, 8, 14)]
    merged = _merge_digits(dets, config)
    assert len(merged) == 2


def test_merge_takes_minimum_confidence():
    config = Config()
    dets = [_det(1, 10, 10, 8, 14, conf=0.8), _det(2, 20, 12, 8, 14, conf=0.5)]
    merged = _merge_digits(dets, config)
    assert len(merged) == 1
    assert merged[0].confidence == 0.5
