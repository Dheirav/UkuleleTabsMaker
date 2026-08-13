import numpy as np

from src.models.schema import DigitDetection
from src.parsing.reconstruct import reconstruct_notes


def _frame(ts, bar_lines, digits):
    roi = np.zeros((100, 200, 3), dtype=np.uint8)
    return {
        "timestamp": ts,
        "roi": roi,
        "bar_lines": bar_lines,
        "digits": digits,
    }


def test_reconstruct_estimates_speed_and_times():
    digits = [DigitDetection(value=3, bbox=(10, 10, 8, 12), confidence=0.9, string_index=1, x_center=40)]
    frames = [
        _frame(0.0, [50], digits),
        _frame(1.0, [100], digits),
    ]

    result = reconstruct_notes(frames)

    assert result.speed_px_per_s is not None
    assert result.speed_px_per_s > 0
    assert len(result.notes) == 2
    assert result.notes[0].time > 0.0
    assert result.notes[1].time > result.notes[0].time
    assert len(result.bar_times) == 2


def test_reconstruct_deduplicates_by_time_string_fret():
    digits_a = [DigitDetection(value=5, bbox=(10, 10, 8, 12), confidence=0.6, string_index=2, x_center=30)]
    digits_b = [DigitDetection(value=5, bbox=(10, 10, 8, 12), confidence=0.9, string_index=2, x_center=30)]
    frames = [
        _frame(0.0, [40], digits_a),
        _frame(0.0, [40], digits_b),
    ]

    result = reconstruct_notes(frames)

    assert len(result.notes) == 1
    assert result.notes[0].confidence == 0.9
