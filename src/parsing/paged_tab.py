"""Turn paged tab reads into timed notes.

Timing comes from what the player itself shows: the highlight marks which
measure is sounding and when, and (when present) a playhead gives the exact
instant each x-position is reached. Nothing is inferred from scroll velocity.
"""
from typing import Dict, List, Optional

import numpy as np

from src.app.config import Config
from src.models.schema import Note, ReconstructionResult
from src.vision.paged import playhead_bar_starts


def _playhead_times(scan_data: Dict, page, config: Config) -> Optional[List]:
    """(x, t) samples of the cursor while this page was on screen.

    One entry per x, holding the first time the cursor was seen there. The cursor
    rests on the same pixel for several frames, so the raw samples repeat an x
    many times over — 84% of them at full frame rate — and np.interp needs its
    x values strictly increasing. Fed duplicates it reads off the last one, which
    dates every note to the moment the cursor *left* it rather than arrived, and
    grows worse the more finely the video is sampled.
    """
    heads, fps = scan_data["heads"], scan_data["fps"]
    arrival: Dict[int, float] = {}
    for i in range(page.first_frame, min(page.last_frame + 1, len(heads))):
        if heads[i] < 0:
            continue
        time = i / fps
        if heads[i] not in arrival or time < arrival[heads[i]]:
            arrival[heads[i]] = time
    if len(arrival) < config.playhead_min_positions:
        return None
    return sorted(arrival.items())


def _time_from_playhead(x: float, samples: List) -> Optional[float]:
    xs = [s[0] for s in samples]
    ts = [s[1] for s in samples]
    if x < xs[0] or x > xs[-1]:
        return None
    return float(np.interp(x, xs, ts))


def measure_coverage(pages) -> Dict[str, float]:
    """How much of what the player actually played did we read?

    Recognition accuracy only speaks for pages the reader kept. The highlight
    track is produced by the scan and attaches to every page, including ones
    whose composite was rejected, so it is an independent witness: a measure the
    player highlighted but we emitted nothing for is music we lost.

    The denominator counts highlight spans, and one musical measure can appear as
    two spans when a page turns mid-measure, so treat this as a floor on coverage
    rather than an exact fraction of the score.
    """
    total = 0
    covered = 0
    empty_pages = 0
    for page in pages:
        if not page.measures:
            continue
        if not page.digits:
            empty_pages += 1
        for measure in page.measures:
            total += 1
            if any(measure.x0 <= d.x_center <= measure.x1 for d in page.digits):
                covered += 1
    return {
        "measures_highlighted": float(total),
        "measures_with_notes": float(covered),
        "coverage": float(covered / total) if total else 0.0,
        "pages_without_digits": float(empty_pages),
    }


def notes_from_pages(pages, scan_data: Dict, config: Config) -> ReconstructionResult:
    notes: List[Note] = []
    bar_times: List[float] = []

    for page in pages:
        if not page.digits:
            continue
        samples = (_playhead_times(scan_data, page, config)
                   if config.use_playhead else None)

        for measure in page.measures:
            span = float(measure.x1 - measure.x0)
            if span <= 0:
                continue
            inside = [d for d in page.digits if measure.x0 <= d.x_center <= measure.x1]
            if not inside:
                continue
            bar_times.append(measure.t0)
            for det in inside:
                time = _time_from_playhead(det.x_center, samples) if samples else None
                if time is None:
                    frac = (det.x_center - measure.x0) / span
                    time = measure.t0 + frac * (measure.t1 - measure.t0)
                notes.append(Note(
                    time=float(max(time, 0.0)),
                    string_index=det.string_index,
                    fret=det.value,
                    confidence=det.confidence,
                ))

    notes.sort(key=lambda n: (n.time, n.string_index))
    # The highlight alone misses a bar line whenever two bars in a row carry the
    # same music, because this player holds the sounding measure in one place on
    # screen and the display barely changes. The cursor snapping back to the left
    # marks those, and the two sources together cover what either misses on its
    # own — the cursor cannot mark the first bar, having nothing to snap back
    # from. Clustering drops the duplicates where both saw the same bar.
    if config.use_playhead:
        bar_times.extend(playhead_bar_starts(scan_data, config))
    # A page turn can also split one measure's highlight in two, which would
    # otherwise emit a spurious bar line a few frames after the real one.
    bar_times = _cluster(sorted(set(bar_times)), config.bar_time_cluster_tolerance_s)
    return ReconstructionResult(notes=notes, bar_times=bar_times, speed_px_per_s=None)


def _cluster(times: List[float], tolerance: float) -> List[float]:
    clustered: List[float] = []
    for t in times:
        if clustered and t - clustered[-1] <= tolerance:
            continue
        clustered.append(t)
    return clustered
