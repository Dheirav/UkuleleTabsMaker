from typing import List, Dict, Any, Optional
from collections import defaultdict

import numpy as np

from src.app.config import Config
from src.models.schema import Note, ReconstructionResult


def _pair_velocities(frames: List[Dict[str, Any]]) -> List[float]:
    velocities = []
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]
        if not prev["bar_lines"] or not curr["bar_lines"]:
            continue
        prev_x = float(np.median(prev["bar_lines"]))
        curr_x = float(np.median(curr["bar_lines"]))
        dt = curr["timestamp"] - prev["timestamp"]
        if dt <= 0:
            continue
        velocities.append((curr_x - prev_x) / dt)
    return velocities


def _robust_median(velocities: List[float]) -> Optional[float]:
    if not velocities:
        return None
    median = float(np.median(velocities))
    mad = float(np.median(np.abs(np.array(velocities) - median)))
    if mad <= 1e-6:
        return median
    filtered = [v for v in velocities if abs(v - median) <= 3.0 * mad]
    if not filtered:
        return median
    return float(np.median(filtered))


def _estimate_speed(frames: List[Dict[str, Any]]) -> Optional[float]:
    return _robust_median(_pair_velocities(frames))


def _estimate_speed_profile(
    frames: List[Dict[str, Any]], config: Optional[Config] = None
) -> Optional[Dict[float, float]]:
    window_s = config.speed_estimation_window_s if config else 3.0
    samples = []
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]
        if not prev["bar_lines"] or not curr["bar_lines"]:
            continue
        prev_x = float(np.median(prev["bar_lines"]))
        curr_x = float(np.median(curr["bar_lines"]))
        dt = curr["timestamp"] - prev["timestamp"]
        if dt <= 0:
            continue
        t_mid = (prev["timestamp"] + curr["timestamp"]) / 2.0
        samples.append((t_mid, (curr_x - prev_x) / dt))
    if not samples:
        return None
    velocities = [v for _, v in samples]
    global_median = float(np.median(velocities))
    mad = float(np.median(np.abs(np.array(velocities) - global_median)))
    threshold = 3.0 * mad if mad > 1e-6 else float("inf")
    cleaned = [(t, v) for t, v in samples if abs(v - global_median) <= threshold]
    if not cleaned:
        cleaned = samples

    speeds: Dict[float, float] = {}
    times = sorted({f["timestamp"] for f in frames})
    for t in times:
        window = [v for tt, v in cleaned if abs(tt - t) <= window_s]
        speeds[t] = float(np.median(window)) if window else global_median
    return speeds


def _time_for_x(frame_time: float, x: float, speed_px_s: Optional[float]) -> float:
    if speed_px_s is None or speed_px_s == 0:
        return frame_time
    speed = abs(speed_px_s)
    return frame_time + (x / speed)


def _cluster_times(times: List[float], tolerance: float = 0.25) -> List[float]:
    if not times:
        return []
    clusters: List[List[float]] = []
    for t in sorted(times):
        if clusters and t - clusters[-1][-1] <= tolerance:
            clusters[-1].append(t)
        else:
            clusters.append([t])
    return [float(np.median(c)) for c in clusters]


def _dedup_notes(notes: List[Note], tolerance: float = 0.15) -> List[Note]:
    groups: Dict[Any, List[Note]] = defaultdict(list)
    for note in sorted(notes, key=lambda n: (n.time, n.string_index, n.fret)):
        key = (note.string_index, note.fret)
        lst = groups[key]
        if lst and note.time - lst[-1].time <= tolerance:
            if note.confidence > lst[-1].confidence:
                lst[-1] = note
        else:
            lst.append(note)
    result = [n for lst in groups.values() for n in lst]
    return sorted(result, key=lambda n: n.time)


def _chunk_signature(frame: dict) -> tuple:
    return tuple(
        sorted((d.bbox, d.value, d.string_index) for d in frame["digits"])
    )


def reconstruct_chunked_notes(
    frames: List[dict], config: Optional[Config] = None
) -> ReconstructionResult:
    frames = sorted(frames, key=lambda f: f["timestamp"])
    chunks: List[List[dict]] = []
    for frame in frames:
        if chunks and _chunk_signature(frame) == _chunk_signature(chunks[-1][-1]):
            chunks[-1].append(frame)
        else:
            chunks.append([frame])

    notes: List[Note] = []
    bar_times: List[float] = []
    for ci, chunk in enumerate(chunks):
        start = chunk[0]["timestamp"]
        if ci + 1 < len(chunks):
            end = chunks[ci + 1][0]["timestamp"]
        else:
            gap = chunk[-1]["timestamp"] - chunk[0]["timestamp"]
            end = start + (gap if gap > 0 else 0.25)
        duration = max(end - start, 0.01)

        seen_notes = set()
        seen_bars = set()
        xs = []
        chunk_notes = []
        for frame in chunk:
            for det in frame["digits"]:
                key = (det.x_center, det.value, det.string_index)
                if key in seen_notes:
                    continue
                seen_notes.add(key)
                xs.append(det.x_center)
                chunk_notes.append((det.x_center, det.value, det.confidence))
            for bx in frame["bar_lines"]:
                if bx in seen_bars:
                    continue
                seen_bars.add(bx)
                xs.append(bx)

        if not xs:
            continue
        x_min, x_max = min(xs), max(xs)
        span = x_max - x_min

        def to_time(x: float) -> float:
            t = start + ((x - x_min) / span) * duration if span > 0 else start
            return float(min(max(t, start), end))

        for x, value, confidence in chunk_notes:
            notes.append(
                Note(
                    time=to_time(x),
                    string_index=config.single_line_string_index if config else 0,
                    fret=value,
                    confidence=confidence,
                )
            )
        for bx in seen_bars:
            bar_times.append(to_time(bx))

    tolerance = config.note_dedup_tolerance_s if config else 0.15
    notes = _dedup_notes(notes, min(tolerance, 0.02))
    notes = sorted(notes, key=lambda n: n.time)
    bar_tolerance = config.bar_time_cluster_tolerance_s if config else 0.25
    bar_times = _cluster_times(bar_times, bar_tolerance)
    return ReconstructionResult(notes=notes, bar_times=bar_times, speed_px_per_s=None)


def reconstruct_notes(frames: List[dict], config: Optional[Config] = None) -> ReconstructionResult:
    global_speed = _estimate_speed(frames)
    profile = _estimate_speed_profile(frames, config)
    notes: List[Note] = []
    bar_times: List[float] = []
    support: Dict[Any, set] = defaultdict(set)

    for frame in frames:
        if profile is not None:
            speed = profile.get(frame["timestamp"], global_speed)
        else:
            speed = global_speed
        for det in frame["digits"]:
            time = _time_for_x(frame["timestamp"], det.x_center, speed)
            time = max(time, 0.0)
            notes.append(
                Note(
                    time=time,
                    string_index=det.string_index,
                    fret=det.value,
                    confidence=det.confidence,
                )
            )
            support[(det.string_index, det.value)].add(frame["timestamp"])
        for bar_x in frame["bar_lines"]:
            bar_times.append(_time_for_x(frame["timestamp"], bar_x, speed))

    if config is not None:
        min_frames = max(config.note_min_frames, 1)
        high_conf = config.note_high_confidence
        notes = [
            n
            for n in notes
            if len(support.get((n.string_index, n.fret), ())) >= min_frames
            or n.confidence >= high_conf
        ]

    tolerance = config.note_dedup_tolerance_s if config else 0.15
    notes = _dedup_notes(notes, tolerance)

    bar_tolerance = config.bar_time_cluster_tolerance_s if config else 0.25
    bar_times = _cluster_times(bar_times, bar_tolerance)
    return ReconstructionResult(notes=notes, bar_times=bar_times, speed_px_per_s=global_speed)
