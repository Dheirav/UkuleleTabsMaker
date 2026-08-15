import os
import cv2
import numpy as np
from typing import List, Tuple, Dict

from src.app.config import Config
from src.models.schema import FrameSample
from src.utils.hash import average_hash, hamming_distance
from src.vision.region import detect_tab_region
from src.vision.lines import detect_string_lines


def _edge_score(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 80, 160)
    return float(edges.mean())


def _hist_diff(gray_a: np.ndarray, gray_b: np.ndarray) -> float:
    hist_a = cv2.calcHist([gray_a], [0], None, [64], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [64], [0, 256])
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA))


def sample_frames(
    video_path: str,
    config: Config,
    return_stats: bool = False,
) -> List[FrameSample] | Tuple[List[FrameSample], Dict[str, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    base_step = max(int(fps / config.base_sample_fps), 1)

    frames: List[FrameSample] = []
    prev_gray = None
    prev_hash = None
    last_kept_time = 0.0
    frame_id = 0
    total_read = 0
    considered = 0
    skipped_dark = 0
    skipped_dedup = 0
    skipped_other = 0
    skipped_tab_filter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        total_read += 1
        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if frame_id % base_step != 0:
            frame_id += 1
            continue
        considered += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(gray.mean())
        keep = False

        if prev_gray is None:
            keep = brightness >= config.min_frame_brightness
        else:
            edge_delta = abs(_edge_score(gray) - _edge_score(prev_gray))
            hist_delta = _hist_diff(gray, prev_gray)
            if edge_delta > config.scene_edge_threshold or hist_delta > config.hist_diff_threshold:
                keep_ratio = len(frames) / float(considered) if considered else 0.0
                keep = True
            if (timestamp - last_kept_time) >= config.max_sample_gap_s:
                keep = True

        if brightness < config.min_frame_brightness:
            keep = False
            skipped_dark += 1

        if keep:
            if config.tab_filter_enabled:
                region = detect_tab_region(frame, config)
                x, y, w, h = region.bbox
                roi = frame[y : y + h, x : x + w]
                string_lines = detect_string_lines(roi, config)
                if region.confidence < config.tab_min_confidence or len(string_lines) < config.tab_min_string_lines:
                    skipped_tab_filter += 1
                    frame_id += 1
                    prev_gray = gray
                    continue
            if config.dedup_hamming_threshold <= 0:
                frames.append(FrameSample(frame_id=frame_id, timestamp=timestamp, image=frame))
                if config.save_sampled_frames:
                    out_dir = os.path.join(config.output_dir, config.sampled_frames_subdir)
                    os.makedirs(out_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(out_dir, f"frame_{frame_id:06d}.png"), frame)
                last_kept_time = timestamp
            else:
                frame_hash = average_hash(gray)
                if prev_hash is None or hamming_distance(frame_hash, prev_hash) > config.dedup_hamming_threshold:
                    frames.append(FrameSample(frame_id=frame_id, timestamp=timestamp, image=frame))
                    if config.save_sampled_frames:
                        out_dir = os.path.join(config.output_dir, config.sampled_frames_subdir)
                        os.makedirs(out_dir, exist_ok=True)
                        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_id:06d}.png"), frame)
                    prev_hash = frame_hash
                    last_kept_time = timestamp
                else:
                    skipped_dedup += 1
        else:
            skipped_other += 1

        prev_gray = gray
        frame_id += 1

    reported_total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_s = float(total_read) / float(fps) if fps else 0.0
    cap.release()
    if total_read == 0 and reported_total > 0:
        raise RuntimeError(
            f"Decoded 0 of ~{int(reported_total)} frames from {video_path}. "
            "The container opened but no frame could be decoded, which usually "
            "means the codec is unsupported by this OpenCV build."
        )
    relaxed_threshold = None
    if considered > 0 and config.dedup_hamming_threshold > 0:
        keep_ratio = len(frames) / float(considered)
        if keep_ratio < config.min_keep_ratio:
            relaxed_threshold = max(config.dedup_hamming_threshold - 2, 1)
            relaxed: List[FrameSample] = []
            if config.save_sampled_frames:
                out_dir = os.path.join(config.output_dir, config.sampled_frames_subdir)
                os.makedirs(out_dir, exist_ok=True)
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                raise FileNotFoundError(f"Could not open video: {video_path}")
            frame_id = 0
            prev_hash = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_id % base_step != 0:
                    frame_id += 1
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = float(gray.mean())
                if brightness < config.min_frame_brightness:
                    frame_id += 1
                    continue
                frame_hash = average_hash(gray)
                if prev_hash is None or hamming_distance(frame_hash, prev_hash) > relaxed_threshold:
                    timestamp = frame_id / fps if fps else 0.0
                    relaxed.append(FrameSample(frame_id=frame_id, timestamp=timestamp, image=frame))
                    if config.save_sampled_frames:
                        cv2.imwrite(os.path.join(out_dir, f"frame_{frame_id:06d}.png"), frame)
                    prev_hash = frame_hash
                frame_id += 1
            cap.release()
            frames = relaxed
    stats = {
        "fps": float(fps),
        "base_step": float(base_step),
        "total_frames_read": float(total_read),
        "frames_considered": float(considered),
        "frames_kept": float(len(frames)),
        "skipped_dark": float(skipped_dark),
        "skipped_dedup": float(skipped_dedup),
        "skipped_other": float(skipped_other),
        "skipped_tab_filter": float(skipped_tab_filter),
        "min_frame_brightness": float(config.min_frame_brightness),
        "duration_s": float(duration_s),
        "save_sampled_frames": float(1.0 if config.save_sampled_frames else 0.0),
        "dedup_disabled": float(1.0 if config.dedup_hamming_threshold <= 0 else 0.0),
        "tab_filter_enabled": float(1.0 if config.tab_filter_enabled else 0.0),
        "tab_min_confidence": float(config.tab_min_confidence),
        "tab_min_string_lines": float(config.tab_min_string_lines),
    }
    if relaxed_threshold is not None:
        stats["relaxed_dedup_threshold"] = float(relaxed_threshold)
    return (frames, stats) if return_stats else frames
