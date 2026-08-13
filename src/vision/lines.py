import itertools

import cv2
import numpy as np
from typing import List

from src.app.config import Config


def _cluster_positions(positions: List[int], threshold: int) -> List[int]:
    if not positions:
        return []
    positions = sorted(positions)
    clusters = [[positions[0]]]
    for pos in positions[1:]:
        if abs(pos - clusters[-1][-1]) <= threshold:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])
    return [int(sum(c) / len(c)) for c in clusters]


def detect_string_lines(roi: np.ndarray, config: Config) -> List[int]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    edges = cv2.Canny(gray, 40, 120)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(roi.shape[1] // 12, 15), 1))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    min_len = max(int(roi.shape[1] * config.string_line_min_span_ratio), 80)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=min_len, maxLineGap=10)
    y_positions = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(y1 - y2) <= config.line_angle_tolerance:
                y_positions.append(int((y1 + y2) / 2))

    clustered = _cluster_positions(y_positions, config.string_cluster_px)
    if len(clustered) >= config.string_count:
        best_window = None
        best_score = float("inf")
        for i in range(len(clustered) - config.string_count + 1):
            window = sorted(clustered)[i : i + config.string_count]
            spacings = [window[j + 1] - window[j] for j in range(len(window) - 1)]
            if not spacings:
                continue
            span = max(window) - min(window)
            spacing_var = float(np.var(spacings))
            score = spacing_var * 1000.0 + span
            if score < best_score:
                best_score = score
                best_window = (span, window)
        return best_window[1]
    return sorted(clustered)


def consensus_staff(abs_line_lists: List[List[int]], config: Config) -> List[int]:
    all_lines: List[int] = []
    for abs_lines in abs_line_lists:
        all_lines.extend(abs_lines)
    if len(all_lines) < config.string_count:
        return []

    clusters = []
    for pos in sorted(all_lines):
        if clusters and abs(pos - clusters[-1][1]) <= config.string_cluster_px:
            count, mean = clusters[-1]
            clusters[-1] = (count + 1, (mean * count + pos) / (count + 1))
        else:
            clusters.append((1, float(pos)))
    if len(clusters) < config.string_count:
        return []

    best = None
    for combo in itertools.combinations(range(len(clusters)), config.string_count):
        window = [clusters[j][1] for j in combo]
        spacings = [window[j + 1] - window[j] for j in range(len(window) - 1)]
        spacing_var = float(np.var(spacings))
        support = min(clusters[j][0] for j in combo)
        score = spacing_var * 1000.0 + (max(window) - min(window)) / max(support, 1)
        if best is None or score < best[0]:
            best = (score, window)
    if best is None:
        return []

    mean_spacing = (best[1][-1] - best[1][0]) / (config.string_count - 1)
    if float(np.var([best[1][j + 1] - best[1][j] for j in range(len(best[1]) - 1)])) > (mean_spacing * 0.25) ** 2:
        return []
    return [int(round(pos)) for pos in best[1]]


def apply_consensus_staff(string_lines: List[int], region_y: int, consensus: List[int]) -> List[int]:
    if not consensus:
        return string_lines
    return [int(round(pos - region_y)) for pos in consensus]


def detect_bar_lines(roi: np.ndarray, config: Config) -> List[int]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    grad_x = cv2.convertScaleAbs(grad_x)
    _, thresh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(roi.shape[0] // 4, 12)))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    min_len = max(int(roi.shape[0] * config.bar_line_min_span_ratio), 30)
    lines = cv2.HoughLinesP(thresh, 1, np.pi / 180, threshold=40, minLineLength=min_len, maxLineGap=8)
    x_positions = []
    if lines is not None:
        for x1, y1, x2, y2 in lines[:, 0]:
            if abs(x1 - x2) <= config.line_angle_tolerance:
                x_positions.append(int((x1 + x2) / 2))
    clustered = _cluster_positions(x_positions, max(config.string_cluster_px, 4))
    return sorted(set(clustered))
