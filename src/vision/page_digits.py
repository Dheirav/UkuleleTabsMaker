"""Digit extraction from a clean static tab-page composite."""
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.app.config import Config
from src.models.schema import DigitDetection


def find_string_lines(gray: np.ndarray, config: Config) -> List[int]:
    """Rows spanned by a staff line.

    Keyed on contrast against the page background rather than an absolute
    darkness: renderers draw the staff anywhere from near-black to pale grey, and
    an absolute cut silently finds no lines on the pale ones — which collapses
    every note onto string 0 instead of failing visibly.
    """
    background = float(np.median(gray))
    dark = (gray < background - config.string_line_contrast).astype(np.uint8)
    rows = np.where(dark.mean(axis=1) > config.string_line_row_ratio)[0]
    if len(rows) == 0:
        return []
    groups, current = [], [rows[0]]
    for r in rows[1:]:
        if r - current[-1] <= 2:
            current.append(r)
        else:
            groups.append(current)
            current = [r]
    groups.append(current)
    return [int(np.mean(g)) for g in groups]


def strip_rules(gray: np.ndarray, config: Config) -> np.ndarray:
    """Remove the long horizontal string lines and vertical bar lines."""
    ink = (gray < config.page_ink_threshold).astype(np.uint8) * 255
    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(gray.shape[1] // 20, 40), 1))
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, max(gray.shape[0] // 3, 20)))
    horizontals = cv2.morphologyEx(ink, cv2.MORPH_OPEN, h_kernel)
    verticals = cv2.morphologyEx(ink, cv2.MORPH_OPEN, v_kernel)
    return cv2.subtract(cv2.subtract(ink, horizontals), verticals)


def glyph_components(mask: np.ndarray, config: Config):
    """(x, y, w, h, glyph) where glyph holds only that component's pixels, so
    neighbouring rules and glyphs cannot leak into the classifier."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < config.glyph_min_height or area < config.glyph_min_area or w < 3:
            continue
        if h > mask.shape[0] * config.glyph_max_height_ratio:
            continue
        glyph = (labels[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        out.append((int(x), int(y), int(w), int(h), glyph))
    return sorted(out, key=lambda c: c[0])


def _group_multidigit(components, gap: int):
    """Merge horizontally adjacent glyphs on the same row (e.g. '1' '2' -> 12)."""
    groups = []
    for comp in components:
        if groups:
            px, py, pw, ph = groups[-1][-1][:4]
            same_row = abs((comp[1] + comp[3] / 2) - (py + ph / 2)) < max(ph, comp[3]) * 0.6
            if same_row and comp[0] - (px + pw) <= gap:
                groups[-1].append(comp)
                continue
        groups.append([comp])
    return groups


def read_page(composite: np.ndarray, classifier, config: Config) -> List[DigitDetection]:
    gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    lines = find_string_lines(gray, config)
    components = glyph_components(strip_rules(gray, config), config)
    if not components:
        return []
    median_w = float(np.median([c[2] for c in components]))
    gap = max(int(median_w * config.glyph_merge_gap_ratio), 3)

    detections: List[DigitDetection] = []
    for group in _group_multidigit(components, gap):
        digits, scores = [], []
        for comp in group:
            value, score = classifier.classify(comp[4])
            if value is None or score < config.glyph_min_score:
                digits = []
                break
            digits.append(value)
            scores.append(score)
        if not digits:
            continue
        value = int("".join(str(d) for d in digits))
        if value > config.max_fret:
            continue
        x = min(c[0] for c in group)
        y = min(c[1] for c in group)
        x1 = max(c[0] + c[2] for c in group)
        y1 = max(c[1] + c[3] for c in group)
        y_center = (y + y1) / 2
        string_index = (int(np.argmin([abs(y_center - s) for s in lines]))
                        if lines else 0)
        detections.append(DigitDetection(
            value=value,
            bbox=(x, y, x1 - x, y1 - y),
            confidence=float(min(scores)),
            string_index=string_index,
            x_center=int((x + x1) / 2),
        ))
    return detections


def collect_sample_glyphs(pages, config: Config, limit: int = 400) -> List[np.ndarray]:
    """Glyph bitmaps used to identify the video's font before classifying."""
    sample: List[np.ndarray] = []
    for page in pages:
        if page.composite is None:
            continue
        gray = cv2.cvtColor(page.composite, cv2.COLOR_BGR2GRAY)
        for comp in glyph_components(strip_rules(gray, config), config):
            sample.append(comp[4])
            if len(sample) >= limit:
                return sample
    return sample
