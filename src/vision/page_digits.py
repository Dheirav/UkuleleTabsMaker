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
    # The first and last rows of the strip are where the video's black surround
    # meets the page. That edge is dark and runs the full width, so it reads as a
    # staff line and shifts every note onto its neighbour's string.
    margin = config.string_line_edge_margin
    rows = rows[(rows >= margin) & (rows < gray.shape[0] - margin)]
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
    return _evenly_spaced(([int(np.mean(g)) for g in groups]), config)


def _consistent(run: List[int], config: Config) -> bool:
    """Whether these lines sit at one spacing, as a staff does by construction."""
    if len(run) < 3:
        return False
    gaps = [b - a for a, b in zip(run, run[1:])]
    middle = sorted(gaps)[len(gaps) // 2]
    if middle <= 0:
        return False
    return all(abs(g - middle) <= config.string_line_spacing_tolerance * middle
               for g in gaps)


def _maximal_runs(lines: List[int], config: Config) -> List[List[int]]:
    """Every staff the lines could be, none of them part of a longer one.

    Maximality is what makes the count mean anything: the top four lines of a
    five-line staff are just as evenly spaced as the staff itself, so without it
    a notation staff offers a spurious four-line reading of its own.
    """
    out: List[List[int]] = []
    total = len(lines)
    for start in range(total):
        for end in range(start + 3, total + 1):
            run = lines[start:end]
            if not _consistent(run, config):
                continue
            if start > 0 and _consistent(lines[start - 1:end], config):
                continue
            if end < total and _consistent(lines[start:end + 1], config):
                continue
            out.append(run)
    return out


def _evenly_spaced(lines: List[int], config: Config) -> List[int]:
    """The lines of the staff the fret numbers are written on.

    Tab for a ukulele has one line per string, so the staff wanted here is the
    one with exactly four. That is worth saying explicitly because the genre this
    reader is being pointed at prints standard notation above the tab, and a
    notation staff has five lines and is therefore the *longer* run: taking the
    longest, as this did while every video was tab alone, reads the melody line
    as if it were tab and hands every note a string it does not have.

    Where nothing has four lines the longest run is still the best guess, which
    keeps this honest on a video showing an instrument with some other number of
    strings.
    """
    if len(lines) < 3:
        return lines
    runs = _maximal_runs(lines, config)
    if not runs:
        return lines
    strings = len(config.tuning)
    exact = [run for run in runs if len(run) == strings]
    if exact:
        return max(exact, key=lambda run: run[-1] - run[0])
    return max(runs, key=len)


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
    """The frets on a page. See read_page_detail for what it declined."""
    return read_page_detail(composite, classifier, config)[0]


def read_page_detail(composite: np.ndarray, classifier, config: Config):
    """(frets, where it declined) — the second is what makes a blank bar legible.

    A tie is drawn "(4)", and the brackets group with the digit as though it were
    a multi-digit fret. Neither bracket classifies, so the group is dropped whole
    — which is right, since a tie is a note still ringing rather than one to
    pluck. But a bar holding only a tie then looks identical to a bar the reader
    failed on, and it is not: one had nothing to print, the other lost music.
    """
    gray = cv2.cvtColor(composite, cv2.COLOR_BGR2GRAY)
    lines = find_string_lines(gray, config)
    components = glyph_components(strip_rules(gray, config), config)
    if not components:
        return [], []
    median_w = float(np.median([c[2] for c in components]))
    gap = max(int(median_w * config.glyph_merge_gap_ratio), 3)

    detections: List[DigitDetection] = []
    declined: List[int] = []
    for group in _group_multidigit(components, gap):
        digits, scores = [], []
        for comp in group:
            value, score = classifier.classify(comp[4])
            if value is None or score < config.glyph_min_score:
                digits = []
                break
            digits.append(value)
            scores.append(score)
        left = min(c[0] for c in group)
        right = max(c[0] + c[2] for c in group)
        if not digits:
            declined.append(int((left + right) / 2))
            continue
        value = int("".join(str(d) for d in digits))
        if value > config.max_fret:
            declined.append(int((left + right) / 2))
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
    return detections, declined


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
