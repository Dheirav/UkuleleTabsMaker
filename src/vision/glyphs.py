"""Glyph classification for cleanly rendered tab digits.

Tab-player screencasts render digits in a single consistent UI font, so matching
against font-rendered reference glyphs beats a model trained on handwriting.
The font family is identified per video from the glyphs themselves.
"""
from typing import List, Optional, Tuple

import cv2
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional dependency
    Image = ImageDraw = ImageFont = None


CANDIDATE_FONTS = (
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",
    "/usr/share/fonts/opentype/urw-base35/NimbusSans-Bold.otf",
    "/usr/share/fonts/opentype/urw-base35/URWGothic-Book.otf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
)

NORM = 32
ASPECT_PENALTY = 0.35
# Overlap alone cannot tell a 3 from an 8. A 3 is very nearly an 8 with the left
# of each bowl removed, so it sits inside the 8 template and scores well against
# it, and these videos render the digits thickly enough to close the counters up
# further. Holes are what actually differ, and they are cheap to count: none for
# 1 2 3 5 7, one for 0 4 6 9, two for 8.
HOLE_PENALTY = 0.30
MIN_HOLE_AREA = 0.01


def normalize_glyph(binary: np.ndarray) -> Optional[np.ndarray]:
    """Tight-crop, scale the longest side to NORM, and centre on a NORM square."""
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    crop = binary[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = crop.shape
    scale = (NORM - 4) / float(max(h, w))
    crop = cv2.resize(
        crop,
        (max(int(round(w * scale)), 1), max(int(round(h * scale)), 1)),
        interpolation=cv2.INTER_AREA,
    )
    out = np.zeros((NORM, NORM), np.float32)
    y0 = (NORM - crop.shape[0]) // 2
    x0 = (NORM - crop.shape[1]) // 2
    out[y0: y0 + crop.shape[0], x0: x0 + crop.shape[1]] = crop
    peak = out.max()
    return out / peak if peak else out


def _holes(binary: np.ndarray) -> int:
    """Enclosed background regions, which is what separates 8 from 3 and 0 from 7.

    Counted on the glyph's own mask, so a neighbouring digit cannot close a gap
    that is really open. Specks are ignored: a thick stroke leaves ragged pixels
    against the edge of a counter, and each of those would otherwise be a hole.
    """
    padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    background = (padded == 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(background, 4)
    outer = labels[0, 0]
    floor = MIN_HOLE_AREA * float(binary.shape[0] * binary.shape[1])
    return sum(1 for i in range(1, count)
               if i != outer and stats[i, cv2.CC_STAT_AREA] >= floor)


def _aspect(binary: np.ndarray) -> float:
    ys, xs = np.where(binary > 0)
    return (xs.max() - xs.min() + 1) / float(ys.max() - ys.min() + 1)


def build_font_templates(path: str) -> Optional[List[Tuple[np.ndarray, int, float]]]:
    if ImageFont is None:
        return None
    try:
        font = ImageFont.truetype(path, 64)
    except OSError:
        return None
    templates = []
    for digit in range(10):
        img = Image.new("L", (96, 128), 0)
        ImageDraw.Draw(img).text((24, 24), str(digit), fill=255, font=font)
        arr = np.array(img)
        binary = (arr > 96).astype(np.uint8) * 255
        norm = normalize_glyph(binary)
        if norm is None:
            return None
        templates.append((norm, digit, _aspect(binary), _holes(binary)))
    return templates


def _match(binary: np.ndarray, templates) -> Tuple[Optional[int], float]:
    norm = normalize_glyph(binary)
    if norm is None:
        return None, -1.0
    aspect = _aspect(binary)
    holes = _holes(binary)
    best, best_score = None, -1.0
    for template, digit, t_aspect, t_holes in templates:
        denom = np.sqrt((norm * norm).sum() * (template * template).sum()) + 1e-9
        overlap = float((norm * template).sum() / denom)
        penalty = min(abs(aspect - t_aspect) / max(t_aspect, 0.05), 1.0)
        score = overlap - ASPECT_PENALTY * penalty
        score -= HOLE_PENALTY * min(abs(holes - t_holes), 2)
        if score > best_score:
            best_score, best = score, digit
    return best, best_score


class GlyphClassifier:
    """Picks the best-fitting font for a video, then classifies with it."""

    def __init__(self, sample_glyphs: List[np.ndarray]):
        self.font_path: Optional[str] = None
        self.fit: float = 0.0
        self.templates = None
        sample = [g for g in sample_glyphs if g is not None][:400]
        if not sample:
            return
        for path in CANDIDATE_FONTS:
            templates = build_font_templates(path)
            if templates is None:
                continue
            scores = sorted((s for _, s in (_match(g, templates) for g in sample) if s > 0),
                            reverse=True)
            if not scores:
                continue
            # A page also carries non-digit ink (the "TAB" clef, tempo marks), which
            # no digit font can explain. Score each font on the glyphs it fits best
            # so that ink cannot drag the vote onto the wrong family.
            keep = max(int(len(scores) * 0.7), 1)
            trimmed = float(np.mean(scores[:keep]))
            if trimmed > self.fit:
                self.fit, self.font_path, self.templates = trimmed, path, templates

    def classify(self, binary: np.ndarray) -> Tuple[Optional[int], float]:
        if self.templates is None:
            return None, 0.0
        return _match(binary, self.templates)
