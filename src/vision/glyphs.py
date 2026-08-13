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
        templates.append((norm, digit, _aspect(binary)))
    return templates


def _match(binary: np.ndarray, templates) -> Tuple[Optional[int], float]:
    norm = normalize_glyph(binary)
    if norm is None:
        return None, -1.0
    aspect = _aspect(binary)
    best, best_score = None, -1.0
    for template, digit, t_aspect in templates:
        denom = np.sqrt((norm * norm).sum() * (template * template).sum()) + 1e-9
        overlap = float((norm * template).sum() / denom)
        penalty = min(abs(aspect - t_aspect) / max(t_aspect, 0.05), 1.0)
        score = overlap - ASPECT_PENALTY * penalty
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
