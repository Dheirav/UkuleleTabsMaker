import cv2
import numpy as np
from typing import Tuple

from src.app.config import Config
from src.models.schema import TabRegion


def detect_tab_region(frame: np.ndarray, config: Config) -> TabRegion:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 140)
    edges = cv2.dilate(edges, None, iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape[:2]
    best_bbox: Tuple[int, int, int, int] = (0, int(h * 0.35), w, int(h * 0.3))
    best_score = 0.1

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area == 0:
            continue
        ratio = cw / float(ch)
        area_ratio = area / float(w * h)
        if ratio < config.tab_region_aspect_min:
            continue
        if not (config.tab_region_min_ratio <= area_ratio <= config.tab_region_max_ratio):
            continue
        score = ratio * area_ratio
        if score > best_score:
            best_score = score
            pad = int(0.02 * w)
            x = max(x - pad, 0)
            y = max(y - pad, 0)
            cw = min(cw + 2 * pad, w - x)
            ch = min(ch + 2 * pad, h - y)
            best_bbox = (x, y, cw, ch)

    return TabRegion(bbox=best_bbox, confidence=min(best_score, 1.0))
