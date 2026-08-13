import cv2
import numpy as np
from typing import List, Tuple

from src.app.config import Config
from src.models.schema import DigitDetection
from src.vision.ocr_fallback import ocr_digits
from src.vision.cnn_classifier import load_cnn_classifier, predict_digit


_CNN_MODEL = None


def _build_templates(font_scale: float = 0.6) -> dict:
    templates = {}
    for digit in range(0, 13):
        text = str(digit)
        img = np.zeros((28, 28), dtype=np.uint8)
        cv2.putText(img, text, (2, 22), cv2.FONT_HERSHEY_SIMPLEX, font_scale, 255, 2)
        templates[digit] = img
    return templates


def _match_template(patch: np.ndarray, templates: dict) -> Tuple[int, float]:
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch = cv2.resize(patch, (28, 28))
    patch = patch.astype(np.uint8)
    best_digit = 0
    best_score = -1.0
    for candidate in (patch, 255 - patch):
        for digit, tmpl in templates.items():
            res = cv2.matchTemplate(candidate, tmpl, cv2.TM_CCOEFF_NORMED)
            score = float(res.max())
            if score > best_score:
                best_score = score
                best_digit = digit
    return best_digit, best_score


def _remove_long_runs(mask: np.ndarray, max_run: int) -> np.ndarray:
    out = mask.copy()
    for row in range(mask.shape[0]):
        cols = np.where(mask[row] > 0)[0]
        if len(cols) == 0:
            continue
        runs = np.split(cols, np.where(np.diff(cols) > 1)[0] + 1)
        for run in runs:
            if len(run) > max_run:
                out[row, run] = 0
    return out


def _extract_candidates(roi: np.ndarray, config: Config) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if np.median(gray) < 128:
        gray = 255 - gray
    threshold = int(np.median(gray)) - 110
    threshold = max(90, min(170, threshold))
    mask = (gray < threshold).astype(np.uint8) * 255
    max_run = max(80, int(roi.shape[1] * 0.05))
    mask = _remove_long_runs(mask, max_run)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    bboxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        aspect = w / float(h) if h else 0
        if area < config.digit_min_area or area > config.digit_max_area:
            continue
        if h < config.digit_min_height or h > config.digit_max_height:
            continue
        if w < 3:
            continue
        if aspect < config.digit_min_aspect or aspect > config.digit_max_aspect:
            continue
        bboxes.append((int(x), int(y), int(w), int(h)))
    return sorted(bboxes, key=lambda b: b[0])


def _filter_tab_band(
    bboxes: List[Tuple[int, int, int, int]], region_h: int, config: Config
) -> List[Tuple[int, int, int, int]]:
    if not config.tab_band_filter or len(bboxes) < 8:
        return bboxes
    centers = sorted(y + h // 2 for x, y, w, h in bboxes)
    median_h = float(np.median([h for _, _, _, h in bboxes]))
    band_h = max(int(3 * median_h), 30)
    band_h = min(band_h, int(region_h * 0.6))
    if band_h < 15:
        return bboxes
    best_start = centers[0]
    best_count = 0
    left = 0
    for right in range(len(centers)):
        while centers[right] - centers[left] >= band_h:
            left += 1
        count = right - left + 1
        if count > best_count:
            best_count = count
            best_start = centers[left]
    if best_count < len(bboxes) * 0.45:
        return bboxes
    pad = int(band_h * 0.2)
    kept = [
        b
        for b in bboxes
        if best_start - pad <= b[1] + b[3] // 2 < best_start + band_h + pad
    ]
    if not kept:
        return bboxes
    kept_median_h = float(np.median([h for _, _, _, h in kept]))
    kept = [b for b in kept if b[3] <= 2.2 * kept_median_h]
    return kept if kept else bboxes


def detect_tab_mode(roi: np.ndarray, string_lines: List[int], config: Config) -> str:
    if config.tab_mode in ("single", "multi"):
        return config.tab_mode
    if len(string_lines) >= config.string_count:
        return "multi"
    bboxes = _filter_tab_band(_extract_candidates(roi, config), roi.shape[0], config)
    if not string_lines or not bboxes:
        return "multi"
    centers = [y + h // 2 for _, y, _, h in bboxes]
    band_lo, band_hi = min(centers), max(centers)
    margin = 12
    within = sum(1 for s in string_lines if band_lo - margin <= s <= band_hi + margin)
    return "single" if within <= 2 else "multi"


def _upscale_patch(patch: np.ndarray, config: Config) -> np.ndarray:
    if len(patch.shape) == 3:
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    h, w = patch.shape[:2]
    if h >= config.digit_upscale_min_h or w >= 2 * config.digit_upscale_min_h:
        return patch
    scale = max(3, int(config.digit_upscale_min_h / max(h, 1)) + 1)
    scale = min(scale, 8)
    return cv2.resize(patch, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)


def _extract_candidates_single(
    roi: np.ndarray, string_lines: List[int], config: Config
) -> List[Tuple[int, int, int, int]]:
    if len(string_lines) < 2:
        return _extract_candidates(roi, config)
    top = max(string_lines[-2], 0)
    bottom = min(string_lines[-1], roi.shape[0])
    if bottom - top < 10:
        return _extract_candidates(roi, config)
    band = roi[top:bottom]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / float(h)
        if area < config.digit_min_area or area > config.digit_max_area:
            continue
        if h < 8 or w < 4:
            continue
        if aspect < config.digit_min_aspect or aspect > config.digit_max_aspect:
            continue
        bboxes.append((x, y + top, w, h))
    return sorted(bboxes, key=lambda b: b[0])


def _merge_digits(digits: List[DigitDetection], config: Config) -> List[DigitDetection]:
    merged: List[DigitDetection] = []
    skip = set()
    for i, det in enumerate(digits):
        if i in skip:
            continue
        run = det
        skip.add(i)
        for j in range(i + 1, len(digits)):
            other = digits[j]
            if j in skip or other.string_index != run.string_index:
                continue
            gap = other.bbox[0] - (run.bbox[0] + run.bbox[2])
            if not (0 <= gap <= config.digit_merge_gap_px):
                break
            candidate = int(f"{run.value}{other.value}")
            if candidate > config.max_fret:
                break
            top = min(run.bbox[1], other.bbox[1])
            bottom = max(run.bbox[1] + run.bbox[3], other.bbox[1] + other.bbox[3])
            bbox = (
                run.bbox[0],
                top,
                (other.bbox[0] + other.bbox[2]) - run.bbox[0],
                bottom - top,
            )
            run = DigitDetection(
                value=candidate,
                bbox=bbox,
                confidence=min(run.confidence, other.confidence),
                string_index=run.string_index,
                x_center=bbox[0] + bbox[2] // 2,
            )
            skip.add(j)
        merged.append(run)
    return merged


def _filter_by_string_lines(
    bboxes: List[Tuple[int, int, int, int]], string_lines: List[int], config: Config
) -> List[Tuple[int, int, int, int]]:
    if not string_lines:
        return bboxes
    if len(string_lines) >= 2:
        spacing = (max(string_lines) - min(string_lines)) / (len(string_lines) - 1)
        margin = max(spacing * config.digit_line_offset_ratio, 8.0)
    else:
        margin = 12.0
    kept = [
        b
        for b in bboxes
        if min(abs((b[1] + b[3] // 2) - s) for s in string_lines) <= margin
    ]
    if kept and len(kept) < 0.4 * len(bboxes):
        return bboxes
    return kept


def detect_digits(
    roi: np.ndarray,
    string_lines: List[int],
    config: Config,
    force_single: bool = False,
) -> List[DigitDetection]:
    templates = _build_templates()
    if force_single:
        bboxes = _extract_candidates_single(roi, string_lines, config)
    else:
        bboxes = _extract_candidates(roi, config)
        bboxes = _filter_by_string_lines(bboxes, string_lines, config)
    digits: List[DigitDetection] = []
    cnn_model = None
    if config.use_cnn:
        global _CNN_MODEL
        if _CNN_MODEL is None:
            _CNN_MODEL = load_cnn_classifier(config.cnn_weights_path)
        cnn_model = _CNN_MODEL
    for x, y, w, h in bboxes:
        patch = _upscale_patch(roi[y : y + h, x : x + w], config)
        best_digit = None
        score = 0.0
        if cnn_model is not None:
            pred = predict_digit(cnn_model, patch)
            if pred is not None:
                best_digit, score = pred
        if best_digit is None or score < config.cnn_confidence_threshold:
            best_digit, score = _match_template(patch, templates)
            if score < 0.35 and config.ocr_fallback:
                text = ocr_digits(patch)
                if text.isdigit():
                    best_digit = int(text)
                    score = 0.4
        if best_digit is None or best_digit > config.max_fret:
            continue
        if score < config.digit_min_confidence:
            continue
        if force_single or not string_lines:
            string_index = config.single_line_string_index
        else:
            y_center = y + h // 2
            dists = [abs(y_center - s) for s in string_lines]
            string_index = int(np.argmin(dists))
        digits.append(
            DigitDetection(
                value=best_digit,
                bbox=(x, y, w, h),
                confidence=min(max(score, 0.0), 1.0),
                string_index=string_index,
                x_center=x + w // 2,
            )
        )

    return _merge_digits(digits, config)
