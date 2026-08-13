"""Static paged tab readers.

Tab-player screencasts do not scroll: the notation is static and turns a page at
a time, while a coloured highlight marks the measure being played. This module
streams a video once to find page boundaries and highlight timing, then makes a
clean composite of each page for recognition.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.app.config import Config


@dataclass
class MeasureSpan:
    x0: int
    x1: int
    t0: float
    t1: float


@dataclass
class Page:
    index: int
    first_frame: int
    last_frame: int
    t0: float
    t1: float
    composite: Optional[np.ndarray] = None
    measures: List[MeasureSpan] = field(default_factory=list)
    digits: List = field(default_factory=list)
    instability: float = 0.0


def highlight_mask(bgr: np.ndarray) -> np.ndarray:
    """Warm (yellow) 'currently playing' fill used by tab players."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return (r > 195) & (g > 190) & (b < r - 18) & (b < g - 12)


def playhead_mask(bgr: np.ndarray) -> np.ndarray:
    """Saturated blue vertical cursor used by some players."""
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    return (b > 140) & (g < 130) & (r < 150) & (b - g > 60) & (b - r > 50)


def notation_ink(strip: np.ndarray, config: Config) -> np.ndarray:
    """Neutral dark ink: the notation itself.

    Players draw the notation in near-black and their moving chrome — the blue
    playhead, the warm measure highlight — in saturated colour. Keying on low
    saturation therefore tracks the notation while ignoring the chrome, which is
    what a page-change signal must do.
    """
    lo = strip.min(axis=2).astype(np.int16)
    hi = strip.max(axis=2).astype(np.int16)
    return ((hi < config.page_ink_threshold) &
            ((hi - lo) < config.ink_max_saturation)).astype(np.uint8) * 255


def page_signature(strip: np.ndarray, config: Optional[Config] = None) -> np.ndarray:
    """Unit-normalised column profile of the notation ink.

    Notation is laid out along x, so a page turn moves every glyph and the whole
    profile decorrelates. Collapsing to columns keeps that signal sharp: a coarse
    2D map of sparse ink is mostly blank cells either way, and separates a real
    turn from frame noise by too thin a margin to threshold reliably across
    different renderers.
    """
    config = config or Config()
    ink = notation_ink(strip, config)
    # Drop the staff rules first. They run the full width and are identical on
    # every page, so wherever a renderer draws them dark they dominate each
    # column and bury the glyph signal that actually distinguishes pages.
    rules = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(ink.shape[1] // 20, 40), 1)))
    glyph_ink = cv2.subtract(ink, rules).astype(np.float32) / 255.0
    columns = glyph_ink.sum(axis=0).reshape(1, -1)
    columns = cv2.resize(columns, (config.page_profile_bins, 1),
                         interpolation=cv2.INTER_AREA).ravel()
    norm = float(np.linalg.norm(columns))
    return columns / norm if norm > 0 else columns


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - correlation between two unit-normalised profiles; 0 means identical."""
    return float(max(0.0, 1.0 - float(np.dot(a, b))))


def find_content_rows(video_path: str, config: Config, probes: int = 40) -> Tuple[int, int]:
    """Rows carrying actual content, so letterboxing is dropped once per video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    acc, seen = None, 0
    for i in np.linspace(0, max(total - 1, 0), probes).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        row = (gray > 40).mean(axis=1)
        acc = row if acc is None else acc + row
        seen += 1
    cap.release()
    if acc is None or seen == 0:
        return 0, 0
    rows = np.where(acc / seen > 0.5)[0]
    if len(rows) == 0:
        return 0, 0
    return int(rows.min()), int(rows.max()) + 1


def measure_scroll(video_path: str, config: Config) -> float:
    """Median |horizontal drift| in px/s.

    Probes consecutive pairs spread across the whole video rather than a single
    run at the start: the opening seconds are often a title card or an intro
    animation, which says nothing about how the tab behaves later.
    """
    y0, y1 = find_content_rows(video_path, config)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    probes = max(config.paged_motion_probes, 2)
    drifts = []
    for start in np.linspace(0, max(total - 3, 0), probes).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start))
        prev = None
        for _ in range(config.paged_motion_pair_frames):
            ret, frame = cap.read()
            if not ret:
                break
            strip = frame[y0:y1] if y1 > y0 else frame
            if strip.shape[1] > config.scan_max_width:
                scale = config.scan_max_width / float(strip.shape[1])
                strip = cv2.resize(strip, (config.scan_max_width,
                                           max(int(round(strip.shape[0] * scale)), 1)),
                                   interpolation=cv2.INTER_AREA)
            else:
                scale = 1.0
            gray = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev is not None:
                (dx, _), response = cv2.phaseCorrelate(prev, gray)
                if response > 0.3:
                    drifts.append(abs(dx) / scale * fps)
            prev = gray
    cap.release()
    return float(np.median(drifts)) if drifts else 0.0


def is_paged(video_path: str, config: Config) -> bool:
    if config.paged_mode == "paged":
        return True
    if config.paged_mode == "scrolling":
        return False
    return measure_scroll(video_path, config) <= config.paged_max_scroll_px_s


def scan(video_path: str, config: Config) -> Dict:
    """Single streaming pass: page signatures, highlight spans, playhead x."""
    y0, y1 = find_content_rows(video_path, config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    sig_prev = None
    sig_ref = None
    pending = 0
    boundaries: List[int] = []
    signatures: List[np.ndarray] = []
    diffs: List[float] = []
    spans: List[Optional[Tuple[int, int]]] = []
    heads: List[int] = []
    idx = 0
    scale = 1.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        strip = frame[y0:y1] if y1 > y0 else frame
        # Per-frame work runs on a width-capped copy: the signature is reduced to
        # 256px anyway, and column positions are scaled back to source pixels.
        if strip.shape[1] > config.scan_max_width:
            scale = config.scan_max_width / float(strip.shape[1])
            strip = cv2.resize(strip, (config.scan_max_width,
                                       max(int(round(strip.shape[0] * scale)), 1)),
                               interpolation=cv2.INTER_AREA)
        sig = page_signature(strip, config)
        # Compare against the current page's reference, not the previous frame.
        # A static page holds its notation exactly, so this stays near zero inside
        # a page and jumps hard at a turn — a far wider margin than frame-to-frame
        # differencing, which only ever sees the instant of the turn itself.
        signatures.append(sig)
        # Bootstrap trace against a running reference at a permissive threshold.
        if sig_ref is None:
            sig_ref = sig
            diffs.append(0.0)
        else:
            d = signature_distance(sig, sig_ref)
            diffs.append(d)
            if d > config.page_change_threshold:
                pending += 1
                if pending >= config.page_confirm_frames:
                    boundaries.append(idx - pending + 1)
                    sig_ref = sig
                    pending = 0
            else:
                pending = 0
        sig_prev = sig

        hl = highlight_mask(strip)
        cols = np.where(hl.sum(0) > strip.shape[0] * config.highlight_row_ratio)[0]
        spans.append((int(cols.min() / scale), int(cols.max() / scale))
                     if len(cols) > 20 * scale else None)

        ph = playhead_mask(strip).sum(0)
        heads.append(int(np.argmax(ph) / scale)
                     if ph.max() > strip.shape[0] * 0.3 else -1)
        idx += 1
    cap.release()
    return {"fps": fps, "y0": y0, "y1": y1, "n": idx, "scan_scale": scale,
            "diffs": diffs, "spans": spans, "heads": heads,
            "boundaries": boundaries,
            "signatures": np.asarray(signatures, dtype=np.float32) if signatures else None}


def detect_boundaries(signatures: np.ndarray, threshold: float,
                      config: Config) -> List[int]:
    """Frames where the notation stops matching the current page's reference."""
    cuts: List[int] = []
    reference = None
    pending = 0
    for i in range(len(signatures)):
        sig = signatures[i]
        if reference is None:
            reference = sig
            continue
        if signature_distance(sig, reference) > threshold:
            pending += 1
            if pending >= config.page_confirm_frames:
                cuts.append(i - pending + 1)
                reference = sig
                pending = 0
        else:
            pending = 0
    return cuts


def segment_pages(scan_data: Dict, config: Config) -> List[Page]:
    fps = scan_data["fps"]
    guard = config.page_guard_frames
    if scan_data.get("signatures") is not None:
        cuts = detect_boundaries(scan_data["signatures"],
                                 config.page_change_threshold, config)
    elif "boundaries" in scan_data:
        cuts = list(scan_data["boundaries"])
    else:  # legacy frame-to-frame differencing
        cuts = [i for i, d in enumerate(scan_data["diffs"])
                if d > config.page_change_threshold]
    merged: List[int] = []
    for c in cuts:
        if merged and c - merged[-1] <= config.page_cut_merge_frames:
            merged[-1] = c
        else:
            merged.append(c)

    bounds, start = [], 0
    for c in merged:
        bounds.append((start, c - 1 - guard))
        start = c + guard
    bounds.append((start, scan_data["n"] - 1))

    pages = []
    for lo, hi in bounds:
        if hi - lo < config.page_min_frames:
            continue
        pages.append(Page(index=len(pages), first_frame=lo, last_frame=hi,
                          t0=lo / fps, t1=(hi + 1) / fps))
    return pages


def composite_pages(video_path: str, pages: List[Page], scan_data: Dict,
                    config: Config) -> None:
    """Median over each page's frames removes the moving playhead and highlight."""
    y0, y1 = scan_data["y0"], scan_data["y1"]

    # Sample indices per page, gathered in one sequential pass. Seeking to an
    # arbitrary frame makes the decoder restart from the preceding keyframe, so
    # per-page seeking costs far more than simply decoding the file in order.
    wanted: Dict[int, List[Page]] = {}
    for page in pages:
        count = page.last_frame - page.first_frame + 1
        idxs = np.linspace(page.first_frame, page.last_frame,
                           min(count, config.page_composite_samples)).astype(int)
        for i in idxs:
            wanted.setdefault(int(i), []).append(page)

    cap = cv2.VideoCapture(video_path)
    # Pages are contiguous and ordered, so a page can be finished and freed as
    # soon as the read position passes it: only one page's frames are ever held.
    buffers: Dict[int, List[np.ndarray]] = {}
    pending = sorted(pages, key=lambda p: p.first_frame)
    done: List[Tuple[Page, List[np.ndarray]]] = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for page in wanted.get(idx, ()):
            strip = frame[y0:y1] if y1 > y0 else frame
            buffers.setdefault(page.index, []).append(strip.copy())
        while pending and pending[0].last_frame < idx:
            finished = pending.pop(0)
            done.append((finished, buffers.pop(finished.index, [])))
        idx += 1
    cap.release()
    for page in pending:
        done.append((page, buffers.pop(page.index, [])))

    for page, stack in done:
        if not stack:
            continue
        composite = np.median(np.stack(stack), axis=0).astype(np.uint8)
        # A span that straddles a page turn medians into a ghost of two pages.
        # Real pages hold still, so their frames agree with the composite; measure
        # that agreement and refuse to read a page that does not hold.
        # Jaccard distance on the ink, not raw pixel disagreement: notation covers
        # only a few percent of a page, so a half-ghosted composite still agrees on
        # almost every (blank) pixel. Normalising by ink area makes the difference
        # between "holds still" and "straddles a turn" plain.
        reference = notation_ink(composite, config) > 0
        residuals = []
        for frame in stack:
            ink = notation_ink(frame, config) > 0
            union = float((ink | reference).sum())
            if union > 0:
                residuals.append(float((ink ^ reference).sum()) / union)
        page.instability = float(np.mean(residuals)) if residuals else 1.0
        # Kept as a diagnostic, not a gate. Rejecting an unstable composite double
        # -gates what the classifier already handles: it scores every glyph and
        # declines the ones it cannot match, so a ghosted region yields nothing
        # while the clean interior still reads. Gating here as well threw away
        # whole pages for contamination confined to their margins.
        page.composite = composite


def track_measures(scan_data: Dict, config: Config) -> List[MeasureSpan]:
    """Every measure the player highlighted, over the whole video.

    Deliberately independent of page segmentation: the highlight is drawn by the
    player, so this sequence is the same whatever the page threshold does. That
    makes it the one stable axis to key ground truth against, and the only way a
    measure the reader never saw can be counted as missing rather than silently
    dropped from both sides of the comparison.
    """
    spans, fps = scan_data["spans"], scan_data["fps"]
    signatures = scan_data.get("signatures")
    tol = config.highlight_span_tolerance_px
    out: List[MeasureSpan] = []
    current, current_start, start_sig = None, None, None
    for i, span in enumerate(spans):
        same = (current is not None and span is not None
                and abs(span[0] - current[0]) < tol and abs(span[1] - current[1]) < tol)
        # The highlight alone does not identify a measure. Players redraw the page
        # with the highlight in the same place, so successive measures can occupy
        # identical coordinates; without this, a run of them merges into one block.
        # The fixed threshold keeps this axis independent of the reader's tunable
        # page threshold, so retuning segmentation does not move the truth.
        if same and signatures is not None and start_sig is not None:
            if signature_distance(signatures[i], start_sig) > config.measure_content_threshold:
                same = False
        if same:
            current = (min(current[0], span[0]), max(current[1], span[1]))
            continue
        if current is not None:
            out.append(MeasureSpan(current[0], current[1], current_start / fps, i / fps))
        current, current_start = span, i
        start_sig = signatures[i] if signatures is not None else None
    if current is not None:
        out.append(MeasureSpan(current[0], current[1], current_start / fps,
                               len(spans) / fps))
    return [m for m in out
            if (m.t1 - m.t0) >= config.measure_min_duration_s
            and (m.x1 - m.x0) >= config.measure_min_width_px]


def attach_measures(pages: List[Page], scan_data: Dict, config: Config) -> None:
    """Collapse per-frame highlight boxes into stable measure spans per page."""
    spans, fps = scan_data["spans"], scan_data["fps"]
    tol = config.highlight_span_tolerance_px
    for page in pages:
        current, current_start, out = None, None, []
        for i in range(page.first_frame, page.last_frame + 1):
            sp = spans[i] if i < len(spans) else None
            same = (current is not None and sp is not None
                    and abs(sp[0] - current[0]) < tol and abs(sp[1] - current[1]) < tol)
            if same:
                current = (min(current[0], sp[0]), max(current[1], sp[1]))
                continue
            if current is not None:
                out.append(MeasureSpan(current[0], current[1], current_start / fps, i / fps))
            current, current_start = sp, i
        if current is not None:
            out.append(MeasureSpan(current[0], current[1], current_start / fps,
                                   (page.last_frame + 1) / fps))
        page.measures = [
            m for m in out
            if (m.t1 - m.t0) >= config.measure_min_duration_s
            and (m.x1 - m.x0) >= config.measure_min_width_px
        ]
