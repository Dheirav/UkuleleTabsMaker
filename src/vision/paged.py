"""Static paged tab readers.

Tab-player screencasts do not scroll: the notation is static and turns a page at
a time, while a coloured highlight marks the measure being played. This module
streams a video once to find page boundaries and highlight timing, then makes a
clean composite of each page for recognition.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.app.config import Config


# Enough probes to average out an intro card or a dark passage. Each one is a
# seek, which makes the decoder restart from the preceding keyframe, so they are
# the most expensive frames in the run.
CONTENT_ROW_PROBES = 16


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
    # Reduce to the signature's own resolution before anything else. The result
    # is a page_profile_bins-wide profile whatever the input size, and the rule
    # removal below costs width squared times height, so paying it at source
    # resolution dominated the entire scan while buying no discrimination.
    small = cv2.resize(ink, (config.page_profile_bins, config.page_signature_rows),
                       interpolation=cv2.INTER_AREA)
    # Drop the staff rules first. They run the full width and are identical on
    # every page, so wherever a renderer draws them dark they dominate each
    # column and bury the glyph signal that actually distinguishes pages.
    rules = cv2.morphologyEx(
        small, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(small.shape[1] // 20, 40), 1)))
    glyph_ink = cv2.subtract(small, rules).astype(np.float32) / 255.0
    columns = glyph_ink.sum(axis=0)
    norm = float(np.linalg.norm(columns))
    return columns / norm if norm > 0 else columns


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - correlation between two unit-normalised profiles; 0 means identical."""
    return float(max(0.0, 1.0 - float(np.dot(a, b))))


def find_content_rows(video_path: str, config: Config, probes: int = CONTENT_ROW_PROBES,
                      on_step: Optional[Callable[[int], None]] = None) -> Tuple[int, int]:
    """Rows carrying actual content, so letterboxing is dropped once per video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    acc, seen = None, 0
    for step, i in enumerate(np.linspace(0, max(total - 1, 0), probes).astype(int)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, frame = cap.read()
        if on_step:
            on_step(step + 1)
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


def measure_scroll(video_path: str, config: Config,
                   on_step: Optional[Callable[[int, int], None]] = None,
                   content_rows: Optional[Tuple[int, int]] = None) -> float:
    """Median |horizontal drift| in px/s.

    Probes consecutive pairs spread across the whole video rather than a single
    run at the start: the opening seconds are often a title card or an intro
    animation, which says nothing about how the tab behaves later.
    """
    probes = max(config.paged_motion_probes, 2)
    steps = (0 if content_rows else CONTENT_ROW_PROBES) + probes
    y0, y1 = content_rows or find_content_rows(
        video_path, config, CONTENT_ROW_PROBES,
        (lambda i: on_step(i, steps)) if on_step else None)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    drifts = []
    base = steps - probes
    for probe, start in enumerate(np.linspace(0, max(total - 3, 0), probes).astype(int)):
        if on_step:
            on_step(base + probe + 1, steps)
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


def is_paged(video_path: str, config: Config,
             on_step: Optional[Callable[[int, int], None]] = None,
             content_rows: Optional[Tuple[int, int]] = None) -> bool:
    if config.paged_mode == "paged":
        return True
    if config.paged_mode == "scrolling":
        return False
    scroll = measure_scroll(video_path, config, on_step, content_rows)
    return scroll <= config.paged_max_scroll_px_s


def sample_stride(video_fps: float, config: Config) -> int:
    """How many video frames one scan sample covers.

    Nothing the scan looks for moves at frame rate: a page is on screen for
    seconds and the shortest measure for a third of one. Sampling at a fixed rate
    keeps the same time resolution on a 60fps upload as on a 30fps one, and
    leaves the decoder skipping frames it never has to convert or copy.
    """
    if config.scan_stride_hz <= 0:
        return 1
    return max(1, int(video_fps // config.scan_stride_hz))


def scan(video_path: str, config: Config,
         on_frame: Optional[Callable[[int, int], None]] = None,
         content_rows: Optional[Tuple[int, int]] = None) -> Dict:
    """Single streaming pass: page signatures, highlight spans, playhead x.

    Indices in the returned arrays count *samples*, not video frames, and "fps"
    is the sample rate, so every downstream time is index / fps as before. The
    video's own frame rate and the stride are carried alongside for the one job
    that needs them: seeking the file again to composite pages.
    """
    y0, y1 = content_rows or find_content_rows(video_path, config)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise FileNotFoundError(f"Could not open video: {video_path}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = sample_stride(video_fps, config)
    fps = video_fps / stride
    expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    sig_prev = None
    sig_ref = None
    pending = 0
    boundaries: List[int] = []
    signatures: List[np.ndarray] = []
    diffs: List[float] = []
    spans: List[Optional[Tuple[int, int]]] = []
    heads: List[int] = []
    idx = 0
    frame_no = 0
    scale = 1.0
    while True:
        # Frames between samples still have to be decoded, but grab() leaves out
        # the colour conversion and copy that read() would spend on them.
        if frame_no % stride:
            if not cap.grab():
                break
            frame_no += 1
            continue
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
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
        if on_frame:
            on_frame(frame_no, max(expected, frame_no))
    cap.release()
    return {"fps": fps, "video_fps": video_fps, "stride": stride,
            "y0": y0, "y1": y1, "n": idx, "scan_scale": scale,
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
                    config: Config,
                    on_frame: Optional[Callable[[int, int], None]] = None) -> None:
    """Median over each page's frames removes the moving playhead and highlight."""
    y0, y1 = scan_data["y0"], scan_data["y1"]
    # Pages are bounded in scan samples; the file is read in video frames.
    stride = int(scan_data.get("stride", 1))

    # Sample indices per page, gathered in one sequential pass. Seeking to an
    # arbitrary frame makes the decoder restart from the preceding keyframe, so
    # per-page seeking costs far more than simply decoding the file in order.
    wanted: Dict[int, Page] = {}
    for page in pages:
        count = page.last_frame - page.first_frame + 1
        idxs = np.linspace(page.first_frame, page.last_frame,
                           min(count, config.page_composite_samples)).astype(int)
        for i in idxs:
            wanted.setdefault(int(i) * stride, page)
    if not wanted:
        return

    cap = cv2.VideoCapture(video_path)
    last_wanted = max(wanted)
    # Pages are contiguous and ordered, so only one is ever being filled: each is
    # composited and its frames released the moment the next page's first sample
    # arrives. Banking every page's frames until the end instead costs
    # pages x samples x frame size — gigabytes on a five-minute video.
    #
    # One buffer, reused. Allocating and freeing a fresh stack per page churns
    # tens of megabytes at a time, which glibc stops returning to the OS after
    # the first few rounds, so resident memory climbs even though nothing is held.
    buffer: Optional[np.ndarray] = None
    current: Optional[Page] = None
    filled = 0
    idx = 0
    while idx <= last_wanted:
        # Frames nobody asked for still have to be decoded — later frames depend
        # on them — but grab() skips the colour conversion and copy that read()
        # would spend on them, which is most of the per-frame cost.
        page = wanted.get(idx)
        if page is not None:
            ret, frame = cap.read()
            if not ret:
                break
            strip = frame[y0:y1] if y1 > y0 else frame
            if buffer is None:
                buffer = np.empty((config.page_composite_samples,) + strip.shape,
                                  dtype=np.uint8)
            if page is not current:
                if current is not None:
                    _finish_page(current, buffer[:filled], config)
                current, filled = page, 0
            if filled < len(buffer):
                buffer[filled] = strip
                filled += 1
        elif not cap.grab():
            break
        idx += 1
        if on_frame:
            on_frame(min(idx, last_wanted), last_wanted)
    cap.release()
    if current is not None and buffer is not None:
        _finish_page(current, buffer[:filled], config)


def _finish_page(page: Page, stack, config: Config) -> None:
    """Median the page's frames into one clean image, then drop the frames."""
    frames = np.asarray(stack)
    if len(frames) == 0:
        return
    # Sample the probes first: partitioning reorders values per pixel, so
    # afterwards no row of the buffer is a real frame any more.
    probes = [_shrink(frames[i], config)
              for i in np.linspace(0, len(frames) - 1, min(len(frames), 8)).astype(int)]
    # An odd count lets the middle element stand in for the median directly,
    # skipping the float conversion and averaging np.median does for an even one.
    if len(frames) % 2 == 0 and len(frames) > 1:
        frames = frames[:-1]
    middle = len(frames) // 2
    frames.partition(middle, axis=0)  # in place, so no per-page temporary
    composite = frames[middle].copy()
    page.instability = _instability(composite, probes, config)
    # The composite is kept whatever its instability. Rejecting an unstable one
    # double-gates what the classifier already handles: it scores every glyph and
    # declines the ones it cannot match, so a ghosted region yields nothing while
    # the clean interior still reads. Gating here as well threw away whole pages
    # for contamination confined to their margins.
    page.composite = composite


def _shrink(image: np.ndarray, config: Config, factor: int = 2) -> np.ndarray:
    return cv2.resize(image, (max(image.shape[1] // factor, 1),
                              max(image.shape[0] // factor, 1)),
                      interpolation=cv2.INTER_AREA)


def _instability(composite: np.ndarray, probes: List[np.ndarray],
                 config: Config) -> float:
    """How far the page moved while it was on screen, as a Jaccard distance.

    A span that straddles a page turn medians into a ghost of two pages, and this
    says how badly. Jaccard on the ink, not raw pixel disagreement: notation
    covers only a few percent of a page, so a half-ghosted composite still agrees
    on almost every (blank) pixel.

    Reported, not acted on, so it is measured from a handful of frames at reduced
    size. Checking every frame at full resolution cost four times the compositing
    it was describing.
    """
    reference = notation_ink(_shrink(composite, config), config) > 0
    residuals = []
    for probe in probes:
        ink = notation_ink(probe, config) > 0
        union = float((ink | reference).sum())
        if union > 0:
            residuals.append(float((ink ^ reference).sum()) / union)
    return float(np.mean(residuals)) if residuals else 1.0


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


def playhead_bar_starts(scan_data: Dict, config: Config) -> List[float]:
    """Times the cursor jumped back to the left, which is a bar beginning.

    The highlight cannot mark bar lines on its own here. This player keeps the
    sounding measure in a fixed place on screen and moves the notation underneath
    it, so the highlight occupies the same columns in every bar and never changes
    to signal a new one. Splitting on notation change instead fails whenever the
    music repeats a bar: the display then barely differs, which is exactly when
    two bars were run together into one.

    The cursor has no such trouble. It sweeps the measure once and snaps back,
    and that snap is unambiguous however similar the two bars look.
    """
    heads, fps = scan_data["heads"], scan_data["fps"]
    seen = [x for x in heads if x >= 0]
    if len(seen) < 2:
        return []
    # A fraction of the cursor's own travel, so this does not depend on the
    # video's resolution or on how wide a bar is drawn.
    threshold = (max(seen) - min(seen)) * config.playhead_reset_ratio
    if threshold <= 0:
        return []
    starts: List[float] = []
    previous = None
    for i, x in enumerate(heads):
        if x < 0:
            continue
        if previous is not None and previous - x >= threshold:
            starts.append(i / fps)
        previous = x
    return starts


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
