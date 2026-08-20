import cv2
import numpy as np
import pytest

from src.app.config import Config
from src.models.schema import DigitDetection
from src.parsing.paged_tab import notes_from_pages
from src.vision.glyphs import (GlyphClassifier, _holes, _match, build_font_templates,
                               normalize_glyph)
from src.vision.paged import (MeasureSpan, Page, highlight_mask, page_signature,
                              segment_pages, signature_distance)
from src.vision.page_digits import (_on_the_staff, find_string_lines, glyph_components,
                                    read_page, read_page_detail, strip_rules)


FONT = "/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf"
# Fixtures render in a *different* real font than the top candidate, so the
# tests also exercise per-video font selection.
FIXTURE_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
requires_font = pytest.mark.skipif(
    build_font_templates(FONT) is None or build_font_templates(FIXTURE_FONT) is None,
    reason="reference fonts unavailable",
)


def render_page(values, width=1200, height=200, text_extra=()):
    """A synthetic tab page shaped like a real renderer's: digits drawn in a real
    UI font, with each string line broken around its glyphs so nothing touches a rule."""
    from PIL import Image, ImageDraw, ImageFont

    lines = [60, 90, 120, 150]
    pil = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(pil)
    font = ImageFont.truetype(FIXTURE_FONT, 28)
    boxes = []
    for text, string_index, x in [(str(v), s, x) for v, s, x in values] + list(text_extra):
        x0, y0, x1, y1 = draw.textbbox((0, 0), text, font=font)
        top = lines[string_index] - (y1 - y0) // 2 - y0
        draw.text((x, top), text, fill=(20, 20, 20), font=font)
        boxes.append((string_index, x - 4, x + (x1 - x0) + 4))

    img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    for i, y in enumerate(lines):
        gaps = sorted(b[1:] for b in boxes if b[0] == i)
        cursor = 0
        for gx0, gx1 in gaps:
            if gx0 > cursor:
                cv2.line(img, (cursor, y), (min(gx0, width - 1), y), (40, 40, 40), 1)
            cursor = max(cursor, gx1)
        if cursor < width - 1:
            cv2.line(img, (cursor, y), (width - 1, y), (40, 40, 40), 1)
    return img, lines


def test_find_string_lines_recovers_staff():
    img, lines = render_page([])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = find_string_lines(gray, Config())
    assert len(found) == 4
    for expected, got in zip(lines, found):
        assert abs(expected - got) <= 2


@pytest.mark.parametrize("rule_grey", [40, 150, 200, 232])
def test_find_string_lines_handles_pale_staff(rule_grey):
    """Renderers draw the staff anywhere from near-black to pale grey. Missing a
    pale staff is silent and dumps every note onto string 0, so it must not
    depend on an absolute darkness cut."""
    img, lines = render_page([(3, 3, 400)])
    # redraw the rules at the requested lightness
    img[:] = np.where((img == 40).all(axis=2)[..., None], rule_grey, img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    found = find_string_lines(gray, Config())
    assert len(found) == 4, f"staff at grey {rule_grey} not found"
    for expected, got in zip(lines, found):
        assert abs(expected - got) <= 2


@requires_font
def test_notes_are_assigned_to_their_own_string():
    """A collapsed staff silently reads every note as string 0."""
    img, _ = render_page([(5, 0, 200), (3, 1, 400), (7, 2, 600), (1, 3, 800)])
    config = Config()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = glyph_components(strip_rules(gray, config), config)
    classifier = GlyphClassifier([c[4] for c in comps])
    detections = read_page(img, classifier, config)
    assert [(d.string_index, d.value) for d in detections] == [
        (0, 5), (1, 3), (2, 7), (3, 1)]


def test_strip_rules_removes_staff_but_keeps_digits():
    img, _ = render_page([(3, 3, 400)])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = strip_rules(gray, Config())
    # the long horizontal rules are gone, the glyph survives
    assert mask.sum() > 0
    assert (mask > 0).mean(axis=1).max() < 0.5


def test_glyph_components_isolate_pixels():
    img, _ = render_page([(1, 3, 200), (2, 3, 400)])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = glyph_components(strip_rules(gray, Config()), Config())
    assert len(comps) == 2
    for x, y, w, h, glyph in comps:
        assert glyph.shape == (h, w)
        assert glyph.max() == 255


def test_normalize_glyph_is_scale_invariant():
    small = np.zeros((10, 6), np.uint8)
    small[1:9, 2:4] = 255
    big = cv2.resize(small, (24, 40), interpolation=cv2.INTER_NEAREST)
    a, b = normalize_glyph(small), normalize_glyph(big)
    assert a is not None and b is not None
    overlap = (a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum())
    assert overlap > 0.9


@requires_font
def test_classifier_reads_rendered_digits():
    values = [(v, 3, 60 + i * 110) for i, v in enumerate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
    img, _ = render_page(values)
    config = Config()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = glyph_components(strip_rules(gray, config), config)
    classifier = GlyphClassifier([c[4] for c in comps])
    detections = read_page(img, classifier, config)
    assert [d.value for d in detections] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


@requires_font
def test_read_page_merges_two_digit_frets():
    # a renderer draws fret 12 as the single string "12", which segments into
    # two components that must be regrouped
    img, _ = render_page([(12, 3, 300), (5, 3, 600)])
    config = Config()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = glyph_components(strip_rules(gray, config), config)
    assert len(comps) == 3
    classifier = GlyphClassifier([c[4] for c in comps])
    detections = read_page(img, classifier, config)
    assert [d.value for d in detections] == [12, 5]


@requires_font
def test_read_page_rejects_non_digit_glyphs():
    """The 'TAB' clef letters must not be read as frets."""
    # a realistic page: the clef plus a normal run of notes
    img, _ = render_page([(5, 3, 300), (3, 3, 420), (7, 3, 540), (1, 3, 660),
                          (2, 3, 780), (0, 3, 900)],
                         text_extra=[("T", 0, 60), ("A", 1, 60), ("B", 2, 60)])
    config = Config()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    comps = glyph_components(strip_rules(gray, config), config)
    assert len(comps) == 9  # clef letters and digits are all segmented...
    classifier = GlyphClassifier([c[4] for c in comps])
    values = [d.value for d in read_page(img, classifier, config)]
    assert values == [5, 3, 7, 1, 2, 0]  # ...but only the digits survive scoring


def test_highlight_mask_selects_warm_fill():
    img = np.full((40, 60, 3), 245, np.uint8)
    img[:, 20:40] = (185, 250, 245)  # BGR yellow wash
    mask = highlight_mask(img)
    cols = np.where(mask.sum(0) > 20)[0]
    assert cols.min() >= 20 and cols.max() <= 39


def test_page_signature_ignores_highlight_fill():
    """A moving highlight must not read as a page change."""
    img, _ = render_page([(1, 3, 300), (2, 3, 500)])
    tinted = img.copy()
    band = tinted[:, 200:600]
    pale = band.max(axis=2) > 200  # wash the background, leave the ink alone
    band[pale] = (185, 250, 245)
    diff = float(np.mean(page_signature(img) != page_signature(tinted)))
    assert diff < Config().page_change_threshold


def test_segment_pages_splits_on_boundaries():
    config = Config()
    n = 60
    pages = segment_pages({"boundaries": [30], "fps": 30.0, "n": n}, config)
    assert len(pages) == 2
    assert pages[0].first_frame == 0
    assert pages[1].last_frame == n - 1


def test_segment_pages_without_boundaries_is_one_page():
    config = Config()
    pages = segment_pages({"boundaries": [], "fps": 30.0, "n": 90}, config)
    assert len(pages) == 1


def test_segment_pages_legacy_diffs_path():
    """The frame-to-frame path still honours page_change_threshold."""
    config = Config()
    n = 60
    diffs = [0.01] * n
    diffs[30] = config.page_change_threshold + 0.5
    pages = segment_pages({"diffs": diffs, "fps": 30.0, "n": n}, config)
    assert len(pages) == 2


def test_page_signature_separates_pages_far_better_than_noise():
    """A turn must sit well clear of within-page variation, or no single
    threshold can transfer between renderers."""
    config = Config()
    page_a, _ = render_page([(1, 3, 200), (2, 3, 400), (4, 3, 600)])
    # same page, one glyph redrawn a pixel over (encoder noise)
    page_a2, _ = render_page([(1, 3, 201), (2, 3, 400), (4, 3, 600)])
    # a different page: every glyph moves
    page_b, _ = render_page([(5, 3, 260), (7, 3, 480), (0, 3, 700)])

    same = signature_distance(page_signature(page_a, config),
                              page_signature(page_a2, config))
    turn = signature_distance(page_signature(page_a, config),
                              page_signature(page_b, config))
    assert same < config.page_change_threshold
    assert turn > config.page_change_threshold
    assert turn > same * 5


def test_signature_distance_is_zero_for_identical_pages():
    config = Config()
    img, _ = render_page([(3, 3, 300)])
    sig = page_signature(img, config)
    assert signature_distance(sig, sig) == pytest.approx(0.0, abs=1e-6)


def _page_with(digits, measures):
    page = Page(index=0, first_frame=0, last_frame=60, t0=0.0, t1=2.0)
    page.digits = digits
    page.measures = measures
    return page


def test_notes_timed_by_measure_highlight():
    config = Config()
    config.use_playhead = False
    digits = [
        DigitDetection(value=1, bbox=(0, 0, 10, 10), confidence=0.9, string_index=3, x_center=100),
        DigitDetection(value=2, bbox=(0, 0, 10, 10), confidence=0.9, string_index=3, x_center=300),
    ]
    page = _page_with(digits, [MeasureSpan(x0=100, x1=300, t0=4.0, t1=8.0)])
    result = notes_from_pages([page], {"heads": [], "fps": 30.0}, config)
    assert [n.fret for n in result.notes] == [1, 2]
    assert result.notes[0].time == pytest.approx(4.0)
    assert result.notes[1].time == pytest.approx(8.0)
    assert result.bar_times == [4.0]


def test_notes_outside_highlighted_measures_are_dropped():
    """Pages overlap; only what the player actually highlighted is emitted."""
    config = Config()
    config.use_playhead = False
    digits = [
        DigitDetection(value=7, bbox=(0, 0, 10, 10), confidence=0.9, string_index=0, x_center=20),
        DigitDetection(value=1, bbox=(0, 0, 10, 10), confidence=0.9, string_index=0, x_center=150),
    ]
    page = _page_with(digits, [MeasureSpan(x0=100, x1=300, t0=1.0, t1=2.0)])
    result = notes_from_pages([page], {"heads": [], "fps": 30.0}, config)
    assert [n.fret for n in result.notes] == [1]


def test_playhead_timing_preferred_over_interpolation():
    config = Config()
    fps = 10.0
    # cursor sweeps x=0..100 over frames 0..10
    heads = [int(i * 10) for i in range(11)]
    digits = [
        DigitDetection(value=3, bbox=(0, 0, 10, 10), confidence=0.9, string_index=0, x_center=50),
    ]
    page = Page(index=0, first_frame=0, last_frame=10, t0=0.0, t1=1.1)
    page.digits = digits
    page.measures = [MeasureSpan(x0=0, x1=100, t0=0.0, t1=10.0)]
    result = notes_from_pages([page], {"heads": heads, "fps": fps}, config)
    # linear interpolation over the measure would give 5.0s; the cursor says 0.5s
    assert result.notes[0].time == pytest.approx(0.5, abs=0.05)


def test_coverage_counts_highlighted_measures_without_notes():
    """A measure the player highlighted but we read nothing for is lost music."""
    from src.parsing.paged_tab import measure_coverage

    read = Page(index=0, first_frame=0, last_frame=10, t0=0.0, t1=1.0)
    read.measures = [MeasureSpan(0, 100, 0.0, 1.0), MeasureSpan(100, 200, 1.0, 2.0)]
    read.digits = [DigitDetection(value=1, bbox=(0, 0, 8, 8), confidence=0.9,
                                  string_index=0, x_center=50)]
    stats = measure_coverage([read])
    assert stats["measures_highlighted"] == 2
    assert stats["measures_with_notes"] == 1
    assert stats["coverage"] == pytest.approx(0.5)


def test_coverage_sees_measures_on_rejected_pages():
    """A page whose composite was rejected still carries highlight spans, which
    is what makes coverage independent of page segmentation."""
    from src.parsing.paged_tab import measure_coverage

    rejected = Page(index=0, first_frame=0, last_frame=10, t0=0.0, t1=1.0)
    rejected.measures = [MeasureSpan(0, 100, 0.0, 1.0)]
    rejected.digits = []  # composite was dropped as unstable
    stats = measure_coverage([rejected])
    assert stats["measures_highlighted"] == 1
    assert stats["measures_with_notes"] == 0
    assert stats["coverage"] == 0.0
    assert stats["pages_without_digits"] == 1


def test_coverage_is_one_when_every_measure_read():
    from src.parsing.paged_tab import measure_coverage

    page = Page(index=0, first_frame=0, last_frame=10, t0=0.0, t1=1.0)
    page.measures = [MeasureSpan(0, 100, 0.0, 1.0), MeasureSpan(100, 200, 1.0, 2.0)]
    page.digits = [
        DigitDetection(value=1, bbox=(0, 0, 8, 8), confidence=0.9, string_index=0, x_center=50),
        DigitDetection(value=2, bbox=(0, 0, 8, 8), confidence=0.9, string_index=0, x_center=150),
    ]
    assert measure_coverage([page])["coverage"] == pytest.approx(1.0)


def _scan_with_spans(spans, fps=10.0):
    return {"spans": spans, "fps": fps, "n": len(spans)}


def test_track_measures_is_independent_of_page_segmentation():
    """Truth keys off this, so it must not move when pages do."""
    from src.vision.paged import track_measures

    config = Config()
    # highlight sits on one measure, then jumps to the next
    spans = [(0, 200)] * 40 + [(200, 400)] * 40
    measures = track_measures(_scan_with_spans(spans), config)
    assert len(measures) == 2
    assert measures[0].x0 == 0 and measures[0].x1 == 200
    assert measures[1].x0 == 200 and measures[1].x1 == 400
    assert measures[0].t1 == pytest.approx(4.0)


def test_track_measures_spans_a_page_turn():
    """A measure straddling a page turn is one measure, not two. The per-page
    tracker splits it, which is why truth cannot be keyed off pages."""
    from src.vision.paged import attach_measures, track_measures

    config = Config()
    spans = [(0, 200)] * 80
    scan = _scan_with_spans(spans)
    assert len(track_measures(scan, config)) == 1

    # the same highlight, cut across two pages
    a = Page(index=0, first_frame=0, last_frame=39, t0=0.0, t1=4.0)
    b = Page(index=1, first_frame=40, last_frame=79, t0=4.0, t1=8.0)
    attach_measures([a, b], scan, config)
    assert len(a.measures) + len(b.measures) == 2  # one measure, seen as two


def test_track_measures_drops_flickers():
    from src.vision.paged import track_measures

    config = Config()
    spans = [(0, 200)] * 40 + [(500, 700)] * 2 + [(200, 400)] * 40
    measures = track_measures(_scan_with_spans(spans), config)
    assert len(measures) == 2  # the 2-frame flicker is below the duration floor


def test_unstable_composites_are_kept_for_the_classifier_to_judge():
    """Instability is a diagnostic, not a gate. Contamination is usually confined
    to a page's margins while the interior reads perfectly, and the glyph
    classifier already declines what it cannot match — so rejecting whole pages
    here discarded clean music to guard a failure already handled downstream."""
    import inspect
    from src.vision import paged as paged_module

    source = inspect.getsource(paged_module._finish_page)
    assert "page.composite = composite" in source
    assert "config.page_max_instability" not in source


def _write_pages_video(path, pages=6, frames_per_page=8, size=(160, 60)):
    """A tiny video that turns a page every few frames."""
    width, height = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0,
                             (width, height))
    for index in range(pages):
        frame = np.full((height, width, 3), 255, np.uint8)
        cv2.putText(frame, str(index), (10 + index * 20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        for _ in range(frames_per_page):
            writer.write(frame)
    writer.release()
    return pages, frames_per_page


def test_pages_are_composited_as_each_finishes_not_all_at_the_end(tmp_path, monkeypatch):
    """Holding every page's sampled frames until the read ends costs
    pages x samples x frame size — gigabytes on a five-minute video, enough to
    take the machine down. Each page must be composited and freed in flight."""
    from src.vision import paged as paged_module

    path = tmp_path / "pages.mp4"
    count, per_page = _write_pages_video(path)
    config = Config()
    pages = [Page(index=i, first_frame=i * per_page, last_frame=(i + 1) * per_page - 1,
                  t0=0.0, t1=1.0) for i in range(count)]
    scan_data = {"y0": 0, "y1": 60}

    seen_frames, finished = [], []
    real_finish = paged_module._finish_page

    def spy(page, stack, config_):
        finished.append((page.index, seen_frames[-1] if seen_frames else 0))
        real_finish(page, stack, config_)

    monkeypatch.setattr(paged_module, "_finish_page", spy)
    paged_module.composite_pages(str(path), pages, scan_data, config,
                                 lambda i, total: seen_frames.append(i))

    assert all(p.composite is not None for p in pages)
    last_frame = seen_frames[-1]
    # The first page must be done and its frames released while most of the file
    # is still unread, not banked up for a final pass over everything.
    assert finished[0][0] == 0
    assert finished[0][1] < last_frame / 2
    assert [index for index, _ in finished] == list(range(count))


def test_bar_lines_come_from_the_cursor_snapping_back():
    """This player holds the sounding measure in one place on screen, so the
    highlight occupies the same columns in every bar and cannot mark a new one.
    Where the music repeats a bar the display barely changes either, and the two
    bars were being run together. The cursor's snap back to the left is
    unambiguous however alike the bars look."""
    from src.vision.paged import playhead_bar_starts

    config = Config()
    sweep = list(range(200, 900, 20))          # cursor crossing one bar
    heads = sweep * 3                          # three identical bars in a row
    starts = playhead_bar_starts({"heads": heads, "fps": 10.0}, config)
    assert len(starts) == 2                    # two snaps back, after bars 1 and 2
    assert starts[0] == pytest.approx(len(sweep) / 10.0)


def test_a_cursor_that_never_snaps_back_marks_no_bars():
    from src.vision.paged import playhead_bar_starts

    heads = list(range(100, 900, 10))          # one continuous sweep
    assert playhead_bar_starts({"heads": heads, "fps": 10.0}, Config()) == []


def test_cursor_jitter_is_not_read_as_a_bar_line():
    """The threshold is a fraction of the cursor's own travel, so it does not
    depend on the video's resolution or on how wide a bar is drawn."""
    from src.vision.paged import playhead_bar_starts

    heads = []
    for x in range(200, 900, 20):
        heads.extend([x, x - 8, x])            # a few pixels of wobble
    assert playhead_bar_starts({"heads": heads, "fps": 10.0}, Config()) == []


def test_bar_lines_survive_a_video_with_no_cursor():
    from src.vision.paged import playhead_bar_starts

    assert playhead_bar_starts({"heads": [-1] * 50, "fps": 10.0}, Config()) == []


def test_the_letterbox_edge_is_not_read_as_a_string():
    """Where the video's black surround meets the page, the boundary row is dark
    and runs the full width, so it reads as a staff line. That shifted every note
    onto its neighbour's string and produced a fifth string on a four-string
    instrument."""
    config = Config()
    strip = np.full((216, 900), 245, np.uint8)
    strip[0, :] = 0            # letterbox boundary at the top
    strip[215, :] = 0          # and at the bottom
    for y in (51, 89, 126, 163):
        strip[y, :] = 60       # the real staff
    assert find_string_lines(strip, config) == [51, 89, 126, 163]


def test_unevenly_spaced_darkness_is_not_mistaken_for_a_staff():
    """Selection is on spacing, so a stray rule is dropped for sitting at the
    wrong distance. A stray that happens to land a whole string-width away is
    indistinguishable from a real string by this rule and would survive."""
    config = Config()
    strip = np.full((216, 900), 245, np.uint8)
    for y in (51, 89, 126, 163):
        strip[y, :] = 60
    strip[30, :] = 60          # 21px from the staff, against its 38px spacing
    assert find_string_lines(strip, config) == [51, 89, 126, 163]


def test_tab_is_read_rather_than_the_notation_printed_above_it():
    """The genre this reader is being pointed at prints standard notation over
    the tab. A notation staff has five lines to the tab's four, so it is the
    *longer* run — and taking the longest read the melody line as if it were tab
    and handed every note a string the instrument does not have."""
    config = Config()
    strip = np.full((400, 900), 245, np.uint8)
    for y in (79, 97, 114, 132, 149):      # notation staff, tightly spaced
        strip[y, :] = 60
    for y in (262, 289, 315, 341):         # tab staff, one line per string
        strip[y, :] = 60
    assert find_string_lines(strip, config) == [262, 289, 315, 341]


def test_a_clean_staff_is_left_alone():
    config = Config()
    strip = np.full((200, 900), 245, np.uint8)
    for y in (40, 80, 120, 160):
        strip[y, :] = 60
    assert find_string_lines(strip, config) == [40, 80, 120, 160]


def test_a_video_that_decodes_nothing_fails_loudly(tmp_path):
    """AV1 opens and reports a frame count, then fails on every frame read. The
    run used to carry on and write a confidently empty tab sheet."""
    import cv2 as _cv2
    from src.vision import paged as paged_module

    path = tmp_path / "undecodable.mp4"
    path.write_bytes(b"not a video, but a file that exists")

    class _Cap:
        def isOpened(self): return True
        def get(self, prop): return 300.0
        def read(self): return False, None
        def grab(self): return False
        def set(self, *a): return True
        def release(self): return None

    original = _cv2.VideoCapture
    paged_module.cv2.VideoCapture = lambda *a, **k: _Cap()
    try:
        with pytest.raises(RuntimeError, match="No frames could be decoded"):
            paged_module.scan(str(path), Config(), content_rows=(0, 100))
    finally:
        paged_module.cv2.VideoCapture = original


def test_a_moving_cursor_is_found_whatever_colour_it_is():
    """The cursor is the one thing on the page that moves. Keying on colour needs
    a new rule per renderer — this channel draws a saturated blue bar on one song
    and a pale cyan hairline on another, and the hairline fails four of the five
    colour tests written for the bar."""
    from src.vision.paged import playhead_by_motion

    config = Config()
    page = np.full((80, 400, 3), 240, np.uint8)
    before = page.copy()
    before[:, 100:103] = 170          # a faint cursor, nothing like saturated blue
    after = page.copy()
    after[:, 140:143] = 170           # ...one step further on
    x, strength = playhead_by_motion(before, after, config)
    assert 138 <= x <= 144
    assert strength > config.playhead_motion_min_delta


def test_a_page_turn_is_not_mistaken_for_a_cursor():
    """A page turn darkens columns too, but across most of the width."""
    from src.vision.paged import playhead_by_motion

    config = Config()
    before = np.full((80, 400, 3), 240, np.uint8)
    after = np.full((80, 400, 3), 150, np.uint8)   # the whole page changed
    x, _ = playhead_by_motion(before, after, config)
    assert x == -1


def test_a_still_page_reports_no_cursor():
    from src.vision.paged import playhead_by_motion

    page = np.full((80, 400, 3), 240, np.uint8)
    x, _ = playhead_by_motion(page, page.copy(), Config())
    assert x == -1


def _page_with_declined(measures, digits, declined):
    page = Page(index=0, first_frame=0, last_frame=10, t0=0.0, t1=1.0)
    page.measures = measures
    page.digits = digits
    page.declined = declined
    return page


def test_a_bar_holding_only_a_tie_is_not_counted_as_music_lost():
    """A tie is a note still ringing, not one to pluck, so the reader declines it
    on purpose. Counting those as lost blamed the reader for being right, and read
    as one bar in six missing on a slow piece full of held notes."""
    from src.parsing.paged_tab import measure_coverage

    played = MeasureSpan(0, 100, 0.0, 1.0)
    tied = MeasureSpan(200, 300, 1.0, 2.0)
    note = DigitDetection(value=3, bbox=(10, 0, 8, 12), confidence=1.0,
                          string_index=0, x_center=50)
    page = _page_with_declined([played, tied], [note], [250])
    cov = measure_coverage([page])
    assert cov["measures_with_notes"] == 1
    assert cov["measures_held_only"] == 1
    assert cov["measures_lost"] == 0
    assert cov["coverage"] == 1.0        # nothing was lost
    assert cov["note_coverage"] == 0.5   # but only half the bars print a note


def test_a_bar_where_nothing_was_found_still_counts_as_lost():
    """The one kind of empty bar that means something went wrong."""
    from src.parsing.paged_tab import measure_coverage

    played = MeasureSpan(0, 100, 0.0, 1.0)
    empty = MeasureSpan(200, 300, 1.0, 2.0)
    note = DigitDetection(value=3, bbox=(10, 0, 8, 12), confidence=1.0,
                          string_index=0, x_center=50)
    cov = measure_coverage([_page_with_declined([played, empty], [note], [])])
    assert cov["measures_lost"] == 1
    assert cov["coverage"] == 0.5


def test_the_notation_above_the_tab_is_not_read_as_frets():
    """These pages print standard notation over the tab, and the notation area
    carries real digits of its own — a bar number, a time signature. Read whole,
    they all land on string 0, that being the tab line nearest the staff above.
    """
    from PIL import Image, ImageDraw, ImageFont
    config = Config()
    page = Image.new("L", (1200, 400), 245)
    draw = ImageDraw.Draw(page)
    for y in (60, 78, 96, 114, 132):          # notation staff
        draw.line([(0, y), (1199, y)], fill=60)
    for y in (250, 280, 310, 340):            # tab staff, one line per string
        draw.line([(0, y), (1199, y)], fill=60)
    font = ImageFont.truetype(FIXTURE_FONT, 24)
    draw.text((20, 30), "12", fill=20, font=font)      # the bar number
    for x, y in ((200, 250), (400, 280), (600, 310)):
        draw.rectangle([x - 10, y - 14, x + 18, y + 14], fill=245)
        draw.text((x, y - 13), "3", fill=20, font=font)
    strip = cv2.cvtColor(np.array(page), cv2.COLOR_GRAY2BGR)

    classifier = GlyphClassifier(
        [c[4] for c in glyph_components(strip_rules(np.array(page), config), config)])
    detections, _ = read_page_detail(strip, classifier, config)
    assert len(detections) == 3, "the three fret numbers should still be read"
    for detection in detections:
        y_center = detection.bbox[1] + detection.bbox[3] / 2
        assert y_center > 200, f"read notation ink at y={y_center} as fret {detection.value}"


def test_a_fret_number_just_off_the_staff_is_kept():
    """Digits are written on the lines, but the tallest sit proud of the outer
    ones. Cutting to the staff exactly would drop them."""
    config = Config()
    lines = [262, 289, 315, 341]
    just_above = (100, 262 - 12, 14, 20, None)
    assert _on_the_staff([just_above], lines, config) == [just_above]


@requires_font
def test_a_bold_three_is_not_read_as_an_eight():
    """Overlap alone cannot separate them: a 3 is very nearly an 8 with the left
    of each bowl removed, so it sits inside the 8 template and scores well
    against it — and the heavier the stroke, the closer the two get. One video
    read 20 of its 80 glyphs as the eighth fret out of a beginner arrangement
    that has none. Holes are what actually differ."""
    from PIL import Image, ImageDraw, ImageFont
    bold = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    templates = build_font_templates(FONT)
    img = Image.new("L", (96, 128), 0)
    ImageDraw.Draw(img).text((24, 24), "3", fill=255, font=ImageFont.truetype(bold, 64))
    binary = (np.array(img) > 96).astype(np.uint8) * 255
    assert _holes(binary) == 0
    assert _match(binary, templates)[0] == 3


@requires_font
def test_holes_are_counted_as_the_digit_has_them():
    """None for 1 2 3 5 7, one for 0 4 6 9, two for 8."""
    from PIL import Image, ImageDraw, ImageFont
    expected = {0: 1, 1: 0, 2: 0, 3: 0, 4: 1, 5: 0, 6: 1, 7: 0, 8: 2, 9: 1}
    for digit, holes in expected.items():
        img = Image.new("L", (96, 128), 0)
        ImageDraw.Draw(img).text((24, 24), str(digit),
                                 fill=255, font=ImageFont.truetype(FIXTURE_FONT, 64))
        binary = (np.array(img) > 96).astype(np.uint8) * 255
        assert _holes(binary) == holes, digit
