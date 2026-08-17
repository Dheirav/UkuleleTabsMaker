from dataclasses import dataclass
from typing import Tuple


@dataclass
class Config:
    output_dir: str = "./outputs"
    num_workers: int = 0
    base_sample_fps: float = 2.0
    max_sample_gap_s: float = 1.0
    scene_edge_threshold: float = 0.12
    hist_diff_threshold: float = 0.25
    dedup_hamming_threshold: int = 0
    min_frame_brightness: float = 12.0
    min_keep_ratio: float = 0.08
    save_sampled_frames: bool = False
    sampled_frames_subdir: str = "sampled_frames"
    tab_filter_enabled: bool = True
    tab_min_confidence: float = 0.3
    tab_min_string_lines: int = 4
    tab_region_min_ratio: float = 0.05
    tab_region_max_ratio: float = 0.85
    tab_region_aspect_min: float = 3.0
    string_count: int = 4
    string_cluster_px: int = 6
    string_line_min_span_ratio: float = 0.4
    bar_line_min_span_ratio: float = 0.3
    line_angle_tolerance: float = 10.0
    digit_min_area: int = 30
    digit_max_area: int = 2000
    digit_min_aspect: float = 0.2
    digit_max_aspect: float = 3.0
    digit_min_height: int = 8
    digit_max_height: int = 60
    digit_merge_gap_px: int = 6
    max_fret: int = 15
    digit_line_offset_ratio: float = 0.35
    digit_min_confidence: float = 0.35
    note_min_frames: int = 2
    note_high_confidence: float = 0.6
    tab_band_filter: bool = True
    tab_mode: str = "auto"
    digit_upscale_min_h: int = 16
    single_line_string_index: int = 0
    use_cnn: bool = True
    cnn_weights_path: str = "./models/digit_cnn.pth"
    cnn_confidence_threshold: float = 0.5
    ocr_fallback: bool = True
    time_quantum_s: float = 0.25
    # Notes this close together are one strum and share a printed column.
    chord_window_s: float = 0.05
    # Ceiling on the dashes standing for a rest, so a long silence cannot run a
    # system off the page.
    max_gap_dashes: int = 12
    # How many dashes a short note is worth. One, measured against the recorded
    # sound: a dash per short note reproduces the real gaps between onsets at
    # r=0.997, and every finer setting is worse, not better — the extra width
    # pushes long gaps into max_gap_dashes, where they clip and stop being
    # proportional to anything. 2 gives 0.98, 3 gives 0.93, 6 gives 0.65.
    spacing_resolution: int = 1
    note_dedup_tolerance_s: float = 0.15
    bar_time_cluster_tolerance_s: float = 0.25
    speed_estimation_window_s: float = 3.0
    tuning: Tuple[str, ...] = ("G", "C", "E", "A")

    # --- paged (static) tab reader ---
    paged_mode: str = "auto"  # auto | paged | scrolling
    scan_max_width: int = 960
    paged_motion_probes: int = 30
    paged_motion_pair_frames: int = 3
    paged_max_scroll_px_s: float = 4.0
    page_change_threshold: float = 0.10
    page_profile_bins: int = 512
    page_signature_rows: int = 64
    scan_stride_hz: float = 20.0
    page_confirm_frames: int = 3
    page_cut_merge_frames: int = 4
    page_guard_frames: int = 3
    page_min_frames: int = 5
    # What tells a playback highlight from a warm-coloured background. The mask
    # keys on colour, so a cream wall behind a player's hands matches it on every
    # frame; the difference is that a real highlight steps from measure to
    # measure and a wall never moves. Measured over the labelled clips, which
    # hold 5 to 43 distinct positions and travel several times the highlight's
    # own width, against 1 position and no travel for a background match.
    highlight_min_frame_share: float = 0.05
    highlight_min_positions: int = 3
    highlight_min_travel: float = 0.5
    # Odd, so the median is one of the samples rather than a mean of two. The
    # count only has to outvote the playhead and highlight at each pixel, which
    # is a fraction of the page's frames however many are taken.
    page_composite_samples: int = 21
    page_max_instability: float = 0.20
    page_ink_threshold: int = 170
    ink_max_saturation: int = 60
    string_line_row_ratio: float = 0.5
    string_line_contrast: int = 8
    # Rows this close to the strip's top or bottom are the letterbox boundary,
    # not notation.
    string_line_edge_margin: int = 3
    # How far apart two staff gaps may be and still count as the same staff.
    string_line_spacing_tolerance: float = 0.25
    glyph_min_height: int = 8
    glyph_min_area: int = 20
    glyph_max_height_ratio: float = 0.8
    glyph_merge_gap_ratio: float = 0.45
    glyph_min_score: float = 0.75
    highlight_row_ratio: float = 0.25
    highlight_span_tolerance_px: int = 25
    measure_min_duration_s: float = 0.35
    measure_content_threshold: float = 0.08
    measure_min_width_px: int = 60
    use_playhead: bool = True
    # How far the cursor must jump backwards, as a fraction of its own travel,
    # to count as the start of a new bar rather than detection noise.
    playhead_reset_ratio: float = 0.25
    # Finding the cursor by what moves rather than by its colour. A moving cursor
    # darkens its column by tens of levels; a frame where nothing moved scores
    # about one, so this sits far below the signal and far above the noise.
    playhead_motion_min_delta: float = 8.0
    # A cursor shifts a handful of columns. A page turn shifts most of them, and
    # must not be mistaken for one.
    playhead_motion_max_width: float = 0.05
    # Below this share of samples showing a coloured cursor, the colour rule is
    # taken to have missed this renderer entirely and motion takes over.
    playhead_colour_min_share: float = 0.2
    # Distinct cursor positions a page needs before its notes are timed from the
    # cursor rather than from where they sit in the bar. Counts positions, not
    # frames: forty frames of a resting cursor are one position's worth of
    # evidence. Deliberately permissive — sparse cursor evidence still beats the
    # fallback, and 2 through 6 all measure identically on the timing benchmark,
    # so this sits in the middle of that flat range rather than at the edge of it.
    playhead_min_positions: int = 4
    # How much of a bar's width the cursor must be seen to cross before its times
    # are trusted for that bar. Below this the bar is timed by position instead —
    # all of it, never half and half. Low because the cursor beats the fallback
    # even on partial evidence: 0 and 0.25 measure identically on the timing
    # benchmark, while 0.5 costs a short-barred clip half its bars and sextuples
    # its 90th-percentile error.
    playhead_min_coverage: float = 0.25
