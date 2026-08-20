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
    # --- timing from the soundtrack ---
    # For a tab drawn over a playthrough there is no highlight and no cursor, but
    # the audio is the notes themselves. Measured on two videos of that kind
    # whose timing is otherwise known, onsets land within 15 to 17ms of the note
    # they belong to, against the 50ms the visual reader is scored at.
    audio_sample_rate: int = 22050
    audio_frame: int = 1024
    audio_hop: int = 256          # 11.6ms, well inside the 50ms scoring window
    audio_log_gain: float = 100.0
    audio_onset_delta: float = 0.5
    audio_threshold_window_s: float = 0.30
    audio_min_gap_s: float = 0.06
    # A soprano ukulele runs C4 to about A5; the margin either side leaves room
    # for an octave slip rather than letting one fall off the end of the range.
    audio_min_midi: int = 55
    audio_max_midi: int = 88
    audio_harmonics: int = 5
    audio_pitch_skip_s: float = 0.01
    audio_pitch_window_s: float = 0.12
    audio_pitch_fft: int = 32768
    # How far a note may sit from the onset claimed for it. Wider than the error
    # actually seen, because the cost of leaving a note untimed is worse than the
    # cost of timing it slightly wrong.
    audio_match_window_s: float = 0.40
    audio_pitch_bonus: float = 1.0
    audio_octave_bonus: float = 0.5
    audio_skip_penalty: float = 0.3
    # Notes within this fraction of the page width share an x, so they are one
    # chord and take one time between them.
    audio_chord_x_ratio: float = 0.008
    # A page is turned by hand, a little after or before the music moves on.
    audio_page_margin_s: float = 1.0
    # Where the audio route may be used at all, and when it must give up.
    use_audio_timing: bool = True
    audio_min_onsets: int = 8
    # Agreement is a ratio, and a ratio over a handful of notes is noise. The
    # videos of this kind read 90, 121 and 446 notes; ten means recognition
    # failed, and ten notes can clear a 90% bar by chance.
    audio_min_attacks: int = 20
    audio_min_matched_share: float = 0.75
    # Measured across five videos whose timing is known independently: the four
    # that align well agree on pitch 95% to 98% of the time, the one that does
    # not agrees 88%, and a video whose tab is misread agrees 70%.
    audio_min_pitch_agreement: float = 0.90

    # --- tab drawn over a live video ---
    # Two thirds of tab videos on YouTube lay a tab band over a shot of someone
    # playing. find_content_rows only strips letterboxing, so it keeps the live
    # half, and every page turn is then buried under the player's hands: one such
    # video segmented into 3 pages over 114s and read 17 glyphs.
    #
    # The tab holds still between page turns and the video under it never does,
    # so the split is visible in per-row motion. Measured over the local videos,
    # the moving half of an overlay runs 2.6 to 17.9 while a screencast peaks at
    # 1.25 and usually at 0.0, so the thresholds sit in that gap rather than on a
    # guess. Both sides must be substantial: a screencast has no moving half at
    # all and must keep today's behaviour.
    overlay_probe_pairs: int = 40
    overlay_still_motion: float = 0.5
    overlay_moving_motion: float = 1.5
    overlay_min_still_ratio: float = 0.20
    overlay_min_moving_ratio: float = 0.15
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
    # How far beyond the outermost string line a fret number may sit. Digits are
    # written on the lines, so this only has to cover the tallest of them.
    glyph_staff_margin_ratio: float = 1.0
    # A bar line runs the staff's full height in one column. Measured against
    # the staff rather than the composite, because the composite also holds a
    # notation staff and a slab of video.
    bar_line_coverage: float = 0.9
    bar_line_merge_px: int = 4
    bar_line_edge_margin_ratio: float = 0.02
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
