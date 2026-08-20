import argparse
import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Dict, List, Optional

from src.app.config import Config
from src.app.progress import STAGES, Progress, ProgressFn
from src.models.schema import FrameSample, TabSheet
from src.utils.logging import setup_logging
from src.video.downloader import cached_title, download_youtube
from src.video.sampler import sample_frames
from src.vision.region import detect_tab_region
from src.vision.lines import detect_string_lines, detect_bar_lines, consensus_staff, apply_consensus_staff
from src.vision.digits import detect_digits, detect_tab_mode
from src.parsing.reconstruct import reconstruct_notes, reconstruct_chunked_notes
from src.parsing.parser import parse_measures
from src.vision import paged
from src.vision.glyphs import GlyphClassifier
from src.vision.page_digits import collect_sample_glyphs, read_page_detail
from src.parsing.audio_timing import audio_diagnosis, notes_from_audio
from src.parsing.paged_tab import measure_coverage, notes_from_pages
from src.output.json import write_json
from src.output.pdf import write_pdf


logger = logging.getLogger(__name__)


def download_video(url: str, config: Config,
                   progress_cb: Optional[Callable[[float, str], None]] = None):
    return download_youtube(url, config.output_dir, progress_cb)


def _title_from_filename(video_path: str) -> str:
    """A readable name from a file: "sherma_song-final.mp4" -> "Sherma Song Final"."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    words = re.split(r"[\s_\-.]+", stem)
    words = [w for w in words if w]
    if not words:
        return ""
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


def sample_video(video_path: str, config: Config) -> List[FrameSample]:
    frames, _ = sample_frames(video_path, config, return_stats=True)
    return frames


def _extract_structure(frame: FrameSample, config: Config) -> tuple:
    region = detect_tab_region(frame.image, config)
    x, y, w, h = region.bbox
    roi = frame.image[y : y + h, x : x + w]
    return region, detect_string_lines(roi, config)


def _extract_frame(item: tuple, config: Config) -> dict:
    frame, region, string_lines = item
    x, y, w, h = region.bbox
    roi = frame.image[y : y + h, x : x + w]
    bar_lines = detect_bar_lines(roi, config)
    mode = detect_tab_mode(roi, string_lines, config)
    digits = detect_digits(roi, string_lines, config, force_single=(mode == "single"))
    return {
        "timestamp": frame.timestamp,
        "region": region,
        "string_lines": string_lines,
        "bar_lines": bar_lines,
        "digits": digits,
        "mode": mode,
    }


def _run_parallel(func: Callable, items: list, workers: int) -> list:
    if workers > 1 and len(items) > 1:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(func, items))
        except Exception as exc:
            logger.warning("parallel frame extraction failed (%s); falling back to sequential", exc)
    return [func(item) for item in items]


def extract_tab_frames(frames: List[FrameSample], config: Config) -> List[dict]:
    workers = config.num_workers
    if workers <= 0:
        workers = os.cpu_count() or 1

    structures = _run_parallel(partial(_extract_structure, config=config), frames, workers)
    abs_line_lists = [
        [line + region.bbox[1] for line in lines] for region, lines in structures
    ]
    consensus = consensus_staff(abs_line_lists, config) if len(frames) > 1 else []
    if consensus:
        logger.info("using consensus staff lines %s across %d frames", consensus, len(frames))

    items = [
        (frame, region, apply_consensus_staff(lines, region.bbox[1], consensus))
        for frame, (region, lines) in zip(frames, structures)
    ]
    return _run_parallel(partial(_extract_frame, config=config), items, workers)


def reconstruct_sheet(
    tab_frames: List[dict],
    config: Config,
    source: str,
) -> TabSheet:
    modes = [f.get("mode", "multi") for f in tab_frames]
    single = modes and modes.count("single") >= max(1, len(modes) // 2)
    tab_mode = "single" if single else "multi"
    if single:
        reconstruction = reconstruct_chunked_notes(tab_frames, config)
    else:
        reconstruction = reconstruct_notes(tab_frames, config)
    measures = parse_measures(reconstruction.notes, reconstruction.bar_times)
    return TabSheet(
        notes=reconstruction.notes,
        measures=measures,
        metadata={"source_url": source, "tab_mode": tab_mode},
    )


def write_outputs(
    sheet: TabSheet,
    config: Config,
    sampling_stats: Optional[Dict] = None,
) -> Dict[str, str]:
    os.makedirs(config.output_dir, exist_ok=True)
    if sampling_stats:
        report_path = os.path.join(config.output_dir, "sampling_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(sampling_stats, f, indent=2)
    json_path = os.path.join(config.output_dir, "tabs.json")
    pdf_path = os.path.join(config.output_dir, "tabs.pdf")
    write_json(sheet, json_path)
    write_pdf(sheet, config, pdf_path, sheet.metadata.get("title"))
    return {"json": json_path, "pdf": pdf_path}


def run_paged_pipeline(
    video_path: str,
    config: Config,
    source: str,
    progress: Progress,
    content_rows: Optional[tuple] = None,
) -> tuple:
    """Static paged tab: read each page once, take timing from the player's own
    highlight and playhead rather than estimating scroll speed."""
    progress.stage("scan")
    scan_data = paged.scan(
        video_path, config,
        lambda i, total: progress.tick(i / max(total, 1), f"frame {i} of {total}"),
        content_rows)

    # A video that marks nothing on the page can still be timed by the sound of
    # it being played, so a failed highlight is no longer the end of the road --
    # but the audio has to answer for itself further down, and refuses there if
    # what it hears is not what the page shows.
    unreadable = paged.highlight_diagnosis(scan_data, config)
    use_audio = False
    if unreadable:
        if not config.use_audio_timing:
            raise paged.UnreadableVideo(unreadable)
        use_audio = True
        logger.info("no usable highlight; timing from the soundtrack instead")

    pages = paged.segment_pages(scan_data, config)
    duration = scan_data["n"] / scan_data["fps"]
    logger.info("paged mode: %d pages over %.1fs", len(pages), duration)
    progress.stage("pages", f"{len(pages)} pages over {duration:.0f}s")
    paged.composite_pages(
        video_path, pages, scan_data, config,
        lambda i, total: progress.tick(i / max(total, 1),
                                       f"{len(pages)} pages · frame {i} of {total}"))
    if not use_audio:
        paged.attach_measures(pages, scan_data, config)

    progress.stage("read")
    classifier = GlyphClassifier(collect_sample_glyphs(pages, config))
    font_name = os.path.basename(classifier.font_path or "none")
    logger.info("glyph font: %s (fit %.3f)", font_name, classifier.fit)
    read_progress = progress.counter(len(pages))
    for done, page in enumerate(pages, start=1):
        if page.composite is not None:
            page.digits, page.declined = read_page_detail(
                page.composite, classifier, config)
        read_progress(done)
    progress.note(f"font {font_name} · {sum(len(p.digits) for p in pages)} glyphs")

    progress.stage("timing")
    audio_stats: dict = {}
    if use_audio:
        reconstruction, audio_stats = notes_from_audio(pages, video_path, config)
        refused = audio_diagnosis(audio_stats, config)
        if refused:
            raise paged.UnreadableVideo(f"{unreadable}\n\n{refused}")
        logger.info("timed from audio: %d onsets, %.0f%% of the tab matched",
                    int(audio_stats["audio_onsets"]),
                    100.0 * audio_stats["audio_matched_share"])
    else:
        reconstruction = notes_from_pages(pages, scan_data, config)
    measures = parse_measures(reconstruction.notes, reconstruction.bar_times)
    sheet = TabSheet(
        notes=reconstruction.notes,
        measures=measures,
        metadata={
            "source_url": source,
            "tab_mode": "paged",
            "timing": "audio" if use_audio else "highlight",
            "pages": str(len(pages)),
            "font": os.path.basename(classifier.font_path or "none"),
        },
    )
    # Coverage counts highlighted measures that produced notes, so it means
    # nothing on a video timed by its soundtrack -- there are no highlighted
    # measures to count. Reporting it anyway wrote `coverage: 0.0` into the
    # sampling report, which reads as "none of the music was captured" rather
    # than "this is not the question here".
    coverage = {} if use_audio else measure_coverage(pages)
    if coverage:
        logger.info("coverage: %d/%d highlighted measures produced notes (%.1f%%)",
                    int(coverage["measures_with_notes"]),
                    int(coverage["measures_highlighted"]),
                    100.0 * coverage["coverage"])
    progress.note(f"{len(reconstruction.notes)} notes in {len(measures)} measures")
    stats = {
        "mode": "paged",
        "pages": float(len(pages)),
        "frames": float(scan_data["n"]),
        "fps": float(scan_data["fps"]),
        "playhead_frames": float(sum(1 for h in scan_data["heads"] if h >= 0)),
        **({} if use_audio else
           {"measures_tracked": float(sum(len(p.measures) for p in pages))}),
        "glyph_font_fit": float(classifier.fit),
        **audio_stats,
        # Worst page ghosting seen. High means a page boundary landed mid-turn.
        "worst_page_instability": float(max((p.instability for p in pages), default=0.0)),
        "scan_stride": float(scan_data.get("stride", 1)),
        **coverage,
    }
    return sheet, stats


def run_pipeline(
    url: str,
    config: Config,
    progress_cb: Optional[ProgressFn] = None,
    video_path: Optional[str] = None,
) -> TabSheet:
    stages = [key for key, _ in STAGES]
    if video_path:
        stages.remove("download")
    progress = Progress(progress_cb, stages)

    title = ""
    if not video_path:
        progress.stage("download", "contacting YouTube")
        download = download_video(url, config, progress.tick)
        video_path, title = download.path, download.title
    else:
        # A file the caller handed us: the best name available is its own, or
        # the title stored beside it when this video was downloaded earlier.
        title = cached_title(os.path.dirname(os.path.abspath(video_path))) or \
            _title_from_filename(video_path)

    source = url if url else (video_path or "local")
    progress.stage("probe", "checking whether the tab scrolls")
    # Found once and shared: locating the content rows costs 40 seeks, and both
    # the scroll probe and the scan need the same answer. Where the tab is drawn
    # over a live video this is the tab's own band, so neither the scroll probe
    # nor the page signature is reading the player instead of the notation.
    content_rows = paged.find_tab_rows(video_path, config)
    paged_video = paged.is_paged(
        video_path, config,
        lambda i, total: progress.tick(i / max(total, 1), f"probe {i} of {total}"),
        content_rows)

    if paged_video:
        sheet, stats = run_paged_pipeline(video_path, config, source, progress,
                                          content_rows)
        if title:
            sheet.metadata["title"] = title
        progress.stage("write", "txt · pdf · json")
        write_outputs(sheet, config, stats)
        progress.done()
        return sheet

    progress.stage("scan", "scrolling tab")
    frames, sampling_stats = sample_frames(video_path, config, return_stats=True)

    progress.stage("pages", f"{len(frames)} frames")
    tab_frames = extract_tab_frames(frames, config)

    progress.stage("timing")
    sheet = reconstruct_sheet(tab_frames, config, source)
    if title:
        sheet.metadata["title"] = title

    progress.stage("write", "txt · pdf · json")
    write_outputs(sheet, config, sampling_stats)

    progress.done()
    return sheet


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Ukulele tab extractor")
    parser.add_argument("url", nargs="?", default="", help="YouTube URL")
    parser.add_argument("--output", default="./outputs", help="Output directory")
    parser.add_argument("--video-path", default="", help="Use an existing video file")
    parser.add_argument("--save-frames", action="store_true", help="Save sampled frames for debugging")
    parser.add_argument("--workers", type=int, default=0, help="Number of vision worker processes (0 = auto)")
    parser.add_argument("--plain", action="store_true",
                        help="Log lines and JSON instead of the interactive display")
    parser.add_argument("--queue", default="",
                        help="A file of videos, one per line, to read in one go")
    args = parser.parse_args()

    if args.queue:
        from src.app.tui import run_queue_app
        run_queue_app(args.queue, args.output, args.workers, plain=args.plain)
        return

    # No target and a terminal to draw on: run as an app rather than demanding
    # the target be spelled correctly on the command line first time.
    if not args.url and not args.video_path and not args.plain:
        from src.app.tui import run_app
        run_app(args.output, args.workers)
        return

    setup_logging()
    config = Config(output_dir=args.output, num_workers=args.workers)
    if args.save_frames:
        config.save_sampled_frames = True
    if not args.url and not args.video_path:
        raise SystemExit("Provide a YouTube URL or --video-path")
    try:
        sheet = run_pipeline(args.url, config, video_path=args.video_path or None)
    except paged.UnreadableVideo as exc:
        # Say so and stop. Writing a sheet anyway is worse than writing nothing:
        # nobody checks a tab against the video it came from, which is exactly
        # when a confident wrong answer does its damage.
        print(json.dumps({"error": "unreadable", "reason": str(exc)}))
        raise SystemExit(2)
    print(json.dumps({"notes": len(sheet.notes), "measures": len(sheet.measures)}))
