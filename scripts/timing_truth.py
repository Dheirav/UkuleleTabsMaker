"""Timing ground truth for the paged tab reader.

  python scripts/timing_truth.py stub  [--sample N] [id ...]   # seed truth to check
  python scripts/timing_truth.py dump  [id ...]                # crops to check it against
  python scripts/timing_truth.py score [--stride-hz N] [id ...]  # score the reader

The note truth in benchmark/measures/ says *which* notes a clip contains. It says
nothing about *when* they sound, so a change to timing cannot be shown to help or
hurt — which is how a scan-rate change once cost timing precision unnoticed.

The reference here is the player's own playhead, read at full frame rate: a note
sounds at the instant the cursor crosses it. That is the same thing a listener
would mark, it is visible in a single frame so a person can confirm it, and it is
independent of what the reader does — the reader samples the video at a fraction
of frame rate and interpolates the cursor between samples, and both of those are
sources of error this measures.

What it cannot catch: if the playhead detector itself is biased, truth and reader
share the bias. A constant bias shifts every note equally and leaves the rhythm
intact, so the metrical figures below are reported alongside as an independent
witness — they depend on the notes' spacing, not on where the cursor was.
"""
import json
import os
import statistics
import sys
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app.config import Config
from src.parsing.paged_tab import notes_from_pages
from src.vision import paged
from src.vision.glyphs import GlyphClassifier
from src.vision.page_digits import collect_sample_glyphs, read_page

BENCH = os.path.join(ROOT, "benchmark")
CLIPS = os.path.join(BENCH, "clips.json")
VIDEOS = os.path.join(BENCH, "videos")
TRUTH = os.path.join(BENCH, "timing")
CROPS = os.path.join(BENCH, "timing_crops")

MATCH_WINDOW_S = 1.5   # furthest a reader note may sit from its truth note and still be it


def load_clips(argv: List[str]) -> List[dict]:
    clips = json.load(open(CLIPS))
    wanted = set(argv) if argv else None
    out = []
    for clip in clips:
        if wanted and clip["id"] not in wanted:
            continue
        # Same footage as another clip under a second labelling. Scoring both
        # counts one video twice.
        if clip.get("duplicate_of") and not wanted:
            continue
        # A clip may carry its own path: songs the user ran through the tool
        # live under outputs/, and are worth scoring without being copied.
        path = clip.get("path") or os.path.join(VIDEOS, f"{clip['id']}.mp4")
        if not os.path.exists(path):
            print(f"  skip {clip['id']} (not downloaded)")
            continue
        clip["path"] = path
        out.append(clip)
    return out


def read_clip(path: str, config: Config):
    """Full read of a clip: pages, digits and the highlight's measure track."""
    scan = paged.scan(path, config)
    pages = paged.segment_pages(scan, config)
    paged.composite_pages(path, pages, scan, config)
    paged.attach_measures(pages, scan, config)
    classifier = GlyphClassifier(collect_sample_glyphs(pages, config))
    for page in pages:
        if page.composite is not None:
            page.digits = read_page(page.composite, classifier, config)
    return scan, pages


def reference_config() -> Config:
    """Every frame, so the cursor is never interpolated across a gap."""
    config = Config()
    config.scan_stride_hz = 0.0
    return config


def crossings(scan: Dict, pages) -> List[dict]:
    """When the cursor reaches each note, at full frame rate.

    The cursor sweeps a page once, so the frame whose cursor sits nearest a
    note's centre is the instant that note sounds. Ties go to the earliest frame:
    a cursor that pauses on a note should be credited with arriving, not leaving.
    """
    heads, fps = scan["heads"], scan["fps"]
    out: List[dict] = []
    for page in pages:
        window = range(page.first_frame, min(page.last_frame + 1, len(heads)))
        seen = [(i, heads[i]) for i in window if heads[i] >= 0]
        if not seen:
            continue
        for det in page.digits:
            frame, distance = None, None
            for i, x in seen:
                gap = abs(x - det.x_center)
                if distance is None or gap < distance:
                    frame, distance = i, gap
            if frame is None or distance > 40:   # cursor never reached it
                continue
            out.append({
                "time": round(frame / fps, 4),
                "frame": int(frame),
                "x": int(det.x_center),
                "string": int(det.string_index),
                "fret": int(det.value),
                "page": int(page.index),
            })
    out.sort(key=lambda n: n["time"])
    return out


# --- metrical statistics: an independent witness ---------------------------

def _bar_spans(bars) -> List[Tuple[float, float]]:
    """Bar spans, with merged ones split back apart.

    The highlight keeps time to within a few percent, so a tracked measure
    running at twice the median is not a long bar — it is two bars the tracker
    failed to separate, because consecutive bars carrying the same phrase look
    identical to it. Left merged, every note inside one is measured against a
    span twice its true length, and lands nowhere near a beat.
    """
    spans = [(m.t0, m.t1) for m in bars if m.t1 > m.t0]
    if not spans:
        return []
    median = statistics.median(t1 - t0 for t0, t1 in spans)
    out: List[Tuple[float, float]] = []
    for t0, t1 in spans:
        count = max(int(round((t1 - t0) / median)), 1) if median > 0 else 1
        step = (t1 - t0) / count
        out.extend((t0 + i * step, t0 + (i + 1) * step) for i in range(count))
    return out


SUBDIVISIONS = (3, 4, 6, 8, 12, 16)


def metrical(times: List[float], bars) -> Tuple[Optional[float], int]:
    """How close the notes sit to a regular division of the bar.

    Returns the mean distance from the nearest division as a fraction of one,
    with the division that fits best. 0% means every note lands on a beat; 25% is
    what unrelated times would score.

    The division has to be searched, not assumed. A bar of six notes measured
    against a grid of eight scores near random however perfect the timing — which
    is exactly the mistake that made a hand-verified reference look worse than
    the reader it was meant to judge.

    Depends only on where notes sit between bar lines, and the bar lines come
    from the highlight rather than the cursor, so this witnesses errors the
    cursor and the reader would otherwise share.
    """
    spans = _bar_spans(bars)
    positions = []
    for time in times:
        span = next(((a, b) for a, b in spans if a <= time <= b), None)
        if span is not None:
            positions.append((time - span[0]) / (span[1] - span[0]))
    if not positions:
        return None, 0
    # Offsets stay in units of one slot, which makes grids comparable: whatever
    # the division, unrelated times average a quarter of a slot away from it.
    # Measuring as a fraction of the bar instead would rank the finest grid best
    # every time, since a smaller slot is never further from anything.
    scores = {}
    for divisions in SUBDIVISIONS:
        offsets = [abs(p * divisions - round(p * divisions)) for p in positions]
        scores[divisions] = statistics.mean(offsets)
    best = min(scores.values())
    # Prefer the plainest reading: a bar of six also fits a grid of twelve.
    divisions = min(d for d in SUBDIVISIONS if scores[d] <= best + 0.02)
    return scores[divisions], divisions


def cmd_stub(clips, sample: int) -> None:
    os.makedirs(TRUTH, exist_ok=True)
    config = reference_config()
    for clip in clips:
        scan, pages = read_clip(clip["path"], config)
        notes = crossings(scan, pages)
        if not notes:
            print(f"  {clip['id']}: no playhead found, cannot time this clip")
            continue
        if sample and sample < len(notes):
            step = len(notes) / float(sample)
            notes = [notes[int(i * step)] for i in range(sample)]
        path = os.path.join(TRUTH, f"{clip['id']}.json")
        json.dump({"verified": False, "schema": "timing/1",
                   "source": "playhead crossing at full frame rate",
                   "video_fps": scan["fps"], "notes": notes},
                  open(path, "w"), indent=2)
        print(f"  {clip['id']}: {len(notes)} note times -> {path}")


def cmd_dump(clips) -> None:
    """A crop per truth note: the cursor should be sitting on that note."""
    config = reference_config()
    for clip in clips:
        truth_path = os.path.join(TRUTH, f"{clip['id']}.json")
        if not os.path.exists(truth_path):
            print(f"  {clip['id']}: no truth file, run stub first")
            continue
        truth = json.load(open(truth_path))
        out_dir = os.path.join(CROPS, clip["id"])
        os.makedirs(out_dir, exist_ok=True)
        rows = paged.find_content_rows(clip["path"], config)
        cap = cv2.VideoCapture(clip["path"])
        for i, note in enumerate(truth["notes"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, note["frame"])
            ok, frame = cap.read()
            if not ok:
                continue
            y0, y1 = rows
            strip = frame[y0:y1] if y1 > y0 else frame
            x0 = max(note["x"] - 160, 0)
            x1 = min(note["x"] + 160, strip.shape[1])
            crop = strip[:, x0:x1].copy()
            band = np.full((26, crop.shape[1], 3), 255, np.uint8)
            cv2.putText(band, f"{i}: fret {note['fret']} str {note['string']} "
                              f"t={note['time']:.2f}s", (4, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            cv2.imwrite(os.path.join(out_dir, f"{i:03d}.png"),
                        np.vstack([band, crop]))
        cap.release()
        print(f"  {clip['id']}: {len(truth['notes'])} crops -> {out_dir}")


def _match(truth: List[dict], notes) -> Tuple[List[float], int]:
    """Pair each truth note with the reader's nearest note of the same fret.

    Matching on identity rather than order means a reader that drops or invents a
    note is reported as unmatched instead of silently shifting every later pairing
    and turning one fault into a whole clip of apparent timing error.
    """
    used = set()
    errors: List[float] = []
    unmatched = 0
    for want in truth:
        best, best_gap = None, None
        for i, note in enumerate(notes):
            if i in used or note.string_index != want["string"] or note.fret != want["fret"]:
                continue
            gap = abs(note.time - want["time"])
            if gap <= MATCH_WINDOW_S and (best_gap is None or gap < best_gap):
                best, best_gap = i, gap
        if best is None:
            unmatched += 1
            continue
        used.add(best)
        errors.append(best_gap)
    return errors, unmatched


def cmd_score(clips, stride_hz: Optional[float] = None) -> None:
    config = Config()          # the reader as it actually ships
    if stride_hz is not None:  # ...or with a scan rate to be judged
        config.scan_stride_hz = stride_hz
        print(f"scan rate: {'every frame' if stride_hz == 0 else f'{stride_hz:g}/s'}")
    print(f"{'clip':24s} {'notes':>6s} {'matched':>8s} {'median':>8s} {'p90':>8s} "
          f"{'<=50ms':>7s} {'grid:reader':>12s} {'grid:truth':>11s}")
    totals: List[float] = []
    for clip in clips:
        truth_path = os.path.join(TRUTH, f"{clip['id']}.json")
        if not os.path.exists(truth_path):
            print(f"{clip['id']:24s}   no truth file")
            continue
        truth = json.load(open(truth_path))
        scan, pages = read_clip(clip["path"], config)
        notes = notes_from_pages(pages, scan, config).notes
        bars = paged.track_measures(scan, config)

        errors, unmatched = _match(truth["notes"], notes)
        if not errors:
            print(f"{clip['id']:24s}   nothing matched")
            continue
        totals.extend(errors)
        errors.sort()
        median = statistics.median(errors)
        p90 = errors[int(len(errors) * 0.9) - 1]
        close = 100.0 * sum(1 for e in errors if e <= 0.05) / len(errors)
        grid_reader, div_reader = metrical([n.time for n in notes], bars)
        grid_truth, div_truth = metrical([n["time"] for n in truth["notes"]], bars)
        flag = "" if truth.get("verified") else "  (unverified)"
        print(f"{clip['id']:24s} {len(truth['notes']):6d} "
              f"{len(errors):4d}/{len(truth['notes']):<3d} "
              f"{median*1000:7.0f}ms {p90*1000:7.0f}ms {close:6.0f}% "
              f"{(grid_reader or 0)*100:9.1f}%/{div_reader:<2d} "
              f"{(grid_truth or 0)*100:8.1f}%/{div_truth:<2d}{flag}")
    if totals:
        totals.sort()
        print(f"\nTOTAL {len(totals)} notes  median error={statistics.median(totals)*1000:.0f}ms  "
              f"p90={totals[int(len(totals)*0.9)-1]*1000:.0f}ms  "
              f"within 50ms={100.0*sum(1 for e in totals if e <= 0.05)/len(totals):.0f}%")
        print("grid = mean distance from the nearest beat, as a fraction of one beat, "
              "with the bar division that fits best")
        print("       0% is perfectly on the beat, 25% is what unrelated times score")


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        return
    command, rest = argv[0], argv[1:]
    sample = 0
    if "--sample" in rest:
        index = rest.index("--sample")
        sample = int(rest[index + 1])
        rest = rest[:index] + rest[index + 2:]
    stride_hz = None
    if "--stride-hz" in rest:
        index = rest.index("--stride-hz")
        stride_hz = float(rest[index + 1])
        rest = rest[:index] + rest[index + 2:]
    clips = load_clips(rest)
    if command == "stub":
        cmd_stub(clips, sample)
    elif command == "dump":
        cmd_dump(clips)
    elif command == "score":
        cmd_score(clips, stride_hz)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
