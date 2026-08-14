"""Measure-keyed ground truth for the paged tab reader.

  python scripts/measure_truth.py dump  [id ...]   # one crop per highlighted measure
  python scripts/measure_truth.py stub  [--blank] [--sample N] [id ...]  # seed truth
  python scripts/measure_truth.py score [id ...]   # score against benchmark/measures/

Truth is keyed to the measures the player highlighted, not to page indices. The
highlight is drawn by the player, so that sequence does not move when page
segmentation changes — which page-keyed truth did, making it useless for
validating the very changes it needed to guard.

It also closes a blind spot. Page-keyed truth was built from pages the reader
kept, so a measure it never read could not register as a miss. Here every
highlighted measure is labelled, from a raw frame when no composite exists, so
missing music is counted as missing.
"""
import json
import os
import random
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app.config import Config
from src.vision import paged
from src.vision.glyphs import GlyphClassifier
from src.vision.page_digits import collect_sample_glyphs, find_string_lines, read_page

BENCH = os.path.join(ROOT, "benchmark")
CLIPS = os.path.join(BENCH, "clips.json")
VIDEOS = os.path.join(BENCH, "videos")
TRUTH = os.path.join(BENCH, "measures")
CROPS = os.path.join(BENCH, "measure_crops")


def load_clips(argv: List[str]) -> List[dict]:
    clips = json.load(open(CLIPS))
    wanted = set(argv) if argv else None
    out = []
    for clip in clips:
        if wanted and clip["id"] not in wanted:
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


def analyse(clip: dict, config: Config):
    scan = paged.scan(clip["path"], config)
    pages = paged.segment_pages(scan, config)
    paged.composite_pages(clip["path"], pages, scan, config)
    paged.attach_measures(pages, scan, config)
    classifier = GlyphClassifier(collect_sample_glyphs(pages, config))
    for page in pages:
        if page.composite is not None:
            page.digits = read_page(page.composite, classifier, config)
    measures = paged.track_measures(scan, config)
    return scan, pages, classifier, measures


def detections_by_measure(measures, pages) -> List[List[List[int]]]:
    """What the reader produced for each highlighted measure."""
    out = []
    for measure in measures:
        found: List[Tuple[int, List[int]]] = []
        for page in pages:
            if not page.digits:
                continue
            if page.t1 <= measure.t0 or page.t0 >= measure.t1:
                continue  # page was not on screen during this measure
            for det in page.digits:
                if measure.x0 <= det.x_center <= measure.x1:
                    found.append((det.x_center, [det.string_index, det.value]))
        out.append([pair for _, pair in sorted(found, key=lambda f: f[0])])
    return out


def _page_at(pages, measure):
    for page in pages:
        if page.composite is not None and page.t0 <= measure.t0 < page.t1:
            return page
    return None


def _measure_composite(video_path, scan, measure):
    """Median of the frames this measure was sounding over.

    The cursor sweeps this very span, so medianing across it takes the cursor
    out. A single frame leaves the cursor sitting on top of a digit, and a digit
    hidden behind the cursor is exactly the one a labeller gets wrong — which
    would write truth that agrees with nothing.
    """
    y0, y1 = scan["y0"], scan["y1"]
    video_fps = scan.get("video_fps", scan["fps"])
    cap = cv2.VideoCapture(video_path)
    frames = []
    for t in np.linspace(measure.t0 + 0.05, max(measure.t1 - 0.05, measure.t0 + 0.06), 15):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * video_fps))
        ok, frame = cap.read()
        if ok:
            frames.append(frame[y0:y1] if y1 > y0 else frame)
    cap.release()
    if not frames:
        return None
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def _label_strings(crop, config):
    """Draw a named guide on each staff line, so which string a digit sits on is
    read off the picture rather than judged by eye."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    lines = find_string_lines(gray, config)
    crop = cv2.copyMakeBorder(crop, 0, 0, 46, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
    names = list(reversed(config.tuning))
    for k, y in enumerate(lines[:len(names)]):
        cv2.line(crop, (0, y), (44, y), (0, 140, 255), 1)
        cv2.putText(crop, f"{k}:{names[k]}", (1, y + 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 90, 200), 1)
    return crop


def crop_for(video_path, scan, pages, measure, config):
    """The measure's slice, with the cursor medianed away and the strings named.

    Taken from the measure's own frames rather than from the page composite: a
    page can span several measures, and its composite is the median of all of
    them, so a page that turned mid-measure ghosts exactly the notes in dispute.
    """
    page = _page_at(pages, measure)
    source = _measure_composite(video_path, scan, measure)
    if source is None:
        if page is None:
            return None
        source = page.composite
    pad = 40
    x0 = max(int(measure.x0) - pad, 0)
    x1 = min(int(measure.x1) + pad, source.shape[1])
    if x1 <= x0:
        return None
    crop = source[:, x0:x1].copy()
    # Mark the measure's own boundaries. Padding pulls in neighbouring notes, and
    # judging the edge from a faint highlight tint is how mislabelled truth gets
    # written; only what lies between these two lines belongs to this measure.
    for edge in (int(measure.x0) - x0, int(measure.x1) - x0):
        if 0 <= edge < crop.shape[1]:
            cv2.line(crop, (edge, 0), (edge, crop.shape[0]), (200, 0, 200), 2)
    return _label_strings(crop, config), page is not None


def cmd_dump(clips, config):
    os.makedirs(CROPS, exist_ok=True)
    for clip in clips:
        print(f"[{clip['id']}]")
        scan, pages, classifier, measures = analyse(clip, config)
        out_dir = os.path.join(CROPS, clip["id"])
        os.makedirs(out_dir, exist_ok=True)
        detected = detections_by_measure(measures, pages)
        rows, meta = [], []
        for i, measure in enumerate(measures):
            got = crop_for(clip["path"], scan, pages, measure, config)
            has_composite = False
            if got is not None:
                crop, has_composite = got
                cv2.imwrite(os.path.join(out_dir, f"m{i:03d}.png"), crop)
                scaled = cv2.resize(crop, (1000, 170))
                margin = np.full((170, 150, 3), 255, np.uint8)
                cv2.putText(margin, f"m{i:03d}", (6, 74),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 2)
                cv2.putText(margin, "no composite" if not has_composite else "",
                            (6, 104), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                rows.append(np.hstack([margin, scaled]))
            meta.append({"index": i, "t0": round(measure.t0, 2), "t1": round(measure.t1, 2),
                         "x0": measure.x0, "x1": measure.x1,
                         "from_composite": has_composite,
                         "detected": detected[i]})
        for start in range(0, len(rows), 12):
            chunk = rows[start:start + 12]
            cv2.imwrite(os.path.join(out_dir, f"sheet_{start // 12:02d}.png"),
                        np.vstack(chunk))
        json.dump({"measures": meta}, open(os.path.join(out_dir, "measures.json"), "w"),
                  indent=2)
        no_comp = sum(1 for m in meta if not m["from_composite"])
        read = sum(1 for m in meta if m["detected"])
        print(f"  {len(measures)} measures, {read} read, {no_comp} without a composite"
              f" -> {out_dir}")


def cmd_stub(clips, config, blank=False, sample=0):
    """Seed a truth file.

    Seeding from detections turns labelling into a review of the reader's own
    answer. That is biased in the direction that matters least: a misread note is
    visible on the page and gets caught, but a note the reader never emitted
    leaves nothing on screen to prompt you. Almost every error found so far has
    been a deletion, which is exactly the case that review is worst at catching.

    --blank writes empty measures instead, so notes are transcribed from the crops
    with no sight of what the system produced. Required for held-out clips, where
    the labels have to be independent evidence rather than a graded homework.
    """
    os.makedirs(TRUTH, exist_ok=True)
    for clip in clips:
        src = os.path.join(CROPS, clip["id"], "measures.json")
        if not os.path.exists(src):
            print(f"  {clip['id']}: run dump first")
            continue
        dst = os.path.join(TRUTH, f"{clip['id']}.json")
        if os.path.exists(dst):
            print(f"  {clip['id']}: truth exists, not overwriting")
            continue
        if clip.get("heldout") and not blank:
            print(f"  {clip['id']}: held out — use 'stub --blank', labels must not be "
                  f"seeded from detections")
            continue
        meta = json.load(open(src))["measures"]
        if sample and sample < len(meta):
            # A random sample estimates accuracy as well as an exhaustive pass, at a
            # fraction of the labelling. It also keeps the fixtures a scattered
            # sample of each piece rather than a complete transcription of it.
            rng = random.Random(f"{clip['id']}/{sample}")  # stable across re-runs
            chosen = sorted(rng.sample(range(len(meta)), sample))
            meta = [meta[i] for i in chosen]
        json.dump({"verified": False, "schema": "measures/1",
                   "blind": bool(blank), "heldout": bool(clip.get("heldout")),
                   "sampled": bool(sample), "sample_size": len(meta),
                   "measures": [{"index": m["index"], "t0": m["t0"], "t1": m["t1"],
                                 "notes": [] if blank else m["detected"]} for m in meta]},
                  open(dst, "w"), indent=2)
        how = "blank (blind)" if blank else "seeded from detections"
        scope = f"sample of {len(meta)}" if sample else f"all {len(meta)}"
        print(f"  {clip['id']}: {scope} measures, {how}")


def align(truth: List[Tuple], hyp: List[Tuple]) -> Dict[str, int]:
    n, m = len(truth), len(hyp)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if truth[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    i, j, hits, subs, dels, ins = n, m, 0, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if truth[i - 1] == hyp[j - 1] else 1
            if dp[i, j] == dp[i - 1, j - 1] + cost:
                hits, subs = (hits + 1, subs) if cost == 0 else (hits, subs + 1)
                i, j = i - 1, j - 1
                continue
        if i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
            continue
        ins += 1
        j -= 1
    return {"hits": hits, "subs": subs, "dels": dels, "ins": ins}


def cmd_score(clips, config):
    total = {"hits": 0, "subs": 0, "dels": 0, "ins": 0}
    measures_total = measures_read = 0
    unverified = []
    print(f"{'clip':<22}{'notes':>7}{'hit':>6}{'sub':>5}{'del':>5}{'ins':>5}"
          f"{'recall':>9}{'prec':>7}{'meas':>7}{'cover':>8}")
    for clip in clips:
        path = os.path.join(TRUTH, f"{clip['id']}.json")
        if not os.path.exists(path):
            print(f"{clip['id']:<22}  no truth file")
            continue
        doc = json.load(open(path))
        if not doc.get("verified"):
            unverified.append(clip["id"])

        _, pages, _, measures = analyse(clip, config)
        detected = detections_by_measure(measures, pages)

        agg = {"hits": 0, "subs": 0, "dels": 0, "ins": 0}
        covered = 0
        truth_measures = doc["measures"]
        for truth_measure in truth_measures:
            # Key on the measure's own index, not its position in the file. A
            # sampled truth file holds measures 3, 4, 9, ... and comparing its
            # first entry against the first measure read scores the whole clip
            # against the wrong music — as near zero as makes no difference,
            # which reads exactly like a reader that has completely failed.
            index = truth_measure["index"]
            t = [tuple(x) for x in truth_measure["notes"]]
            h = [tuple(x) for x in (detected[index] if index < len(detected) else [])]
            if t and h:
                covered += 1
            for key, val in align(t, h).items():
                agg[key] += val
        n_truth = agg["hits"] + agg["subs"] + agg["dels"]
        n_hyp = agg["hits"] + agg["subs"] + agg["ins"]
        recall = agg["hits"] / n_truth if n_truth else 0.0
        precision = agg["hits"] / n_hyp if n_hyp else 0.0
        with_notes = sum(1 for m in truth_measures if m["notes"])
        cover = covered / with_notes if with_notes else 0.0
        print(f"{clip['id']:<22}{n_truth:>7}{agg['hits']:>6}{agg['subs']:>5}"
              f"{agg['dels']:>5}{agg['ins']:>5}{recall:>8.1%}{precision:>7.1%}"
              f"{with_notes:>7}{cover:>8.1%}")
        for key in total:
            total[key] += agg[key]
        measures_total += with_notes
        measures_read += covered

    n_truth = total["hits"] + total["subs"] + total["dels"]
    n_hyp = total["hits"] + total["subs"] + total["ins"]
    if n_truth:
        recall = total["hits"] / n_truth
        precision = total["hits"] / n_hyp if n_hyp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        cover = measures_read / measures_total if measures_total else 0.0
        print(f"\nTOTAL notes={n_truth} recall={recall:.2%} precision={precision:.2%} "
              f"F1={f1:.2%} | measures={measures_total} coverage={cover:.2%}")
    if unverified:
        print(f"WARNING unverified truth (stubs, not hand-checked): {', '.join(unverified)}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"dump", "stub", "score"}:
        print(__doc__)
        raise SystemExit(2)
    command, argv = sys.argv[1], sys.argv[2:]
    blank = "--blank" in argv
    sample = 0
    if "--sample" in argv:
        i = argv.index("--sample")
        sample = int(argv[i + 1])
        argv = argv[:i] + argv[i + 2:]
    argv = [a for a in argv if a != "--blank"]
    clips = load_clips(argv)
    if not clips:
        raise SystemExit("no clips available; run scripts/fetch_benchmark.py")
    if command == "stub":
        cmd_stub(clips, Config(), blank=blank, sample=sample)
    else:
        {"dump": cmd_dump, "score": cmd_score}[command](clips, Config())


if __name__ == "__main__":
    main()
