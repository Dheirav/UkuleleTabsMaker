"""Page-keyed accuracy harness for the paged tab reader.

NOT the headline accuracy metric. Use scripts/measure_truth.py for that.

Truth here is keyed to page indices, so any change to page segmentation moves
the labels underneath the score: the same glyphs land on differently cut pages
and register as insertions rather than hits. Lowering page_change_threshold in
42566e1 took silksong from 100% to 30.1% precision that way, without a single
note being read differently. Measure-keyed truth exists precisely because of
this, and reports 98.94% recall / 100% precision on the same reader.

Read the scores below as a segmentation diagnostic. A drop here means pages are
being cut differently, which may be an improvement; it is not evidence that
recognition regressed. NEXT.md records the bisect behind those numbers.

  python scripts/benchmark.py dump   [id ...]   # page composites + montage, for labelling
  python scripts/benchmark.py score  [id ...]   # score detections against benchmark/truth
  python scripts/benchmark.py stub   [id ...]   # write a truth stub to be checked by eye

Recognition is scored against hand-checked ground truth: for each page, the
ordered (string_index, fret) sequence read off the page composite. Detected and
true sequences are aligned by edit distance, so a dropped or spurious glyph does
not smear into a run of substitutions.

Timing is NOT scored. The measure highlight is itself the timing source, so
labelling onsets from it would be circular; the harness reports structural
diagnostics (monotonic, duplicate-free, plausible fret range) instead.
"""
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.app.config import Config
from src.app.main import run_paged_pipeline
from src.app.progress import STAGES, Progress
from src.vision import paged
from src.vision.glyphs import GlyphClassifier
from src.vision.page_digits import collect_sample_glyphs, read_page
from src.parsing.paged_tab import measure_coverage

BENCH = os.path.join(ROOT, "benchmark")
CLIPS = os.path.join(BENCH, "clips.json")
VIDEOS = os.path.join(BENCH, "videos")
TRUTH = os.path.join(BENCH, "truth")
PAGES = os.path.join(BENCH, "pages")
RESULTS = os.path.join(BENCH, "results")


def load_clips(argv: List[str]) -> List[dict]:
    clips = json.load(open(CLIPS))
    wanted = set(argv) if argv else None
    out = []
    for clip in clips:
        if wanted and clip["id"] not in wanted:
            continue
        path = os.path.join(VIDEOS, f"{clip['id']}.mp4")
        if not os.path.exists(path):
            print(f"  skip {clip['id']} (not downloaded)")
            continue
        clip["path"] = path
        out.append(clip)
    return out


def analyse(clip: dict, config: Config):
    """Page composites + per-page detections, without writing tab outputs."""
    scan = paged.scan(clip["path"], config)
    pages = paged.segment_pages(scan, config)
    paged.composite_pages(clip["path"], pages, scan, config)
    paged.attach_measures(pages, scan, config)
    classifier = GlyphClassifier(collect_sample_glyphs(pages, config))
    for page in pages:
        if page.composite is not None:
            page.digits = read_page(page.composite, classifier, config)
    return scan, pages, classifier


def detected_sequences(pages) -> Dict[str, List[List[int]]]:
    return {
        str(p.index): [[d.string_index, d.value]
                       for d in sorted(p.digits, key=lambda d: d.x_center)]
        for p in pages
    }


def align(truth: List[Tuple], hypothesis: List[Tuple]) -> Dict[str, int]:
    """Levenshtein alignment; returns hit / substitution / deletion / insertion."""
    n, m = len(truth), len(hypothesis)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    dp[:, 0] = np.arange(n + 1)
    dp[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if truth[i - 1] == hypothesis[j - 1] else 1
            dp[i, j] = min(dp[i - 1, j] + 1, dp[i, j - 1] + 1, dp[i - 1, j - 1] + cost)
    i, j = n, m
    hits = subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if truth[i - 1] == hypothesis[j - 1] else 1
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


def structural(notes) -> Dict[str, object]:
    times = [n.time for n in notes]
    dup = sum(1 for a, b in zip(notes, notes[1:])
              if abs(a.time - b.time) < 1e-6 and a.fret == b.fret
              and a.string_index == b.string_index)
    return {
        "notes": len(notes),
        "monotonic": all(a <= b for a, b in zip(times, times[1:])),
        "exact_duplicates": dup,
        "fret_min": min((n.fret for n in notes), default=None),
        "fret_max": max((n.fret for n in notes), default=None),
        "strings": sorted({n.string_index for n in notes}),
    }


def cmd_dump(clips, config):
    os.makedirs(PAGES, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    for clip in clips:
        print(f"[{clip['id']}]")
        scan, pages, classifier = analyse(clip, config)
        out_dir = os.path.join(PAGES, clip["id"])
        os.makedirs(out_dir, exist_ok=True)
        rows = []
        for page in pages:
            if page.composite is None:
                continue
            cv2.imwrite(os.path.join(out_dir, f"page_{page.index:03d}.png"), page.composite)
            # Label in a margin, never over the composite. A label drawn on the
            # page hides glyphs at the left edge, and this montage is exactly what
            # the ground truth gets read from.
            scaled = cv2.resize(page.composite, (1280, 240))
            margin = np.full((240, 96, 3), 255, np.uint8)
            cv2.putText(margin, f"p{page.index:03d}", (4, 128),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            rows.append(np.hstack([margin, scaled]))
        if rows:
            cv2.imwrite(os.path.join(out_dir, "montage.png"), np.vstack(rows))
        json.dump(
            {"pages": detected_sequences(pages),
             "font": os.path.basename(classifier.font_path or "none"),
             "font_fit": classifier.fit,
             "playhead_frames": sum(1 for h in scan["heads"] if h >= 0),
             "frames": scan["n"]},
            open(os.path.join(RESULTS, f"{clip['id']}.json"), "w"), indent=2)
        print(f"  {len(pages)} pages -> {out_dir}  font={os.path.basename(classifier.font_path or 'none')}"
              f" fit={classifier.fit:.3f}")


def cmd_stub(clips, config):
    """Seed a truth file from detections. MUST be checked against the montage."""
    os.makedirs(TRUTH, exist_ok=True)
    for clip in clips:
        result_path = os.path.join(RESULTS, f"{clip['id']}.json")
        if not os.path.exists(result_path):
            print(f"  {clip['id']}: run dump first")
            continue
        truth_path = os.path.join(TRUTH, f"{clip['id']}.json")
        if os.path.exists(truth_path):
            print(f"  {clip['id']}: truth exists, not overwriting")
            continue
        pages = json.load(open(result_path))["pages"]
        json.dump({"verified": False, "pages": pages}, open(truth_path, "w"), indent=2)
        print(f"  {clip['id']}: stub written, set verified=true after checking by eye")


def cmd_score(clips, config):
    total = {"hits": 0, "subs": 0, "dels": 0, "ins": 0}
    unverified = []
    print("Page-keyed scores: a segmentation diagnostic, not the accuracy figure.")
    print("Re-cutting pages moves these numbers without changing a single note.")
    print("For recognition accuracy run scripts/measure_truth.py score.\n")
    print(f"{'clip':<22}{'truth':>7}{'hit':>6}{'sub':>5}{'del':>5}{'ins':>5}{'recall':>9}{'prec':>8}")
    for clip in clips:
        truth_path = os.path.join(TRUTH, f"{clip['id']}.json")
        if not os.path.exists(truth_path):
            print(f"{clip['id']:<22}  no truth file")
            continue
        truth_doc = json.load(open(truth_path))
        if not truth_doc.get("verified"):
            unverified.append(clip["id"])
        truth_pages = truth_doc["pages"]

        # Silent progress: the harness prints its own scores, and a second
        # progress display would tear through them.
        sheet, stats = run_paged_pipeline(clip["path"], config, clip["id"],
                                          Progress(None, STAGES))
        _, pages, _ = analyse(clip, config)
        hyp_pages = detected_sequences(pages)

        agg = {"hits": 0, "subs": 0, "dels": 0, "ins": 0}
        for key in sorted(set(truth_pages) | set(hyp_pages), key=int):
            t = [tuple(x) for x in truth_pages.get(key, [])]
            h = [tuple(x) for x in hyp_pages.get(key, [])]
            for k, v in align(t, h).items():
                agg[k] += v
        n_truth = agg["hits"] + agg["subs"] + agg["dels"]
        n_hyp = agg["hits"] + agg["subs"] + agg["ins"]
        recall = agg["hits"] / n_truth if n_truth else 0.0
        precision = agg["hits"] / n_hyp if n_hyp else 0.0
        print(f"{clip['id']:<22}{n_truth:>7}{agg['hits']:>6}{agg['subs']:>5}"
              f"{agg['dels']:>5}{agg['ins']:>5}{recall:>8.1%}{precision:>8.1%}")
        for k in total:
            total[k] += agg[k]
        diag = structural(sheet.notes)
        cov = measure_coverage(pages)
        print(f"{'':<22}  notes={diag['notes']} monotonic={diag['monotonic']} "
              f"dups={diag['exact_duplicates']} frets={diag['fret_min']}-{diag['fret_max']} "
              f"strings={diag['strings']}")
        print(f"{'':<22}  coverage={cov['coverage']:.1%} "
              f"({int(cov['measures_with_notes'])}/{int(cov['measures_highlighted'])} "
              f"highlighted measures read)")

    n_truth = total["hits"] + total["subs"] + total["dels"]
    n_hyp = total["hits"] + total["subs"] + total["ins"]
    if n_truth:
        recall = total["hits"] / n_truth
        precision = total["hits"] / n_hyp if n_hyp else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        print(f"\nTOTAL glyphs={n_truth} recall={recall:.2%} precision={precision:.2%} F1={f1:.2%}")
    if unverified:
        print(f"WARNING unverified truth (stubs, not hand-checked): {', '.join(unverified)}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"dump", "stub", "score"}:
        print(__doc__)
        raise SystemExit(2)
    command, argv = sys.argv[1], sys.argv[2:]
    clips = load_clips(argv)
    if not clips:
        raise SystemExit("no clips available; run scripts/fetch_benchmark.py")
    config = Config()
    {"dump": cmd_dump, "stub": cmd_stub, "score": cmd_score}[command](clips, config)


if __name__ == "__main__":
    main()
