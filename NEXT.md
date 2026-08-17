# Open items

Written 2026-08-16, at the end of the AV1 decoding work. Everything below is
either unfinished or something the next person should know before touching the
accuracy numbers.

## Repo state

- `main` is **1 commit ahead of `origin/main`**: `f3d2092` (accuracy harness fix),
  unpushed.
- This file is untracked. Commit it or delete it, but do not leave it as the only
  record.

## Closed, for reference

AV1 sources used to decode to zero frames. Fixed and shipped:

- `fc1b493` pins `opencv-python==4.14.0.94` in `requirements-opencv.txt`, adds a
  zero-frame guard to `sample_frames()`, and adds `tests/fixtures/av1_sample.mp4`.
- `1e44e3b` removes the stale pin from `pyproject.toml`.
- `73fa0f0` ranks download formats by resolution then bitrate rather than by codec.
- `f3d2092` lets `scripts/benchmark.py score` run again.

Verified end to end on a real 1080p AV1 download: identical note sequence to the
H.264 copy of the same song, 110 notes, 22 measures, timing drift 8.7ms max.

**Install now takes two commands.** `pip install -r requirements.txt` alone gives
no OpenCV. An existing venv will not upgrade itself — it keeps 4.10.0.84 and
silently fails on AV1 until someone reinstalls. `test_opencv_build_decodes_av1`
catches it, but only when someone runs pytest.

## 1. verity_minecraft: measure 23 falls into a page-composite gap

Diagnosed 2026-08-16. verity's entire 88.9% is one measure. Every other measure
in the sampled truth reads perfectly; measure 23 (t 24.45-25.00, truth
`(1,0) (2,3) (1,0) (2,0)`) accounts for all 4 deletions and the one uncovered
measure.

**Cause.** Page composites do not cover the whole time a page is on screen:

    page 17: t 23.65-24.30   digits=7
    page 18: t 25.15-25.45   digits=6     <-- 0.85s gap
    measure 23: t 24.45-25.00             <-- lands entirely inside the gap

Page 17 carries measure 23's digits (x=1127 `(1,0)`, 1587 `(2,3)`, 1742 `(1,0)`),
but its span ends at 24.30, before measure 23 begins. Measure 22 spans x 384-1918
and overlaps page 17 in time, so it claims those digits by x-range. Measure 23,
matching no page at all, gets nothing.

**This is not only a scoring artifact.** The two code paths differ:

- `scripts/measure_truth.py:73` `detections_by_measure` matches pages to measures
  by *time overlap*, so measure 23 scores zero.
- `src/parsing/paged_tab.py:128` `notes_from_pages` iterates `page.measures`, so
  it does not hit the gap the same way — but it still assigns by x-range, and the
  emitted tab shows the damage: those three notes come out at t=23.77, 24.17,
  24.30, roughly half a second early, inside measure 22 instead of 23. The fourth
  note `(2,0)` is absent entirely.

**Before fixing, decide the model.** Options, none free:

- Extend each page's span to the next page's start, so no measure falls in a gap.
  Careful: measure 22 (x 384-1918) and measure 23 (x 0-1218) then both overlap
  page 17 and both claim the digits between x 384 and 1218, turning deletions
  into duplicates.
- Attach each measure to the nearest page in time when none overlaps.
- Make per-measure x-ranges disjoint, or discriminate by which staff line a digit
  sits on. `detections_by_measure` has no vertical filter at all: a digit is
  claimed by any measure whose x-range contains it, whatever line it is on.

Any of these moves every clip's numbers, so re-score all five afterwards with
`scripts/measure_truth.py score` — not `benchmark.py`.

## 2. Decide what `scripts/benchmark.py` is for

It scores against **page-keyed** truth. `scripts/measure_truth.py` scores against
**measure-keyed** truth and is the one the README's 99.7%/100% claim comes from
(it reports 98.94%/100% today, so that claim reproduces).

`measure_truth.py`'s own docstring explains that page-keyed truth moves whenever
page segmentation changes, "making it useless for validating the very changes it
needed to guard" — which is why measure-keyed truth exists.

The problem: `benchmark.py score` now runs again, and reports 63.5%/41.9%. Anyone
who runs it will think accuracy collapsed. Either retire the script or say in its
docstring that its scores are segmentation-sensitive and not the headline metric.

Bisect of the page-keyed scores, so nobody repeats it:

| commit | verity R/P | silksong R/P |
|---|---|---|
| `693d59e` baseline | 36.3 / 38.1 | 100 / 100 |
| `42566e1` page_change_threshold 0.30 → 0.10 | 73.4 / 62.3 | 62.5 / 30.1 |
| `c0d65cb` | 74.2 / 58.6 | 76.0 / 26.8 |
| `7ae44c9` harness breaks here | 74.2 / 58.6 | 76.0 / 26.8 |
| `3181418` stop banking page frames | 22.6 / 23.1 | 74.0 / 27.2 |
| HEAD | 22.6 / 23.1 | 74.0 / 27.2 |

Both drops come from commits that change page segmentation, which is exactly what
page-keyed truth cannot measure. Reverting the threshold to 0.30 on current code
moves silksong to 100%/56.5% but drops verity to 12.1%/13.0% — not a fix, just
the metric moving around. Treat these numbers as diagnostic, not as accuracy.

Worth knowing: `42566e1`'s message claims "precision holds at 100%". That was
measured on the measure-keyed harness. The page-keyed harness still worked at
that commit and disagreed. `7ae44c9` then broke the page-keyed harness, and
`3181418` — the very next commit — moved the numbers furthest, with nothing left
running to notice.

## 3. No CI

There are no workflows at all. Nothing runs the tests unless someone remembers.
This matters more than usual now that installing takes two commands, and it is
what makes `tests/fixtures/av1_sample.mp4` worth having. A workflow that runs both
install steps and then pytest would cover it.

## 4. Unguarded captures in `src/vision/paged.py`

Four `VideoCapture` loops without the zero-frame guard that `sample_frames()` now
has (`src/video/sampler.py:116`). No live bug — no current codec trips them, and
the sampler runs first and would raise before those are reached. Consistency work.
