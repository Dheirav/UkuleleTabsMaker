# Open items

## 0. Generality: the reader covers one genre of video, not one channel

Tested 2026-08-17 against three videos chosen from outside the benchmark. None
produced a usable tab, and they failed in three different ways:

| video | channel | what it is | result |
|---|---|---|---|
| Zombie | Ukulele Easy Tabs | notation + TAB staff, page-advancing | no highlight in 4316 frames |
| One Summer's Day | salamander | empty fretboard grid, positions light up | no printed fret numbers at all |
| Overtaken | salamander | ASCII tab over a cream blanket | mask matched the blanket, 100% of frames |

Two of the three are squarely in scope and both fail for the same reason: they
are **tutorial videos with a tab drawn on top**, not **tab-player screencasts**.
The second genre is what this reader was built for, and only it draws a playback
highlight — which is not merely how measures are found but the sole timing
source. `scan` derives spans from it, `attach_measures` builds on those, and
`notes_from_pages` iterates them. No highlight is not degraded timing, it is no
notes.

Note the two salamander videos differ from each other more than from other
channels. Generality here is per-genre, not per-channel.

**The genres are now counted.** 17 videos across 12 channels, sampled from three
YouTube searches on 2026-08-17:

| what the video is | count | reader today |
|---|---|---|
| screencast with a warm measure highlight | 3 | **readable** |
| highlight present, different style (orange outline box) | 1 | refused |
| playhead cursor only, no measure highlight | 1 | refused |
| tab drawn over live video, no timing marker at all | **10** | refused |
| not song tabs (how-to-read lessons) | 2 | n/a |

Of the 15 that are actually song tabs, **about 3 — a fifth — can be read today**.
Two thirds carry no playback marker of any kind.

Treat the fraction as directional. The sample follows YouTube's ranking and
three query phrasings, and six of the ten no-marker videos come from just two
channels.

**Cheapest win: make the highlight adaptive.** One sampled video steps an orange
outline box through the music — a real measure highlight, found in 0% of frames,
because the mask wants a warm fill and this is an outline in another hue. The
cursor-only video is the same story. Neither is missing a timing signal; the
colour rule cannot see it. `playhead_by_motion` already solved this one layer
down by keying on what moves rather than what colour it is. Doing the same for
the highlight plausibly takes 3 readable to 5 of 15.

Both are saved as test cases: youtu.be/wmXnzxOcJRI (orange outline) and
youtu.be/oDGo0LDH9UE (cursor only), against youtu.be/21MVI1aOJQI and
youtu.be/4BLgpXCS1po which already read.

**The big one: timing for the no-marker genre.** No highlight work reaches those
ten. They are consistent and cleanly rendered — a notation staff above a TAB
strip, advancing a page at a time — so recognition should do well; only timing is
absent. Most print standard notation, whose note durations are a complete timing
source already on the page. Audio onsets are the alternative.

**The gate was checked against four of these**, none of which it had been tuned
on: both warm-highlight videos read, both others refused, no false accept and no
false refusal.

**Done: the gate.** `paged.highlight_diagnosis` refuses a video it cannot time
rather than emitting a sheet, because Overtaken produced 18 notes and a PDF off
a blanket, and nobody checks a tab against the video it came from. Do not treat
a passing gate as a promise of accuracy: it establishes only that there is a
moving highlight to time against.

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

**Refined 2026-08-17.** The digits are on page 18, not 17. Page 18 reads
`(1,0) (2,3) (1,0) (2,0)` at x 70/530/685/1145 — exactly truth measure 23 — then
`(0,3) (0,0)`, exactly measure 24. Recognition is not the problem: those glyphs
are read correctly and sit in `page.digits`.

They never reach the tab. Page 18 has 6 digits and **zero attached measures**, and
`notes_from_pages` iterates `page.measures`, so the page contributes nothing.
`segment_pages` holds `page_guard_frames` back from each cut so composites never
average a page mid-turn, which leaves page 18 with a 6-frame window against
`page_min_frames = 5`; its measures fall in the guard and are dropped.

**Fixed 2026-08-17** by tracking the highlight *backward* from the previous
page's last frame, so the guard belongs to the incoming page that is already on
screen through it.

This was tried and reverted earlier on the timing harness alone, which was the
wrong call: it measures 36 onsets across two clips and saw only the cost. The
sheet metric measures 379 notes across five and shows the gain:

| | before | after |
|---|---|---|
| sheet recall | 85.75% | **91.82%** |
| sheet precision | 88.08% | **92.31%** |
| printed in the right bar | 87.84% | **93.30%** |
| reached the sheet at all | 97.63% | 98.42% |
| recognition (`score`) | 98.94% | 98.94% — unchanged |
| timing p90 | 60ms | 90ms |
| timing within 50ms | 83% | 81% |

All five clips improve, silksong most (87.3% → 97.3%). The timing cost is real
and concentrated in silksong's p90: notes recovered near a page turn arrive with
less precise onsets than notes read mid-page. A note in the wrong bar is the
worse fault — it is visible in the printed sheet, where a 200ms onset error
inside the correct bar is not — so this trade is taken deliberately. Revisit it
if onset precision starts to matter more than layout.

Tracking *forward* into the following guard was also tried: 68 → 88 notes on
verity with visible duplicates, and no harness moved. Not taken.

**Also ruled out: the threshold.** `page_change_threshold` is already at its best
value; it trades one clip against the other and 0.10 wins:

| threshold | verity | silksong | TOTAL |
|---|---|---|---|
| **0.10 (current)** | 88.9% | 100% | **98.94% / 100%** |
| 0.20 | 100% | 87.3% / 72.2% | 96.31% / 90.80% |
| 0.30 | 100% | 76.4% / 73.7% | 93.14% / 92.17% |

**The measurement gap is now closed.** `measure_truth.py sheet` scores the notes
that reach the sheet, against the same truth `score` uses. Built 2026-08-17,
after both attempts above changed the output visibly while leaving `score`
unmoved at 98.94%.

    recognised on the composite (score)   98.94%
    reached the sheet at all              97.63%
    printed in the right bar (sheet)      85.75%

**This reframes the problem.** Notes are not being lost — 97.6% of them reach the
sheet. They are being printed in the wrong bar: of everything that arrives, only
87.8% lands in its own measure. Recognition is not the weak link and neither is
loss; bar attribution is, and it is worth roughly twelve points.

Per clip, `recall` counts a note only in its own bar and `in bar` is how much of
what arrived was placed correctly:

| clip | recall | in bar |
|---|---|---|
| reference_clip | 83.7% | 87.2% |
| verity_minecraft | 77.8% | 82.4% |
| silksong_sherma | 87.3% | 87.3% |
| danganronpa_kyoko | 94.4% | 94.4% |
| chainsaw_man | 81.0% | 85.0% |

Every clip is affected, including the two that score 100% on recognition. So
this is not verity's bug — verity is where a page-turn made it visible.

Use `sheet` to judge any change to attachment or segmentation; `score` cannot
see them. Its note-to-bar assignment is covered by `tests/test_sheet_metric.py`.

**Options, none free:**

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
