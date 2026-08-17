# Next session

Last updated 2026-08-17. `main` is in sync with `origin/main`, 151 tests pass.

---

## Start here: pick one

### A. Timing for videos with no playback marker — recommended

Two thirds of tab videos on YouTube draw a tab over someone playing and never
mark which measure is sounding (see *Coverage* below). The reader takes all of
its timing from that marker, so it refuses every one of them. This is the only
task that reaches the majority of videos.

**Do not start by reading rhythm out of the notation.** Parsing note heads,
stems and beams is a large job. There is a cheaper framing: the tab already says
*which* notes are played and *in what order*. Only *when* is missing. Audio
onset detection gives a sequence of onset times, and aligning two monotonic
sequences — known notes against detected onsets — is a small, well-understood
problem. No transcription and no rhythm parsing.

It also sidesteps what defeated the last attempt: nothing has to be told apart
from a moving hand, because no visual marker is involved.

Prototype on youtu.be/RsN_OCnpnN4 (Zombie, Ukulele Easy Tabs). It is a clean
notation+TAB band over a playthrough, its page segmentation already works — 6
pages found — and its recognition should be fine. Only timing is absent. Success
is checkable by ear before any harness exists.

Open questions worth settling early: what happens when the video's audio is a
backing track rather than the instrument, and whether page turns give enough
anchoring to keep alignment from drifting over several minutes.

### B. Adaptive highlight, take two

Reaches perhaps 2 more videos in 15. Attempted and reverted — read *Dead ends*
before touching it, because the obvious approaches are already spent.

The unsolved part is not finding the marker. It is telling a marker from a
player's hand once the mask is loose enough to see both. Two ideas untried:

- Require the marker to hold a constant height and vertical position. A box sits
  on the staff; hands wander vertically.
- Confine it to rows where notation ink actually lives, rather than to rows that
  merely look like paper.

### C. Note timing inside the bar

The quality lever for the genre that already works. `reference_clip` places only
92.6% of its notes in the right bar despite its measure boundaries being almost
exactly right, so the error is in note *times*, not bar edges. Look at
`_playhead_times` and the fallback in `src/parsing/paged_tab.py` that places a
note by `frac = (x_center - measure.x0) / span`.

Smallest scope, highest certainty, and `sheet` can judge it directly.

---

## How to judge a change

Three harnesses. They measure different things and are easy to mix up — that
mistake cost a whole investigation this week.

| harness | scores | use it for |
|---|---|---|
| `measure_truth.py score` | glyphs read off page composites | recognition changes |
| `measure_truth.py sheet` | notes that reach `tabs.json` | anything touching attachment, segmentation or timing |
| `timing_truth.py score` | onset accuracy, 36 notes, 2 clips | timing precision only |
| `benchmark.py score` | page-keyed truth — **not the headline metric** | segmentation diagnosis only |

Current numbers, five labelled clips:

    recognised on the composite     98.94% recall / 100% precision
    reached the sheet at all        98.42%
    printed in the right bar        91.82% recall / 92.31% precision
    onsets within 50ms              81%

**`score` cannot see whether a note reaches the sheet.** Two fixes were judged
by it and misjudged: one reverted that should have shipped, one that looked
harmless and was not. If a change touches how measures attach to pages, use
`sheet`.

**`benchmark.py` reports around 63%** and that is not a regression. Its truth is
keyed to page indices, so re-cutting pages moves the score without a single note
being read differently. It prints a warning saying so.

---

## Coverage: what the reader actually handles

17 videos across 12 channels, sampled from three YouTube searches:

| what the video is | count | reader today |
|---|---|---|
| screencast with a warm measure highlight | 3 | **readable** |
| highlight present, different style (orange outline box) | 1 | refused |
| playhead cursor only, no measure highlight | 1 | refused |
| tab drawn over live video, no timing marker at all | **10** | refused |
| not song tabs (how-to-read lessons) | 2 | n/a |

About a fifth of song-tab videos read today. Directional, not precise: the
sample follows YouTube's ranking, and six of the ten no-marker videos come from
two channels.

The split is by **genre, not channel** — two videos from the same channel
differed more from each other than from other channels.

Test videos, with known outcomes:

- read today: youtu.be/21MVI1aOJQI, youtu.be/4BLgpXCS1po
- refused, has a marker the mask cannot see: youtu.be/wmXnzxOcJRI (orange
  outline), youtu.be/oDGo0LDH9UE (cursor only)
- refused, no marker at all: youtu.be/RsN_OCnpnN4 (Zombie)
- out of scope entirely: youtu.be/l1u1OBNyUGI (fretboard grid, no printed fret
  numbers — a different program, not a tuning of this one)

**The refusal gate** (`paged.highlight_diagnosis`) refuses a video it cannot
time instead of emitting a sheet, after one produced 18 notes and a PDF off a
cream blanket that matched the highlight mask on every frame. A passing gate
promises only that there is a moving highlight — never accuracy.

---

## Dead ends — do not repeat

**Adaptive highlight by colour-agnostic masking.** Three attempts, measured on
the orange-box video:

| attempt | median span | travel | verdict |
|---|---|---|---|
| saturation alone | 1184px of 1253 | 0.06 | that is the frame |
| confined to neutral page rows | 1169px | 0.36 | still the frame |
| plus a strict tall-column test (0.40) | 165px | 4.93 | the box, correctly |

The third finds the box — an outline is caught by its two tall vertical sides,
so the column test must be *strict*, which is the opposite of the obvious guess.
It still cannot ship: with it, a video with no highlight at all passes the gate,
because a player's moving hands make tall coloured columns that shift about.

Forward-advance was the obvious separator and fails outright: `reference_clip`,
a good clip, advances forward 52% of the time, and so does the video with no
highlight.

**Raising `page_change_threshold`.** Already at its best value. It trades clips
against each other: 0.10 gives 98.94% overall, 0.20 gives 96.31%, 0.30 gives
93.14%.

**Judging an attachment change on the timing harness alone.** It scores 36
onsets across two clips and sees the cost without the benefit.

**A "notes all on one string" sanity rule.** `reference_clip` legitimately uses
only string 3. It would reject the cleanest clip in the benchmark.

---

## Fixed this week, for context

- **AV1 sources decoded to zero frames.** `opencv-python` pinned to 4.14.0.94;
  earlier releases and 5.0.0.93 are all broken. Verified end to end.
- **Notes lost at page turns.** `segment_pages` holds a guard back from each cut;
  a measure highlighted inside that guard was attached to no page, so its notes
  never reached the sheet. Fixed by tracking the highlight back from the previous
  page. Sheet recall 85.75% → 91.82%, every clip improved.
- **The accuracy harness could not run.** `benchmark.py score` had raised
  `TypeError` on every invocation since `7ae44c9`.

---

## Small items

- **No CI at all.** Nothing runs the tests unless someone remembers. A workflow
  doing both install steps then pytest would cover it.
- **Four unguarded `VideoCapture` loops** in `src/vision/paged.py`, missing the
  zero-frame guard `sample_frames()` has. No live bug.

## Gotcha

**Installing takes two commands.** `pip install -r requirements.txt` alone gives
no OpenCV — `requirements-opencv.txt` installs separately with `--no-deps`,
because the release that decodes AV1 declares a numpy bound that conflicts with
torch's. An existing venv will not upgrade itself; it keeps 4.10.0.84 and fails
silently on AV1 until someone reinstalls. `test_opencv_build_decodes_av1` catches
it, but only when someone runs pytest.
