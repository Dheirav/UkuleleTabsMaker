# Next session

Last updated 2026-08-19. Branch `audio-timing`, 177 tests pass.

The audio timing route is built and works. No video passes its gate yet, and the
reason is recognition, not timing.

---

## Start here: pick one

### A. Recognition on notation+TAB pages — recommended

This is the last thing between the reader and the two thirds of tab videos it
cannot touch. Everything else on that path now works.

A tab drawn over a playthrough is timed by its soundtrack (see *How the audio
route works*), and the route refuses when the notes it times do not sound at the
pitch the tab says. The three overlay videos to hand agree 68%, 70% and 85%
against a threshold of 90%, so all three are refused. The gate is right to refuse
them: on one, 28 of 110 glyphs are read as the eighth fret in a beginner
arrangement that plainly has none.

Font fit on these pages runs 0.55 to 0.68, against a good deal better on the
screencasts the reader was built against. Likely causes, none yet investigated:

- The composite is a wide thin strip, and the glyphs are small in it.
- The page carries a notation staff as well as a tab staff, so `strip_rules`
  has beams, stems and note heads to remove that it never saw before.
- These renderers use fonts the candidate list may not hold.

**Success is measurable without hand-labelling anything.** Pitch agreement is
reported by the audio route and needs no truth: the soundtrack is an independent
witness to what the fret numbers say. Raise agreement on
`ho_easytabs_perfect`, `ho_ukealong_greensleeves` and `ho_cheats_wonderful` above
90% and those videos start producing sheets.

### A2. Bar lines for audio-timed sheets

Smaller, and worth doing before any audio-timed sheet is shown to anyone. The
audio route returns no bar times at all, so a whole song prints as one measure.
The highlight used to supply them and there is no highlight here.

The bar lines are drawn on the page — `strip_rules` already finds verticals in
order to remove them. Keeping their x positions and turning them into times
through the same alignment would do it.

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
| audio pitch agreement | fret numbers against the pitches heard | recognition on videos with **no truth at all** |

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

## How the audio route works

For a tab that marks nothing on the page, timing comes from the soundtrack. Two
thirds of tab videos are that shape, and they carry the one thing a screencast
usually does not: audio of the very notes the tab shows, played once, in order.

1. `paged.find_overlay_band` cuts the tab away from the video of the player, by
   per-row motion between consecutive frames. Exactly one still band is required,
   which is what tells an overlay from a scrolling player.
2. `src/audio/onsets.py` finds onsets by spectral flux and reads a pitch at each
   by harmonic salience. ffmpeg and numpy only — no new dependency, which matters
   because opencv-python and torch already disagree about numpy's version.
3. `src/parsing/audio_timing.py` aligns the page's notes against those onsets,
   one page at a time, scoring on pitch and paying for insertions and deletions
   on both sides. The page turn is the anchor that stops a bad run dragging the
   rest of the song out of step.

**Two gates, and the second is the one that matters.** `audio_diagnosis` refuses
on how much of the tab found a sound, and on how much of it agreed on pitch.
Matching alone proves very little: where the audio holds three times as many
onsets as the page holds notes, every note finds *a* sound. One video matched
100% of its notes while agreeing with 5% of them on pitch.

Measured against timing taken from the highlight, on four videos carrying both:

| video | matched | pitch | jitter | within 50ms |
|---|---|---|---|---|
| Perfect (outputs/) | 95% | 96% | 19ms | 81% |
| Un Poco Loco | 98% | 96% | 18ms | 79% |
| KICKBACK | 89% | 98% | 18ms | 76% |
| USSEWA | 90% | 95% | 25ms | 68% |
| Snowman | 63% | 88% | 321ms | 14% |

Snowman is the failure the first gate catches. KICKBACK sits a constant 109ms out
and still scores well, because the number reported is jitter about the median —
a constant offset shifts every note equally and the sheet, which spaces notes by
the gaps between them, cannot show it.

---

## Coverage: what the reader actually handles

17 videos across 12 channels, sampled from three YouTube searches:

| what the video is | count | reader today |
|---|---|---|
| screencast with a warm measure highlight | 3 | **readable** |
| highlight present, different style (orange outline box) | 1 | refused |
| playhead cursor only, no measure highlight | 1 | refused |
| tab drawn over live video, no timing marker at all | **10** | refused — timing now works, recognition does not |
| not song tabs (how-to-read lessons) | 2 | n/a |

About a fifth of song-tab videos read today. Directional, not precise: the
sample follows YouTube's ranking, and six of the ten no-marker videos come from
two channels.

The split is by **genre, not channel** — two videos from the same channel
differed more from each other than from other channels.

Test videos, with known outcomes:

- read today: youtu.be/21MVI1aOJQI, youtu.be/4BLgpXCS1po
- refused, has a marker the mask cannot see: youtu.be/wmXnzxOcJRI (orange
  outline), youtu.be/oDGo0LDH9UE (cursor only, and a scrolling player rather
  than an overlay — `ho_anbu_rolling` locally)
- refused for want of recognition, timing now works: `ho_easytabs_perfect`,
  `ho_ukealong_greensleeves`, `ho_cheats_wonderful` locally
- out of scope entirely: youtu.be/l1u1OBNyUGI (fretboard grid, no printed fret
  numbers — a different program, not a tuning of this one)

**youtu.be/RsN_OCnpnN4 (Zombie) can no longer be fetched.** YouTube answers 403
to the media streams on every rung of the format ladder, and the player-client
overrides do not help. The downloader reads a cookies.txt path from
`YTDLP_COOKIES`, which is the way back in. `ho_easytabs_perfect` is the same
channel and the same shape, and stood in for it.

**The refusal gates.** `paged.highlight_diagnosis` refuses a video it cannot
time instead of emitting a sheet, after one produced 18 notes and a PDF off a
cream blanket that matched the highlight mask on every frame. A video it turns
down now falls through to the audio route rather than stopping there, and
`audio_timing.audio_diagnosis` refuses in turn if the soundtrack cannot account
for the page. Passing either promises only that something was found to time
against — never accuracy.

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

**Judging the audio route by how much of the tab found a sound.** Useless on its
own. A video with three onsets per note matched 100% of them and agreed with 5%
on pitch; the times were arbitrary. Pitch agreement is the signal.

**Timing a screencast from its soundtrack.** The labelled clips carry the
original song, not the tab: verity holds 112 onsets against 36 notes, and
squeezing the detector down to 36 still lands only 4 of 12 truth notes. The audio
route is for playthroughs, and the clips that have truth are exactly the ones it
does not suit.

**Raising `page_change_threshold`.** Already at its best value. It trades clips
against each other: 0.10 gives 98.94% overall, 0.20 gives 96.31%, 0.30 gives
93.14%.

**Judging an attachment change on the timing harness alone.** It scores 36
onsets across two clips and sees the cost without the benefit.

**A "notes all on one string" sanity rule.** `reference_clip` legitimately uses
only string 3. It would reject the cleanest clip in the benchmark.

---

## Landed this week

- **The tab is found under an overlay.** One video went from 3 pages and 17
  glyphs to 7 and 110. Two others read *fewer* glyphs, correctly: what they had
  been reading was the picture of the player.
- **Tab is read rather than the notation above it.** `find_string_lines` took the
  longest run of evenly spaced lines, and a notation staff has five to the tab's
  four. Notes were landing on `string_index` 4, which no ukulele has.
- **Timing from the soundtrack**, with its own refusal gate.

## Fixed earlier, for context

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
