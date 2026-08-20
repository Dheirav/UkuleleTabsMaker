# Next session

Last updated 2026-08-20. `main`, 199 tests pass, 9 commits ahead of `origin/main`.

The audio timing route is merged. No video passes its gate yet, and the reason is
recognition, not timing.

---

## Start here: pick one

### A. Recognition on notation+TAB pages — recommended

The last thing between the reader and the two thirds of tab videos it cannot
touch. Everything else on that path now works: the tab band is found under an
overlay, the tab staff is read rather than the notation above it, the soundtrack
supplies times, and the page supplies bar lines.

A tab drawn over a playthrough is timed by its soundtrack (see *How the audio
route works*), and the route refuses when the notes it times do not sound at the
pitch the tab says. The three overlay videos to hand agree 82%, 73% and 86%
against a threshold of 90%, so all three are refused.

**Success is measurable without hand-labelling anything.** Pitch agreement is
reported by the audio route and needs no truth: the soundtrack is an independent
witness to what the fret numbers say. Raise agreement on `ho_easytabs_perfect`,
`ho_ukealong_greensleeves` or `ho_cheats_wonderful` above 90% and that video
starts producing sheets. Iterate off cached page composites rather than
re-scanning -- a scan is a minute a video and the composites do not change.

Where the remaining disagreement actually is, measured rather than assumed: only
7%, 6% and 1% of it is unexplained. The rest is the instrument still ringing --
the pitch heard at an attack is a note the page shows within one attack either
side. So recognition is already 93% to 99% consistent with the audio, and the
gap to 90% exact agreement may not be closable by reading the page better.

**That makes the threshold itself the open question, and it is a judgement, not
a measurement.** Scoring agreement over a +-1 window instead would pass all
three. It was not done, for two reasons worth knowing before revisiting it:
against a null model a random pitch counts as explained 19% to 30% of the time
at +-1 (26% to 44% at +-3), and chance-corrected the three sit at 0.88 to 0.94 --
straddling Snowman at 0.91, a video known to time badly. On pitch evidence alone
these three cannot be told from one that fails. See *Judging the audio route*.

### A2. A second witness for timing, free

Bar durations. A steady tempo gives bars of near-equal length, so the spread of
bar durations witnesses the note times without any truth: measured as
interquartile range over median, `ho_cheats_wonderful` sits at 0.16 while the
other two sit at 0.57 and 0.58. On `ho_easytabs_perfect` the bar lines are found
cleanly and evenly -- 3 to 4 per system at consistent x -- and the durations
still run 5.88, 8.54, 1.39, 0.73 before settling to a steady 4s, which is the
note times being wrong rather than the bar lines.

Two independent no-truth witnesses on the same question is worth more than one.

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
| bar-duration spread | how even the bars are, IQR over median | note *times* on those same videos |

Current numbers, four labelled clips (316 notes):

    recognised on the composite     98.73% recall / 100% precision
    reached the sheet at all        99.05%
    printed in the right bar        93.04% recall / 92.16% precision
    onsets within 50ms              81%

These rose about a point when the benchmark stopped scoring one video twice.
Nothing was read differently; see *A benchmark that counted a video twice*.

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
4. `page_digits.find_bar_lines` reads the bar lines off the staff -- the columns
   carrying ink from its top line to its bottom -- and they are dated by the
   notes either side of them. Without those a whole song parses as one measure.

**Three gates, and the last is the one that matters.** `audio_diagnosis` refuses
a run that read too little of the tab to judge, then on how much of the tab found
a sound, then on how much of it agreed on pitch. Each catches something the
others do not:

- **Sample size.** Both other figures are ratios. A run that read ten notes out
  of a whole song reported nine agreeing and passed at exactly the 90% bar.
- **Matched share.** Catches a soundtrack that is the original recording rather
  than the instrument: Snowman matched 63% where the good ones match 89% to 98%.
- **Pitch agreement.** Matching alone proves very little -- where the audio holds
  three times as many onsets as the page holds notes, every note finds *a* sound.
  One video matched 100% of its notes while agreeing with 5% of them on pitch.

Measured against timing taken from the highlight, on four videos carrying both:

| video | matched | pitch | jitter | within 50ms |
|---|---|---|---|---|
| Perfect (outputs/) | 95% | 96% | 19ms | 81% |
| Un Poco Loco | 98% | 96% | 18ms | 79% |
| KICKBACK | 89% | 98% | 18ms | 76% |
| USSEWA | 90% | 95% | 25ms | 68% |
| Snowman | 63% | 88% | 321ms | 14% |

Snowman is the failure the matched-share gate catches. KICKBACK sits a constant
109ms out and still scores well, because the number reported is jitter about the
median — a constant offset shifts every note equally and the sheet, which spaces
notes by the gaps between them, cannot show it.

**The emitting path has never run end to end on a real video.** Every video to
hand is refused, so bar lines, alignment and gating are each tested in isolation
and on cached composites, and "a real video produces a real sheet" is still
unproven. First video through the gate, read the sheet before believing it.

---

## Judging the audio route

Neither harness reaches it: `measure_truth` scores glyphs and notes against
hand-labelled truth, and the videos this route serves have none. Two figures
stand in, and both are computable on a video nobody has labelled.

**Pitch agreement** — the fraction of timed notes sounding at the pitch the tab
claims. Reported in `sampling_report.json` as `audio_pitch_agreement`.

**Bar-duration spread** — interquartile range over median. A steady tempo gives
near-equal bars, so an irregular spread witnesses bad note times even where the
bar lines themselves are demonstrably right.

Before loosening the pitch threshold, know what was already measured. Counting a
pitch as agreeing when it matches any note within one attack either side -- which
is fair, since a ukulele rings for seconds -- would pass all three overlay
videos. Against that stands a null model: draw a pitch at random from the
instrument's range and it counts as explained anyway, 19% to 30% of the time at
+-1 and 26% to 44% at +-3, because a chord-dense page shows most of the scale
nearby. Chance-corrected, the three overlay videos score 0.88 to 0.94 and
Snowman -- which times badly, at 321ms jitter -- scores 0.91, sitting among them.

So on pitch evidence alone a video that times well cannot be told from one that
does not, and the strict threshold is what stands between the route and a
confident wrong sheet. It is a judgement about which error costs more, and the
project's answer elsewhere has been that a false accept is worse than a refusal.
A hand-labelled no-marker clip would replace the judgement with a measurement,
and is the only thing that would.

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

## A benchmark that counted a video twice

`benchmark/videos/reference_clip.mp4` and `outputs/In The Poo/video.mp4` are the
same file, byte for byte, and both were registered as clips with their own
verified truth. Between them they were 161 of the benchmark's 379 notes -- 42% of
the headline, one video, its failures weighted double.

The two labellings do not agree with each other, which is the useful part: 18
measures and 98 notes against 10 and 63, over overlapping spans of the same
footage, scoring 89.8% against 85.7% recall. A second opinion on one video is
worth keeping; counting it as a second video is not.

The duplicate now carries `duplicate_of` in `clips.json` and every harness skips
it unless asked for by name, so the labelling survives and can still be scored
on its own. De-duplicating moved the headline about a point in each figure
without a single note being read differently.

**Check `md5sum` before adding a clip.** Videos arrive here twice, under the
title the tool gave them and under whatever they were called when downloaded.

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

**Counting a glyph's holes against the image rather than its ink.** The hole
test that separates 3 from 8 compares a detected glyph to a rendered reference,
and the reference is drawn on a roomy canvas while the glyph is cropped to its
own bounds. A speck threshold taken from the image area therefore means two
different things, and at bold weights it read the reference 8 as having one hole
and declined every real 8. Caught only by testing every font on the candidate
list; DejaVuSans, which the fixtures use, counts correctly either way.

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
- **Only ink on the tab staff is read.** Notation above and lyrics below were
  being read as frets, all landing on string 0.
- **A 3 is told from an 8 by its holes.** Overlap alone cannot: a 3 sits inside
  the 8 template. One video read 20 of 80 glyphs as the eighth fret.
- **Timing from the soundtrack**, with its refusal gates.
- **Bar lines**, so a soundtrack-timed sheet is divided into measures.

## Verified before merging

The five labelled clips are one renderer. The classifier and staff changes touch
every video, so the branch was run against `main` over all 23 songs under
`outputs/` -- same script, both versions, main in a worktree. **All 23 read
identically**: same note counts, same fret and string histograms.

That comparison is worth repeating for any change to `glyphs.py` or
`page_digits.py`, and worth doing right. Two ways it can lie, both of which it
did first time round: `outputs/*/tabs.json` is whenever that song was last run,
not `main`, so it is not a baseline; and comparing note lists positionally counts
tie-order as difference, because `sort(key=(time, string_index))` leaves notes
sharing both in insertion order. Compare histograms, not positions.

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
