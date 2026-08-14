# Ukulele Tabs Maker

Turn a YouTube ukulele tutorial into a tab sheet you can print and play from.

It reads the notation off the screen — this is computer vision, not audio
transcription — and recovers each note's fret, string and time from what the
player itself shows.

Measured against hand-checked ground truth on five clips (379 notes): **98.9%
recall, 100% precision**, with onsets a median of 17ms from the video's own
cursor. Across a library of 23 songs and 2,145 bars, 8 bars were lost.

## Features
- YouTube ingestion via yt-dlp, preferring H.264 because OpenCV cannot decode
  AV1 on many platforms and reads it as an empty video
- Automatic detection of paged vs scrolling tab videos
- **Paged mode** (tab-player screencasts): page segmentation, per-page
  compositing, and timing taken from the player's own measure highlight and
  cursor — found by its movement rather than its colour, so a renderer that
  draws a pale hairline works as well as one that draws a blue bar
- **Scrolling mode** (legacy): adaptive sampling, Hough line detection, and
  scroll-speed reconstruction
- Digit recognition against font-rendered reference glyphs, with the font
  identified per video
- PDF and JSON output, the sheet drawn as engraved notation
- Read a whole list of videos in one go
- Web UI with basic playback view
- CLI + Dockerized web service

## How paged mode works

Most ukulele tutorials that show tabs are screen recordings of a tab player
(Songsterr and similar). Those do not scroll: the notation is static, the view
turns a page at a time, and a coloured highlight marks the measure being played.

The pipeline exploits that directly:

1. **Mode detection** — phase-correlate consecutive frames. Near-zero horizontal
   drift means the tab is static, so paged mode is used.
2. **Page segmentation** — an edge-map signature per frame ignores the flat
   highlight fill but changes sharply when the notation does, marking page turns.
3. **Compositing** — the median over each page's frames removes the moving
   playhead and highlight, leaving one clean static image per page.
4. **Recognition** — string lines and bar lines are removed morphologically, each
   glyph is isolated to its own connected component, and glyphs are matched
   against reference digits rendered in the font that best fits the video.
   Non-digit ink (the `TAB` clef) falls below the score threshold and is dropped.
5. **Timing** — the measure highlight gives each measure's start and end; a
   playhead, where the player draws one, gives the exact instant each x-position
   is reached. No scroll velocity is estimated, because there is none.

Only notes inside a highlighted measure are emitted, which de-duplicates the
overlap between consecutive pages: each measure is highlighted exactly once.

## Setup

### Local (Python)
1. Install system dependencies:
   - `ffmpeg`
   - `tesseract-ocr`
2. Create a virtual environment and install Python deps:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

### Docker
```bash
docker build -t ukulele-tabs .
docker run -p 8000:8000 ukulele-tabs
```

## CLI Usage
Run it with no arguments to get the interactive app: paste a URL or a video
path at the prompt and watch each stage report what it is doing.
```bash
python main.py
```
```
  Ukulele Tabs  ·  YouTube tab video → tab sheet
  Paste a YouTube URL or a video file path. Enter on its own quits.

  > https://www.youtube.com/watch?v=VIDEO_ID

  ██████████████████████████░░░░░░░░░░░░░░  42%  0:11

  ✓ download video     8.4 MB of 8.4 MB · 6.1 MB/s
  ✓ inspect video      probe 70 of 70
  ▸ scan frames        frame 1171 of 1570
    build pages
    read notation
    work out timing
    write files
```
Each source gets its own directory under `--output`, and a video already
downloaded there is reused instead of fetched again. The prompt accepts Windows
and `\\wsl.localhost\...` paths as pasted — the shell would eat the backslashes,
the prompt does not.

Scripted use takes the target on the command line, which prints log lines and a
JSON summary instead of the live display:
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --output ./outputs
python main.py --video-path ./clip.mp4 --output ./outputs
```
Each song gets its own folder under `--output`, named after the song, holding:
- `tabs.pdf` — the sheet, drawn as notation
- `tabs.json` — every note with its time, string, fret and position on the page
- `video.mp4` and `title.txt`, so a re-run reads the copy already fetched

The title is only known once the video has been fetched, so a run starts in a
working folder and is renamed when the title arrives. An index records where each
source ended up, and a sheet's own record of its source is searched when the
index cannot answer, so the same song is never fetched twice.

Save sampled frames for debugging (scrolling mode only):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --output ./outputs --save-frames
```

## Reading a whole list
```bash
python main.py --queue queue.txt   # copy queue.sample.txt to start
```
One video per line, a URL or a path; blank lines and `#` comments are ignored.
A failure is recorded and stepped over — one video offered only in a codec this
build cannot decode should not cost you the rest of the list — and videos already
read are skipped, so adding a line later costs that line rather than the list.

## Web UI
Start locally:
```bash
python -m src.web.app
```
Open http://localhost:8000

## How the sheet is laid out
Tab carries no note values, so horizontal space is the only thing expressing
rhythm. Every distinct onset gets its own column — notes within
`chord_window_s` sound together and share one — and the gap before a column is
set in proportion to the silence before it, measured in units of the piece's own
quick note (a low percentile of its onset gaps) rather than in seconds. A piece
played at half speed therefore lays out identically, and a long rest opens up
without running off the page (`max_gap_dashes`).

Strings run down the staff the way tab is drawn: A on the top line, G on the
bottom. A note's `string_index` in `tabs.json` is its staff line counting from
the top, so index 0 is the A string.

A fret in brackets is a tie — a note still ringing from the bar before, not one
to pluck — and is not printed as a struck note. `sampling_report.json` counts
those bars separately from bars where nothing was found at all, because only the
second means music was lost. Reported together, a slow piece full of held notes
looked like one bar in six missing when nothing was wrong.

## Notes
- Best results come from clear tutorial videos with high-contrast tab overlays.
- Paged mode runs at roughly 4x real time and holds memory flat in the length of
  the video: one page's sampled frames at a time, not every page's. Sampling is
  capped at `scan_stride_hz` (20/s) because nothing it looks for — a page turn, a
  measure highlight — moves at frame rate.
- The digit recognizer uses a lightweight CNN when weights are present; template matching is the fallback.

## CNN Digit Classifier
Train the lightweight CNN and store weights:
```bash
python scripts/train_digit_cnn.py
```
This saves weights to `./models/digit_cnn.pth`. The pipeline will auto-load them when `use_cnn=True`.

## Example Run
```bash
python main.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --output ./outputs/example
```

## Project Structure
```
src/
  app/        Pipeline config and orchestration
  video/      Downloader + adaptive sampler
  vision/     Region detection, line detection, digits
  parsing/    Temporal reconstruction + parsing
  output/     Text/JSON/PDF output
  web/        Flask UI
  models/     Data models
  utils/      Shared helpers
```

## Accuracy harness

Recognition accuracy is measured against hand-checked ground truth so that
changes to the vision code can be scored instead of eyeballed. Every number in
this README comes from these harnesses; where a claim could not be measured it
is not made.

There are two harnesses, because *which* notes and *when* they sound fail in
different ways and a change can easily improve one while quietly hurting the
other.

### Which notes — `scripts/measure_truth.py`
```bash
python scripts/fetch_benchmark.py                       # download benchmark/clips.json
python scripts/measure_truth.py dump  [clip ...]        # one crop per highlighted measure
python scripts/measure_truth.py stub  [--blank] [--sample N] [clip ...]
python scripts/measure_truth.py score [clip ...]        # against benchmark/measures/
```
Truth is keyed to the measures the player highlighted rather than to page
indices, so it does not move when page segmentation changes, and a measure the
reader never read still counts as missing. Label with `--blank` so the truth is
written blind instead of being a review of the reader's own guesses. `score`
aligns the read and true `(string, fret)` sequences by edit distance, so one
dropped glyph costs one deletion rather than smearing into a run of
substitutions, and it warns for any file still marked unverified.

### When they sound — `scripts/timing_truth.py`
```bash
python scripts/timing_truth.py stub  [--sample N] [clip ...]
python scripts/timing_truth.py dump  [clip ...]              # crops to check truth against
python scripts/timing_truth.py score [--stride-hz N] [clip ...]
```
The reference is the player's blue cursor read at full frame rate: a note sounds
when the cursor crosses it. That is checkable in a single frame — `dump` writes
one crop per truth note, and the cursor should be sitting on the note — and it is
a different visual cue from the measure highlight, so it is not merely restating
what the reader assumed.

`score` reports median and 90th-percentile onset error against that reference,
and `--stride-hz` re-runs the reader at a different scan rate so a sampling
change can be judged rather than guessed at. Alongside it reports a metrical
figure: how close the notes sit to a regular division of the bar, with the
division searched rather than assumed. Bar lines come from the highlight, not the
cursor, so that figure witnesses errors the cursor and reader would otherwise
share — but it is a weak instrument, and the onset error is the number to trust.

Videos are H.264-only: OpenCV cannot reliably decode AV1 on many platforms, and a
benchmark that silently fails to decode is worse than one that refuses to.

`benchmark/videos/`, `pages/`, and `results/` are gitignored; `clips.json` and
`truth/` are the versioned parts.
