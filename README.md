# Ukulele Tabs Maker

Extract ukulele tab sheets from YouTube tutorial videos that display scrolling tabs. This is a computer-vision pipeline (not audio processing) and reconstructs time-aligned tabs from on-screen notation.

## Features
- YouTube ingestion via yt-dlp
- Automatic detection of paged vs scrolling tab videos
- **Paged mode** (tab-player screencasts): page segmentation, per-page compositing,
  and timing taken from the player's own measure highlight and playhead
- **Scrolling mode** (legacy): adaptive sampling, Hough line detection, and
  scroll-speed reconstruction
- Digit recognition against font-rendered reference glyphs, with the font
  identified per video
- Text, JSON, and PDF output
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
Outputs are written to the run directory:
- `tabs.txt`
- `tabs.pdf`
- `tabs.json`

Save sampled frames for debugging (scrolling mode only):
```bash
python main.py "https://www.youtube.com/watch?v=VIDEO_ID" --output ./outputs --save-frames
```

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
changes to the vision code can be scored instead of eyeballed.

```bash
python scripts/fetch_benchmark.py            # download clips listed in benchmark/clips.json
python scripts/benchmark.py dump             # page composites + montage.png per clip
python scripts/benchmark.py stub  <clip>     # seed a truth file from detections
python scripts/benchmark.py score            # score against benchmark/truth/
```

Workflow: `dump` writes one composite per page plus a `montage.png`; `stub`
seeds `benchmark/truth/<clip>.json` from the current detections; you then check
each page against the montage, correct the sequences, and set `"verified": true`.
`score` reports results and warns for any truth file still marked unverified, so
a stub can never be mistaken for a checked label.

Scoring aligns the detected and true `(string_index, fret)` sequences per page by
edit distance, so one dropped glyph costs one deletion instead of smearing into a
run of substitutions. Videos are H.264-only: OpenCV cannot reliably decode AV1 on
many platforms, and a benchmark that silently fails to decode is worse than one
that refuses to.

**Timing is deliberately not scored.** The measure highlight is itself the timing
source, so labelling onsets from it would be circular. The harness reports
structural diagnostics instead — monotonicity, duplicate notes, fret range, and
which strings were used.

`benchmark/videos/`, `pages/`, and `results/` are gitignored; `clips.json` and
`truth/` are the versioned parts.
