"""When notes are struck, and at what pitch, read off the soundtrack.

Two thirds of tab videos draw a tab over a shot of someone playing and mark
nothing on the page, so there is no highlight and no cursor to take timing from.
But those videos carry the one thing a tab-player screencast usually does not:
audio of the very notes the tab shows, played once, in order.

The tab already says *which* notes are played and *in what order*. Only *when* is
missing, and an onset is a far smaller thing to detect than a note is to
transcribe. Pitch is read alongside, not to identify notes -- the tab has already
said what they are -- but to give the alignment something to match on when the
count of onsets and the count of notes disagree.

Nothing here needs a new dependency. ffmpeg already decodes video for the rest of
the pipeline and numpy already carries an FFT, which is the whole of the method.
That matters more than it sounds: opencv-python and torch already disagree about
numpy's version badly enough to need a separate requirements file, and an audio
library would have to be squeezed into the same gap.
"""
import subprocess
from typing import List, Optional, Tuple

import numpy as np

from src.app.config import Config

# Where a note sits on the fretboard, in MIDI numbers. string_index counts staff
# lines from the top, so 0 is the A string; config.tuning runs the other way.
# The G is the high reentrant one a soprano ukulele is normally strung with.
OPEN_STRING_MIDI = (69, 64, 60, 67)


def load_audio(video_path: str, config: Config) -> np.ndarray:
    """Mono PCM, decoded by the ffmpeg the project already requires."""
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", video_path, "-f", "s16le", "-ac", "1",
         "-ar", str(config.audio_sample_rate), "-"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def onset_envelope(audio: np.ndarray, config: Config) -> Tuple[np.ndarray, float]:
    """Positive spectral change per frame, and the rate those frames come at.

    The magnitudes are log-compressed first. A ukulele rings on for a long time
    after it is struck, so a note played over the tail of the last one is a small
    change in absolute terms and an obvious one in proportional terms; on raw
    magnitudes a loud chorus hides every onset in a quiet verse.
    """
    frame, hop = config.audio_frame, config.audio_hop
    if len(audio) < frame:
        return np.zeros(0), config.audio_sample_rate / hop
    count = 1 + (len(audio) - frame) // hop
    window = np.hanning(frame).astype(np.float32)
    index = np.arange(frame)[None, :] + hop * np.arange(count)[:, None]
    spectra = np.abs(np.fft.rfft(audio[index] * window, axis=1))
    spectra = np.log1p(spectra * config.audio_log_gain)
    flux = np.maximum(np.diff(spectra, axis=0), 0.0).sum(axis=1)
    # One leading zero so a flux frame and an audio frame share an index.
    return np.concatenate([[0.0], flux]), config.audio_sample_rate / hop


def pick_onsets(flux: np.ndarray, rate: float, config: Config) -> np.ndarray:
    """Peaks standing clear of a running median, in seconds.

    The threshold has to follow the music rather than sit at one level: the same
    piece is loud in its chorus and quiet in its verse, and a fixed number either
    floods the quiet part with onsets that are not there or goes deaf to it.
    """
    if len(flux) == 0:
        return np.zeros(0)
    scale = np.percentile(flux, 95) or 1.0
    normalised = flux / scale
    half = max(int(config.audio_threshold_window_s * rate / 2), 1)
    padded = np.pad(normalised, half, mode="edge")
    window = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    threshold = np.median(window, axis=1) + config.audio_onset_delta
    gap = max(int(config.audio_min_gap_s * rate), 1)
    peaks: List[int] = []
    for i in range(1, len(normalised) - 1):
        if normalised[i] < threshold[i]:
            continue
        if normalised[i] < normalised[i - 1] or normalised[i] < normalised[i + 1]:
            continue
        if peaks and i - peaks[-1] < gap:
            if normalised[i] > normalised[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return np.array(peaks, dtype=float) / rate


def _candidate_hz(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    midi = np.arange(config.audio_min_midi, config.audio_max_midi + 1)
    return midi, 440.0 * 2.0 ** ((midi.astype(float) - 69.0) / 12.0)


def estimate_pitch(audio: np.ndarray, time: float, config: Config) -> Optional[int]:
    """The MIDI note sounding just after `time`, by harmonic salience.

    A plucked string puts as much energy in its overtones as in its fundamental,
    so the tallest peak in the spectrum is often an octave or a fifth above the
    note actually played. Scoring each candidate by the sum of its own harmonics
    counts that energy towards the right note instead of against it.
    """
    rate = config.audio_sample_rate
    start = int((time + config.audio_pitch_skip_s) * rate)
    length = int(config.audio_pitch_window_s * rate)
    segment = audio[start:start + length]
    if len(segment) < length // 2:
        return None
    segment = segment * np.hanning(len(segment))
    size = config.audio_pitch_fft
    spectrum = np.abs(np.fft.rfft(segment, size))
    midi, hz = _candidate_hz(config)
    bin_hz = rate / size
    scores = np.zeros(len(midi))
    for harmonic in range(1, config.audio_harmonics + 1):
        bins = np.round(hz * harmonic / bin_hz).astype(int)
        valid = bins < len(spectrum) - 3
        # A peak is taken over a few neighbouring bins because a real string is
        # never exactly in tune with the grid, and reading one bin alone would
        # miss it by a fraction of a semitone.
        for i in np.where(valid)[0]:
            lo, hi = max(bins[i] - 3, 0), min(bins[i] + 4, len(spectrum))
            scores[i] += spectrum[lo:hi].max() / harmonic
    if not scores.any():
        return None
    return int(midi[int(scores.argmax())])


def note_midi(string_index: int, fret: int) -> Optional[int]:
    """What the tab says a note should sound as."""
    if not 0 <= string_index < len(OPEN_STRING_MIDI):
        return None
    return OPEN_STRING_MIDI[string_index] + fret


def detect(video_path: str, config: Config) -> List[Tuple[float, Optional[int]]]:
    """Every struck note the soundtrack shows: (time, pitch or None)."""
    audio = load_audio(video_path, config)
    if len(audio) == 0:
        return []
    flux, rate = onset_envelope(audio, config)
    times = pick_onsets(flux, rate, config)
    return [(float(t), estimate_pitch(audio, float(t), config)) for t in times]
