"""Timing a tab by the sound of it being played.

For a tab drawn over a playthrough there is no highlight and no cursor, but the
soundtrack is the notes themselves. The page says which notes and in what order;
the audio says when something was struck and roughly at what pitch.
"""
import numpy as np
import pytest

from src.app.config import Config
from src.audio.onsets import (estimate_pitch, note_midi, onset_envelope,
                              pick_onsets)
from src.parsing.audio_timing import (align, audio_diagnosis, fill_gaps,
                                      group_attacks)


class Digit:
    """Stands in for a DigitDetection, which needs only these three fields here."""

    def __init__(self, x, string_index, value):
        self.x_center = x
        self.string_index = string_index
        self.value = value
        self.confidence = 1.0


def pluck(midi, at, duration=0.5, rate=22050):
    """A struck string: a hard attack, harmonics, and a decaying tail."""
    hz = 440.0 * 2.0 ** ((midi - 69) / 12.0)
    n = int(duration * rate)
    t = np.arange(n) / rate
    wave = sum(np.sin(2 * np.pi * hz * h * t) / h for h in (1, 2, 3, 4))
    return int(at * rate), (wave * np.exp(-4.0 * t)).astype(np.float32)


def track(notes, seconds=6.0, rate=22050):
    audio = np.zeros(int(seconds * rate), np.float32)
    for midi, at in notes:
        start, wave = pluck(midi, at, rate=rate)
        end = min(start + len(wave), len(audio))
        audio[start:end] += wave[:end - start]
    return audio


# --- reading the soundtrack -------------------------------------------------

def test_every_struck_note_is_heard():
    config = Config()
    played = [(60, 0.5), (64, 1.5), (67, 2.5), (69, 3.5)]
    flux, rate = onset_envelope(track(played), config)
    found = pick_onsets(flux, rate, config)
    for _, when in played:
        assert np.min(np.abs(found - when)) <= 0.05


def test_silence_yields_no_onsets():
    config = Config()
    flux, rate = onset_envelope(np.zeros(22050 * 3, np.float32), config)
    assert len(pick_onsets(flux, rate, config)) == 0


def test_a_quiet_passage_is_not_gone_deaf_to():
    """The threshold follows the music. A fixed one either floods the quiet part
    with onsets that are not there or cannot hear it at all."""
    config = Config()
    audio = track([(60, 0.5), (64, 1.5)])
    quiet = track([(67, 3.5), (69, 4.5)]) * 0.08
    flux, rate = onset_envelope(audio + quiet, config)
    found = pick_onsets(flux, rate, config)
    for when in (3.5, 4.5):
        assert np.min(np.abs(found - when)) <= 0.08


@pytest.mark.parametrize("midi", [60, 64, 67, 69, 72])
def test_the_pitch_of_a_struck_note_is_recovered(midi):
    """A plucked string puts as much energy in its overtones as its fundamental,
    so the tallest peak is often an octave up from the note actually played."""
    config = Config()
    audio = track([(midi, 1.0)])
    assert estimate_pitch(audio, 1.0, config) == midi


def test_the_tab_says_what_pitch_to_expect():
    assert note_midi(0, 0) == 69      # open A string
    assert note_midi(2, 0) == 60      # open C string
    assert note_midi(0, 3) == 72      # third fret of the A string
    assert note_midi(9, 0) is None    # not a string this instrument has


# --- matching the page against the sound ------------------------------------

def test_notes_stacked_at_one_x_are_one_attack():
    """Tab draws a chord vertically. Timing its notes separately would print one
    chord as a run of single notes a few milliseconds apart."""
    config = Config()
    digits = [Digit(100, 0, 0), Digit(102, 1, 2), Digit(400, 2, 3)]
    groups = group_attacks(digits, 1000.0, config)
    assert [len(g) for g in groups] == [2, 1]


def test_a_sound_the_tab_does_not_show_is_stepped_over():
    """The audio holds strums and count-ins. Matching nearest-first would let one
    spurious onset shunt the rest of the song along by one."""
    config = Config()
    attacks = [[Digit(10, 0, 0)], [Digit(50, 0, 2)], [Digit(90, 1, 3)]]
    onsets = [(1.0, note_midi(0, 0)), (1.4, 42),
              (1.5, note_midi(0, 2)), (2.0, note_midi(1, 3))]
    assert align(attacks, onsets, config) == {0: 1.0, 1: 1.5, 2: 2.0}


def test_a_note_the_audio_missed_leaves_the_rest_in_step():
    config = Config()
    attacks = [[Digit(10, 0, 0)], [Digit(50, 0, 2)], [Digit(90, 1, 3)]]
    onsets = [(1.0, note_midi(0, 0)), (2.0, note_midi(1, 3))]
    placed = align(attacks, onsets, config)
    assert placed[0] == 1.0 and placed[2] == 2.0


def test_pitch_decides_between_two_readings():
    """Order alone cannot say which of two onsets is the note; pitch can."""
    config = Config()
    attacks = [[Digit(10, 2, 0)]]        # open C, MIDI 60
    onsets = [(1.0, 75), (1.6, 60)]
    assert align(attacks, onsets, config) == {0: 1.6}


def test_an_unplaced_note_is_still_given_a_time():
    """A note the audio missed is still a note; dropping it loses music the page
    plainly shows."""
    config = Config()
    attacks = [[Digit(0, 0, 0)], [Digit(50, 0, 1)], [Digit(100, 0, 2)]]
    times = fill_gaps(attacks, {0: 10.0, 2: 12.0}, 10.0, 12.0)
    assert times[0] == pytest.approx(10.0)
    assert times[2] == pytest.approx(12.0)
    assert 10.0 < times[1] < 12.0


# --- refusing what it cannot time -------------------------------------------

def test_a_video_with_no_audio_is_refused():
    config = Config()
    reason = audio_diagnosis({"audio_onsets": 0.0, "audio_attacks": 90.0}, config)
    assert reason is not None and "no notes could be heard" in reason


def test_a_backing_track_is_refused():
    """Where the soundtrack is the original recording the onsets are drums and
    voice, and few of the page's notes find one at all."""
    config = Config()
    reason = audio_diagnosis({"audio_onsets": 400.0, "audio_attacks": 724.0,
                              "audio_matched_share": 0.63,
                              "audio_pitch_agreement": 0.88}, config)
    assert reason is not None and "does not play what the tab shows" in reason


def test_matching_everything_at_the_wrong_pitch_is_refused():
    """Where the audio holds three times as many onsets as the page holds notes,
    every note finds *a* sound whether or not it is the right one. One video
    matched 100% of its notes while agreeing with 70% of them on pitch."""
    config = Config()
    reason = audio_diagnosis({"audio_onsets": 323.0, "audio_attacks": 90.0,
                              "audio_matched_share": 1.0,
                              "audio_pitch_agreement": 0.70}, config)
    assert reason is not None and "not the notes on the page" in reason


def test_a_video_the_soundtrack_can_time_is_allowed_through():
    config = Config()
    assert audio_diagnosis({"audio_onsets": 100.0, "audio_attacks": 100.0,
                            "audio_matched_share": 0.95,
                            "audio_pitch_agreement": 0.96}, config) is None


def test_a_missing_decoder_does_not_end_the_run(monkeypatch):
    """This route is reached only after the highlight has been ruled out. A
    machine without ffmpeg should be told the video cannot be timed — which is
    true, and which the caller already knows how to say — not handed a
    traceback."""
    import subprocess
    from src.audio.onsets import load_audio

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    assert len(load_audio("anything.mp4", Config())) == 0


def test_a_video_whose_audio_will_not_decode_is_refused_in_words(monkeypatch):
    import subprocess
    from src.audio.onsets import detect

    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ffmpeg")

    monkeypatch.setattr(subprocess, "run", missing)
    config = Config()
    assert detect("anything.mp4", config) == []
    reason = audio_diagnosis({"audio_onsets": 0.0, "audio_attacks": 90.0}, config)
    assert reason is not None and "soundtrack" in reason


def test_a_run_that_read_almost_nothing_is_refused():
    """Agreement is a ratio and a ratio over a handful of notes is noise. A run
    that read ten notes out of a whole song had already failed at recognition,
    and nine of those ten agreeing on pitch cleared a 90% bar by chance."""
    config = Config()
    reason = audio_diagnosis({"audio_onsets": 323.0, "audio_attacks": 10.0,
                              "audio_matched_share": 1.0,
                              "audio_pitch_agreement": 0.9}, config)
    assert reason is not None and "too little of the tab was read" in reason
