"""Give a page of notes its times by matching them to what was played.

The page says which notes sound and in what order; the soundtrack says when
something was struck and roughly at what pitch. Neither is complete on its own --
the page has no clock, and the audio does not know a strum from a note the tab
omits -- so the two are aligned against each other.

Aligning is done a page at a time rather than over the whole video. A page turn
is a hard anchor: what is drawn on a page is played while that page is up. That
keeps a run of wrong matches from dragging the rest of the song out of step,
which is the failure that would otherwise make this useless over several minutes.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.app.config import Config
from src.audio.onsets import note_midi
from src.parsing.paged_tab import _cluster
from src.vision.page_digits import find_bar_lines, find_string_lines

Onset = Tuple[float, Optional[int]]


def group_attacks(digits: Sequence, page_width: float, config: Config) -> List[List]:
    """Notes struck together, in the order they are played.

    Tab stacks a chord vertically at one x, so notes sharing an x are one attack
    and must be given one time between them -- timing them separately would print
    a chord as a run of single notes a few milliseconds apart.
    """
    if not digits:
        return []
    tolerance = max(page_width * config.audio_chord_x_ratio, 1.0)
    ordered = sorted(digits, key=lambda d: d.x_center)
    groups: List[List] = [[ordered[0]]]
    for digit in ordered[1:]:
        if digit.x_center - groups[-1][-1].x_center <= tolerance:
            groups[-1].append(digit)
        else:
            groups.append([digit])
    return groups


def _match_score(attack: Sequence, onset: Onset, config: Config) -> float:
    """How well a struck sound answers to a group of notes on the page."""
    pitch = onset[1]
    if pitch is None:
        return 0.0
    wanted = {midi for midi in (note_midi(d.string_index, d.value) for d in attack)
              if midi is not None}
    if not wanted:
        return 0.0
    if pitch in wanted:
        return config.audio_pitch_bonus
    # An octave out is the overtone series being read one rung up, not a
    # different note; it is still evidence this is the right attack.
    if any((pitch - w) % 12 == 0 for w in wanted):
        return config.audio_octave_bonus
    return 0.0


def align(attacks: List[List], onsets: List[Onset],
          config: Config) -> Dict[int, float]:
    """Which onset belongs to which attack, keeping both in their own order.

    A plain nearest-onset rule cannot do this: the audio holds strums, count-ins
    and notes the tab leaves out, so the two sequences differ in length and every
    spurious onset would shunt the rest of the song along by one. Insertions and
    deletions are allowed on both sides and paid for, and pitch decides which
    reading is worth paying for.
    """
    n, m = len(attacks), len(onsets)
    if n == 0 or m == 0:
        return {}
    penalty = config.audio_skip_penalty
    best = np.full((n + 1, m + 1), -np.inf)
    best[0, :] = -penalty * np.arange(m + 1)
    best[:, 0] = -penalty * np.arange(n + 1)
    for i in range(1, n + 1):
        score_row = [_match_score(attacks[i - 1], onsets[j - 1], config)
                     for j in range(1, m + 1)]
        for j in range(1, m + 1):
            best[i, j] = max(best[i - 1, j - 1] + score_row[j - 1],
                             best[i - 1, j] - penalty,
                             best[i, j - 1] - penalty)
    times: Dict[int, float] = {}
    i, j = n, m
    while i > 0 and j > 0:
        diagonal = best[i - 1, j - 1] + _match_score(attacks[i - 1], onsets[j - 1], config)
        if best[i, j] == diagonal:
            times[i - 1] = onsets[j - 1][0]
            i, j = i - 1, j - 1
        elif best[i, j] == best[i - 1, j] - penalty:
            i -= 1
        else:
            j -= 1
    return times


def fill_gaps(attacks: List[List], times: Dict[int, float],
              t0: float, t1: float) -> List[float]:
    """Times for the attacks the alignment could not place.

    Spread between the matched attacks either side, in proportion to how far
    across the page each one sits. A note the audio missed is still a note, and
    dropping it would lose music the page plainly shows.
    """
    if not attacks:
        return []
    xs = [float(np.mean([d.x_center for d in group])) for group in attacks]
    known = sorted(times)
    if not known:
        span = max(xs[-1] - xs[0], 1e-6)
        return [t0 + (x - xs[0]) / span * (t1 - t0) for x in xs]
    anchor_x = [xs[i] for i in known] or [xs[0]]
    anchor_t = [times[i] for i in known]
    if len(anchor_x) == 1:
        anchor_x = [anchor_x[0] - 1.0, anchor_x[0] + 1.0]
        anchor_t = [anchor_t[0], anchor_t[0]]
    return [float(np.interp(x, anchor_x, anchor_t)) for x in xs]


def time_page(digits: Sequence, onsets: List[Onset], t0: float, t1: float,
              page_width: float, config: Config) -> List[Tuple[Any, float]]:
    """Every note on one page, paired with the moment it sounds."""
    attacks = group_attacks(digits, page_width, config)
    if not attacks:
        return []
    margin = config.audio_page_margin_s
    inside = [o for o in onsets if t0 - margin <= o[0] <= t1 + margin]
    times = align(attacks, inside, config)
    filled = fill_gaps(attacks, times, t0, t1)
    out: List[Tuple[Any, float]] = []
    for group, time in zip(attacks, filled):
        for digit in group:
            out.append((digit, float(max(time, 0.0))))
    return out


def bar_times_from_x(bar_xs: Sequence, placed: Sequence[Tuple[Any, float]],
                     t0: float, t1: float) -> List[float]:
    """When the music reached each bar line, from where the notes around it fell.

    A bar line has no sound of its own, so it is dated by the notes either side
    of it -- the same notes the soundtrack has already timed. Past the outermost
    note the bar line is pinned to the page's own bounds rather than
    extrapolated, because a system's closing bar sits beyond every note in it and
    a speed estimated from two notes would throw it a long way out.
    """
    if not bar_xs or not placed:
        return []
    xs = [float(d.x_center) for d, _ in placed]
    ts = [float(t) for _, t in placed]
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    xs = [xs[i] for i in order]
    ts = [ts[i] for i in order]
    return [float(np.interp(float(x), xs, ts, left=t0, right=t1)) for x in bar_xs]


def notes_from_audio(pages, video_path: str, config: Config):
    """A sheet timed by the soundtrack, for a video that marks nothing on the page.

    Returns the notes and, alongside, how much of the tab the audio could
    actually account for. The caller needs that: a video whose soundtrack is the
    original recording rather than the player's own instrument aligns badly, and
    is better refused than printed.
    """
    from src.models.schema import Note, ReconstructionResult
    from src.audio import onsets as audio_onsets

    detected = audio_onsets.detect(video_path, config)
    notes: List[Note] = []
    bars: List[float] = []
    matched = attacks_total = agreed = 0
    for page in pages:
        if not page.digits or page.composite is None:
            continue
        width = float(page.composite.shape[1])
        groups = group_attacks(page.digits, width, config)
        attacks_total += len(groups)
        margin = config.audio_page_margin_s
        inside = [o for o in detected if page.t0 - margin <= o[0] <= page.t1 + margin]
        placed = align(groups, inside, config)
        matched += len(placed)
        by_time = {onset[0]: onset for onset in inside}
        for index, when in placed.items():
            if _match_score(groups[index], by_time[when], config) >= config.audio_pitch_bonus:
                agreed += 1
        placed_notes = time_page(page.digits, detected, page.t0, page.t1,
                                 width, config)
        for digit, time in placed_notes:
            notes.append(Note(
                time=float(time),
                string_index=digit.string_index,
                fret=digit.value,
                confidence=digit.confidence,
                x=float(digit.x_center),
            ))
        # The page turn is itself a bar line: a system begins a new bar.
        bars.append(float(page.t0))
        staff = find_string_lines(cv2.cvtColor(page.composite, cv2.COLOR_BGR2GRAY),
                                  config)
        bars.extend(bar_times_from_x(
            find_bar_lines(page.composite, staff, config),
            placed_notes, page.t0, page.t1))
    notes.sort(key=lambda n: (n.time, n.string_index))
    share = matched / attacks_total if attacks_total else 0.0
    agreement = agreed / matched if matched else 0.0
    bar_times = _cluster(sorted(bars), config.bar_time_cluster_tolerance_s)
    result = ReconstructionResult(notes=notes, bar_times=bar_times,
                                  speed_px_per_s=None)
    return result, {"audio_onsets": float(len(detected)),
                    "audio_attacks": float(attacks_total),
                    "audio_matched_share": float(share),
                    "audio_pitch_agreement": float(agreement)}


def audio_diagnosis(stats: Dict[str, float], config: Config) -> Optional[str]:
    """Why the soundtrack cannot time this video, in words, or None if it can.

    The tell is how much of the tab the audio accounts for. Where the soundtrack
    is the player's own instrument nearly every note on the page is struck in it,
    and matching runs 89% to 98%. Where it is the original recording instead --
    drums, voice, a full band -- the onsets are not the tab's notes and matching
    falls to 63%, with the notes it does place landing a third of a second out.
    """
    if stats.get("audio_onsets", 0) < config.audio_min_onsets:
        return ("no notes could be heard in the soundtrack — this reader falls back "
                "to timing a tab by the sound of it being played when the video "
                "marks nothing on the page, and it needs audio to do that.")
    share = stats.get("audio_matched_share", 0.0)
    if share < config.audio_min_matched_share:
        return (f"the soundtrack does not play what the tab shows — only "
                f"{share:.0%} of the notes on the page could be matched to a "
                f"sound, against the 89% or better seen when a video carries the "
                f"player's own instrument. A backing track times nothing.")
    # Matching alone proves very little. Where the audio holds three times as
    # many onsets as the page holds notes, every note finds *a* sound whether or
    # not it is the right one, and one video matched 100% of its notes while
    # agreeing with barely any of them on pitch. Pitch is what says the alignment
    # found the note rather than merely something audible nearby.
    agreement = stats.get("audio_pitch_agreement", 0.0)
    if agreement < config.audio_min_pitch_agreement:
        return (f"the notes timed from the soundtrack are not the notes on the "
                f"page — only {agreement:.0%} of them sound at the pitch the tab "
                f"says they should, against 95% or better where this works. "
                f"Either the tab is being misread or the audio is not the "
                f"instrument, and the times that follow would be guesswork.")
    return None
