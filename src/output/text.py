"""Laying a read-out sheet onto the page.

Tab has no note values, so the only thing carrying rhythm is horizontal space.
The layout therefore gives every distinct onset its own column and sets the gap
before it in proportion to the silence before it, measured against the piece's
own typical note gap rather than a fixed number of seconds.

Bucketing time into fixed-width slots instead — the obvious approach — quietly
loses music: any two notes falling in the same slot on the same string collide,
and one of them is simply not printed. On a piece whose notes average a quarter
of a second apart, a quarter-second slot drops one note in six.
"""
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple

from src.app.config import Config
from src.models.schema import TabSheet


System = List[Tuple[str, str]]  # (string label, rendered line)


@dataclass
class Column:
    """One printed column: the frets sounding together at a single instant."""
    time: float
    frets: Dict[int, str] = field(default_factory=dict)
    pad: int = 1        # dashes before this column, standing for the gap
    bar: bool = False   # a bar line falls immediately before it

    @property
    def width(self) -> int:
        return max((len(text) for text in self.frets.values()), default=1)


def _string_labels(sheet: TabSheet, config: Config) -> Sequence[str]:
    """String names down the staff, top line first.

    A note's string_index is the staff line it was found on, counting from the
    top, and tab draws the highest-pitched string on top — so for a ukulele the
    top line is A and the bottom one G. config.tuning runs the other way, from
    the fourth string to the first, so it has to be reversed to label rows.
    Reading it straight across mirrors every note onto the wrong string.
    """
    if sheet.metadata.get("tab_mode") == "single":
        return ("",)
    return tuple(reversed(config.tuning))


def columns(sheet: TabSheet, config: Config) -> List[Column]:
    """Every note, in order, grouped into the instants they sound at.

    Notes close enough together are one strum and share a column. A note landing
    on a string that column has already spoken for cannot be part of the same
    strum, so it opens a new one — which is what keeps the layout lossless.
    """
    rows = len(_string_labels(sheet, config))
    single = sheet.metadata.get("tab_mode") == "single"
    out: List[Column] = []
    for note in sorted(sheet.notes, key=lambda n: (n.time, n.string_index)):
        row = 0 if single else note.string_index
        if not 0 <= row < rows:
            continue
        if (out and note.time - out[-1].time <= config.chord_window_s
                and row not in out[-1].frets):
            out[-1].frets[row] = str(note.fret)
        else:
            out.append(Column(time=note.time, frets={row: str(note.fret)}))
    return out


def _space(cols: List[Column], config: Config) -> None:
    """Set each column's gap from the piece's own sense of a short note.

    The unit is a low percentile of the gaps between onsets — the piece's quick
    note, near enough. A run of those then reads evenly and anything held longer
    opens up in proportion. An absolute unit cannot do this: the same 0.25s is
    one note in a fast piece and a whole bar in a slow one.

    Low rather than middle because a rest must not set the scale. Take the median
    of a piece that is half quick notes and half long holds and every quick note
    collapses into the same column width as its neighbour.
    """
    gaps = [b.time - a.time for a, b in zip(cols, cols[1:])]
    positive = sorted(g for g in gaps if g > 0)
    unit = positive[int(len(positive) * 0.25)] if positive else 0.0
    for column, gap in zip(cols[1:], gaps):
        if unit <= 0:
            column.pad = 1
        else:
            column.pad = min(max(int(round(gap / unit)), 1), config.max_gap_dashes)


def _mark_bars(cols: List[Column], sheet: TabSheet) -> None:
    times = [c.time for c in cols]
    for measure in sheet.measures:
        index = bisect_left(times, measure.end_time)
        if 0 < index < len(cols):
            cols[index].bar = True


def layout(sheet: TabSheet, config: Config, width: int = 72) -> List[List[Column]]:
    """Columns split into systems that fit the given character width."""
    cols = columns(sheet, config)
    if not cols:
        return []
    _space(cols, config)
    _mark_bars(cols, sheet)
    cols[0].pad = 1  # a system opens flush against its bar line

    label_width = max((len(l) for l in _string_labels(sheet, config)), default=0)
    budget = max(width - label_width - 2, 8)
    systems: List[List[Column]] = []
    current: List[Column] = []
    used = 0
    for column in cols:
        cost = column.pad + column.width + (1 if column.bar else 0)
        if current and used + cost > budget:
            systems.append(current)
            current, used = [], 0
        current.append(column)
        used += cost
    if current:
        systems.append(current)
    return systems


def build_systems(sheet: TabSheet, config: Config,
                  width: int = 72) -> List[System]:
    """Wrap the tab into fixed-width systems, the way a printed sheet reads.

    Columns are padded to the widest fret in the column so every string stays
    aligned, and a system is only broken between columns, never inside one.
    """
    labels = _string_labels(sheet, config)
    systems: List[System] = []
    for block in layout(sheet, config, width):
        lines = ["" for _ in labels]
        for column in block:
            prefix = "-" * column.pad
            if column.bar:
                prefix = "|" + prefix
            for row in range(len(labels)):
                text = column.frets.get(row, "")
                lines[row] += prefix + (text.ljust(column.width, "-") if text
                                        else "-" * column.width)
        systems.append([(label, line + "|") for label, line in zip(labels, lines)])
    return systems


def render_text_tab(sheet: TabSheet, config: Config) -> str:
    """Plain-text tab sheet, wrapped into systems."""
    if not sheet.notes:
        return "No notes detected."
    systems = build_systems(sheet, config)
    if not systems:
        return "No notes detected."

    single = sheet.metadata.get("tab_mode") == "single"
    header = "Ukulele Tabs (single line)" if single else "Ukulele Tabs (GCEA)"
    label_width = max(len(label) for label, _ in systems[0])

    out = [header, ""]
    for system in systems:
        for label, line in system:
            out.append(f"{label.rjust(label_width)}|{line}")
        out.append("")
    return "\n".join(out).rstrip() + "\n"
