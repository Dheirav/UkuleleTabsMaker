from typing import List, Optional, Sequence, Tuple

from src.app.config import Config
from src.models.schema import TabSheet


System = List[Tuple[str, str]]  # (string label, rendered line)


def _string_labels(sheet: TabSheet, config: Config) -> Sequence[str]:
    if sheet.metadata.get("tab_mode") == "single":
        return ("",)
    return config.tuning


def _grid(sheet: TabSheet, config: Config):
    """cells[string][step] -> fret text, plus the steps a bar line follows."""
    labels = _string_labels(sheet, config)
    rows = len(labels)
    max_time = max((n.time for n in sheet.notes), default=0.0)
    steps = int(max_time / config.time_quantum_s) + 1

    single = sheet.metadata.get("tab_mode") == "single"
    cells: List[List[Optional[str]]] = [[None] * steps for _ in range(rows)]
    for note in sheet.notes:
        step = int(note.time / config.time_quantum_s)
        if step >= steps:
            continue
        row = 0 if single else note.string_index
        if 0 <= row < rows:
            cells[row][step] = str(note.fret)

    bars = set()
    for measure in sheet.measures:
        step = int(measure.end_time / config.time_quantum_s)
        if 0 < step < steps:
            bars.add(step)
    return labels, cells, bars, steps


def build_systems(sheet: TabSheet, config: Config,
                  width: int = 72) -> List[System]:
    """Wrap the tab into fixed-width systems, the way a printed sheet reads.

    Columns are padded to the widest fret in that column so every string stays
    aligned, and a system is only broken between columns, never inside one.
    """
    labels, cells, bars, steps = _grid(sheet, config)
    if steps == 0 or not sheet.notes:
        return []

    label_width = max((len(l) for l in labels), default=0)
    systems: List[System] = []
    lines = ["" for _ in labels]
    used = 0

    def flush():
        nonlocal lines, used
        if used:
            systems.append([(label, line + "|")
                            for label, line in zip(labels, lines)])
        lines = ["" for _ in labels]
        used = 0

    for step in range(steps):
        column_width = max((len(cells[r][step] or "") for r in range(len(labels))), default=1)
        column_width = max(column_width, 1)
        chunk = ["".join((cells[r][step] or "").ljust(column_width, "-")
                         if cells[r][step] else "-" * column_width)
                 for r in range(len(labels))]
        separator = "|" if step in bars else ""
        cost = column_width + len(separator)
        if used and used + cost > width - label_width - 1:
            flush()
        for r in range(len(labels)):
            lines[r] += chunk[r] + separator
        used += cost
    flush()
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
