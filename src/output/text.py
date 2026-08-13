from typing import List

from src.app.config import Config
from src.models.schema import TabSheet


def render_text_tab(sheet: TabSheet, config: Config) -> str:
    if not sheet.notes:
        return "No notes detected."
    max_time = max(note.time for note in sheet.notes)
    steps = int(max_time / config.time_quantum_s) + 1

    if sheet.metadata.get("tab_mode") == "single":
        line = ["-"] * steps
        for note in sheet.notes:
            idx = int(note.time / config.time_quantum_s)
            if idx >= steps:
                continue
            line[idx] = str(note.fret)
        return "Ukulele Tabs (single line)\n" + "|".join(line) + "|"

    lines = [["-"] * steps for _ in range(len(config.tuning))]

    for note in sheet.notes:
        idx = int(note.time / config.time_quantum_s)
        if idx >= steps:
            continue
        fret = str(note.fret)
        row = note.string_index
        if row >= len(lines):
            continue
        lines[row][idx] = fret

    header = "Ukulele Tabs (GCEA)"
    output = [header]
    for string_idx, string_name in enumerate(config.tuning):
        line = "|".join(lines[string_idx])
        output.append(f"{string_name}|{line}|")

    return "\n".join(output)
