from typing import List

from src.models.schema import Note, Measure


def parse_measures(notes: List[Note], bar_times: List[float]) -> List[Measure]:
    if not notes:
        return []
    bar_times = sorted(bar_times)
    measures: List[Measure] = []
    current_start = notes[0].time
    bar_iter = iter(bar_times)
    next_bar = next(bar_iter, None)
    while next_bar is not None and next_bar < current_start:
        next_bar = next(bar_iter, None)

    current_notes: List[Note] = []
    for note in notes:
        while next_bar is not None and note.time >= next_bar:
            if current_notes:
                measures.append(Measure(start_time=current_start, end_time=next_bar, notes=current_notes))
            current_start = next_bar
            current_notes = []
            next_bar = next(bar_iter, None)
        current_notes.append(note)

    end_time = notes[-1].time
    if current_notes:
        measures.append(Measure(start_time=current_start, end_time=end_time, notes=current_notes))
    return measures
