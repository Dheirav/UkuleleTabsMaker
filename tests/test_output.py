import os

import pytest

from src.app.config import Config
from src.models.schema import Measure, Note, TabSheet
from src.output.pdf import write_pdf
from src.output.text import build_systems, columns, render_text_tab


def _sheet(notes, measures=()):
    return TabSheet(
        notes=[Note(t, s, f, 1.0) for t, s, f in notes],
        measures=[Measure(a, b) for a, b in measures],
        metadata={"tab_mode": "paged"},
    )


def test_every_note_reaches_the_page():
    """Bucketing time into fixed slots dropped whichever note landed second in a
    slot on a given string — one note in six on a real piece, silently."""
    config = Config()
    # four notes on the same string, far closer together than any fixed quantum
    sheet = _sheet([(0.00, 3, 1), (0.05, 3, 2), (0.09, 3, 3), (0.12, 3, 4)])
    printed = sum(len(c.frets) for c in columns(sheet, config))
    assert printed == 4
    line = build_systems(sheet, config)[0][3][1]
    assert "1" in line and "2" in line and "3" in line and "4" in line


def test_notes_struck_together_share_one_column():
    config = Config()
    sheet = _sheet([(1.0, 0, 2), (1.0, 1, 0), (1.005, 2, 1)])
    cols = columns(sheet, config)
    assert len(cols) == 1
    assert cols[0].frets == {0: "2", 1: "0", 2: "1"}


def test_a_repeated_string_cannot_join_the_same_column():
    """Two frets on one string cannot sound at once, so this has to open a new
    column rather than overwrite — that overwrite was the lost-note bug."""
    config = Config()
    sheet = _sheet([(1.0, 2, 5), (1.001, 2, 7)])
    cols = columns(sheet, config)
    assert len(cols) == 2
    assert [c.frets[2] for c in cols] == ["5", "7"]


def test_spacing_grows_with_the_gap_before_a_note():
    config = Config()
    # steady eighths, then a note four times as far away
    times = [0.0, 0.25, 0.50, 0.75, 1.75]
    sheet = _sheet([(t, 3, 1) for t in times])
    cols = columns(sheet, config)
    from src.output.text import _space
    _space(cols, config)
    steady = [c.pad for c in cols[1:4]]
    assert steady == [1, 1, 1]
    assert cols[4].pad == 4  # a gap four units long reads four units wide


def test_spacing_is_relative_to_the_piece_not_to_the_clock():
    """The same shape played at half speed must lay out identically: the unit is
    the piece's own typical gap, not an absolute number of seconds."""
    config = Config()
    from src.output.text import _space
    pads = []
    for scale in (1.0, 0.5, 4.0):
        cols = columns(_sheet([(t * scale, 3, 1) for t in (0.0, 0.25, 0.5, 1.5)]), config)
        _space(cols, config)
        pads.append([c.pad for c in cols])
    assert pads[0] == pads[1] == pads[2]


def test_long_rests_cannot_run_off_the_page():
    config = Config()
    from src.output.text import _space
    cols = columns(_sheet([(0.0, 3, 1), (0.25, 3, 1), (300.0, 3, 1)]), config)
    _space(cols, config)
    assert cols[-1].pad == config.max_gap_dashes


def test_strings_are_labelled_down_the_staff_not_up():
    """Tab draws the highest string on top, and string_index is the staff line
    counting from the top — so row 0 is A, not the tuning's first entry."""
    config = Config()
    sheet = _sheet([(0.0, 0, 5)])
    labels = [label for label, _ in build_systems(sheet, config)[0]]
    assert labels == ["A", "E", "C", "G"]
    # the note was found on the top line, so it must print on the top line
    top_line = build_systems(sheet, config)[0][0][1]
    assert "5" in top_line


def test_bar_lines_land_between_the_right_columns():
    config = Config()
    sheet = _sheet([(0.0, 3, 1), (0.25, 3, 2), (0.5, 3, 3)],
                   measures=[(0.0, 0.4)])
    cols = columns(sheet, config)
    from src.output.text import _mark_bars
    _mark_bars(cols, sheet)
    assert [c.bar for c in cols] == [False, False, True]


def test_text_output_reports_when_there_is_nothing_to_show():
    assert render_text_tab(_sheet([]), Config()) == "No notes detected."


def test_pdf_is_written_as_notation_not_as_a_text_dump(tmp_path):
    config = Config()
    sheet = _sheet([(i * 0.25, i % 4, i % 5) for i in range(400)],
                   measures=[(i, i + 1.0) for i in range(20)])
    path = tmp_path / "tabs.pdf"
    write_pdf(sheet, config, str(path), "Test piece")
    assert path.exists() and path.stat().st_size > 2000
    pymupdf = pytest.importorskip("pymupdf")
    doc = pymupdf.open(str(path))
    assert doc.page_count >= 1
    page = doc[0]
    assert "Test piece" in page.get_text()
    # staves and bar lines are drawn, not typed
    assert len(page.get_drawings()) > 20


def test_pdf_handles_an_empty_sheet(tmp_path):
    path = tmp_path / "empty.pdf"
    write_pdf(_sheet([]), Config(), str(path))
    assert os.path.getsize(path) > 0


def test_the_sheet_is_named_after_its_source():
    config = Config()
    sheet = _sheet([(0.0, 3, 1)])
    sheet.metadata["title"] = "Sherma Song - Hollow Knight Silksong"
    assert render_text_tab(sheet, config).splitlines()[0] == \
        "Sherma Song - Hollow Knight Silksong"


def test_an_unnamed_sheet_still_has_a_heading():
    assert render_text_tab(_sheet([(0.0, 3, 1)]), Config()).splitlines()[0] == "Ukulele tab"


@pytest.mark.parametrize("name,expected", [
    ("silksong_sherma.mp4", "Silksong Sherma"),
    ("my-favourite-song.mkv", "My Favourite Song"),
    ("TOTO Africa.mp4", "TOTO Africa"),
])
def test_a_local_file_is_named_after_itself(name, expected):
    from src.app.main import _title_from_filename
    assert _title_from_filename(f"/videos/{name}") == expected


def test_pdf_carries_the_title_into_its_metadata(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    path = tmp_path / "named.pdf"
    sheet = _sheet([(0.0, 3, 1), (0.5, 2, 3)])
    sheet.metadata["title"] = "Kyoko Kirigiri"
    write_pdf(sheet, Config(), str(path), sheet.metadata["title"])
    doc = pymupdf.open(str(path))
    assert doc.metadata.get("title") == "Kyoko Kirigiri"
    assert "Kyoko Kirigiri" in doc[0].get_text()
