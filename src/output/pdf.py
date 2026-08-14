"""Tab as printed notation rather than a picture of a text file.

The staves, bar lines and fret numbers are drawn as graphics, so the sheet does
not depend on the reader having a monospaced font — the previous version wrote
the ASCII art in the default proportional face, where dashes and digits differ
in width and the columns cannot line up by construction.
"""
from typing import List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas

from src.app.config import Config
from src.models.schema import TabSheet
from src.output.text import Column, _string_labels, layout


MARGIN = 18 * mm
STRING_GAP = 4.2 * mm        # between adjacent string lines
SYSTEM_GAP = 12 * mm         # between the bottom of one system and the top of the next
FRET_SIZE = 8.5              # point size of the fret numbers
LABEL_SIZE = 8.0
COLUMNS_PER_SYSTEM = 74      # character cells, matching the text layout


def _system_height(rows: int) -> float:
    return (rows - 1) * STRING_GAP


def _draw_system(c: canvas.Canvas, block: List[Column], labels, top: float,
                 left: float, width: float, max_cell: float) -> None:
    rows = len(labels)
    y_of = [top - row * STRING_GAP for row in range(rows)]
    span = sum(column.pad + column.width + (1 if column.bar else 0)
               for column in block)
    # Justify to the full measure, the way a printed sheet does — but never
    # stretch a short final system across the page, which reads as a mistake.
    cell = min(width / float(span), max_cell) if span else max_cell
    right = left + span * cell

    c.setLineWidth(0.4)
    c.setStrokeColorRGB(0.45, 0.45, 0.45)
    for y in y_of:
        c.line(left, y, right, y)

    c.setFont("Helvetica", LABEL_SIZE)
    c.setFillColorRGB(0.35, 0.35, 0.35)
    for row, label in enumerate(labels):
        if label:
            c.drawRightString(left - 2.2 * mm, y_of[row] - LABEL_SIZE * 0.35, label)

    # Opening and closing bar lines bracket the system.
    c.setStrokeColorRGB(0.25, 0.25, 0.25)
    c.setLineWidth(0.7)
    c.line(left, y_of[0], left, y_of[-1])
    c.line(right, y_of[0], right, y_of[-1])

    x = left
    for column in block:
        if column.bar:
            c.line(x, y_of[0], x, y_of[-1])
            x += cell
        x += column.pad * cell
        for row, text in column.frets.items():
            centre = x + (column.width * cell) / 2.0
            baseline = y_of[row] - FRET_SIZE * 0.34
            # Knock the staff line out behind the digit, the way engraved tab does.
            half = c.stringWidth(text, "Helvetica-Bold", FRET_SIZE) / 2.0 + 0.5 * mm
            c.setFillColorRGB(1, 1, 1)
            c.rect(centre - half, y_of[row] - FRET_SIZE * 0.5,
                   half * 2, FRET_SIZE, stroke=0, fill=1)
            c.setFillColorRGB(0.05, 0.05, 0.05)
            c.setFont("Helvetica-Bold", FRET_SIZE)
            c.drawCentredString(centre, baseline, text)
        x += column.width * cell


def write_pdf(sheet: TabSheet, config: Config, path: str,
              title: Optional[str] = None) -> None:
    page_width, page_height = A4
    c = canvas.Canvas(path, pagesize=A4)
    c.setTitle(title or "Ukulele tab")

    labels = _string_labels(sheet, config)
    systems = layout(sheet, config, COLUMNS_PER_SYSTEM)
    left = MARGIN + 8 * mm            # room for the string labels
    usable = page_width - left - MARGIN
    cell = usable / float(COLUMNS_PER_SYSTEM)

    y = page_height - MARGIN
    c.setFont("Helvetica-Bold", 13)
    c.setFillColorRGB(0.05, 0.05, 0.05)
    c.drawString(MARGIN, y, title or "Ukulele tab")
    y -= 6 * mm
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    tuning = " ".join(labels) if any(labels) else "single line"
    c.drawString(MARGIN, y, f"{len(sheet.notes)} notes · {len(sheet.measures)} measures "
                            f"· tuning {tuning}")
    y -= 10 * mm

    if not systems:
        c.setFont("Helvetica", 10)
        c.setFillColorRGB(0.2, 0.2, 0.2)
        c.drawString(MARGIN, y, "No notes were detected in this video.")
        c.save()
        return

    height = _system_height(len(labels))
    for block in systems:
        if y - height < MARGIN + 10 * mm:
            _footer(c, page_width)
            c.showPage()
            y = page_height - MARGIN
        _draw_system(c, block, labels, y, left, usable, cell * 1.35)
        y -= height + SYSTEM_GAP
    _footer(c, page_width)
    c.save()


def _footer(c: canvas.Canvas, page_width: float) -> None:
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(0.55, 0.55, 0.55)
    c.drawCentredString(page_width / 2.0, MARGIN * 0.6, str(c.getPageNumber()))
