from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def write_pdf(text: str, path: str) -> None:
    c = canvas.Canvas(path, pagesize=letter)
    width, height = letter
    y = height - 40
    for line in text.split("\n"):
        c.drawString(40, y, line)
        y -= 14
        if y < 40:
            c.showPage()
            y = height - 40
    c.save()
