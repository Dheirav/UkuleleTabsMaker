import cv2
import pytesseract
from pytesseract import TesseractNotFoundError


def ocr_digits(image) -> str:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        return pytesseract.image_to_string(gray, config=config).strip()
    except TesseractNotFoundError:
        return ""
