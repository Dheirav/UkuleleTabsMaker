FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ffmpeg \
       tesseract-ocr \
       libglib2.0-0 \
       libsm6 \
       libxext6 \
       libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements-opencv.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir --no-deps -r /app/requirements-opencv.txt

COPY . /app

EXPOSE 8000

CMD ["python", "-m", "src.web.app"]
