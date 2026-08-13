import os
import shutil
import threading
import time
import uuid
from typing import Dict
from urllib.parse import urlparse

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from src.app.config import Config
from src.app.main import run_pipeline
from src.output.text import render_text_tab


app = Flask(__name__)

JOBS: Dict[str, dict] = {}

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

_ALLOWED_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"}
_JOB_TTL_S = 24 * 3600
_MAX_JOBS = 50


def _valid_youtube_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    return parsed.netloc.lower() in _ALLOWED_HOSTS


def _known_job(job_id: str) -> bool:
    return job_id in JOBS


def _cleanup_jobs() -> None:
    now = time.time()
    stale = [
        jid
        for jid, job in JOBS.items()
        if job.get("finished_at") is not None and now - job["finished_at"] > _JOB_TTL_S
    ]
    for jid in stale:
        JOBS.pop(jid, None)
        shutil.rmtree(os.path.join(OUTPUTS_DIR, jid), ignore_errors=True)
    if len(JOBS) > _MAX_JOBS:
        for jid in sorted(JOBS, key=lambda j: JOBS[j].get("finished_at") or float("inf")):
            if len(JOBS) <= _MAX_JOBS:
                break
            JOBS.pop(jid, None)
            shutil.rmtree(os.path.join(OUTPUTS_DIR, jid), ignore_errors=True)


def _progress(job_id: str, stage: str, value: float) -> None:
    JOBS[job_id]["stage"] = stage
    JOBS[job_id]["progress"] = value


def _worker(job_id: str, url: str, output_dir: str) -> None:
    config = Config(output_dir=output_dir)
    try:
        sheet = run_pipeline(url, config, lambda s, v: _progress(job_id, s, v))
        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["sheet"] = sheet
        JOBS[job_id]["text"] = render_text_tab(sheet, config)
    except Exception as exc:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(exc)
    finally:
        JOBS[job_id]["finished_at"] = time.time()


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    _cleanup_jobs()
    url = request.form.get("url", "").strip()
    if not _valid_youtube_url(url):
        return render_template("index.html", error="Please enter a valid YouTube URL."), 400

    job_id = uuid.uuid4().hex
    output_dir = os.path.join(OUTPUTS_DIR, job_id)
    JOBS[job_id] = {"status": "running", "progress": 0.0, "stage": "init"}

    thread = threading.Thread(target=_worker, args=(job_id, url, output_dir), daemon=True)
    thread.start()
    return redirect(url_for("result", job_id=job_id))


@app.route("/status/<job_id>")
def status(job_id: str):
    _cleanup_jobs()
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"status": "missing"}), 404
    return jsonify({"status": job.get("status"), "progress": job.get("progress"), "stage": job.get("stage")})


@app.route("/result/<job_id>")
def result(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        return redirect(url_for("index"))
    return render_template("result.html", job_id=job_id, job=job)


def _send_download(path: str):
    resp = send_file(path, as_attachment=True, conditional=False, max_age=0)
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/download/<job_id>/<kind>")
def download(job_id: str, kind: str):
    if not _known_job(job_id):
        return jsonify({"error": "unknown job"}), 404
    job = JOBS[job_id]
    if job.get("status") != "done":
        return jsonify({"error": "job not finished"}), 409
    base = os.path.join(OUTPUTS_DIR, job_id)
    try:
        if kind == "txt":
            return _send_download(os.path.join(base, "tabs.txt"))
        if kind == "pdf":
            return _send_download(os.path.join(base, "tabs.pdf"))
        if kind == "json":
            return _send_download(os.path.join(base, "tabs.json"))
    except FileNotFoundError:
        return jsonify({"error": "job output missing"}), 404
    return jsonify({"error": "unknown kind"}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
