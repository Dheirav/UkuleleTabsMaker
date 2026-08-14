import os

from src.web.app import OUTPUTS_DIR, app, JOBS, _valid_youtube_url


def test_valid_youtube_url():
    assert _valid_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
    assert _valid_youtube_url("http://youtu.be/abc123")
    assert _valid_youtube_url("https://m.youtube.com/watch?v=abc")


def test_invalid_youtube_url():
    assert not _valid_youtube_url("file:///etc/passwd")
    assert not _valid_youtube_url("https://evil.com/watch")
    assert not _valid_youtube_url("ftp://www.youtube.com/watch")
    assert not _valid_youtube_url("not a url")


def test_process_rejects_invalid_url():
    client = app.test_client()
    resp = client.post("/process", data={"url": "file:///etc/passwd"})
    assert resp.status_code == 400


def test_download_unknown_job_returns_404():
    client = app.test_client()
    resp = client.get("/download/deadbeef/pdf")
    assert resp.status_code == 404


def test_download_traversal_job_returns_404():
    client = app.test_client()
    resp = client.get("/download/%2e%2e/pdf")
    assert resp.status_code == 404


def test_download_finished_job_sends_file(tmp_path, monkeypatch):
    job_id = "testjob"
    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "tabs.pdf"), "w") as f:
        f.write("A |--0--|\n")
    JOBS[job_id] = {"status": "done", "finished_at": 0.0}
    try:
        client = app.test_client()
        resp = client.get(f"/download/{job_id}/pdf")
        assert resp.status_code == 200
        assert "A |--0--|" in resp.get_data(as_text=True)
        assert "no-store" in resp.headers.get("Cache-Control", "")
    finally:
        JOBS.pop(job_id, None)
        os.remove(os.path.join(job_dir, "tabs.pdf"))
        os.rmdir(job_dir)


def test_download_finished_job_missing_file_returns_404():
    JOBS["missingout"] = {"status": "done", "finished_at": 0.0}
    try:
        client = app.test_client()
        resp = client.get("/download/missingout/pdf")
        assert resp.status_code == 404
    finally:
        JOBS.pop("missingout", None)


def _finished_job(coverage):
    """A completed job, as the result page expects to receive it."""
    import time as _time
    from src.app.config import Config
    from src.models.schema import Measure, Note, TabSheet
    from src.output.text import build_systems, render_text_tab

    config = Config()
    notes = [Note(time=i * 0.5, string_index=i % 4, fret=i % 5, confidence=0.9)
             for i in range(12)]
    measures = [Measure(start_time=i * 2.0, end_time=(i + 1) * 2.0, notes=[])
                for i in range(3)]
    sheet = TabSheet(notes=notes, measures=measures,
                     metadata={"tab_mode": "paged", "font": "NimbusSans-Regular.otf"})
    return {
        "status": "done",
        "sheet": sheet,
        "text": render_text_tab(sheet, config),
        "systems": build_systems(sheet, config),
        "finished_at": _time.time(),
        "report": {"mode": "paged", "pages": 12.0, "coverage": coverage,
                   "measures_with_notes": 4.0, "measures_highlighted": 12.0},
    }


def test_low_coverage_is_reported_as_incomplete():
    """A sheet holding a third of the piece must say so; a partial tab that
    looks complete is worse than one that admits it."""
    JOBS["lowcov"] = _finished_job(0.33)
    body = app.test_client().get("/result/lowcov").get_data(as_text=True)
    assert "33%" in body
    assert "Most of this piece is missing" in body
    JOBS.pop("lowcov", None)


def test_high_coverage_is_not_alarming():
    JOBS["highcov"] = _finished_job(0.94)
    body = app.test_client().get("/result/highcov").get_data(as_text=True)
    assert "94%" in body
    assert "Most of this piece is missing" not in body
    JOBS.pop("highcov", None)


def test_result_page_renders_wrapped_systems():
    JOBS["sys"] = _finished_job(0.9)
    body = app.test_client().get("/result/sys").get_data(as_text=True)
    assert 'class="system"' in body
    for label in ("G", "C", "E", "A"):
        assert f"{label}|" in body
    JOBS.pop("sys", None)


def test_error_job_reports_failure():
    JOBS["boom"] = {"status": "error", "error": "Could not open video",
                    "finished_at": 0.0}
    body = app.test_client().get("/result/boom").get_data(as_text=True)
    assert "Processing failed" in body
    assert "Could not open video" in body
    assert "Reading the video" not in body
    JOBS.pop("boom", None)


def test_missing_report_still_renders():
    """Older jobs have no stats file; the page must not blow up."""
    job = _finished_job(0.9)
    job["report"] = {}
    JOBS["noreport"] = job
    resp = app.test_client().get("/result/noreport")
    assert resp.status_code == 200
    JOBS.pop("noreport", None)
