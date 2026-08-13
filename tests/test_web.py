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
    resp = client.get("/download/deadbeef/txt")
    assert resp.status_code == 404


def test_download_traversal_job_returns_404():
    client = app.test_client()
    resp = client.get("/download/%2e%2e/txt")
    assert resp.status_code == 404


def test_download_finished_job_sends_file(tmp_path, monkeypatch):
    job_id = "testjob"
    job_dir = os.path.join(OUTPUTS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)
    with open(os.path.join(job_dir, "tabs.txt"), "w") as f:
        f.write("A |--0--|\n")
    JOBS[job_id] = {"status": "done", "finished_at": 0.0}
    try:
        client = app.test_client()
        resp = client.get(f"/download/{job_id}/txt")
        assert resp.status_code == 200
        assert "A |--0--|" in resp.get_data(as_text=True)
        assert "no-store" in resp.headers.get("Cache-Control", "")
    finally:
        JOBS.pop(job_id, None)
        os.remove(os.path.join(job_dir, "tabs.txt"))
        os.rmdir(job_dir)


def test_download_finished_job_missing_file_returns_404():
    JOBS["missingout"] = {"status": "done", "finished_at": 0.0}
    try:
        client = app.test_client()
        resp = client.get("/download/missingout/txt")
        assert resp.status_code == 404
    finally:
        JOBS.pop("missingout", None)
