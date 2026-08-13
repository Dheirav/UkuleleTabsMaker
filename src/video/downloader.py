import os
from glob import glob

import yt_dlp


_VIDEO_EXTENSIONS = ("mp4", "mkv", "webm", "mov", "m4v", "avi", "flv", "ts", "mpeg")


def _build_opts(outtmpl: str, fmt: str, cookies_path: str | None) -> dict:
    opts = {
        "outtmpl": outtmpl,
        "format": fmt,
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
    }
    client = os.environ.get("YTDLP_YOUTUBE_CLIENT", "").strip()
    if client:
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
    if cookies_path and os.path.exists(cookies_path):
        opts["cookiefile"] = cookies_path
    return opts


def _resolve_downloaded_video(output_dir: str) -> str:
    for ext in _VIDEO_EXTENSIONS:
        candidate = os.path.join(output_dir, f"video.{ext}")
        if os.path.exists(candidate):
            return candidate

    matches = sorted(glob(os.path.join(output_dir, "video.*")))
    if matches:
        return matches[0]

    raise FileNotFoundError("yt-dlp finished without producing a usable video file")


def download_youtube(url: str, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, "video.%(ext)s")
    cookies_path = os.environ.get("YTDLP_COOKIES")

    formats = ["bestvideo*+bestaudio/best", "bestvideo+bestaudio/best", "best"]
    last_error = None
    info = None
    for fmt in formats:
        try:
            with yt_dlp.YoutubeDL(_build_opts(outtmpl, fmt, cookies_path)) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as exc:
            last_error = exc

    if info is None:
        raise last_error

    try:
        return _resolve_downloaded_video(output_dir)
    except FileNotFoundError:
        ext = info.get("ext", "mp4")
        candidate = os.path.join(output_dir, f"video.{ext}")
        if os.path.exists(candidate):
            return candidate
        raise
