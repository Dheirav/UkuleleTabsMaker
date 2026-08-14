import os
from dataclasses import dataclass
from glob import glob
from typing import Callable, Optional

import yt_dlp


@dataclass
class Download:
    path: str
    title: str = ""

    def __fspath__(self) -> str:  # usable anywhere a path is expected
        return self.path

    def __str__(self) -> str:
        return self.path


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


def _progress_hook(callback: Callable[[float, str], None]):
    def hook(status: dict) -> None:
        if status.get("status") != "downloading":
            return
        total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
        got = status.get("downloaded_bytes") or 0
        fraction = (got / total) if total else 0.0
        detail = f"{got / 1e6:.1f} MB"
        if total:
            detail += f" of {total / 1e6:.1f} MB"
        speed = status.get("speed")
        if speed:
            detail += f" · {speed / 1e6:.1f} MB/s"
        callback(min(fraction, 1.0), detail)

    return hook


def download_youtube(url: str, output_dir: str,
                     progress_cb: Optional[Callable[[float, str], None]] = None) -> Download:
    os.makedirs(output_dir, exist_ok=True)
    outtmpl = os.path.join(output_dir, "video.%(ext)s")
    cookies_path = os.environ.get("YTDLP_COOKIES")

    # H.264 first: YouTube serves AV1 for many of these uploads and OpenCV's
    # bundled FFmpeg cannot decode it, so the fastest download is useless. Fall
    # back to unconstrained formats rather than failing outright.
    formats = [
        "bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]",
        "bestvideo*+bestaudio/best",
        "bestvideo+bestaudio/best",
        "best",
    ]
    last_error = None
    info = None
    for fmt in formats:
        opts = _build_opts(outtmpl, fmt, cookies_path)
        if progress_cb:
            opts["progress_hooks"] = [_progress_hook(progress_cb)]
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
            break
        except Exception as exc:
            last_error = exc

    if info is None:
        raise last_error

    title = (info.get("title") or "").strip()
    # Kept beside the video so a cached copy still knows what it is on a re-run.
    if title:
        try:
            with open(os.path.join(output_dir, "title.txt"), "w", encoding="utf-8") as fh:
                fh.write(title)
        except OSError:
            pass

    try:
        return Download(_resolve_downloaded_video(output_dir), title)
    except FileNotFoundError:
        ext = info.get("ext", "mp4")
        candidate = os.path.join(output_dir, f"video.{ext}")
        if os.path.exists(candidate):
            return Download(candidate, title)
        raise


def cached_title(output_dir: str) -> str:
    """The stored title for an already-downloaded video, if there is one."""
    try:
        with open(os.path.join(output_dir, "title.txt"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""
