import logging
import os
import time
from dataclasses import dataclass
from glob import glob
from typing import Callable, List, Optional, Tuple

import yt_dlp


logger = logging.getLogger(__name__)

# Tried in order. Every H.264 rung is exhausted before anything unconstrained,
# because a codec this build cannot decode is worse than a smaller picture:
# OpenCV reads AV1 as zero frames, which used to yield a confidently empty sheet.
FORMAT_LADDER: List[Tuple[str, str]] = [
    ("H.264 up to 1080p",
     "bestvideo[vcodec^=avc1][height<=1080]+bestaudio/best[vcodec^=avc1][height<=1080]"),
    ("H.264 any size", "bestvideo[vcodec^=avc1]+bestaudio/best[vcodec^=avc1]"),
    ("H.264 pre-muxed", "best[vcodec^=avc1]"),
    ("any codec", "bestvideo*+bestaudio/best"),
    ("whatever is offered", "best"),
]
DECODABLE_RUNGS = 3          # how many of the above are known-decodable
ATTEMPTS_PER_FORMAT = 3
RETRY_PAUSE_S = 4.0


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

    last_error = None
    info = None
    for rung, (label, fmt) in enumerate(FORMAT_LADDER):
        # Retry a rung before abandoning it. YouTube answers 403 to a perfectly
        # good format under load, and dropping to the next rung on one refusal
        # trades a decodable video for an undecodable one over a passing hiccup.
        for attempt in range(ATTEMPTS_PER_FORMAT):
            opts = _build_opts(outtmpl, fmt, cookies_path)
            if progress_cb:
                opts["progress_hooks"] = [_progress_hook(progress_cb)]
            try:
                with yt_dlp.YoutubeDL(opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                break
            except Exception as exc:
                last_error = exc
                if attempt + 1 < ATTEMPTS_PER_FORMAT:
                    logger.info("%s attempt %d failed (%s); retrying",
                                label, attempt + 1, str(exc)[:120])
                    time.sleep(RETRY_PAUSE_S)
        if info is not None:
            if rung:
                # Say so. A silent downgrade is how a codec this build cannot
                # read got accepted while the preference looked to be in force.
                logger.warning("could not get %s (%s); fell back to %s",
                               FORMAT_LADDER[0][0], str(last_error)[:120], label)
            if rung >= DECODABLE_RUNGS:
                logger.warning("%s may not be decodable here — if the read finds "
                               "no frames, this is why", label)
            break

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
