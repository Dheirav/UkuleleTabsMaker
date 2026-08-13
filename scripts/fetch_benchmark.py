"""Fetch the benchmark clips listed in benchmark/clips.json.

Videos are local test fixtures only; they are not committed.
"""
import json
import os
import sys

import yt_dlp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CLIPS = os.path.join(ROOT, "benchmark", "clips.json")
VIDEOS = os.path.join(ROOT, "benchmark", "videos")


def fetch(clip: dict) -> str:
    target = os.path.join(VIDEOS, f"{clip['id']}.mp4")
    if os.path.exists(target):
        print(f"  have {clip['id']}")
        return target
    opts = {
        "outtmpl": os.path.join(VIDEOS, f"{clip['id']}.%(ext)s"),
        # H.264 only: OpenCV cannot reliably decode AV1 on many platforms, and a
        # benchmark that silently fails to decode is worse than one that refuses to.
        "format": ("bestvideo[vcodec^=avc1][height<=1080]+bestaudio/"
                   "best[vcodec^=avc1][height<=1080]/"
                   "bestvideo[ext=mp4][height<=1080]+bestaudio/best"),
        "merge_output_format": "mp4",
        "quiet": True,
        "noplaylist": True,
        "retries": 3,
    }
    client = os.environ.get("YTDLP_YOUTUBE_CLIENT", "").strip()
    if client:
        opts["extractor_args"] = {"youtube": {"player_client": [client]}}
    cookies = os.environ.get("YTDLP_COOKIES")
    if cookies and os.path.exists(cookies):
        opts["cookiefile"] = cookies
    url = f"https://www.youtube.com/watch?v={clip['yt']}"
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.extract_info(url, download=True)
    print(f"  got  {clip['id']}")
    return target


def main() -> None:
    os.makedirs(VIDEOS, exist_ok=True)
    clips = json.load(open(CLIPS))
    wanted = sys.argv[1:] or [c["id"] for c in clips]
    for clip in clips:
        if clip["id"] not in wanted:
            continue
        try:
            fetch(clip)
        except Exception as exc:
            print(f"  FAIL {clip['id']}: {exc}")


if __name__ == "__main__":
    main()
