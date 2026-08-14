"""Where a song's files live.

A folder named after the song is worth more than one named after a hash of its
URL — it is the only name you see in a file browser. But the title is not known
until the video has been fetched, so a run starts in a working folder keyed to
its source and is renamed once the title arrives.

An index maps each source to its folder, so a second run finds the first one
under whatever name it ended up with and does not fetch it again.
"""
import hashlib
import json
import os
import re
from typing import Dict, Optional

INDEX_NAME = ".library.json"
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_SPACE = re.compile(r"\s+")


def slug(title: str, limit: int = 80) -> str:
    """A title turned into a folder name, keeping it readable.

    Only the characters a filesystem refuses are removed — spaces, commas and
    brackets are what make a title legible at a glance, and mangling them into
    underscores buys nothing.
    """
    name = _UNSAFE.sub("", title)
    name = _SPACE.sub(" ", name).strip(" .")
    if len(name) > limit:
        name = name[:limit].rstrip(" .")
    return name or "untitled"


def _index_path(root: str) -> str:
    return os.path.join(root, INDEX_NAME)


def load_index(root: str) -> Dict[str, str]:
    try:
        with open(_index_path(root), encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _save_index(root: str, index: Dict[str, str]) -> None:
    try:
        os.makedirs(root, exist_ok=True)
        with open(_index_path(root), "w", encoding="utf-8") as handle:
            json.dump(index, handle, indent=2, ensure_ascii=False)
    except OSError:
        pass


def working_dir(source: str, root: str) -> str:
    """The folder to read this source into — the one it already has, or a new one."""
    known = load_index(root).get(source)
    if known:
        candidate = os.path.join(root, known)
        if os.path.isdir(candidate):
            return candidate
    found = _search_for_source(source, root)
    if found:
        return found
    return os.path.join(root, hashlib.sha1(source.encode()).hexdigest()[:12])


def _search_for_source(source: str, root: str) -> Optional[str]:
    """Look for a sheet already made from this source but not in the index.

    Each sheet records the source it came from, so the folders can be searched
    when the index cannot answer — for anything read before the index existed,
    or if it is lost. Without this the same song is fetched again and lands
    beside itself under a numbered name.
    """
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name, "tabs.json")
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as handle:
                meta = json.load(handle).get("metadata") or {}
        except (OSError, ValueError):
            continue
        if meta.get("source_url") == source:
            return os.path.join(root, name)
    return None


def finished_dir(source: str, root: str, current: str, title: str) -> str:
    """Rename a finished run to its song's name and remember where it went.

    A name already taken by another song gets a numbered suffix rather than
    silently overwriting somebody else's sheet.
    """
    if not title:
        return current
    wanted = slug(title)
    target = os.path.join(root, wanted)
    if os.path.abspath(target) != os.path.abspath(current):
        suffix = 2
        while os.path.exists(target):
            target = os.path.join(root, f"{wanted} ({suffix})")
            suffix += 1
        try:
            os.replace(current, target)
        except OSError:
            target = current
    index = load_index(root)
    index[source] = os.path.basename(target)
    _save_index(root, index)
    return target


def existing_sheet(source: str, root: str) -> Optional[str]:
    """The folder holding this source's finished sheet, if it has one."""
    folder = working_dir(source, root)
    return folder if os.path.exists(os.path.join(folder, "tabs.json")) else None
