"""Running a list of videos in one go.

A queue is a text file of one video per line — a URL or a path — with blank
lines and anything after a # ignored, so a list can be annotated and pruned
without deleting it.

Every item gets its own directory and its own sheet, and a failure is recorded
and stepped over rather than ending the run. One video that is offered only in a
codec this build cannot read should not cost you the eighteen after it.
"""
import os
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.app.config import Config
from src.app.progress import ProgressFn


@dataclass
class QueueItem:
    target: str
    kind: str                       # "url" or "path"
    title: str = ""
    output_dir: str = ""
    notes: int = 0
    measures: int = 0
    coverage: Optional[float] = None
    seconds: float = 0.0
    error: str = ""
    skipped: bool = False

    @property
    def ok(self) -> bool:
        return not self.error and not self.skipped


@dataclass
class QueueResult:
    items: List[QueueItem] = field(default_factory=list)

    @property
    def done(self) -> List[QueueItem]:
        return [i for i in self.items if i.ok]

    @property
    def failed(self) -> List[QueueItem]:
        return [i for i in self.items if i.error]

    @property
    def skipped(self) -> List[QueueItem]:
        return [i for i in self.items if i.skipped]


def read_queue(path: str) -> List[str]:
    """One target per line. Blank lines and #-comments are ignored."""
    out: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.split("#", 1)[0].strip()
            if line:
                out.append(line)
    return out


def run_queue(targets: List[str], output_root: str, workers: int = 0,
              on_item: Optional[Callable[[QueueItem, int, int], None]] = None,
              progress_cb: Optional[ProgressFn] = None,
              skip_existing: bool = True) -> QueueResult:
    """Read every target in turn, carrying on past whatever fails."""
    from src.app.library import existing_sheet, finished_dir, working_dir
    from src.app.main import run_pipeline
    from src.app.tui import normalise_target

    result = QueueResult()
    for index, raw in enumerate(targets):
        kind, target = normalise_target(raw)
        item = QueueItem(target=target or raw, kind=kind or "url")
        result.items.append(item)
        if not kind:
            item.error = "not a URL or a file path"
            if on_item:
                on_item(item, index, len(targets))
            continue
        if kind == "path" and not os.path.exists(target):
            item.error = f"no such file: {target}"
            if on_item:
                on_item(item, index, len(targets))
            continue

        item.output_dir = working_dir(target, output_root)
        done_already = existing_sheet(target, output_root)
        if skip_existing and done_already:
            item.output_dir = done_already
            # Already read. Re-running a queue after adding a line to it should
            # cost the new line, not the whole list again.
            item.skipped = True
            item.title = _stored_title(item.output_dir) or os.path.basename(item.output_dir)
            if on_item:
                on_item(item, index, len(targets))
            continue

        os.makedirs(item.output_dir, exist_ok=True)
        config = Config(output_dir=item.output_dir, num_workers=workers)
        started = time.monotonic()
        try:
            sheet = run_pipeline(target if kind == "url" else "", config,
                                 progress_cb,
                                 video_path=target if kind == "path" else None)
            item.notes = len(sheet.notes)
            item.measures = len(sheet.measures)
            item.title = sheet.metadata.get("title", "")
            item.coverage = _stored_coverage(item.output_dir)
            # Renamed once the title is known: it cannot be known sooner, and a
            # folder called by the song's name is the only one you can find.
            item.output_dir = finished_dir(target, output_root,
                                           item.output_dir, item.title)
        except BaseException as exc:                 # noqa: BLE001 — recorded, not raised
            item.error = str(exc).strip().splitlines()[0][:200] if str(exc) else type(exc).__name__
        item.seconds = time.monotonic() - started
        if on_item:
            on_item(item, index, len(targets))
    return result


def _stored_title(output_dir: str) -> str:
    try:
        with open(os.path.join(output_dir, "title.txt"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _stored_coverage(output_dir: str) -> Optional[float]:
    import json
    try:
        with open(os.path.join(output_dir, "sampling_report.json"), encoding="utf-8") as fh:
            return json.load(fh).get("coverage")
    except (OSError, ValueError):
        return None
