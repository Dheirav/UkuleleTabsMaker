"""Interactive terminal front end.

Running the extractor should not feel like watching a log file. This drives the
pipeline from a prompt and renders one live block: an overall bar, elapsed time,
and a checklist of stages with whatever each one is doing right now. Log records
are captured rather than printed so they cannot tear through the display; real
warnings are shown afterwards.
"""
import hashlib
import logging
import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from src.app.config import Config
from src.app.progress import STAGES, STAGE_LABELS
from src.models.schema import TabSheet


RESET = "\x1b[0m"
DIM = "\x1b[2m"
BOLD = "\x1b[1m"
GREEN = "\x1b[32m"
CYAN = "\x1b[36m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"

_URL_HINTS = ("youtube.com", "youtu.be", "http://", "https://")


# --- input handling ---------------------------------------------------------

def normalise_target(text: str) -> Tuple[str, str]:
    """Classify what the user typed as ("url" | "path" | "", value).

    Accepts Windows and WSL paths as pasted, because that is how a path arrives
    when it is dragged in from Explorer. Bash eats the backslashes before the
    program ever sees them; a prompt does not, so this is the one place they can
    still be understood.
    """
    value = text.strip().strip('"').strip("'").strip()
    if not value:
        return "", ""

    lowered = value.lower()
    if any(hint in lowered for hint in _URL_HINTS):
        url = value if lowered.startswith(("http://", "https://")) else f"https://{value}"
        if urlparse(url).netloc:
            return "url", url

    return "path", _to_posix_path(value)


def _to_posix_path(value: str) -> str:
    r"""Map \\wsl.localhost\Distro\home\me\x and C:\x onto paths this side sees."""
    unc = re.match(r"^\\\\wsl(?:\.localhost|\$)\\[^\\]+(\\.*)$", value, re.IGNORECASE)
    if unc:
        return unc.group(1).replace("\\", "/")
    drive = re.match(r"^([A-Za-z]):[\\/](.*)$", value)
    if drive:
        return f"/mnt/{drive.group(1).lower()}/{drive.group(2).replace(chr(92), '/')}"
    if "\\" in value and "/" not in value:
        return value.replace("\\", "/")
    return os.path.expanduser(value)


def _run_dir(kind: str, value: str, root: str) -> str:
    if kind == "url":
        return os.path.join(root, hashlib.sha1(value.encode()).hexdigest()[:12])
    stem = os.path.splitext(os.path.basename(value))[0] or "video"
    return os.path.join(root, re.sub(r"[^\w.-]+", "-", stem))


# --- live display -----------------------------------------------------------

@dataclass
class _State:
    stages: List[str]
    stage: str = ""
    overall: float = 0.0
    details: dict = field(default_factory=dict)
    finished: bool = False


class Screen:
    """Repaints a fixed block of lines in place."""

    def __init__(self, stream=None) -> None:
        self.stream = stream or sys.stdout
        self.tty = bool(getattr(self.stream, "isatty", lambda: False)()) \
            and not os.environ.get("NO_COLOR")
        self.height = 0
        self._last = 0.0

    def width(self) -> int:
        return max(shutil.get_terminal_size((80, 24)).columns, 40)

    def paint(self, lines: List[str], force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.08:
            return
        self._last = now
        if not self.tty:
            return
        out = []
        if self.height:
            out.append(f"\x1b[{self.height}A")
        for line in lines:
            out.append("\x1b[2K" + line + "\n")
        self.stream.write("".join(out))
        self.stream.flush()
        self.height = len(lines)

    def release(self) -> None:
        """Leave the block on screen and stop repainting over it."""
        self.height = 0

    def hide_cursor(self) -> None:
        if self.tty:
            self.stream.write("\x1b[?25l")
            self.stream.flush()

    def show_cursor(self) -> None:
        if self.tty:
            self.stream.write("\x1b[?25h")
            self.stream.flush()

    def paint_colour(self, text: str, colour: str) -> str:
        return f"{colour}{text}{RESET}" if self.tty else text

    def write(self, text: str = "") -> None:
        self.stream.write(text + "\n")
        self.stream.flush()


def _bar(fraction: float, width: int) -> str:
    filled = int(round(min(max(fraction, 0.0), 1.0) * width))
    return "█" * filled + "░" * (width - filled)


def _elapsed(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def _compose(screen: Screen, state: _State, title: str, started: float) -> List[str]:
    width = min(screen.width() - 4, 76)
    order = state.stages
    position = order.index(state.stage) if state.stage in order else len(order)

    lines = ["", f"  {screen.paint_colour(_clip(title, width), BOLD)}", ""]
    bar_width = max(width - 14, 10)
    lines.append(f"  {screen.paint_colour(_bar(state.overall, bar_width), CYAN)} "
                 f"{int(state.overall * 100):3d}%  {_elapsed(time.monotonic() - started)}")
    lines.append("")
    label_width = 18
    detail_width = max(width - label_width - 4, 12)
    for i, key in enumerate(order):
        label = STAGE_LABELS.get(key, key)
        detail = _clip(state.details.get(key, ""), detail_width)
        if state.finished or i < position:
            mark = screen.paint_colour("✓", GREEN)
            body = f"{label:<{label_width}}"
            body += f" {screen.paint_colour(detail, DIM)}" if detail else ""
        elif i == position:
            mark = screen.paint_colour("▸", CYAN)
            body = screen.paint_colour(f"{label:<{label_width}}", BOLD)
            body += f" {detail}" if detail else ""
        else:
            mark = " "
            body = screen.paint_colour(label, DIM)
        lines.append(f"  {mark} {body}".rstrip())
    lines.append("")
    return lines


def _clip(text: str, width: int) -> str:
    return text if len(text) <= width else text[: max(width - 1, 1)] + "…"


# --- the app ----------------------------------------------------------------

class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _run_job(kind: str, target: str, config: Config, screen: Screen,
             title: str) -> Tuple[Optional[TabSheet], Optional[BaseException],
                                  List[logging.LogRecord]]:
    from src.app.main import run_pipeline

    stages = [key for key, _ in STAGES]
    if kind == "path":
        stages.remove("download")
    state = _State(stages=stages)
    result: dict = {}

    def on_progress(stage: str, value: float, detail: str = "") -> None:
        state.stage = stage
        state.overall = value
        if detail:
            state.details[stage] = detail

    def work() -> None:
        try:
            result["sheet"] = run_pipeline(
                target if kind == "url" else "", config, on_progress,
                video_path=target if kind == "path" else None)
        except BaseException as exc:  # reported in the caller's frame
            result["error"] = exc

    capture = _Capture()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = [capture]
    root.setLevel(logging.INFO)

    started = time.monotonic()
    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    screen.hide_cursor()
    try:
        while thread.is_alive():
            screen.paint(_compose(screen, state, title, started))
            time.sleep(0.05)
        state.finished = "error" not in result
        if state.finished:
            state.overall = 1.0
        screen.paint(_compose(screen, state, title, started), force=True)
    except KeyboardInterrupt:
        screen.release()
        screen.show_cursor()
        raise
    finally:
        root.handlers, root.level = saved_handlers, saved_level
        screen.show_cursor()
    screen.release()
    return result.get("sheet"), result.get("error"), capture.records


def _report(output_dir: str) -> dict:
    import json
    try:
        with open(os.path.join(output_dir, "sampling_report.json"), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _summarise(screen: Screen, sheet: TabSheet, config: Config,
               records: List[logging.LogRecord]) -> None:
    from src.output.text import build_systems

    report = _report(config.output_dir)
    facts = []
    if report.get("pages"):
        facts.append(f"{int(report['pages'])} pages")
    if report.get("measures_highlighted"):
        facts.append(f"{int(report['measures_with_notes'])} of "
                     f"{int(report['measures_highlighted'])} measures read")
    facts.append(f"{len(sheet.notes)} notes")
    if sheet.metadata.get("font"):
        facts.append(f"font {sheet.metadata['font']}")
    screen.write("  " + screen.paint_colour(" · ".join(facts), DIM))

    coverage = report.get("coverage")
    if coverage is not None and coverage < 0.85:
        colour = YELLOW if coverage >= 0.6 else RED
        note = ("some passages are missing" if coverage >= 0.6
                else "most of this piece is missing — treat it as a partial draft")
        screen.write("  " + screen.paint_colour(
            f"! only {coverage * 100:.0f}% of the measures were read, {note}", colour))

    warnings = [r for r in records if r.levelno >= logging.WARNING]
    for record in warnings[:5]:
        screen.write("  " + screen.paint_colour(f"! {record.getMessage()}", YELLOW))

    screen.write()
    systems = build_systems(sheet, config)
    for system in systems[:2]:
        for label, line in system:
            screen.write(f"  {label:>2}|{line}")
        screen.write()
    if len(systems) > 2:
        screen.write("  " + screen.paint_colour(
            f"… {len(systems) - 2} more systems in the files below", DIM))
        screen.write()

    for kind in ("txt", "pdf", "json"):
        screen.write("  " + screen.paint_colour(
            os.path.join(config.output_dir, f"tabs.{kind}"), DIM))
    screen.write()


def run_app(output_root: str = "./outputs", workers: int = 0) -> None:
    screen = Screen()
    screen.write()
    screen.write("  " + screen.paint_colour("Ukulele Tabs", BOLD)
                 + screen.paint_colour("  ·  YouTube tab video → tab sheet", DIM))
    screen.write("  " + screen.paint_colour(
        "Paste a YouTube URL or a video file path. Enter on its own quits.", DIM))
    screen.write()

    while True:
        try:
            raw = input("  > ")
        except (EOFError, KeyboardInterrupt):
            screen.write()
            return
        kind, target = normalise_target(raw)
        if not kind:
            screen.write()
            return
        if kind == "path" and not os.path.exists(target):
            screen.write("  " + screen.paint_colour(f"No such file: {target}", RED))
            screen.write()
            continue

        config = Config(output_dir=_run_dir(kind, target, output_root),
                        num_workers=workers)
        os.makedirs(config.output_dir, exist_ok=True)
        if kind == "url":
            cached = _cached_video(config.output_dir)
            if cached:
                screen.write("  " + screen.paint_colour(
                    "already downloaded, reading the local copy", DIM))
                kind, target = "path", cached

        title = os.path.basename(target) if kind == "path" else target
        try:
            sheet, error, records = _run_job(kind, target, config, screen, title)
        except KeyboardInterrupt:
            screen.write()
            screen.write("  " + screen.paint_colour("stopped", YELLOW))
            screen.write()
            return

        if error is not None:
            screen.write("  " + screen.paint_colour(f"Failed: {error}", RED))
            screen.write()
        elif sheet is not None:
            _summarise(screen, sheet, config, records)


def _cached_video(output_dir: str) -> Optional[str]:
    for name in sorted(os.listdir(output_dir)) if os.path.isdir(output_dir) else []:
        if name.startswith("video.") and os.path.getsize(os.path.join(output_dir, name)) > 0:
            return os.path.join(output_dir, name)
    return None
