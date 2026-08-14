"""Stage-weighted progress reporting.

The pipeline's stages take wildly different amounts of time, so a bar driven by
"stages finished" jumps and then stalls. Each stage instead owns a band of the
overall figure proportional to how long it actually takes, and reports its own
fraction inside that band. Callers get one monotonic 0..1 number plus a detail
string describing what is happening right now.
"""
from typing import Callable, Dict, List, Optional, Sequence, Tuple


ProgressFn = Callable[[str, float, str], None]

# (key, label) in the order they run. The label is what a user sees.
STAGES: List[Tuple[str, str]] = [
    ("download", "download video"),
    ("probe", "inspect video"),
    ("scan", "scan frames"),
    ("pages", "build pages"),
    ("read", "read notation"),
    ("timing", "work out timing"),
    ("write", "write files"),
]

# Rough share of wall-clock, measured on the benchmark clips.
WEIGHTS: Dict[str, float] = {
    "download": 30.0,
    "probe": 6.0,
    "scan": 32.0,
    "pages": 18.0,
    "read": 8.0,
    "timing": 3.0,
    "write": 3.0,
}

STAGE_LABELS: Dict[str, str] = dict(STAGES)


class Progress:
    """Reports a stage's own 0..1 fraction as a share of the whole run."""

    def __init__(self, cb: Optional[ProgressFn] = None,
                 stages: Optional[Sequence[str]] = None) -> None:
        self.cb = cb
        self.stages: List[str] = list(stages) if stages else [k for k, _ in STAGES]
        total = sum(WEIGHTS.get(s, 1.0) for s in self.stages) or 1.0
        self.bands: Dict[str, Tuple[float, float]] = {}
        acc = 0.0
        for stage in self.stages:
            share = WEIGHTS.get(stage, 1.0) / total
            self.bands[stage] = (acc, acc + share)
            acc += share
        self.current: str = self.stages[0] if self.stages else "done"
        self.detail: str = ""

    def stage(self, name: str, detail: str = "") -> None:
        """Enter a stage. Unknown stages still report, at the tail of the bar."""
        self.current = name
        self.detail = detail
        self.tick(0.0, detail)

    def tick(self, fraction: float, detail: Optional[str] = None) -> None:
        if detail is not None:
            self.detail = detail
        low, high = self.bands.get(self.current, (1.0, 1.0))
        fraction = min(max(fraction, 0.0), 1.0)
        self._emit(self.current, low + (high - low) * fraction, self.detail)

    def note(self, detail: str) -> None:
        """Update the detail text without claiming extra progress."""
        low, high = self.bands.get(self.current, (1.0, 1.0))
        self.detail = detail
        self._emit(self.current, low, detail)

    def counter(self, total: int) -> Callable[[int], None]:
        """A per-item callback for stages that loop a known number of times."""
        total = max(int(total), 1)

        def report(done: int) -> None:
            self.tick(done / total, f"{min(done, total)} of {total}")

        return report

    def done(self, detail: str = "") -> None:
        self.current = "done"
        self.detail = detail
        self._emit("done", 1.0, detail)

    def _emit(self, stage: str, value: float, detail: str) -> None:
        if self.cb is not None:
            self.cb(stage, value, detail)
