from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
import numpy as np


BBox = Tuple[int, int, int, int]


@dataclass
class FrameSample:
    frame_id: int
    timestamp: float
    image: np.ndarray


@dataclass
class TabRegion:
    bbox: BBox
    confidence: float


@dataclass
class DigitDetection:
    value: int
    bbox: BBox
    confidence: float
    string_index: int
    x_center: int


@dataclass
class TabFrame:
    timestamp: float
    roi: np.ndarray
    region: TabRegion
    string_lines: List[int]
    bar_lines: List[int]
    digits: List[DigitDetection]


@dataclass
class Note:
    time: float
    string_index: int
    fret: int
    confidence: float


@dataclass
class Measure:
    start_time: float
    end_time: float
    notes: List[Note] = field(default_factory=list)


@dataclass
class TabSheet:
    notes: List[Note]
    measures: List[Measure]
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class ReconstructionResult:
    notes: List[Note]
    bar_times: List[float]
    speed_px_per_s: Optional[float]
