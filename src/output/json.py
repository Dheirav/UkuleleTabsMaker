import json
from dataclasses import asdict
from typing import Any

from src.models.schema import TabSheet


def write_json(sheet: TabSheet, path: str) -> None:
    data: Any = asdict(sheet)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
