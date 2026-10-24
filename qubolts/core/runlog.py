"""Append-only run log. Every solve can be recorded so backends are compared honestly.

One JSON object per line. Load it with pandas (``pd.read_json(path, lines=True)``)
or Spark (``spark.read.json(path)``) and you have the benchmark table.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from typing import Any


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, **fields: Any) -> dict[str, Any]:
        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "host": platform.node(),
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=_default) + "\n")
        return row

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]


def _default(o: Any) -> Any:
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)
