"""A durable record of what a run actually did."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO


@dataclass
class MetricsLog:
    """Append-only JSONL metrics for one run."""

    path: Path
    _handle: TextIO | None = field(default=None, repr=False)
    _start: float = field(default_factory=time.monotonic, repr=False)

    def __init__(self, path: Path | str, *, resume: bool = False) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("a" if resume else "w", encoding="utf-8")
        self._start = time.monotonic()

    def __enter__(self) -> "MetricsLog":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    def write(self, event: str, **fields: Any) -> None:
        """One record. Flushed immediately -- a buffered record is a lost record."""
        if self._handle is None:
            raise RuntimeError("metrics log is closed")
        record = {"event": event, "elapsed_s": round(self.elapsed_s, 3), **fields}
        self._handle.write(json.dumps(record, default=str) + "\n")
        self._handle.flush()

    def start(self, config: dict[str, Any], **fields: Any) -> None:
        """Record the resolved config. Call once, before the first step."""
        self.write("start", config=config, **fields)

    def step(self, step: int, **fields: Any) -> None:
        self.write("step", step=step, **fields)

    def eval(self, step: int, **fields: Any) -> None:
        self.write("eval", step=step, **fields)

    def note(self, message: str, **fields: Any) -> None:
        """Something worth recording that is not a metric -- a stall, a skipped batch."""
        self.write("note", message=message, **fields)

    def finish(self, **fields: Any) -> None:
        self.write("finish", **fields)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def read(path: Path | str) -> list[dict[str, Any]]:
    """Parse a metrics log, skipping any trailing partial line."""
    records: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


@dataclass
class Throughput:
    """Tokens (or samples) per second, over a window rather than since the start."""

    window: int = 50
    _times: list[float] = field(default_factory=list, repr=False)
    _counts: list[int] = field(default_factory=list, repr=False)

    def record(self, count: int, duration_s: float) -> None:
        self._counts.append(int(count))
        self._times.append(float(duration_s))
        if len(self._times) > self.window:
            self._times.pop(0)
            self._counts.pop(0)

    @property
    def per_second(self) -> float:
        total = sum(self._times)
        return sum(self._counts) / total if total > 0 else 0.0
