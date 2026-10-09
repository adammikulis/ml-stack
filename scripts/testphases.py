"""Where a pytest run's wall time went: collection, session setup, tests and teardown.

Loaded into every ``scripts/test`` run as a pytest plugin. It only reads the reports pytest
already makes, so it adds no work to the run; the controller writes the phases to the file
named by ``PHASES_ENV`` and ``scripts/test`` prints them after the run.

* collection: from the plugin loading to the first test starting (worker start and collection).
* setup: the longest first-test setup on any worker, which is where session fixtures run.
* tests: the test calls, averaged over the workers.
* teardown: the longest last-test teardown on any worker, where session fixtures finish.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

PHASES_ENV = "MLSTACK_PHASES_FILE"
OVERHEAD_SECONDS = 5.0
OVERHEAD_SHARE = 0.25
SHARE_FLOOR_SECONDS = 3.0
"""A share of wall time only counts once the overhead is also this large: a one-test run is all overhead."""


@dataclass(frozen=True)
class Phases:
    """One run's seconds by phase; ``wall`` is the plugin's own span, queue wait excluded."""

    wall: float
    collection: float
    setup: float
    tests: float
    teardown: float

    @property
    def fixed(self) -> float:
        """Everything except the test calls."""
        return max(0.0, self.wall - self.tests)


def line(phases: Phases) -> str:
    """The one-line breakdown, with a warning when the fixed overhead is large."""
    text = (f"test: phases collection {phases.collection:.1f} s, session setup {phases.setup:.1f} s, "
            f"tests {phases.tests:.1f} s, teardown {phases.teardown:.1f} s "
            f"(fixed {phases.fixed:.1f} s of {phases.wall:.1f} s)")
    if too_slow(phases):
        text += (f"\ntest: WARNING fixed overhead {phases.fixed:.1f} s is over {OVERHEAD_SECONDS:g} s "
                 f"or {OVERHEAD_SHARE:.0%} of wall time: find the fixture or import that is slow")
    return text


def too_slow(phases: Phases) -> bool:
    """Whether the fixed overhead is past the absolute or the relative limit."""
    return phases.fixed > OVERHEAD_SECONDS or (
        phases.fixed > SHARE_FLOOR_SECONDS and phases.fixed > OVERHEAD_SHARE * phases.wall)


def read(path: Path) -> Phases | None:
    """The phases a run wrote, or None when it wrote none."""
    try:
        return Phases(**json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, TypeError):
        return None


class Recorder:
    """Collects phase times from the reports the controller receives."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.began = time.time()
        self.first_start = 0.0
        self.calls = 0.0
        self.first_setup: dict[str, float] = {}
        self.last_teardown: dict[str, float] = {}

    def pytest_runtest_logreport(self, report) -> None:
        """Note one report's duration under its phase and worker."""
        node = getattr(report, "node", None)
        worker = node.gateway.id if node is not None else "main"
        if not self.first_start:
            self.first_start = getattr(report, "start", time.time())
        if report.when == "setup":
            self.first_setup.setdefault(worker, report.duration)
        elif report.when == "call":
            self.calls += report.duration
        elif report.when == "teardown":
            self.last_teardown[worker] = report.duration

    def pytest_sessionfinish(self, session) -> None:
        """Write the phases to the file ``scripts/test`` reads."""
        workers = max(1, len(self.first_setup))
        collection = max(0.0, (self.first_start or time.time()) - self.began)
        phases = Phases(time.time() - self.began, collection, max(self.first_setup.values(), default=0.0),
                        self.calls / workers, max(self.last_teardown.values(), default=0.0))
        Path(self.path).write_text(json.dumps(asdict(phases)), encoding="utf-8")


def pytest_configure(config) -> None:
    """Start recording on the controller, and on nothing else."""
    path = os.environ.get(PHASES_ENV)
    if path and not hasattr(config, "workerinput"):
        config.pluginmanager.register(Recorder(path), "mlstack-phases")
