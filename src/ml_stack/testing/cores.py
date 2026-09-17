"""How many xdist workers a suite may take, asked of the machine's broker.

Several suites at once on one machine each taking every core is what makes every one of
them slow and every timing taken beside them meaningless. `pytest_workers` asks the broker
for cores and gets what is free, and the grant is released when this process ends. With no
broker running -- a fresh clone, a machine that never started one -- the caller's own
default stands: the cost of being wrong here is a slow run, never a wrong answer.
"""

from __future__ import annotations

import os

from ml_stack.serve import broker_wire
from ml_stack.serve.broker import BrokerError

__all__ = ["pytest_workers", "workers_for"]


def workers_for(default: int) -> int:
    """Cores this process may run tests on, or ``default`` when no broker answers."""
    try:
        return int(broker_wire.cores(default)["cores"])
    except (BrokerError, OSError, ValueError, KeyError):
        return default


def pytest_workers(config: object, default: int | None = None) -> int | None:
    """Set ``-n`` from the broker unless the command line named it. Returns what was set.

    A pytest plugin calls this from ``pytest_configure``; a run that passed ``-n`` itself
    means it, and is left alone.
    """
    asked = getattr(config, "invocation_params", None)
    args = list(getattr(asked, "args", ()) or ())
    if any(a == "-n" or a.startswith(("-n", "--numprocesses")) for a in args):
        return None
    workers = workers_for(default or os.cpu_count() or 1)
    config.option.numprocesses = workers  # type: ignore[attr-defined]
    return workers
