"""What the test runner does beyond running the tests it was asked for: behind the `test-runner-extras` feature, off by default.

Plain `scripts/test` runs the named or affected tests, in the order pytest finds them. With the feature on
(`ml-stack features enable test-runner-extras`) it also starts the files that took longest last time first
(`testorder`) and remembers each full run's time for `scripts/test ratchet`.
"""
from __future__ import annotations

from pathlib import Path

import testhistory

from ml_stack import features

NAME = "test-runner-extras"


def on() -> bool:
    """Whether this machine turned the extras on."""
    return features.enabled(NAME)


def learned_order(root: Path, environment: dict, plugins: list[str]) -> str | None:
    """Reorder by the recorded durations when the extras are on and there are some: the snapshot's path, or None.

    Adds the ``testorder`` plugin to ``plugins`` and the snapshot to ``environment`` as a side effect.
    """
    if "DEV_TEST_ORDER" in environment or not on():
        return None
    recorded = testhistory.load(testhistory.history_path(root))
    if not recorded:
        return None
    import testorder  # a pytest plugin: imported only when a run is about to use it
    order = testorder.write(recorded)
    environment["DEV_TEST_ORDER"] = order
    plugins.extend(["-p", "testorder"])
    return order
