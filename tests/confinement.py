"""Marker for tests that only run inside the macOS confinement kernel."""
import os

import pytest


def needs_confinement(test):
    """Skip ``test`` unless the confinement kernel launched this run."""
    return pytest.mark.skipif("DEV_TEST_CONFINEMENT_CANARY" not in os.environ,
                              reason="runs under scripts/test --confine")(test)
