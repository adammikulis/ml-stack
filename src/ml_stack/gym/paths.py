"""Simulation artifact directories."""

from ml_stack.home import cache


def artifact_root():
    """Return the directory holding simulation recordings."""
    return cache("gym")
