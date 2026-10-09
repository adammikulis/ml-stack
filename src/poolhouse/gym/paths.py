"""Simulation artifact directories."""

from poolhouse.home import cache


def artifact_root():
    """Return the directory holding simulation recordings."""
    return cache("gym")
