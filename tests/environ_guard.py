"""One test's change to ``os.environ`` is undone before the next test starts.

Several entry points set a marker for the rest of their process (a hook sets
``POOLHOUSE_NONINTERACTIVE``, a worker sets ``POOLHOUSE_AGENT``) because each runs as its own
process. A test that calls one in-process leaves that marker behind, and on a serial run
(``-n 0``) every later test that registers a person sees an agent and is refused. The
fixture is imported by ``conftest.py`` and applies to every test.
"""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def environment_is_restored():
    """Put ``os.environ`` back as it was when the test began."""
    before = dict(os.environ)
    yield
    if dict(os.environ) != before:
        os.environ.clear()
        os.environ.update(before)
