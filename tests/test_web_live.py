"""One smoke test against the real internet, run only on purpose: ``POOLHOUSE_NET=1``.

It finds a regulator's datasheet from its part number, then reads the pin table and the
package pages out of the PDF.
"""

from __future__ import annotations

import os

import pytest

from poolhouse.datasheet import find
from poolhouse.sources import datasheet


def test_a_datasheet_is_found_downloaded_and_read(tmp_path):
    if not os.environ.get("POOLHOUSE_NET"):
        pytest.skip("set POOLHOUSE_NET=1 to reach the real internet")
    pytest.importorskip("ddgs", reason="poolhouse[web]")
    pytest.importorskip("pymupdf", reason="pymupdf is the test-only PDF writer here (AGPL, opt-in): pip install pymupdf")
    got = find("TPS62130", "Texas Instruments", dest=tmp_path)
    assert got.matched == 1.0 and got.pages > 10
    assert {"SW", "PG", "FB"} <= {p.name for p in datasheet.pin_tables(got.path)}
    assert {o.kind for o in datasheet.outline_pages(got.path)} >= {"package_outline"}
