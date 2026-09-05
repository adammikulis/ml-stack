"""``human_bytes``: the one way a size is written for a person."""

from __future__ import annotations

import pytest

from ml_stack.units import human_bytes

K, M, G, T = 1024, 1024**2, 1024**3, 1024**4


@pytest.mark.parametrize("size, said", [
    (0, "0B"),
    (512, "512B"),
    (1023, "1023B"),
    (K, "1.0K"),
    (1536, "1.5K"),
    (M, "1.0M"),
    (700 * M, "700.0M"),
    (G, "1.0G"),
    (24 * G, "24.0G"),
    (1023 * G, "1023.0G"),
    (T, "1.0T"),
    (3 * T, "3.0T"),
])
def test_a_size_is_written_in_the_largest_unit_it_fills(size, said):
    assert human_bytes(size) == said


def test_a_size_under_a_gigabyte_is_not_written_as_gigabytes():
    """A dispatcher's refusal read '0.0G' for every model under a gigabyte."""
    assert human_bytes(300 * M) == "300.0M"
    assert human_bytes(12 * K) == "12.0K"


def test_a_float_is_taken_as_readily_as_an_int():
    assert human_bytes(1.5 * G) == "1.5G"
