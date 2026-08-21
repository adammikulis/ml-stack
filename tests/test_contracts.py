"""The contract data itself, and the one rule that travels with it."""

from __future__ import annotations

import json

import pytest
from mainspring.contracts import Budget, contracts_dir, fits, largest_that_fits, tiers

GIB = 1024**3


def test_ladder_is_descending():
    """'The first that fits' and 'the largest that fits' are only the same walk if the
    ladder is ordered. ``largest_that_fits`` relies on that and deliberately does not
    re-sort, so an entry inserted in the wrong place must fail here rather than quietly
    return a smaller model than the machine could have run."""
    ladder = tiers()
    sizes = [tier.min_ram_gb for tier in ladder]
    assert sizes == sorted(sizes, reverse=True), (
        f"model_tiers.json is out of order: {[(t.id, t.min_ram_gb) for t in ladder]}"
    )


def test_tier_ids_are_unique():
    ids = [tier.id for tier in tiers()]
    assert len(ids) == len(set(ids)), f"duplicate tier id in {ids}"


def test_every_contract_file_is_valid_json():
    for path in sorted(contracts_dir().glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))


def test_floor_rung_needs_no_download():
    """The last rung must be reachable on any machine and must not require a fetch:
    an embedded image ships its own weights and may have no network at first boot."""
    floor = tiers()[-1]
    assert floor.min_ram_gb <= 1
    assert floor.gguf_repo is None


class TestFitsRefusesToDemoteOnIgnorance:
    """A missing measurement is not evidence that a model is too big.

    Treating it as such silently hands the user a worse model than their machine could
    run, and nothing in the logs says why.
    """

    def test_unknown_size_is_unknown_not_too_big(self):
        assert fits(None, total_bytes=8 * GIB) == "unknown"

    def test_unknown_stays_unknown_even_on_a_tiny_machine(self):
        assert fits(None, total_bytes=1) == "unknown"

    def test_a_measured_model_still_gets_a_real_verdict(self):
        budget = Budget(0.7, 0.25, 1 * GIB)
        assert fits(1 * GIB, total_bytes=64 * GIB, budget=budget) == "fits"
        assert fits(60 * GIB, total_bytes=64 * GIB, budget=budget) == "too_big"


def test_overhead_floor_dominates_for_small_models():
    """Weights are the floor, not the total. A 200 MB model does not cost 250 MB -- the
    KV cache alone is bigger than that, which is what the 1 GiB floor encodes."""
    budget = Budget(0.7, 0.25, 1 * GIB)
    from mainspring.contracts import weights_plus_overhead_bytes

    assert weights_plus_overhead_bytes(200 * 1024**2, budget) == 200 * 1024**2 + 1 * GIB


def test_largest_that_fits_walks_down():
    big = largest_that_fits(128 * GIB)
    small = largest_that_fits(16 * GIB)
    assert big.min_ram_gb > small.min_ram_gb


def test_a_tiny_machine_still_gets_a_tier():
    """Never raise, never return None: the floor rung always applies."""
    assert largest_that_fits(512 * 1024**2).id == tiers()[-1].id


def test_availability_filter_skips_absent_models():
    """Passing an availability set restricts the walk to what is actually pulled."""
    ladder = tiers()
    ollama_tiers = [t for t in ladder if t.ollama]
    smallest_available = ollama_tiers[-1]

    chosen = largest_that_fits(512 * GIB, available={smallest_available.ollama})
    assert chosen.id == smallest_available.id


def test_none_availability_means_do_not_filter_not_nothing_available():
    """Same refusal-to-demote-on-ignorance rule: not knowing what is installed must not
    be read as 'nothing is installed'."""
    unfiltered = largest_that_fits(512 * GIB, available=None)
    assert unfiltered.id == tiers()[0].id


def test_budget_profiles_exist_and_mobile_is_stricter():
    """A mobile low-memory killer takes the process rather than swapping, so the
    mobile headroom has to be real."""
    desktop = Budget.for_profile("desktop")
    mobile = Budget.for_profile("mobile")
    assert mobile.usable_fraction_of_total < desktop.usable_fraction_of_total


def test_unknown_profile_raises():
    from mainspring.contracts import ContractError

    with pytest.raises(ContractError, match="no budget profile"):
        Budget.for_profile("toaster")  # type: ignore[arg-type]
