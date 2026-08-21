"""Read the shared contracts. Standard library only.

The contracts are DATA (``contracts/*.json`` at the repo root), not code, so a
non-Python host can consume the same tier ladder and sampler surface directly. This
package is the Python reader for them, plus the one piece of logic that has to travel
with the data: deciding which tier a machine can actually hold.

Device tier: importable with nothing but the standard library.
"""

from __future__ import annotations

from mainspring.contracts.loader import (
    ContractError,
    contracts_dir,
    grammar,
    load,
    sampling_schema,
)
from mainspring.contracts.tiers import (
    Budget,
    Tier,
    fits,
    largest_that_fits,
    tiers,
    weights_plus_overhead_bytes,
)

__all__ = [
    "Budget",
    "ContractError",
    "Tier",
    "contracts_dir",
    "fits",
    "grammar",
    "largest_that_fits",
    "load",
    "sampling_schema",
    "tiers",
    "weights_plus_overhead_bytes",
]
