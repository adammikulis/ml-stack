"""The decision models on this machine, asked of the model library as kind ``decision``.

Classification is a label (see `poolhouse.hub.kinds`): finding a model here grants it nothing,
and anything that serves one still takes a Broker lease and the admission it is subject to.
"""

from __future__ import annotations

from pathlib import Path

from poolhouse import hub
from poolhouse.decide import registry
from poolhouse.decide.sources import CONFIG
from poolhouse.hub import kinds


def models() -> list[hub.ModelInfo]:
    """Every installed model the library labels ``decision``."""
    return hub.discover(kind=kinds.DECISION)


def decider_dir(name: str) -> Path | None:
    """The registered decider directory holding the decision model ``name`` stands for, or
    ``None``."""
    for found in hub.installed_find(name, models()):
        for row in registry.listing():
            root = Path(row["path"]).resolve()
            if (root / CONFIG).is_file() and kinds.registered(found.path, [root]):
                return root
    return None
