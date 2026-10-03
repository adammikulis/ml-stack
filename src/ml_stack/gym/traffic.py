"""SUMO installation and packaged intersection configuration."""

import os
from importlib.util import find_spec
from pathlib import Path


def traffic_defaults(config):
    """Resolve the packaged SUMO-RL single-intersection scenario."""
    if find_spec("sumo") is not None:
        import sumo
        os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)
    if not os.environ.get("SUMO_HOME"):
        raise RuntimeError("Set SUMO_HOME to your SUMO installation or install ml-stack[gym-traffic]")
    module = find_spec("sumo_rl")
    if module is None or module.origin is None:
        raise RuntimeError("Install ml-stack[gym-traffic]")
    scenario = Path(module.origin).parent / "nets" / "single-intersection"
    cfg = {"net_file": str(scenario / "single-intersection.net.xml"),
           "route_file": str(scenario / "single-intersection.rou.xml"), "num_seconds": 300,
           "sumo_warnings": False, **config}
    return cfg
