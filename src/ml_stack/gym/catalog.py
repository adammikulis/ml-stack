"""Simulation libraries and installation diagnostics."""

import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec

CAR_ACTIONS = [f"{steer} {drive}" for steer in ("left", "straight", "right")
               for drive in ("brake", "coast", "accelerate")]
ENVIRONMENTS = {
    "car": {"title": "Smart car", "description": "Road navigation with lidar and vehicle dynamics",
                "library": "MetaDrive", "physics": "Bullet", "modules": ["metadrive", "gymnasium"],
                "extra": "gym-driving", "actions": CAR_ACTIONS},
    "traffic-driving": {"title": "Smart traffic city", "description": "SUMO signals and traffic with MetaDrive vehicle physics",
                        "library": "SUMO-RL + MetaDrive", "physics": "Bullet + SUMO",
                        "modules": ["metadrive", "sumo_rl", "gymnasium"], "extra": "gym", "actions": []},
    "warehouse": {"title": "Warehouse robots", "description": "Two robots collecting and delivering orders",
                      "library": "RWARE", "physics": "Discrete logistics", "modules": ["rware", "gymnasium"],
                      "extra": "gym-warehouse", "actions": ["noop", "forward", "left", "right", "toggle load"]},
    "traffic": {"title": "Traffic junction", "description": "Signal control with SUMO traffic simulation",
                    "library": "SUMO-RL", "physics": "SUMO", "modules": ["sumo_rl", "gymnasium"],
                    "extra": "gym-traffic", "actions": []},
}


def catalogue():
    """Return supported environments and missing dependency diagnostics."""
    configured = os.environ.get("ML_STACK_GYM_PYTHON")
    if configured and configured != sys.executable:
        environment = dict(os.environ)
        environment.pop("ML_STACK_GYM_PYTHON", None)
        result = subprocess.run([configured, "-m", "ml_stack.gym.cli", "catalogue"],
                                capture_output=True, text=True, env=environment, timeout=10)
        if result.returncode:
            raise RuntimeError(f"Gym environment did not answer: {result.stderr.strip()}")
        return json.loads(result.stdout)
    entries = []
    for name, spec in ENVIRONMENTS.items():
        missing = [module for module in spec["modules"] if find_spec(module) is None]
        if name in {"traffic", "traffic-driving"} and not shutil.which("sumo") and find_spec("sumo") is None:
            missing.append("SUMO executable")
        entries.append({"id": name, **{k: v for k, v in spec.items() if k != "modules"},
                        "available": not missing, "missing": missing,
                        "install": f"pip install 'ml-stack[{spec['extra']}]'"})
    return entries


def require(environment):
    """Validate an environment and its dependencies."""
    entry = next((row for row in catalogue() if row["id"] == environment), None)
    if entry is None:
        raise ValueError(f"Unknown environment: {environment}")
    if not entry["available"]:
        raise RuntimeError(f"Missing {', '.join(entry['missing'])}. {entry['install']}")
