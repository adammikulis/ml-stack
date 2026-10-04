"""Simulation libraries and installation diagnostics."""

import json
import os
import shutil
import subprocess
import sys
from importlib.util import find_spec

from ml_stack.gym.transport import interpreter
from ml_stack.gym.worlds import schema

CAR_ACTIONS = [f"{steer} {drive}" for steer in ("left", "straight", "right")
               for drive in ("brake", "coast", "accelerate")]
ENVIRONMENTS = {
    "drone": {"title": "Forest search drones", "description": "Native quadrotors searching for hikers and fires with RGB and synthetic thermal cameras",
              "library": "PyFlyt", "physics": "Bullet", "modules": ["PyFlyt", "pybullet", "gymnasium"],
              "extra": "gym-drone", "python": "3.12", "actions": []},
    "car": {"title": "Smart car", "description": "Road navigation with lidar and vehicle dynamics",
                "library": "MetaDrive", "physics": "Bullet", "modules": ["metadrive", "gymnasium"],
                "extra": "gym-driving", "actions": CAR_ACTIONS},
    "traffic-driving": {"title": "Traffic and driving", "description": "SUMO-controlled intersection with MetaDrive vehicle physics",
                        "library": "SUMO-RL + MetaDrive", "physics": "Bullet + SUMO",
                        "modules": ["metadrive", "sumo_rl", "gymnasium"], "extra": "gym", "actions": []},
    "warehouse": {"title": "Warehouse robots", "description": "Robots collecting and delivering orders in generated or authored layouts",
                      "library": "RWARE", "physics": "Discrete logistics", "modules": ["rware", "gymnasium"],
                      "extra": "gym-warehouse", "actions": ["noop", "forward", "left", "right", "toggle load"]},
    "traffic": {"title": "Traffic junction", "description": "Signal control with SUMO traffic simulation",
                    "library": "SUMO-RL", "physics": "SUMO", "modules": ["sumo_rl", "gymnasium"],
                    "extra": "gym-traffic", "actions": []},
}


def catalogue():
    """Return supported environments and missing dependency diagnostics."""
    entries, probes = [], {}
    for name, spec in ENVIRONMENTS.items():
        interpreter_error = None
        try:
            python = interpreter(name)
        except RuntimeError as exc:
            python, interpreter_error = sys.executable, str(exc)
        if python != sys.executable:
            if python not in probes:
                probes[python] = probe(python)
            entry = next(row for row in probes[python] if row["id"] == name)
            entries.append({**entry, "interpreter": python})
            continue
        missing = [module for module in spec["modules"] if find_spec(module) is None]
        if interpreter_error:
            missing.append(interpreter_error)
        if name in {"traffic", "traffic-driving"} and not shutil.which("sumo") and find_spec("sumo") is None:
            missing.append("SUMO executable")
        entries.append({"id": name, **{k: v for k, v in spec.items() if k != "modules"},
                        "world_schema": schema(name), "simulation_modes": ["episode", "world"],
                        "supported_controllers": ["manual", "random", "decider", "ppo", *(["native-idm"] if name == "car" else ["native-patrol"] if name == "drone" else [])],
                        "available": not missing, "missing": missing,
                        "install": f"pip install 'ml-stack[{spec['extra']}]'" +
                                   ("; install the pinned MetaDrive source from docs/studio-gym.md"
                                    if "metadrive" in spec["modules"] else "")})
    return entries


def probe(python):
    """Ask one installed interpreter without forwarding parent interpreter routing."""
    environment = dict(os.environ)
    environment.pop("ML_STACK_GYM_PYTHON", None)
    environment.pop("ML_STACK_GYM_PYTHONS", None)
    result = subprocess.run([python, "-m", "ml_stack.gym.cli", "catalogue"],
                            capture_output=True, text=True, env=environment, timeout=10)
    if result.returncode:
        raise RuntimeError(f"Gym environment did not answer: {result.stderr.strip()}")
    return json.loads(result.stdout)


def require(environment):
    """Validate an environment and its dependencies."""
    entry = next((row for row in catalogue() if row["id"] == environment), None)
    if entry is None:
        raise ValueError(f"Unknown environment: {environment}")
    if not entry["available"]:
        raise RuntimeError(f"Missing {', '.join(entry['missing'])}. {entry['install']}")
