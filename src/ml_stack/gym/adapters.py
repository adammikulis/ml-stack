"""Gymnasium wrappers over specialist simulators."""

import base64
import io
import itertools

from ml_stack.gym.car_world import make_car_world
from ml_stack.gym.catalog import CAR_ACTIONS, require
from ml_stack.gym.driving import geometry as driving_geometry
from ml_stack.gym.drone_definition import ACTIONS as DRONE_ACTIONS
from ml_stack.gym.drone_world import make_drone_environment
from ml_stack.gym.episode_environment import make_episode
from ml_stack.gym.values import json_value
from ml_stack.gym.world_traffic import make_traffic_world
from ml_stack.gym.world_warehouse import make_warehouse_world
from ml_stack.gym.worlds import configure_world


def make_environment(name, config=None, seed=0):
    """Create a native environment with training-compatible actions."""
    require(name)
    cfg = dict(config or {})
    mode = cfg.pop("simulation_mode", "episode")
    cfg.pop("learning_mode", None)
    if mode not in {"episode", "world"}:
        raise ValueError("simulation_mode must be episode or world")
    provenance = None
    if name == "drone":
        return make_drone_environment(cfg, seed, mode)
    if "world" in cfg and (mode == "episode" or name == "car"):
        cfg, provenance = configure_world(name, cfg, seed)
    if mode == "world":
        if name == "car":
            env = make_car_world(cfg)
        elif name == "warehouse":
            env = make_warehouse_world(cfg, seed)
        else:
            env = make_traffic_world(cfg, seed, combined=name == "traffic-driving")
    else:
        if name == "car" and "world_seed" in cfg:
            cfg.pop("world_seed")
        env = make_episode(name, cfg)
    if provenance:
        env.world_provenance = provenance
    return env


def actions(name, env):
    """Return names and native actions in matching order."""
    if name == "drone":
        return DRONE_ACTIONS, list(range(len(DRONE_ACTIONS)))
    if name == "car":
        return CAR_ACTIONS, list(range(9))
    space = env.action_space
    if hasattr(space, "nvec"):
        joint = list(itertools.product(*(range(int(n)) for n in space.nvec)))
        labels = ("noop", "forward", "left", "right", "toggle load")
        return [" / ".join(labels[a] for a in row) for row in joint], joint
    return [f"phase {i}" for i in range(space.n)], list(range(space.n))


def render_state(name, env):
    """Return native geometry and a PNG frame when available."""
    native = env.unwrapped
    if hasattr(env, "render_state"):
        geometry = env.render_state()
        camera = getattr(native, "main_camera", None)
        frame = camera.perceive(to_float=False) if camera is not None else None
        return geometry, png_frame(frame)
    if name == "traffic-driving":
        return native.geometry(), None
    if name == "car":
        geometry = driving_geometry(native)
        camera = getattr(native, "main_camera", None)
        frame = camera.perceive(to_float=False) if camera is not None else None
    elif name == "warehouse":
        geometry = {"robots": [{"x": a.x, "y": a.y, "direction": int(a.dir.value),
                                "carrying": a.carrying_shelf is not None} for a in native.agents],
                    "shelves": [{"x": s.x, "y": s.y} for s in native.shelfs],
                    "goals": json_value(native.goals),
                    "width": native.grid_size[1], "height": native.grid_size[0]}
        frame = None
    else:
        connection = native.sumo
        geometry = {"heading_unit": "degrees", "vehicles": [{"id": key, "position": list(connection.vehicle.getPosition(key)),
                                  "heading": connection.vehicle.getAngle(key),
                                  "speed": connection.vehicle.getSpeed(key)}
                                 for key in connection.vehicle.getIDList()],
                    "roads": [{"id": key, "points": json_value(connection.lane.getShape(key)),
                               "width": connection.lane.getWidth(key)} for key in connection.lane.getIDList()],
                    "lights": [{"id": key, "state": connection.trafficlight.getRedYellowGreenState(key)}
                               for key in connection.trafficlight.getIDList()]}
        frame = None
    return geometry, png_frame(frame)


def png_frame(frame):
    """Encode an available native camera frame."""
    if frame is None:
        return None
    from PIL import Image
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")
