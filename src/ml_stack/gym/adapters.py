"""Gymnasium wrappers over specialist simulators."""

import base64
import io
import itertools
from pathlib import Path

from ml_stack.gym.catalog import CAR_ACTIONS, require
from ml_stack.gym.driving import geometry as driving_geometry
from ml_stack.gym.traffic import traffic_defaults
from ml_stack.gym.values import json_value


def make_environment(name, config=None):
    """Create a native environment with training-compatible actions."""
    require(name)
    import gymnasium as gym
    import numpy as np

    cfg = dict(config or {})
    if name == "traffic-driving":
        raise RuntimeError("The combined traffic-driving adapter is not installed")
    if name == "car":
        from metadrive import MetaDriveEnv

        class Driving(gym.ActionWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.action_space = gym.spaces.Discrete(9)

            def action(self, action):
                return np.array([(-1., 0., 1.)[int(action) // 3],
                                 (-1., 0., 1.)[int(action) % 3]], dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                if options:
                    raise ValueError("MetaDrive reset options are configured at environment creation")
                if seed is not None:
                    start = self.env.config["start_seed"]
                    count = self.env.config["num_scenarios"]
                    if not start <= seed < start + count:
                        raise ValueError(f"Car scenario seed must be in [{start}, {start + count}); configure start_seed and num_scenarios")
                return self.env.reset(seed=seed)

        cfg.setdefault("use_render", False)
        cfg.setdefault("num_scenarios", 20000)
        cfg.setdefault("start_seed", 0)
        cfg.setdefault("horizon", 1000)
        if cfg.pop("render_preview", False):
            from metadrive.obs.state_obs import LidarStateObservation
            cfg.update(image_observation=True, agent_observation=LidarStateObservation,
                       sensors={"main_camera": ()}, window_size=(960, 540), show_interface=False,
                       show_terrain=False)
            cfg["vehicle_config"] = {**cfg.get("vehicle_config", {}), "show_lidar": True,
                                     "show_side_detector": True, "show_lane_line_detector": True}
        return Driving(MetaDriveEnv(cfg))
    if name == "warehouse":
        import rware  # noqa: F401

        env = gym.make(cfg.pop("env_id", "rware-tiny-2ag-v2"), disable_env_checker=True, **cfg)

        class Warehouse(gym.Wrapper):
            def __init__(self, env):
                super().__init__(env)
                self.observation_space = gym.spaces.flatten_space(env.observation_space)
                self.action_space = gym.spaces.MultiDiscrete([space.n for space in env.action_space])

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                return gym.spaces.flatten(self.env.observation_space, obs), info

            def step(self, action):
                obs, rewards, terminated, truncated, info = self.env.step(tuple(map(int, action)))
                return (gym.spaces.flatten(self.env.observation_space, obs), float(sum(rewards)),
                        bool(terminated), bool(truncated), info)

        return Warehouse(env)
    cfg = traffic_defaults(cfg)
    from sumo_rl import SumoEnvironment
    for key in ("net_file", "route_file"):
        if not Path(cfg[key]).is_file():
            raise ValueError(f"{key} does not exist: {cfg[key]}")
    cfg["single_agent"] = True
    cfg.setdefault("use_gui", False)
    return SumoEnvironment(**cfg)


def actions(name, env):
    """Return names and native actions in matching order."""
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
    if name == "traffic-driving":
        return native.geometry(), None
    if name == "car":
        geometry = driving_geometry(native)
        frame = native.main_camera.perceive(to_float=False) if native.main_camera is not None else None
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
    if frame is None:
        return geometry, None
    from PIL import Image
    output = io.BytesIO()
    Image.fromarray(frame).save(output, format="PNG")
    return geometry, base64.b64encode(output.getvalue()).decode("ascii")
