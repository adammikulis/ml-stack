"""Bounded Gym adapters for native simulators."""

from pathlib import Path

from ml_stack.gym.cosim import make_cosim
from ml_stack.gym.road_rules import stop_environment
from ml_stack.gym.traffic import traffic_defaults


def make_episode(name, config):
    """Construct one bounded native Gym environment."""
    import gymnasium as gym
    import numpy as np

    cfg = dict(config)
    if name == "traffic-driving":
        return make_cosim(cfg)
    if name == "car":
        from metadrive import MetaDriveEnv
        magnitude = float(cfg.pop("steering_magnitude", .35))
        if not 0 < magnitude <= 1:
            raise ValueError("steering_magnitude must be greater than zero and at most one")

        class Driving(gym.ActionWrapper):
            def __init__(self, env):
                super().__init__(env)
                self.action_space = gym.spaces.Discrete(9)

            def action(self, action):
                return np.array([(-magnitude, 0., magnitude)[int(action) // 3],
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
        debug = cfg.pop("sensor_debug", True)
        stop_signs = cfg.pop("stop_signs", False)
        if not isinstance(debug, bool) or not isinstance(stop_signs, bool):
            raise ValueError("sensor_debug and stop_signs must be booleans")
        if cfg.pop("render_preview", False):
            from metadrive.obs.state_obs import LidarStateObservation
            cfg.update(image_observation=True, agent_observation=LidarStateObservation,
                       sensors={"main_camera": ()}, window_size=(960, 540), show_interface=False,
                       show_terrain=False)
            vehicle_config = dict(cfg.get("vehicle_config", {}))
            for flag in ("show_lidar", "show_side_detector", "show_lane_line_detector"):
                vehicle_config.setdefault(flag, debug)
            cfg["vehicle_config"] = vehicle_config
        env = Driving(MetaDriveEnv(cfg))
        env.unwrapped.steering_magnitude = magnitude
        return stop_environment(env) if stop_signs else env
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


