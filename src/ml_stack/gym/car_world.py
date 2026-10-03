"""Persistent MetaDrive traffic with native actor lifecycles."""

import gymnasium as gym
import numpy as np
from metadrive import MultiAgentMetaDrive
from metadrive.policy.env_input_policy import EnvInputPolicy
from metadrive.policy.idm_policy import IDMPolicy

from ml_stack.gym.driving import geometry
from ml_stack.gym.road_rules import checkpoint_on_route, front_progress
from ml_stack.gym.values import json_value


class NativeOrExternalPolicy(EnvInputPolicy):
    def __init__(self, obj, seed):
        super().__init__(obj, seed)
        self.baseline = IDMPolicy(obj, seed)

    def act(self, agent_id):
        if self.engine.external_actions[agent_id] is None:
            action = list(self.baseline.act(agent_id))
            rule = getattr(self.control_object, "studio_stop_checkpoint", None)
            if rule is not None and not rule.completed:
                remaining = rule.distance - front_progress(None, self.control_object)
                speed = self.control_object.speed_km_h / 3.6
                if 0 <= remaining <= max(3, speed * speed / 6 + 2):
                    action[1] = -1. if speed > .1 else 0.
            return action
        return super().act(agent_id)

    def destroy(self):
        self.baseline.destroy()
        super().destroy()

class PersistentTraffic(MultiAgentMetaDrive):
    def _after_vehicle_done(self, obs, rewards, terminated, truncated, info):
        dt = self.config["physics_world_step_size"] * self.config["decision_repeat"]
        for actor, vehicle in self.agents.items():
            rule = getattr(vehicle, "studio_stop_checkpoint", None)
            if rule is not None:
                extra = rule.update(front_progress(self, vehicle), vehicle.speed_km_h / 3.6, dt)
                rewards[actor] += extra
                info[actor].update(stop_rule=rule.state(front_progress(self, vehicle)), stop_rule_reward=extra)
        return super()._after_vehicle_done(obs, rewards, terminated, truncated, info)


class CarWorld(gym.Env):
    def __init__(self, cfg, task_horizon, magnitude, stops):
        self.task_horizon, self.magnitude, self.stops = task_horizon, magnitude, stops
        self.native = PersistentTraffic(cfg)
        self.native.steering_magnitude = magnitude
        self.action_space = gym.spaces.Discrete(9)
        self.initialized = False
        self.ego_id = None
        self.handoffs = self.task_steps = 0
        self.observations = {}
        self.applied_action = None

    def prepare_rules(self):
        if self.stops:
            for vehicle in self.native.agents.values():
                if not hasattr(vehicle, "studio_stop_checkpoint"):
                    vehicle.studio_stop_checkpoint = checkpoint_on_route(self.native, vehicle)

    @property
    def unwrapped(self):
        return self.native

    def select_ego(self):
        if self.ego_id not in self.native.agents:
            old = self.ego_id
            self.ego_id = next(iter(self.native.agents), None)
            self.handoffs += int(old is not None and old != self.ego_id)
        if self.ego_id is None:
            raise RuntimeError("Native world has no active vehicle available")
        self.prepare_rules()
        return self.ego_id

    def select_agent(self, actor):
        if actor not in self.native.agents:
            raise ValueError("Agent is no longer active in this world")
        self.ego_id = actor
        return self.augment(self.observations[actor], actor), self.world_info()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.initialized:
            self.observations, _ = self.native.reset(seed=seed)
            self.initialized = True
        actor = self.select_ego()
        self.task_steps = 0
        observation = self.observations[actor]
        low = np.concatenate((np.zeros_like(observation), [-1., 0., 0.])) if self.stops else np.zeros_like(observation)
        high = np.ones(low.shape, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low.astype(np.float32), high, dtype=np.float32)
        return self.augment(observation, actor), self.world_info()

    def augment(self, observation, actor, state=None):
        if not self.stops:
            return observation
        if state is None:
            vehicle = self.native.agents[actor]
            state = vehicle.studio_stop_checkpoint.state(front_progress(self.native, vehicle))
        return np.concatenate((observation, [np.clip(state["distance_m"] / 100, -1, 1),
                                            min(state["held_seconds"], 1), float(state["completed"])])).astype(np.float32)

    def world_info(self):
        dt = self.native.config["physics_world_step_size"] * self.native.config["decision_repeat"]
        return {"world_steps": self.native.episode_step, "world_time": self.native.episode_step * dt,
                "world_seed": self.native.current_seed,
                "ego_actor_id": self.ego_id, "actor_handoffs": self.handoffs,
                "active_actor_ids": list(self.native.agents), "world_reset_count": 1,
                "applied_native_control": self.applied_action}

    def step(self, action):
        actor = self.select_ego()
        vehicle = self.native.agents[actor]
        controls = None if action is None else [(-self.magnitude, 0., self.magnitude)[int(action) // 3],
                                                (-1., 0., 1.)[int(action) % 3]]
        applied = {key: controls if key == actor else None for key in self.native.agents}
        observations, rewards, terminated, truncated, infos = self.native.step(applied)
        self.applied_action = json_value(vehicle.current_action)
        self.observations = observations
        self.task_steps += 1
        ended = terminated.get(actor, False)
        limited = truncated.get(actor, False) or self.task_steps >= self.task_horizon
        self.select_ego()
        info = {**infos.get(actor, {}), **self.world_info(), "transition_actor_id": actor}
        return self.augment(observations[actor], actor, info.get("stop_rule")), rewards[actor], ended, limited, info

    def decision_state(self, observation):
        vehicle = self.native.agents[self.select_ego()]
        sensor = self.native.observations[self.ego_id]
        state = {"environment": "car", "simulation_mode": "world", **self.world_info(),
                "native_observation": json_value(observation),
                "ego": {"speed_km_h": float(vehicle.speed_km_h), "position_m": json_value(vehicle.position)},
                "sensors": {"lidar_normalized": json_value(sensor.cloud_points),
                            "lidar_range_m": vehicle.config["lidar"]["distance"]}}
        if self.stops:
            state["stop_rule"] = vehicle.studio_stop_checkpoint.state(front_progress(self.native, vehicle))
        return state

    def render_state(self):
        vehicle = self.native.agents[self.select_ego()]
        result = geometry(self.native, vehicle, getattr(vehicle, "studio_stop_checkpoint", None))
        identifiers = {actor.id: key for key, actor in self.native.agents.items()}
        for actor in result["vehicles"]:
            policy = self.native.engine.get_policy(actor["id"])
            agent_id = identifiers.get(actor["id"])
            external = self.native.engine.external_actions.get(agent_id) is not None
            actor.update(agent_id=agent_id, policy=type(policy).__name__,
                         controller="external" if external else "native-idm")
        return {**result, **self.world_info()}

    def close(self):
        self.native.close()



def make_car_world(config):
    """Create a Gym task view over a world initialized exactly once."""
    cfg = dict(config)
    task_horizon = int(cfg.pop("horizon", 1000))
    magnitude = float(cfg.pop("steering_magnitude", .35))
    if task_horizon < 1 or not 0 < magnitude <= 1:
        raise ValueError("horizon must be positive and steering_magnitude in (0, 1]")
    cfg.pop("simulation_mode", None)
    cfg.pop("learning_mode", None)
    stops = cfg.pop("stop_signs", True)
    debug = cfg.pop("sensor_debug", True)
    preview = cfg.pop("render_preview", False)
    cfg.setdefault("map", PersistentTraffic.default_config()["map"] if "map_config" in cfg else "SCSCS")
    cfg.update(horizon=None, allow_respawn=True, agent_policy=NativeOrExternalPolicy,
               action_check=False, use_render=False)
    cfg.setdefault("num_agents", 8)
    cfg.setdefault("traffic_density", .25)
    cfg.setdefault("num_scenarios", 20000)
    cfg.setdefault("start_seed", 0)
    if preview:
        from metadrive.obs.state_obs import LidarStateObservation
        cfg.update(image_observation=True, agent_observation=LidarStateObservation,
                   sensors={"main_camera": ()}, window_size=(960, 540), show_interface=False,
                   show_terrain=False)
    cfg["vehicle_config"] = {**cfg.get("vehicle_config", {}), "show_lidar": debug,
                             "show_side_detector": debug, "show_lane_line_detector": debug}

    return CarWorld(cfg, task_horizon, magnitude, stops)
