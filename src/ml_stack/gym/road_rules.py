"""Stop checkpoints on native MetaDrive routes."""

from ml_stack.gym.values import json_value


def front_progress(native):
    """Locate the vehicle's front bumper along its native route."""
    return native.agent.navigation.travelled_length + native.agent.LENGTH / 2


class StopCheckpoint:
    """A route stop rule independent of native vehicle dynamics."""

    def __init__(self, distance, position, heading, stop_line=None):
        self.distance, self.position, self.heading = distance, position, heading
        self.stop_line = stop_line
        self.held = 0.
        self.completed = self.passed = self.violated = False

    def update(self, progress, speed_m_s, dt):
        remaining = self.distance - progress
        reward = 0.
        if not self.passed:
            if 0 <= remaining <= 3 and speed_m_s <= .5:
                self.held += dt
                if self.held >= 1 and not self.completed:
                    self.completed, reward = True, 5.
            elif not self.completed:
                self.held = 0.
            if remaining < 0:
                self.passed = True
                self.violated = not self.completed
                if self.violated:
                    reward = -10.
        return reward

    def state(self, progress):
        return {"id": "route-stop-1", "position": json_value(self.position),
                "heading": self.heading, "heading_unit": "radians",
                "distance_m": self.distance - progress, "held_seconds": self.held,
                "required_hold_seconds": 1., "speed_threshold_m_s": .5,
                "stop_zone_m": 3.,
                "completed": self.completed, "passed": self.passed, "violated": self.violated,
                "state": "violation" if self.violated else "stopped" if self.completed else "pending",
                "stop_line": json_value(self.stop_line)}


def stop_environment(env):
    """Add route-rule observations and rewards to native Gym transitions."""
    import gymnasium as gym
    import numpy as np

    class RoadRules(gym.Wrapper):
        def __init__(self, native):
            super().__init__(native)
            self.observation_space = gym.spaces.Box(
                low=np.concatenate((native.observation_space.low, [-1., 0., 0.])).astype(np.float32),
                high=np.concatenate((native.observation_space.high, [1., 1., 1.])).astype(np.float32), dtype=np.float32)

        def augment(self, observation):
            state = self.unwrapped.stop_checkpoint.state(front_progress(self.unwrapped))
            return np.concatenate((observation, [np.clip(state["distance_m"] / 100, -1, 1),
                                                min(state["held_seconds"], 1), float(state["completed"])])).astype(np.float32)

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            native = self.unwrapped
            navigation = native.vehicle.navigation
            desired = min(75., navigation.total_length * .35)
            offset = 0.
            for start, end in zip(navigation.checkpoints[:-1], navigation.checkpoints[1:], strict=True):
                lanes = native.current_map.road_network.graph[start][end]
                lane = lanes[0]
                if offset + lane.length >= desired:
                    longitudinal = desired - offset
                    position = lane.position(longitudinal, -lane.width_at(longitudinal) / 2 - 2)
                    heading = lane.heading_theta_at(longitudinal)
                    width = lane.width_at(longitudinal)
                    line = [lane.position(longitudinal, -width / 2),
                            lane.position(longitudinal, (len(lanes) - .5) * width)]
                    native.stop_checkpoint = StopCheckpoint(desired, position, float(heading), line)
                    break
                offset += lane.length
            return self.augment(observation), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            native = self.unwrapped
            dt = native.config["physics_world_step_size"] * native.config["decision_repeat"]
            extra = native.stop_checkpoint.update(front_progress(native),
                                                   native.vehicle.speed_km_h / 3.6, dt)
            info = {**info, "stop_rule_reward": extra,
                    "stop_rule": native.stop_checkpoint.state(front_progress(native))}
            return self.augment(observation), reward + extra, terminated, truncated, info

    return RoadRules(env)
