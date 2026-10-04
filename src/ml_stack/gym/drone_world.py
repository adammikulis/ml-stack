"""Forest search on native PyFlyt quadrotors, PID controllers and Bullet physics."""

import math
from typing import ClassVar

from ml_stack.gym.drone_definition import ACTIONS, build, number
from ml_stack.gym.drone_sensors import capture, forest, instrument_camera


class DroneState:
    """Task bookkeeping around a native single-clock multi-drone Aviary."""

    def initialize(self, config, seed, mode):
        self.config, self.initial_seed, self.mode = dict(config), seed, mode
        self.horizon = number(config.get('task_horizon', 500), 'task_horizon', 1, 100000, True)
        self.native = None
        self.actor, self.world_steps, self.native_world_reset_count = 0, 0, 0
        self.world_definition = config.get('world', {'mode': 'procedural'})
        self.new_world(seed)
        self.task_reset()

    def new_world(self, seed):
        import numpy as np
        from PyFlyt.core import Aviary
        if self.native is not None:
            self.native.disconnect()
        world = dict(self.world_definition)
        if self.mode == 'episode':
            world['seed'] = seed
        cfg, self.world_provenance = build(world, seed)
        self.definition = cfg['forest_definition']
        self.world_seed = cfg['world_seed']
        count = self.definition['n_agents']
        positions = np.array([[0, i * .8, 5] for i in range(count)], dtype=float)
        self.native = Aviary(start_pos=positions, start_orn=np.zeros_like(positions),
                             drone_type='quadx', render=False, np_random=np.random.default_rng(int(self.world_seed)),
                             drone_options={'use_camera': True, 'use_gimbal': True,
                                            'camera_angle_degrees': -45, 'camera_fps': 5,
                                            'camera_resolution': (128, 128)})
        self.native.set_mode(7)
        for drone in self.native.drones:
            instrument_camera(self.native, drone)
        self.setpoints = np.column_stack([positions[:, :2], np.zeros(count), positions[:, 2]])
        for index, point in enumerate(self.setpoints):
            self.native.set_setpoint(index, point)
        self.scenery, self.targets = forest(self.native, self.definition)
        self.native.register_all_new_bodies()
        self.native_world_reset_count += 1
        self.world_steps, self.actor = 0, 0
        capture(self, fresh=True)

    def task_reset(self):
        self.task_steps, self.reported, self.visited = 0, set(), set()

    def reset(self, *, seed=None, options=None):
        if self.mode == 'episode' and (self.task_steps or (seed is not None and seed != self.world_seed)):
            self.new_world(self.initial_seed if seed is None else seed)
        self.task_reset()
        return self.observation(), self.info()

    def observation(self):
        import numpy as np
        state = self.native.state(self.actor).ravel()
        return np.concatenate([state, self.visual_observation]).astype(np.float32)

    def step(self, action):
        import numpy as np
        if isinstance(action, np.ndarray) and action.shape == ():
            action = action.item()
        if isinstance(action, np.integer):
            action = int(action)
        if action is not None and (type(action) is not int or not 0 <= action < len(ACTIONS)):
            raise ValueError('Unknown drone action')
        point = self.setpoints[self.actor]
        offset = {1: (1, 1), 2: (1, -1), 3: (0, 1), 4: (0, -1),
                  5: (3, .5), 6: (3, -.5), 7: (2, .2), 8: (2, -.2)}
        if action in offset:
            axis, increment = offset[action]
            point[axis] += increment
        size = self.definition['size'] / 2
        point[:2] = np.clip(point[:2], -size, size)
        point[3] = np.clip(point[3], 1, 15)
        self.native.set_setpoint(self.actor, point)
        self.patrol(include_selected=action is None)
        for _ in range(24):
            self.native.step()
        self.world_steps += 1
        self.task_steps += 1
        capture(self)
        reward, parts = self.reward(action)
        state = self.native.state(self.actor)
        crashed = bool(state[3, 2] < .2 or np.any(np.abs(state[1, :2]) > math.pi / 2))
        if crashed:
            reward -= 10
            parts['crash'] = -10
        # Persistent world never terminates native physics or resets its actors.
        terminated = crashed and self.mode == 'episode'
        truncated = self.task_steps >= self.horizon
        return self.observation(), reward, terminated, truncated, {**self.info(), 'reward_components': parts}

    def patrol(self, include_selected=False):
        size = self.definition['size'] / 2 - 2
        corners = [(-size, -size), (size, -size), (size, size), (-size, size)]
        for index in range(len(self.native.drones)):
            if include_selected or index != self.actor:
                x, y = corners[(self.world_steps // 80 + index) % 4]
                self.setpoints[index][:2] = [x, y]
                self.setpoints[index][3] = 10 + index
                self.native.set_setpoint(index, self.setpoints[index])

    def reward(self, action):
        position = self.native.state(self.actor)[3]
        cell = (math.floor(position[0] / 2), math.floor(position[1] / 2))
        parts = {'time': -.01, 'coverage': .1 if cell not in self.visited else 0., 'report': 0.}
        self.visited.add(cell)
        if action in {9, 10}:
            kind = 'hiker' if action == 9 else 'fire'
            found = [body for body in self.visible_targets
                     if self.targets[body]['kind'] == kind and body not in self.reported]
            parts['report'] = 5. * len(found) if found else -.5
            self.reported.update(found)
        return sum(parts.values()), parts

    def info(self):
        return {'world_steps': self.world_steps, 'world_time': float(self.native.elapsed_time),
                'world_seed': self.world_seed, 'native_world_reset_count': self.native_world_reset_count,
                'ego_actor_id': f'drone-{self.actor}', 'controlled_agent': f'drone-{self.actor}',
                'active_actor_ids': [f'drone-{i}' for i in range(len(self.native.drones))],
                'task_steps': self.task_steps, 'reported_targets': len(self.reported),
                'native_pid_mode': 7, 'applied_native_control': self.setpoints[self.actor].tolist()}

    def select_agent(self, agent_id):
        ids = self.info()['active_actor_ids']
        if agent_id not in ids:
            raise ValueError('Unknown native drone agent')
        self.actor = ids.index(agent_id)
        capture(self, fresh=self.world_steps == 0)
        return self.observation(), self.info()

    def decision_state(self, observation):
        return {'environment': 'drone', 'agent_id': f'drone-{self.actor}',
                'native_observation': observation.tolist(),
                'proprioception': self.native.state(self.actor).tolist(),
                'camera': self.camera, 'task': 'Search forest; report visible hikers and fires'}

    def render_state(self):
        return {'drones': [{'id': f'drone-{i}', 'position': self.native.state(i)[3].tolist(),
                            'rotation': self.native.state(i)[1].tolist(),
                            'selected': i == self.actor,
                            'controller': 'selected policy' if i == self.actor else 'native PID patrol'} for i in range(len(self.native.drones))],
                'forest': self.scenery, 'size': self.definition['size'], 'camera': self.camera,
                'world_time': float(self.native.elapsed_time), 'world_steps': self.world_steps}

    def close(self):
        if self.native is not None:
            self.native.disconnect()
            self.native = None


def make_drone_environment(config=None, seed=0, mode='episode'):
    import gymnasium as gym
    import numpy as np

    class DroneEnvironment(DroneState, gym.Env):
        metadata: ClassVar[dict] = {'render_modes': []}

        def reset(self, *, seed=None, options=None):
            gym.Env.reset(self, seed=seed)
            return DroneState.reset(self, seed=seed, options=options)

        def __init__(self):
            self.action_space = gym.spaces.Discrete(len(ACTIONS))
            self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(1036,), dtype=np.float32)
            self.initialize(config or {}, seed, mode)

    return DroneEnvironment()
