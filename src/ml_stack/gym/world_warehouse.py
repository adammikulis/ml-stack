"""Persistent RWARE robots and orders with separate learning task boundaries."""

from importlib import import_module

import gymnasium as gym

from ml_stack.gym.observations import decision_state
from ml_stack.gym.worlds import configure_world


def make_warehouse_world(config, seed=0):
    """Construct native RWARE once and keep robots and orders across task resets."""
    cfg = dict(config or {})
    task_horizon = int(cfg.pop('task_horizon', cfg.get('max_steps', 500)))
    if task_horizon < 1:
        raise ValueError('task_horizon must be positive')
    cfg.pop('simulation_mode', None)
    cfg.pop('learning_mode', None)
    initial_seed = int(cfg.get('world', {}).get('seed', seed))
    cfg, provenance = configure_world('warehouse', cfg, initial_seed)
    cfg.update(max_steps=None, max_inactivity_steps=None)
    native = import_module('ml_stack.gym.adapters').make_environment('warehouse', cfg)
    return WarehouseWorld(native, provenance, initial_seed, task_horizon)


class WarehouseWorld(gym.Wrapper):
    def __init__(self, native, provenance, seed, task_horizon):
        super().__init__(native)
        self.native, self.world_provenance = native.unwrapped, provenance
        self.initial_seed, self.task_horizon = seed, task_horizon
        self.initialized = False
        self.task_steps = 0
        self.observation = None
        self.info = {}

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('Construct a new world to change the warehouse layout')
        if not self.initialized:
            self.observation, self.info = self.env.reset(seed=self.initial_seed)
            self.initialized = True
        self.task_steps = 0
        return self.observation, {**self.info, **self.world_info()}

    def world_info(self):
        return {'world_steps': self.native._cur_steps, 'world_time': self.native._cur_steps,
                'world_time_unit': 'native steps', 'world_reset_count': int(self.initialized),
                'active_actor_ids': [str(agent.id) for agent in self.native.agents],
                'world_provenance': self.world_provenance}

    def step(self, action):
        if action is None:
            action = [0] * self.native.n_agents
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        if terminated or truncated:
            raise RuntimeError('Native warehouse world ended unexpectedly')
        self.task_steps += 1
        return self.observation, reward, False, self.task_steps >= self.task_horizon, {**self.info, **self.world_info()}

    def decision_state(self, observation):
        return {**decision_state('warehouse', self, observation), 'simulation_mode': 'world', **self.world_info()}
