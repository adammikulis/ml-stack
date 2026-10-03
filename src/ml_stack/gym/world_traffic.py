"""Persistent native SUMO traffic and SUMO/MetaDrive worlds."""

import json
import math
from importlib import import_module
from pathlib import Path

import gymnasium as gym
import numpy as np

from ml_stack.gym.observations import decision_state
from ml_stack.gym.traffic_world import xml
from ml_stack.gym.world_files import record
from ml_stack.gym.worlds import configure_world


def make_traffic_world(config, seed=0, combined=False):
    """Initialize traffic once and expose bounded learning tasks over its live clock."""
    cfg = dict(config or {})
    task_horizon = int(cfg.pop('task_horizon', cfg.get('num_seconds', 300)))
    cfg.pop('simulation_mode', None)
    cfg.pop('learning_mode', None)
    mode = 'traffic-driving' if combined else 'traffic'
    initial_seed = int(cfg.get('world', {}).get('seed', seed))
    cfg, provenance = configure_world(mode, cfg, initial_seed)
    period = float(cfg.pop('world_demand_period', provenance['definition'].get('vehicle_period', 3)))
    if task_horizon < 1 or not math.isfinite(period) or not .5 <= period <= 60:
        raise ValueError('task_horizon must be positive and world_demand_period in [0.5, 60]')
    cfg.update(num_seconds=float('inf'), single_agent=True, use_gui=False)
    if combined:
        native = import_module('ml_stack.gym.cosim').make_cosim(cfg)
    else:
        native = import_module('sumo_rl').SumoEnvironment(**cfg)
    return TrafficWorld(native, cfg, provenance, {'seed': initial_seed, 'horizon': task_horizon,
                                                  'period': period, 'mode': mode})


class TrafficWorld(gym.Env):
    def __init__(self, native, config, provenance, settings):
        self.native, self.config, self.world_provenance = native, config, provenance
        self.initial_seed, self.task_horizon = settings['seed'], settings['horizon']
        self.period, self.mode = settings['period'], settings['mode']
        self.action_space, self.observation_space = native.action_space, native.observation_space
        self.initialized = False
        self.world_steps = self.task_steps = self.inserted = 0
        self.observation = None
        self.info = {}
        self.route_ids = []

    @property
    def unwrapped(self):
        return self.native

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            raise ValueError('Construct a new world to change native traffic configuration')
        if not self.initialized:
            self.observation, self.info = self.native.reset(seed=self.initial_seed)
            self.initialized = True
            self.rng = np.random.default_rng(self.initial_seed)
            templates = []
            for index, name in enumerate(self.native.sumo.route.getIDList()):
                edges = self.native.sumo.route.getEdges(name)
                if edges:
                    identifier = f'studio-world-route-{index}'
                    self.native.sumo.route.add(identifier, edges)
                    self.route_ids.append(identifier)
                    templates.append({'id': identifier, 'edges': list(edges)})
            if not self.route_ids:
                raise ValueError('Persistent traffic requires native routes for ongoing demand')
            ends = [0.]
            for item in xml(self.config['route_file'], 'routes').iter():
                for key in ('depart', 'end'):
                    value = item.attrib.get(key, '')
                    if value.replace('.', '', 1).isdigit():
                        ends.append(float(value))
            self.next_depart = max(ends) + self.period
            path = Path(self.world_provenance['manifest']).parent
            definition = path / 'continuing-demand.json'
            definition.write_text(json.dumps({'native_api': 'TraCI route.add/vehicle.add',
                                             'seed': self.initial_seed, 'period_seconds': self.period,
                                             'starts_at_seconds': self.next_depart, 'routes': templates}, indent=2))
            files = {name: item['path'] for name, item in self.world_provenance['files'].items()}
            files['continuing_demand'] = definition
            self.world_provenance = record(path, self.world_provenance['backend'],
                                           self.world_provenance['definition'], files)
        self.task_steps = 0
        return self.observation, {**self.info, **self.world_info()}

    def replenish(self):
        through = self.native.sim_step + self.native.delta_time
        while self.next_depart <= through:
            identifier = f'studio-world-vehicle-{self.inserted}'
            route = self.route_ids[int(self.rng.integers(len(self.route_ids)))]
            self.native.sumo.vehicle.add(identifier, route, depart=str(max(self.next_depart, self.native.sim_step)),
                                         departLane='best', departSpeed='max')
            self.inserted += 1
            self.next_depart += self.period

    def world_info(self):
        return {'world_steps': self.world_steps, 'world_time': self.native.sim_step,
                'world_time_unit': 'seconds', 'world_reset_count': int(self.initialized),
                'active_actor_ids': list(self.native.sumo.vehicle.getIDList()),
                'continuing_demand_inserted': self.inserted, 'world_provenance': self.world_provenance}

    def step(self, action):
        self.replenish()
        self.observation, reward, terminated, truncated, self.info = self.native.step(action)
        if terminated or truncated:
            raise RuntimeError('Native traffic world ended unexpectedly')
        self.world_steps += 1
        self.task_steps += 1
        return self.observation, reward, False, self.task_steps >= self.task_horizon, {**self.info, **self.world_info()}

    def decision_state(self, observation):
        return {**decision_state(self.mode, self, observation), 'simulation_mode': 'world', **self.world_info()}

    def close(self):
        self.native.close()
