"""Versioned forest-search definitions; no simulator imports in the application."""

import json
import math

from poolhouse.files import write_json
from poolhouse.gym.world_files import directory, imported, record

ACTIONS = ['hold', 'north', 'south', 'east', 'west', 'ascend', 'descend',
           'yaw left', 'yaw right', 'report hiker', 'report fire']
SCHEMA = {'modes': ['procedural', 'manual'], 'fields': {
    'size': {'type': 'number', 'min': 10, 'max': 100, 'default': 30, 'unit': 'metres'},
    'trees': {'type': 'integer', 'min': 0, 'max': 300, 'default': 60},
    'hikers': {'type': 'integer', 'min': 0, 'max': 20, 'default': 2},
    'fires': {'type': 'integer', 'min': 0, 'max': 20, 'default': 2},
    'n_agents': {'type': 'integer', 'min': 1, 'max': 4, 'default': 2},
    'map_file': {'type': 'file', 'mode': 'manual', 'format': 'Version 1 forest-search JSON'}},
    'constraint': 'Native PyFlyt quadrotor PID and Bullet collisions; visibility-limited synthetic thermal camera',
    'runtime_fields': {'task_horizon': {'type': 'integer', 'min': 1, 'default': 500}}}


def number(value, label, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be a finite number')
    if not low <= value <= high or (integer and not isinstance(value, int)):
        raise ValueError(f'{label} must be between {low} and {high}')
    return value


def validate(definition):
    if not isinstance(definition, dict) or type(definition.get('version')) is not int or definition.get('version') != 1:
        raise ValueError('Forest definition must be a version 1 JSON object')
    size = number(definition.get('size'), 'size', 10, 100)
    number(definition.get('n_agents'), 'n_agents', 1, 4, True)
    for kind, limit in [('trees', 300), ('hikers', 20), ('fires', 20)]:
        rows = definition.get(kind)
        if not isinstance(rows, list) or len(rows) > limit:
            raise ValueError(f'{kind} must be a bounded list')
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f'{kind} entries must be objects')
            number(row.get('x'), 'x', -size / 2, size / 2)
            number(row.get('y'), 'y', -size / 2, size / 2)
            if kind == 'trees':
                number(row.get('height'), 'height', 1, 15)
                number(row.get('radius'), 'radius', .1, 3)
    return definition


def build(world, seed=0):
    if not isinstance(world, dict):
        raise ValueError('world must be an object')
    mode = world.get('mode', 'procedural')
    if mode == 'manual':
        source = imported(world.get('map_file', ''))
        definition = validate(json.loads(source.read_text()))
    elif mode == 'procedural':
        import numpy as np
        seed = number(world.get('seed', seed), 'seed', 0, 2**31 - 1, True)
        size = number(world.get('size', 30), 'size', 10, 100)
        rng = np.random.default_rng(seed)
        definition = {'version': 1, 'seed': seed, 'size': size,
                      'n_agents': number(world.get('n_agents', 2), 'n_agents', 1, 4, True)}
        for kind, default, maximum in [('trees', 60, 300), ('hikers', 2, 20), ('fires', 2, 20)]:
            count = number(world.get(kind, default), kind, 0, maximum, True)
            rows = []
            while len(rows) < count:
                x, y = rng.uniform(-size / 2, size / 2), rng.uniform(-size / 2, size / 2)
                if math.hypot(x, y) < 3:
                    continue
                row = {'x': x, 'y': y}
                if kind == 'trees':
                    row.update(height=rng.uniform(3, 8), radius=rng.uniform(.3, .7))
                rows.append(row)
            definition[kind] = rows
    else:
        raise ValueError('world mode must be procedural or manual')
    path = directory('pyflyt', definition)
    write_json(path / 'forest.json', definition)
    provenance = record(path, 'PyFlyt', definition, {'forest': path / 'forest.json'})
    return {'forest_definition': definition, 'world_seed': seed}, provenance
