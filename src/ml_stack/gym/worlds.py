"""Per-example native world construction and UI configuration descriptors."""

from importlib import import_module

from ml_stack.gym.car_definition import SCHEMA
from ml_stack.gym.traffic import traffic_defaults
from ml_stack.gym.world_schema import TRAFFIC, WAREHOUSE


def schema(environment):
    if environment == 'car':
        return SCHEMA
    if environment == 'warehouse':
        return WAREHOUSE
    if environment in {'traffic', 'traffic-driving'}:
        return TRAFFIC
    return None


def configure_world(environment, config, seed=0):
    """Return native constructor settings and a persisted world manifest."""
    cfg = dict(config or {})
    world = cfg.pop('world', {})
    seed = cfg.pop('world_seed', seed)
    if environment == 'car':
        native, provenance = import_module('ml_stack.gym.car_definition').build(world, seed)
    elif environment == 'warehouse':
        native, provenance = import_module('ml_stack.gym.warehouse_world').build(world, seed)
    elif environment in {'traffic', 'traffic-driving'}:
        cfg = traffic_defaults(cfg)
        native, provenance = import_module('ml_stack.gym.traffic_world').build(world, seed)
    else:
        return cfg, None
    return {**cfg, **native}, provenance
