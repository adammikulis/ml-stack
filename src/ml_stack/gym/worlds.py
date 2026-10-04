"""Per-example native world construction and UI configuration descriptors."""

from ml_stack.gym.car_definition import SCHEMA, build as build_car
from ml_stack.gym.drone_definition import SCHEMA as DRONE_SCHEMA
from ml_stack.gym.traffic import traffic_defaults
from ml_stack.gym.traffic_world import build as build_traffic
from ml_stack.gym.warehouse_world import build as build_warehouse
from ml_stack.gym.world_schema import TRAFFIC, WAREHOUSE


def schema(environment):
    if environment == 'drone':
        return DRONE_SCHEMA
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
        native, provenance = build_car(world, seed)
    elif environment == 'warehouse':
        native, provenance = build_warehouse(world, seed)
        cfg.setdefault('request_queue_size', native['n_agents'])
    elif environment in {'traffic', 'traffic-driving'}:
        cfg = traffic_defaults(cfg)
        native, provenance = build_traffic(world, seed)
    else:
        return cfg, None
    return {**cfg, **native}, provenance
