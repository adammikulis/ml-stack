"""Grounded car input remains compact while raw decision evidence stays unchanged."""

import copy
from types import SimpleNamespace

from poolhouse.gym import simulation
from poolhouse.gym.observations import car_model_state


def test_car_prompt_uses_each_sensor_once_and_keeps_grounded_lane_stop_features():
    raw = {'environment': 'car', 'simulation_mode': 'world', 'native_observation': [123.456789] * 300,
           'world_steps': 42, 'active_actor_ids': ['a', 'b'], 'applied_native_control': [.1, .2],
           'ego': {'speed_km_h': 18, 'heading_radians': .2, 'position_m': [999.12345, 321.3456]},
           'lane': {'heading_radians': .1, 'lateral_offset_m': .4, 'width_m': 3.5},
           'navigation_normalized': [.1, .2], 'stop_rule': {'distance_m': 12, 'completed': False,
                                                         'stop_line': [[5, 4], [6, 3]]},
           'sensors': {'lidar_range_m': 50, 'lidar_normalized': [.123456789, .998765432, 1.]}}
    original = copy.deepcopy(raw)
    model = car_model_state(raw)
    assert raw == original
    assert 'native_observation' not in model and 'world_steps' not in model
    assert 'active_actor_ids' not in model and 'applied_native_control' not in model
    assert model['ego'] == {'speed_km_h': 18, 'heading_radians': .2} and model['lane'] == raw['lane']
    assert model['stop_rule'] == {'distance_m': 12, 'completed': False}
    assert model['navigation_normalized'] == [.1, .2]
    assert model['sensors'] == {'lidar_clear_range_m': 50., 'lidar_ray_count': 3,
                                'lidar_hits_m': [[0, 6.17], [1, 49.94]]}


def test_decision_transport_keeps_exact_raw_input_beside_compact_model_input():
    requests = []
    live = simulation.Simulation.__new__(simulation.Simulation)
    live.decider = SimpleNamespace(status='ready', error=None, pending=None, poll=lambda: None, submit=requests.append)
    live.environment, live.control_revision, live.decision_max_age_s = 'car', 7, 1
    live.state = {'sequence': 11, 'agent_id': 'controlled-car'}
    live.settings = {'model': 'strands'}
    live.names = ['action-' + str(index) for index in range(9)]
    live.native_actions, live.observation = list(range(9)), [.123456789]
    named = {'environment': 'car', 'native_observation': [.123456789],
             'sensors': {'lidar_range_m': 50, 'lidar_normalized': [.123456789, 1.]}}
    action, decision = live.decision_action(named)
    assert action == 3 and decision['fallback']
    request = requests[0]
    assert request['state'] == named and request['observation'] == [.123456789]
    assert request['revision'] == 7 and request['agent_id'] == 'controlled-car'
    assert 'native_observation' not in request['model_state']
    assert request['model_state']['sensors']['lidar_hits_m'] == [[0, 6.17]]
