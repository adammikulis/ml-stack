"""Actual PyFlyt flight, camera visibility and persistent clock proofs."""

import base64
import io
import json

import pytest

from ml_stack.gym.adapters import actions, make_environment, render_state
from ml_stack.gym.drone_definition import build


@pytest.fixture
def native_drone(monkeypatch, tmp_path):
    pytest.importorskip('PyFlyt')
    monkeypatch.setenv('ML_STACK_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(tmp_path))
    definition = {'version': 1, 'size': 20, 'n_agents': 2, 'trees': [],
                  'hikers': [{'x': 5, 'y': 0}], 'fires': [{'x': 6, 'y': 2}]}
    (tmp_path / 'forest.json').write_text(json.dumps(definition))
    env = make_environment('drone', {'simulation_mode': 'world', 'task_horizon': 2,
                                   'world': {'mode': 'manual', 'map_file': 'forest.json'}}, seed=2)
    try:
        yield env
    finally:
        env.close()


@pytest.mark.slow
def test_native_forest_visuals_preserve_tree_colliders(monkeypatch, tmp_path):
    pytest.importorskip('PyFlyt')
    from PIL import Image
    monkeypatch.setenv('ML_STACK_CACHE', str(tmp_path / 'cache'))
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(tmp_path))
    definition = {'version': 1, 'size': 20, 'n_agents': 1,
                  'trees': [{'x': 6, 'y': 2, 'height': 6, 'radius': .7}], 'hikers': [], 'fires': []}
    (tmp_path / 'forest.json').write_text(json.dumps(definition))
    env = make_environment('drone', {'simulation_mode': 'world',
                                   'world': {'mode': 'manual', 'map_file': 'forest.json'}}, seed=2)
    try:
        body = env.scenery[0]['body_id']
        collision = env.native.getCollisionShapeData(body, -1)
        assert collision[0][2] == env.native.GEOM_CYLINDER
        assert collision[0][3][:2] == pytest.approx((6, .7))
        visual = env.native.getVisualShapeData(body)
        assert len(visual) == 4 and visual[0][2] == env.native.GEOM_CYLINDER
        assert visual[0][3][:2] == pytest.approx((2.7, .21))
        assert all(shape[2] == env.native.GEOM_MESH for shape in visual[1:])
        ground = env.native.getVisualShapeData(env.native.planeId)[0]
        assert ground[7] == pytest.approx((.34, .39, .22, 1))
        Image.fromarray(env.rgb).save(tmp_path / 'native-forest.png')
        assert env.camera['rgb'] and env.camera['agent_id'] == 'drone-0'
        assert env.native.getNumBodies() == 3
    finally:
        env.close()


@pytest.mark.slow
def test_native_camera_reward_and_persistent_clock(native_drone):
    import numpy as np
    from PIL import Image
    env = native_drone
    native = env.native
    initial, _ = env.reset(seed=99)
    assert initial.shape == env.observation_space.shape == (1036,)
    assert actions('drone', env)[0][0] == 'hold'
    geometry, frame = render_state('drone', env)
    assert frame is None
    assert len(geometry['drones']) == 2
    camera = geometry['camera']
    for key in ['rgb', 'thermal']:
        image = Image.open(io.BytesIO(base64.b64decode(camera[key])))
        assert image.size == (128, 128)
        assert np.asarray(image).std() > 0
    assert len(env.visible_targets) >= 1
    obs, reward, done, horizon, info = env.step(9)
    assert reward == pytest.approx(sum(info['reward_components'].values()))
    assert info['reward_components']['report'] == 5
    assert not done and not horizon
    assert info['world_time'] == pytest.approx(.2)
    env.step(0)
    before = env.native.state(0).copy()
    env.reset(seed=12345)
    assert env.native is native and env.native_world_reset_count == 1
    np.testing.assert_array_equal(env.native.state(0), before)
    _, _, _, _, info = env.step(3)
    assert info['world_steps'] == 3 and info['world_time'] == pytest.approx(.6)
    clock = info['world_time']
    env.select_agent('drone-1')
    assert env.native.elapsed_time == clock
    assert env.decision_state(obs)['agent_id'] == 'drone-1'


@pytest.mark.slow
def test_native_occlusion_hides_targets_and_thermal(native_drone):
    import numpy as np

    from ml_stack.gym.drone_sensors import capture
    env = native_drone
    assert env.visible_targets
    native = env.native
    shape = native.createCollisionShape(native.GEOM_BOX, halfExtents=[.1, 8, 8])
    visual = native.createVisualShape(native.GEOM_BOX, halfExtents=[.1, 8, 8], rgbaColor=[.1, .2, .1, 1])
    native.createMultiBody(baseMass=0, baseCollisionShapeIndex=shape,
                           baseVisualShapeIndex=visual, basePosition=[2, 0, 5])
    native.register_all_new_bodies()
    capture(env, fresh=True)
    assert env.visible_targets == []
    assert np.max(env.thermal) == 20
    assert env.camera['detections_visible'] == []
    state = env.decision_state(env.observation())
    assert 'hikers' not in state and 'fires' not in state and 'forest' not in state
    _, reward, _, _, info = env.step(9)
    assert info['reward_components']['report'] == -.5
    assert reward == pytest.approx(sum(info['reward_components'].values()))


@pytest.mark.slow
def test_procedural_collision_geometry_seed_and_episode_reset(monkeypatch, tmp_path):
    pytest.importorskip('PyFlyt')
    monkeypatch.setenv('ML_STACK_CACHE', str(tmp_path / 'cache'))
    config = {'task_horizon': 1, 'world': {'trees': 6, 'hikers': 1, 'fires': 1}}
    env = make_environment('drone', config, seed=2)
    try:
        original = env.definition
        assert env.native.getNumBodies() == 3 + 8
        env.step(0)
        env.reset(seed=3)
        assert env.definition != original and env.native_world_reset_count == 2
        env.step(0)
        env.reset(seed=2)
        assert env.definition == original
        assert env.native.getCollisionShapeData(env.scenery[0]['body_id'], -1)
    finally:
        env.close()


@pytest.mark.redteam
@pytest.mark.parametrize('definition', [None, [], {'version': 2}, {'version': 1, 'size': True}])
def test_manual_forest_rejects_invalid_shapes(monkeypatch, tmp_path, definition):
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(tmp_path))
    (tmp_path / 'forest.json').write_text(json.dumps(definition))
    with pytest.raises(ValueError):
        build({'mode': 'manual', 'map_file': 'forest.json'})


@pytest.mark.slow
def test_native_cpu_online_weights_change_frozen_weights_do_not(native_drone):
    import numpy as np
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    torch.set_num_threads(1)
    env = native_drone
    check_env(env, warn=True)
    native = env.native
    policy = PPO('MlpPolicy', env, device='cpu', n_steps=16, batch_size=8,
                 n_epochs=2, seed=3, policy_kwargs={'net_arch': [16]}, verbose=0)
    before = {key: value.detach().clone() for key, value in policy.policy.state_dict().items()}
    start_time = native.elapsed_time
    policy.learn(16)
    after = {key: value.detach().clone() for key, value in policy.policy.state_dict().items()}
    assert any(not torch.equal(before[key], after[key]) for key in before)
    assert env.native is native and env.native_world_reset_count == 1
    assert native.elapsed_time == pytest.approx(start_time + 16 * .2)
    updates = policy._n_updates
    observation, _ = env.reset()
    for _ in range(4):
        action, _ = policy.predict(observation, deterministic=True)
        observation, _, _, _, _ = env.step(action)
    assert policy._n_updates == updates
    for key, value in policy.policy.state_dict().items():
        np.testing.assert_array_equal(value.detach().numpy(), after[key].numpy())
    assert native.elapsed_time == pytest.approx(start_time + 20 * .2)


@pytest.mark.slow
def test_native_patrol_moves_selected_drone_without_task_reset(native_drone):
    import numpy as np
    env = native_drone
    before = env.native.state(0)[3].copy()
    for _ in range(10):
        _, _, _, _, info = env.step(None)
    assert np.linalg.norm(env.native.state(0)[3] - before) > .5
    assert info['native_pid_mode'] == 7
    assert info['applied_native_control'][:2] == [-8., -8.]
    assert env.native_world_reset_count == 1


@pytest.mark.slow
def test_simulation_drone_records_exact_sensor_input_for_reuse(monkeypatch, tmp_path):
    import numpy as np

    from ml_stack.gym.recordings import export_reviewed
    from ml_stack.gym.simulation import Simulation
    pytest.importorskip('PyFlyt')
    monkeypatch.setenv('ML_STACK_CACHE', str(tmp_path / 'cache'))
    simulation = Simulation({'id': 'drone-trajectory-proof', 'environment': 'drone',
                             'controller': 'manual', 'seed': 2,
                             'config': {'simulation_mode': 'world', 'task_horizon': 2,
                                        'world': {'trees': 4, 'hikers': 1, 'fires': 1}}})
    try:
        input_observation = simulation.observation.copy()
        input_camera = dict(simulation.env.camera)
        simulation.manual = 3
        simulation.step()
        row = json.loads((simulation.path / 'trajectory.jsonl').read_text().splitlines()[0])
        np.testing.assert_array_equal(row['transition']['observation'], input_observation)
        assert row['decision']['state']['camera'] == input_camera
        assert row['transition']['reward'] == pytest.approx(sum(row['info']['reward_components'].values()))
        assert row['decision']['state']['camera']['frame_id'] != row['info']['render']['camera']['frame_id']
        assert row['info']['native_pid_mode'] == 7
        assert simulation.settings['versions']['PyFlyt'] == '0.29.0'
        reviews = tmp_path / 'reviews.jsonl'
        reviews.write_text(json.dumps({'episode_id': row['transition']['episode_id'],
                                      'sequence': row['transition']['sequence'], 'label': 'east'}) + '\n')
        destination = tmp_path / 'cases.jsonl'
        export_reviewed(simulation.path / 'trajectory.jsonl', reviews, destination)
        case = json.loads(destination.read_text().splitlines()[0])
        assert case['state']['camera'] == input_camera
        assert case['label'] == 'east'
    finally:
        simulation.env.close()
