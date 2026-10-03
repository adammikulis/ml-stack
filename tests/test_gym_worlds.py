"""Native procedural world reproducibility and manual scenario ownership."""

import shutil
from pathlib import Path

import pytest

from ml_stack.gym.adapters import make_environment
from ml_stack.gym.worlds import configure_world, schema


def test_world_descriptors_expose_native_modes_without_universal_physics():
    assert schema('warehouse')['fields']['layout']['format'].startswith('RWARE ASCII')
    assert schema('traffic-driving')['modes'] == ['procedural', 'manual']
    assert schema('car') is None


def test_warehouse_ascii_validation():
    pytest.importorskip('rware')
    with pytest.raises(ValueError, match='rectangular'):
        configure_world('warehouse', {'world': {'mode': 'manual', 'layout': '..g\nxx..'}})


@pytest.mark.slow
def test_native_warehouse_generated_and_manual_layouts(tmp_path):
    pytest.importorskip('rware')
    config, generated = configure_world('warehouse', {'max_steps': 40}, seed=11)
    again, repeated = configure_world('warehouse', {'max_steps': 40}, seed=11)
    other, distinct = configure_world('warehouse', {'max_steps': 40}, seed=12)
    assert config == again
    assert generated['files'] == repeated['files']
    assert generated['definition'] != distinct['definition']
    assert any(config[key] != other[key] for key in ('shelf_rows', 'shelf_columns', 'column_height'))
    env = make_environment('warehouse', config)
    try:
        observation, _ = env.reset(seed=11)
        assert env.observation_space.contains(observation)
        assert env.unwrapped.grid_size[1] == config['shelf_columns'] * 3 + 1
        env.step([0, 0])
    finally:
        env.close()
    layout = '.......\n.xx.xx.\n.......\n.g...g.'
    config, manual = configure_world('warehouse', {'world': {'mode': 'manual', 'layout': layout}}, seed=4)
    env = make_environment('warehouse', config)
    try:
        env.reset(seed=4)
        assert env.unwrapped.grid_size == (4, 7)
        assert len(env.unwrapped.shelfs) == 4
        assert env.unwrapped.goals == [(1, 3), (5, 3)]
        env.step([0, 0])
        assert Path(manual['files']['layout']['path']).read_text().strip() == layout
    finally:
        env.close()


@pytest.mark.slow
def test_native_sumo_generated_seed_and_manual_files(tmp_path, monkeypatch):
    pytest.importorskip('sumo')
    pytest.importorskip('sumo_rl')
    options = {'num_seconds': 30, 'world': {'demand_seconds': 60}}
    config, first = configure_world('traffic', options, seed=4)
    same, repeated = configure_world('traffic', options, seed=4)
    other, distinct = configure_world('traffic', options, seed=5)
    assert config == same
    with monkeypatch.context() as temporary:
        temporary.setenv('ML_STACK_CACHE', str(tmp_path / 'fresh-cache'))
        _, fresh = configure_world('traffic', options, seed=4)
        assert {name: value['sha256'] for name, value in first['files'].items()} == {
            name: value['sha256'] for name, value in fresh['files'].items()}
    assert first['files'] == repeated['files']
    assert first['files']['network']['sha256'] != distinct['files']['network']['sha256']
    assert first['files']['routes']['sha256'] != distinct['files']['routes']['sha256']
    for supplied in (config, other):
        env = make_environment('traffic', supplied)
        try:
            observation, _ = env.reset(seed=4)
            assert env.observation_space.contains(observation)
            for _ in range(3):
                env.step(0)
            assert len(env.sumo.trafficlight.getIDList()) == 1
            assert env.sumo.simulation.getTime() > 0
            assert env.sumo.vehicle.getIDList()
        finally:
            env.close()
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(tmp_path))
    shutil.copyfile(config['net_file'], tmp_path / 'manual.net.xml')
    shutil.copyfile(config['route_file'], tmp_path / 'manual.rou.xml')
    manual, manifest = configure_world('traffic-driving', {'num_seconds': 30, 'world': {
        'mode': 'manual', 'net_file': 'manual.net.xml', 'route_file': 'manual.rou.xml'}}, seed=4)
    assert manifest['definition']['mode'] == 'manual'
    assert Path(manual['net_file']).read_bytes() == (tmp_path / 'manual.net.xml').read_bytes()
    assert Path(manual['route_file']).read_bytes() == (tmp_path / 'manual.rou.xml').read_bytes()
    env = make_environment('traffic', manual)
    try:
        env.reset(seed=4)
        env.step(0)
        assert env.sumo.trafficlight.getIDList() == ('A0',)
    finally:
        env.close()
    with pytest.raises(ValueError, match='relative paths'):
        configure_world('traffic', {'world': {'mode': 'manual', 'net_file': '../escape', 'route_file': 'manual.rou.xml'}})
