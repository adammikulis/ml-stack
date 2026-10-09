"""Opt-in acceptance with installed verified Strands weights and native car physics."""

import json
import os
import shutil
import time
from pathlib import Path

import pytest

from poolhouse.gym import simulation
from poolhouse.gym.models import model_choices
from poolhouse.serve import broker_wire
from poolhouse.serve.process import kill_process_tree

pytestmark = pytest.mark.slow


@pytest.fixture
def installed_decide_cache(tmp_path, monkeypatch):
    configured = os.environ.get('POOLHOUSE_ACCEPTANCE_DECIDE_CACHE')
    if not configured:
        pytest.skip('set POOLHOUSE_ACCEPTANCE_DECIDE_CACHE to an installed pinned decide cache')
    installed = Path(configured)
    assert (installed / 'snapshots').is_dir()
    cache = tmp_path / 'cache'
    (cache / 'decide').mkdir(parents=True)
    (cache / 'decide' / 'snapshots').symlink_to(installed / 'snapshots', target_is_directory=True)
    if (installed / 'verified.json').is_file():
        shutil.copyfile(installed / 'verified.json', cache / 'decide' / 'verified.json')
    monkeypatch.setenv('POOLHOUSE_CACHE', str(cache))
    monkeypatch.setenv('POOLHOUSE_HOME', str(tmp_path / 'state'))
    monkeypatch.delenv('POOLHOUSE_BROKER_LOCAL', raising=False)
    yield
    record = tmp_path / 'state' / 'broker.json'
    if record.is_file():
        kill_process_tree(json.loads(record.read_text())['pid'])


def _await_native_decision(live, device, engine):
    began = time.monotonic()
    steps, waits = [], 0
    while time.monotonic() - began < 120:
        if live.state.get('decision_result') and (device == 'cpu' or not live.state['decision'].get('fallback')):
            break
        tick = time.monotonic()
        live.step()
        steps.append(time.monotonic() - tick)
        if live.state['decision'].get('fallback'):
            waits += 1
            assert live.state['action'] == 3
        assert live.env.native.engine is engine
        assert live.state['info']['world_reset_count'] == 1
        assert live.state['frame']
        assert live.state.get('decision_readiness', {}).get('status') != 'error', live.state
        time.sleep(.1)
    return steps, waits


def test_cached_strands_reports_native_world_decisions_without_blocking(tmp_path, monkeypatch, installed_decide_cache):
    pytest.importorskip('metadrive')
    monkeypatch.setattr(simulation, 'artifact_root', lambda: tmp_path / 'recordings')
    device = os.environ.get('POOLHOUSE_ACCEPTANCE_DECISION_DEVICE', 'cpu')
    age_limit = 1 if device == 'auto' else 30
    live = simulation.Simulation({'id': 'native-strands', 'environment': 'car', 'seed': 17,
        'controller': 'decider', 'config': {'simulation_mode': 'world', 'horizon': 100000,
            'num_agents': 4, 'traffic_density': .1, 'decision_max_age_s': age_limit,
            'decision_device': device,
            'world': {'mode': 'procedural', 'map': 'SCSCS'}, 'render_preview': True}})
    try:
        engine = live.env.native.engine
        assert live.state['info']['render']['stop_signs']
        assert len(live.state['info']['render']['vehicles']) >= 4
        assert live.state['frame']
        steps, waits = _await_native_decision(live, device, engine)
        result = live.state.get('decision_result')
        assert result, live.state.get('decision_readiness')
        assert 'strands' in result['model'].lower() and result['backend']
        assert result['probabilities'] and result['sequence'] < live.state['sequence']
        assert result['agent_id'] == live.state['agent_id']
        assert result['device'] == live.state['device']
        if device == 'auto':
            assert result['device'] != 'cpu'
            assert not live.state['decision']['fallback'], result
            assert result['input_age_s'] <= age_limit
        elif result['input_age_s'] > age_limit:
            assert live.state['decision']['reason'] == 'stale'
        print({'device': result['device'], 'latency_ms': result['latency_ms'],
               'input_age_s': result['input_age_s'], 'choice': result['choice'],
               'applied': not live.state['decision']['fallback']})
        assert waits >= 2 and len(steps) >= 3
        assert max(steps) < 2, steps
        pid = live.decider.handle.pid
        tick = time.monotonic()
        live.command('pause', {})
        assert time.monotonic() - tick < 2
        assert live.decider is None and live.state['status'] == 'paused'
        assert live.env.native.engine is engine
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        if device == 'auto':
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                claims = broker_wire.status()['claims']
                if not any(held['pid'] == pid for held in claims.values()):
                    break
                time.sleep(.05)
            assert not any(held['pid'] == pid for held in claims.values())
    finally:
        live.stop_decider()
        live.env.close()


def test_installed_smolvlm_reads_native_rgb_thermal_and_pause_releases_lease(tmp_path, monkeypatch, installed_decide_cache):
    if os.environ.get('POOLHOUSE_ACCEPTANCE_VISION') != '1':
        pytest.skip('set POOLHOUSE_ACCEPTANCE_VISION=1 for serialized native vision acceptance')
    pytest.importorskip('PyFlyt')
    model = next(row for row in model_choices()['vision'] if 'smolvlm' in row['label'].lower() and row['available'])
    monkeypatch.setattr(simulation, 'artifact_root', lambda: tmp_path / 'recordings')
    live = simulation.Simulation({'id': 'native-vision', 'environment': 'drone', 'seed': 17,
        'controller': 'native-patrol', 'config': {'simulation_mode': 'world', 'vision_model': model['id'],
            'world': {'mode': 'procedural', 'size': 20, 'trees': 5, 'n_agents': 2, 'hikers': 1, 'fires': 1}}})
    try:
        live.command('play', {})
        began = time.monotonic()
        while time.monotonic() - began < 120 and not live.state.get('perception'):
            live.step()
            assert live.state.get('perception_readiness', {}).get('status') != 'error', live.state['perception_readiness']
            time.sleep(.1)
        result = live.state.get('perception')
        assert result and result['perception']
        assert result['backend'] == 'llama.cpp' and result['model'] == model['id']
        assert result['agent_id'] == live.state['agent_id']
        assert result['camera']['image_hashes'].keys() == {'rgb', 'thermal'}
        assert result['camera']['thermal_kind'] == 'synthetic visible-surface temperature'
        assert 'detections_visible' not in result['camera'] and 'targets' not in result
        print({'model': result['model'], 'latency_ms': result['latency_ms'],
               'actor': result['agent_id'], 'frame_id': result['camera']['frame_id'],
               'perception': result['perception'][:100]})
        pid = live.perception.process.handle.pid
        live.command('pause', {})
        assert live.perception is None
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            servers = broker_wire.status()['servers']
            if not any(h['pid'] == pid for s in servers for h in s['holders']):
                break
            time.sleep(.05)
        assert not any(h['pid'] == pid for s in servers for h in s['holders'])
    finally:
        live.stop_perception()
        live.env.close()
