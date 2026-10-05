"""Installed model choices and camera-only requests against a compatible HTTP server."""

import base64
import contextlib
import io
import json
import sys
import time
from types import SimpleNamespace

import pytest
from conftest import json_reply
from PIL import Image

from ml_stack.gym import models, vision_process

pytestmark = pytest.mark.redteam


def camera():
    output = io.BytesIO()
    Image.new('RGB', (2, 2), (30, 40, 50)).save(output, format='PNG')
    pixels = base64.b64encode(output.getvalue()).decode()
    return {'rgb': pixels, 'thermal': pixels, 'frame_id': 18, 'world_time': 3.6,
            'pose': {'position': [1, 2, 3]}, 'intrinsics': {'width': 2, 'height': 2},
            'thermal_kind': 'synthetic visible-surface temperature',
            'detections_visible': [{'secret': 'OMNISCIENT_HIKER_LOCATION'}]}


def test_catalogue_preserves_registered_and_unsupported_local_models(monkeypatch, tmp_path):
    weights, projector = tmp_path / 'smol.gguf', tmp_path / 'projector.gguf'
    weights.touch()
    projector.touch()
    small = SimpleNamespace(id='smol', name='SmolVLM 256M', path=weights, format='gguf',
                            mmproj=projector, is_complete=True)
    fast = SimpleNamespace(id='fast', name='FastVLM 0.5B', path=weights, format='mlx',
                           mmproj=None, is_complete=True)
    requested = []
    monkeypatch.setattr(models.hub, 'discover', lambda **kwargs: requested.append(kwargs) or [fast, small])
    monkeypatch.setattr(models.registry, 'listing', lambda: [{'name': 'trained', 'path': str(tmp_path)}])
    choices = models.model_choices()
    assert requested == [{'kind': 'vision'}]
    assert choices['decision'][0]['label'] == 'Strands Decider 2B (default)'
    assert choices['decision'][1]['checkpoint'] == str(tmp_path)
    assert choices['vision_default'] == 'smol'
    assert choices['vision'][0]['available']
    assert choices['vision'][1]['status'] == 'unsupported'
    assert 'native vision runtime is not implemented' in choices['vision'][1]['reason']
    with pytest.raises(ValueError, match='native vision runtime'):
        models.selected_vision('fast')
    with pytest.raises(ValueError, match='not installed'):
        models.selected_vision('/outside/unregistered.gguf')


def test_vision_uses_rgb_thermal_pixels_without_simulator_truth(server, monkeypatch):
    requests = []

    def reply(method, path, body):
        if path == '/v1/chat/completions':
            requests.append(json.loads(body))
            return json_reply({'id': 'camera', 'model': 'actual-smol', 'choices': [
                {'index': 0, 'message': {'role': 'assistant', 'content': 'A warm visible shape on the left'},
                 'finish_reason': 'stop'}]})
        return json_reply({'data': [{'id': 'actual-smol'}]})

    service = server(reply)
    leases = []

    def lease(model, **kwargs):
        leases.append((model, kwargs))
        return contextlib.nullcontext(service)

    monkeypatch.setattr(vision_process, 'serve_model', lease)
    monkeypatch.setattr(vision_process, 'selected_vision', lambda _: {
        'id': 'smol', 'model': '/models/smol.gguf', 'mmproj': '/models/projector.gguf'})
    capture = io.StringIO()
    original = {'camera': camera(), 'sequence': 22, 'timestamp': time.monotonic(), 'revision': 7, 'agent_id': 'drone-1'}
    monkeypatch.setattr(sys, 'argv', ['vision', json.dumps('smol')])
    monkeypatch.setattr(sys, 'stdin', io.StringIO(json.dumps(original) + '\n'))
    monkeypatch.setattr(sys, 'stdout', capture)
    vision_process.serve()
    events = [json.loads(line) for line in capture.getvalue().splitlines()]
    assert events[0]['status'] == 'ready'
    result = events[1]['result']
    assert result['perception'] == 'A warm visible shape on the left'
    assert result['sequence'] == 22 and result['revision'] == 7
    assert result['camera']['frame_id'] == 18 and result['camera']['image_hashes']['rgb']
    assert leases[0][1]['reason'] == 'Live Gym RGB and synthetic thermal camera perception'
    assert leases[0][1]['escalate'] is False and leases[0][1]['anyway'] is False
    content = requests[0]['messages'][0]['content']
    assert [part['type'] for part in content] == ['text', 'image_url', 'image_url']
    assert 'OMNISCIENT_HIKER_LOCATION' not in json.dumps(requests)
    assert 'world coordinates' in content[0]['text']


@pytest.mark.parametrize('revision,accepted', [(4, True), (3, False)])
def test_vision_rejects_previous_actor_revision_without_stalling_native_state(monkeypatch, revision, accepted):
    from test_gym_runtime import PendingDecider

    monkeypatch.setattr(vision_process, 'VisionProcess', PendingDecider)
    perception = vision_process.Perception('model')
    simulation = SimpleNamespace(control_revision=4, running=True, state={
        'agent_id': 'drone-1', 'sequence': 8, 'info': {'render': {'camera': camera()}}})
    perception.update(simulation)
    request = perception.process.pending
    assert request['sequence'] == 8
    assert 'detections_visible' not in request['camera']
    perception.process.event = {'status': 'ready', 'result': {
        'revision': revision, 'agent_id': 'drone-1', 'timestamp': time.monotonic(),
        'perception': 'visible warm shape', 'model': 'model'}}
    simulation.state['sequence'] = 9
    perception.update(simulation)
    assert simulation.state['perception_result']['accepted'] is accepted
    assert (simulation.state['perception'] is not None) is accepted
    assert simulation.state['sequence'] == 9
    assert perception.process.pending is None
