"""Hostile browser bodies and planted native world files."""

import json

import pytest
import test_fleet_gym_boundaries as boundaries

from ml_stack.fleet import gym_recording_routes
from ml_stack.gym import car_definition, world_files
from ml_stack.gym.traffic_world import xml

daemon = boundaries.daemon
raw_request = boundaries.raw_request

pytestmark = pytest.mark.redteam


def test_hostile_browser_bodies_cannot_mutate_simulators_or_jobs(daemon):
    for path in ('/ui/gym/sessions', '/ui/workspace/jobs', '/ui/workspace/file'):
        for body in (b'null', b'[]', b'{', b'\xff'):
            assert raw_request(daemon, path, body)[0] == 400
        assert daemon.call(path, method='POST', body={}, ui_header=False)[0] == 403
    assert daemon.runner.snapshot() == []
    assert daemon.call('/ui/workspace/files')[0] == 200


def test_planted_world_symlink_and_traversal_cannot_read_private_file(tmp_path, monkeypatch):
    root = tmp_path / 'files'
    root.mkdir()
    secret = tmp_path / 'secret.json'
    secret.write_text('{"private":"canary"}')
    (root / 'escape.json').symlink_to(secret)
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(root))
    for name in ('escape.json', '../secret.json', str(secret)):
        with pytest.raises(ValueError):
            world_files.imported(name)
    monkeypatch.delenv('ML_STACK_GYM_FILES_ROOT')
    with pytest.raises(ValueError, match='requires the daemon files root'):
        world_files.imported('escape.json')
    assert secret.read_text() == '{"private":"canary"}'


@pytest.mark.parametrize('body', ['[]', 'null', '{', '{"block_sequence":[],"map_config":[]}'])
def test_hand_edited_car_world_rejects_invalid_native_metadata(tmp_path, monkeypatch, body):
    monkeypatch.setenv('ML_STACK_GYM_FILES_ROOT', str(tmp_path))
    monkeypatch.setattr(car_definition, 'directory', lambda *args: tmp_path)
    (tmp_path / 'map.json').write_text(body)
    with pytest.raises(ValueError):
        car_definition.build({'mode': 'manual', 'map_file': 'map.json'}, 0)


def test_sumoworld_entities_includes_and_wrong_root_are_refused(tmp_path):
    for text in ('<!DOCTYPE net [<!ENTITY x SYSTEM "file:///secret">]><net>&x;</net>',
                 '<net><include href="../secret"/></net>', '<routes/>'):
        path = tmp_path / 'network.xml'
        path.write_text(text)
        with pytest.raises(ValueError):
            xml(path, 'net')
    path.write_text('<net/>')
    assert xml(path, 'net').tag == 'net'


def test_recording_review_cannot_label_unseen_actions_or_follow_planted_files(daemon, tmp_path, monkeypatch):
    root = tmp_path / 'records'
    path = root / 'episode'
    path.mkdir(parents=True)
    private = tmp_path / 'private'
    private.write_text('secret-canary')
    row = {'sequence': 1, 'actions': ['stop'], 'frame_path': str(private),
           'transition': {'episode_id': 0, 'sequence': 1}}
    (path / 'trajectory.jsonl').write_text(json.dumps(row) + '\n')
    monkeypatch.setattr(gym_recording_routes, 'artifact_root', lambda: root)
    assert daemon.call('/ui/gym/recordings/episode/frame?sequence=1')[0] == 400
    body = {'episode_id': 0, 'sequence': 1, 'label': 'shell: rm'}
    assert daemon.call('/ui/gym/recordings/episode/review', method='POST', body=body)[0] == 400
    (path / 'reviews.jsonl').symlink_to(private)
    body['label'] = 'stop'
    assert daemon.call('/ui/gym/recordings/episode/review', method='POST', body=body)[0] == 400
    assert private.read_text() == 'secret-canary'
