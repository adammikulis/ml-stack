"""Isolated native graph tasks, authenticated workers and maintained allocation checks."""

from pathlib import Path

import pytest
from workspace_kit import Kit, clean_env

from ml_stack.workspace import device_agent, localagent, resource_allocations as resources, tokens
from ml_stack.workspace.taskboard import TaskBoard


@pytest.fixture
def board(tmp_path, monkeypatch):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    kit.now = [1000.0]
    kit.ws.clock = lambda: kit.now[0]
    kit.parent = kit.agent('lead')
    delegated = kit.ws.delegate(kit.parent, 'worker')
    kit.worker_id = delegated['id']
    kit.child = tokens.read_file(Path(delegated['token_file']))
    monkeypatch.setattr(device_agent.home, 'machine_id', lambda: '1234567890abcdef')
    localagent.save(kit.ws, localagent.Agent('native-worker', 'qwen', identity=kit.worker_id,
                                           profile='coding', pid=555, process_started=42))
    device_agent.bind_worker(kit.ws, kit.owner, 'native-worker')
    kit.status = {'servers': [{'ours': True, 'pid': 999, 'model': 'qwen', 'port': 51548,
                  'holders': [{'lease': 'native-grant', 'pid': 555, 'pid_started': 42}]}]}
    monkeypatch.setattr(resources.broker_wire, 'status', lambda **kw: kit.status)
    monkeypatch.setattr(resources, 'pid_exists', lambda pid: True)
    monkeypatch.setattr(resources, 'started_at', lambda pid: 42)
    kit.board = TaskBoard(kit.ws)
    kit.spec = {'title': 'Inspect native simulation', 'description': 'Collect a reproducible replay.',
                'acceptance': ['Replay passes'], 'source_key': 'repo:demo/sim:issue:42'}
    kit.task = kit.board.create(kit.parent, kit.spec)
    kit.allocation = resources.assign(kit.ws, kit.parent, kit.worker_id, kit.task['id'], 'native-grant')
    return kit


def proposed(kit):
    kit.board.claim(kit.child, kit.task['id'], kit.allocation['allocation_id'])
    return kit.board.submit(kit.child, kit.task['id'],
                           {'artifacts': {'replay.json': 'a' * 64},
                            'checks': [{'name': 'Worker claims tests', 'passed': True}],
                            'summary': 'Native replay proposed',
                            'provenance': {'commit': 'abc123', 'environment': 'native-python',
                                           'model': 'qwen', 'runtime': 'Agents SDK'}})


def accepted():
    return {'accepted': True, 'reason': 'Independent native replay passed',
            'checks': [{'name': 'Replay passes', 'passed': True}]}
