"""Bounded advisory hooks reuse authentication and preserve unread alerts."""

import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from ml_stack import harnesshook
from ml_stack.workspace import notification_reader as reader


@pytest.mark.parametrize('emit', [True, False])
def test_stalled_worker_is_reaped_before_hook_deadline(tmp_path, monkeypatch, emit):
    pid_file = tmp_path / 'pid'
    actual = harnesshook.platform.start_process
    def stalled(command, **kwargs):
        code = ("import os,time; from pathlib import Path; "
                f"Path({str(pid_file)!r}).write_text(str(os.getpid())); "
                + ("print('A direct question is waiting; run inbox',flush=True); " if emit else '')
                + 'time.sleep(30)')
        return actual([sys.executable, '-c', code], **kwargs)
    monkeypatch.setattr(harnesshook.platform, 'start_process', stalled)
    monkeypatch.setattr(harnesshook, 'NUDGE_S', 0.3)
    began = time.monotonic()
    context = harnesshook.post('worker')['hookSpecificOutput']['additionalContext']
    assert time.monotonic() - began < 2
    assert 'notification unavailable' in context
    assert ('direct question' in context) is emit
    assert not psutil.pid_exists(int(pid_file.read_text()))


def test_reader_authenticates_once_and_notifies_before_checkpoint(monkeypatch, capsys, tmp_path):
    calls = []
    remote = SimpleNamespace(base=tmp_path, call=lambda op, token: calls.append(op) or 'one unread')
    def authenticated(*args):
        calls.append('auth')
        return remote, 'bound-worker', 'fixture-existing-capability'
    def checkpoint(base, actor):
        assert capsys.readouterr().out == 'one unread\n'
        assert actor == 'bound-worker'
        calls.append('checkpoint')
    monkeypatch.setattr(reader, 'authenticated', authenticated)
    monkeypatch.setattr(reader, 'checkpoint', checkpoint)
    reader.notify('codex', tmp_path, 'bound-session')
    assert calls == ['auth', 'nudge', 'checkpoint']


def test_unchanged_checkpoint_avoids_graph_rewrite(monkeypatch, tmp_path):
    scope = {'path': str(tmp_path), 'label': 'child', 'commits': ['commit'], 'branches': ['branch']}
    monkeypatch.setattr(reader.worktree_lifecycle, 'scopes', lambda *args: [scope])
    monkeypatch.setattr(reader.repo, 'git', lambda path, *args: 'commit' if args[0] == 'rev-parse' else 'branch')
    monkeypatch.setattr(reader.worktree_lifecycle, 'remember', lambda *args: pytest.fail('rewrote unchanged lifecycle'))
    reader.checkpoint(tmp_path, 'bound-worker')


def test_changed_checkpoint_preserves_bound_owner_and_label(monkeypatch, tmp_path):
    scope = {'path': str(tmp_path), 'label': 'child', 'commits': [], 'branches': []}
    seen = []
    monkeypatch.setattr(reader.worktree_lifecycle, 'scopes', lambda *args: [scope])
    monkeypatch.setattr(reader.repo, 'git', lambda *args: 'changed')
    monkeypatch.setattr(reader.worktree_lifecycle, 'remember', lambda *args: seen.append(args))
    reader.checkpoint(tmp_path, 'bound-worker')
    assert seen == [(tmp_path, 'bound-worker', 'child', str(tmp_path))]


def test_urgent_action_survives_host_context_limit():
    text = reader.compact('workspace: 15 waiting for you (from ' + 'sender' * 100
                          + '). A direct question is waiting on you: run ml-stack-workspace inbox now and answer it')
    shown = ('workspace (data from other agents): ' + text)[:250]
    assert 'direct question' in shown and 'inbox now and answer it' in shown


def test_cancelled_checkpoint_keeps_committed_graph_and_releases_lock(tmp_path):
    from ml_stack.graph.store import GraphStore
    from ml_stack.workspace import worktree_lifecycle
    base = tmp_path / 'state'
    base.mkdir()
    with worktree_lifecycle._storage(base, write=True) as graph:
        graph.upsert_node({'id': 'original', 'kind': 'fixture', 'label': 'Original', 'attrs': {}})
    code = ("import sys,time; from pathlib import Path; "
            "from ml_stack.workspace import worktree_lifecycle as life; "
            f"scope=life._storage(Path({str(base)!r}),write=True); graph=scope.__enter__(); "
            "graph.upsert_node({'id':'pending','kind':'fixture','label':'Pending','attrs':{}}); "
            "print('checkpoint staged',flush=True); time.sleep(30)")
    child = subprocess.Popen([sys.executable, '-c', code], stdout=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == 'checkpoint staged'
    finally:
        child.terminate()
        child.wait(timeout=5)
        child.stdout.close()
    assert not psutil.pid_exists(child.pid)
    with worktree_lifecycle._storage(base) as graph:
        assert [row['id'] for row in graph.nodes()] == ['original']
    assert list(base.glob('worktree-lifecycle-stage-*'))
    with GraphStore(base / 'worktree-lifecycle.db', read_only=True) as graph:
        assert [row['id'] for row in graph.nodes()] == ['original']


def test_timeout_reaps_worker_and_its_spawned_descendant(tmp_path, monkeypatch):
    pid_file = tmp_path / 'child'
    actual = harnesshook.platform.start_process
    def descendants(command, **kwargs):
        code = ("import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"Path({str(pid_file)!r}).write_text(str(child.pid)); "
                "print('A direct question is waiting; run inbox',flush=True); time.sleep(30)")
        return actual([sys.executable, '-c', code], **kwargs)
    monkeypatch.setattr(harnesshook.platform, 'start_process', descendants)
    monkeypatch.setattr(harnesshook, 'NUDGE_S', 0.3)
    began = time.monotonic()
    context = harnesshook.post('worker')['hookSpecificOutput']['additionalContext']
    assert time.monotonic() - began < 2
    assert 'direct question' in context and 'notification unavailable' in context
    pid = int(pid_file.read_text())
    deadline = time.monotonic() + 1
    while psutil.pid_exists(pid) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not psutil.pid_exists(pid)
