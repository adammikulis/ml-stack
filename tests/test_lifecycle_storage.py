"""Interrupted lifecycle staging preserves committed ownership records."""

import os
import subprocess
import sys

import pytest

from ml_stack.workspace import worktree_lifecycle as lifecycle


def seed(base):
    with lifecycle._storage(base, write=True) as graph:
        graph.upsert_node({'id': 'scope', 'kind': 'worktree-lifecycle', 'label': 'worker',
                           'attrs': {'owner': 'worker', 'label': 'worker', 'commits': ['retained']}})


def test_killed_staged_writer_preserves_active_records(tmp_path):
    seed(tmp_path)
    before = (tmp_path / 'worktree-lifecycle.db').read_bytes()
    script = """from pathlib import Path
import os, sys
from ml_stack.workspace.worktree_lifecycle import _storage
with _storage(Path(sys.argv[1]), write=True) as graph:
 graph.upsert_node({'id':'unpublished','kind':'worktree-lifecycle','attrs':{'owner':'worker'}})
 os._exit(23)
"""
    done = subprocess.run([sys.executable, '-c', script, str(tmp_path)], env=os.environ.copy(), check=False)
    assert done.returncode == 23
    assert (tmp_path / 'worktree-lifecycle.db').read_bytes() == before
    assert lifecycle.scopes(tmp_path, 'worker')[0]['commits'] == ['retained']
    assert list(tmp_path.glob('worktree-lifecycle-stage-*'))
    with lifecycle._storage(tmp_path, write=True) as graph:
        graph.upsert_node({'id': 'next', 'kind': 'worktree-lifecycle', 'attrs': {'owner': 'worker', 'label': ''}})
    assert len(lifecycle.scopes(tmp_path, 'worker')) == 2


def test_failed_writer_preserves_evidence_and_committed_graph(tmp_path):
    seed(tmp_path)
    with pytest.raises(RuntimeError, match='interrupted'), lifecycle._storage(tmp_path, write=True) as graph:
        graph.upsert_node({'id': 'unpublished', 'kind': 'worktree-lifecycle', 'attrs': {'owner': 'worker'}})
        raise RuntimeError('interrupted')
    assert len(lifecycle.scopes(tmp_path, 'worker')) == 1
    assert list(tmp_path.glob('worktree-lifecycle-stage-*'))


def test_read_does_not_checkpoint_or_change_committed_database(tmp_path):
    seed(tmp_path)
    before = (tmp_path / 'worktree-lifecycle.db').stat().st_mtime_ns
    assert lifecycle.scopes(tmp_path, 'worker')
    assert (tmp_path / 'worktree-lifecycle.db').stat().st_mtime_ns == before
    assert not list(tmp_path.glob('worktree-lifecycle.db.*'))
