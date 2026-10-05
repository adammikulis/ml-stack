"""Independent scanner snapshots retain strict empty and unknown lock ownership checks."""

import json
import os
import time
from types import SimpleNamespace

import conftest
import pytest


@pytest.mark.parametrize('kind', ['empty','unknown','own'])
def test_multiple_external_snapshots_do_not_exempt_unattributed_lock(tmp_path, monkeypatch, kind):
    sentinel = tmp_path / 'sentinel'
    sentinel.mkdir()
    now = time.time()
    monkeypatch.setattr(conftest, 'ours', lambda record: record['owner_pid'] == os.getpid())
    monkeypatch.setattr(conftest.psutil, 'Process', lambda pid: SimpleNamespace(
        create_time=lambda:now-pid,is_running=lambda:True))
    for name,pid in [('scanner.json',12345),('scanner.json.prev',23456)]:
        (sentinel/name).write_text(json.dumps({'payload':{'pid':pid,'started':now-10,
            'process_started':now-pid,'beat':now,'interval_s':300,'running':True}}))
    text={'empty':'','unknown':'pid 99999 foreign','own':f'pid {os.getpid()} own'}[kind]
    (sentinel/'state.lock').write_text(text)
    assert conftest.file_mtimes(tmp_path).keys() == {'sentinel/state.lock'}
    (sentinel/'state.lock').write_text('pid 23456 scanner scan')
    assert conftest.file_mtimes(tmp_path) == {}
    value = json.loads((sentinel/'scanner.json.prev').read_text())
    value['payload']['process_started'] -= 1
    (sentinel/'scanner.json.prev').write_text(json.dumps(value))
    assert conftest.file_mtimes(tmp_path).keys() == {'sentinel/scanner.json.prev','sentinel/state.lock'}
