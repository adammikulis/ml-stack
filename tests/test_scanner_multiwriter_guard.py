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


@pytest.mark.parametrize('field,value', [('beat',float('nan')),('beat',float('inf')),
    ('beat',time.time()+10000),('started',float('nan')),('interval_s',0),
    ('interval_s',float('inf')),('interval_s',86401)],ids=['nan-beat','infinite-beat','future-beat','nan-start','zero-interval','infinite-interval','huge-interval'])
def test_invalid_scanner_clock_never_attributes_external_write(tmp_path, monkeypatch, field, value):
    now=time.time()
    monkeypatch.setattr(conftest,'ours',lambda record:False)
    monkeypatch.setattr(conftest.psutil,'Process',lambda pid:SimpleNamespace(
        create_time=lambda:now-100,is_running=lambda:True))
    root=tmp_path/'sentinel'
    root.mkdir()
    payload={'pid':12345,'started':now-10,'beat':now,'interval_s':300,'running':True}
    payload[field]=value
    (root/'scanner.json').write_text(json.dumps({'payload':payload}))
    assert not conftest._external_scanner_write(tmp_path,conftest.Path('sentinel/scanner.json'))
