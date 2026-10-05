"""Activity drops identify their actual writer without exempting test-owned or unknown state."""

import json
import os
import time
from types import SimpleNamespace

import conftest
import psutil
import pytest

from ml_stack.activity import writer


def test_activity_drop_records_actual_process_identity_and_write_time(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, 'directory', lambda: tmp_path)
    before = time.time()
    writer._dropped('isolated record failure', before)
    value = json.loads((tmp_path / 'drops.json').read_text())
    assert value['writer_pid'] == os.getpid()
    assert value['writer_started'] == psutil.Process().create_time()
    assert before <= value['writer_at'] <= time.time()
    assert value['dropped'] == 1


@pytest.mark.parametrize('change', ['ours','dead','reused','unknown','other-user','stale','future','missing'])
def test_activity_guard_excludes_only_live_external_matching_writer(tmp_path, monkeypatch, change):
    path = tmp_path / 'activity/u-501/drops.json'
    path.parent.mkdir(parents=True)
    now = time.time()
    record = {'writer_pid':12345,'writer_started':now-20,'writer_at':now}
    live = [True]
    monkeypatch.setattr(conftest, 'ours', lambda entry: entry['owner_pid'] == os.getpid())
    def process(pid):
        if pid == 99999:
            raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(is_running=lambda:live[0],create_time=lambda:now-20,
                               uids=lambda:SimpleNamespace(real=501))
    monkeypatch.setattr(conftest.psutil, 'Process', process)
    path.write_text(json.dumps(record))
    assert conftest.file_mtimes(tmp_path) == {}
    if change == 'ours':
        record['writer_pid'] = os.getpid()
    elif change == 'dead':
        live[0] = False
    elif change == 'reused':
        record['writer_started'] = now-30
    elif change == 'unknown':
        record['writer_pid'] = 99999
    elif change == 'other-user':
        path = tmp_path / 'activity/u-502/drops.json'
        path.parent.mkdir()
    elif change == 'stale':
        record['writer_at'] = now-61
    elif change == 'future':
        record['writer_at'] = now+60
    else:
        record.pop('writer_started')
    path.write_text(json.dumps(record))
    assert path.relative_to(tmp_path).as_posix() in conftest.file_mtimes(tmp_path)
    unrelated = path.with_name('private.json')
    unrelated.write_text(json.dumps(record))
    assert unrelated.relative_to(tmp_path).as_posix() in conftest.file_mtimes(tmp_path)


def test_unavailable_process_provenance_does_not_raise_into_activity_caller(tmp_path, monkeypatch):
    monkeypatch.setattr(writer, 'directory', lambda: tmp_path)
    def unavailable(pid):
        raise psutil.AccessDenied(pid)
    monkeypatch.setattr(writer.psutil, 'Process', unavailable)
    writer._dropped('isolated unavailable introspection', time.time())
    assert not (tmp_path / 'drops.json').exists()


@pytest.mark.parametrize('expiry', ['dead', 'aged'])
def test_external_writer_expiry_does_not_fabricate_a_state_write(tmp_path, monkeypatch, expiry):
    path = tmp_path / 'activity/u-501/drops.json'
    path.parent.mkdir(parents=True)
    now = time.time()
    live = [True]
    clock = [now]
    record = {'writer_pid':12345,'writer_started':now-20,'writer_at':now}
    monkeypatch.setattr(conftest,'ours',lambda entry:False)
    monkeypatch.setattr(conftest.time,'time',lambda:clock[0])
    monkeypatch.setattr(conftest.psutil,'Process',lambda pid:SimpleNamespace(
        is_running=lambda:live[0],create_time=lambda:now-20,uids=lambda:SimpleNamespace(real=501)))
    path.write_text(json.dumps(record))
    before=conftest.file_mtimes(tmp_path,attribute_external=False)
    assert conftest._external_activity_drop_write(tmp_path,path.relative_to(tmp_path))
    if expiry == 'dead':
        live[0]=False
    else:
        clock[0]+=61
    after=conftest.file_mtimes(tmp_path,attribute_external=False)
    assert conftest.real_state_changes(tmp_path,before,after) == []
    record['dropped']=2
    path.write_text(json.dumps(record))
    assert conftest.real_state_changes(tmp_path,before,conftest.file_mtimes(tmp_path,attribute_external=False)) == ['activity/u-501/drops.json']


def test_actual_test_writer_mutation_is_not_attributed_to_external_process(tmp_path, monkeypatch):
    path=tmp_path/'activity'/f'u-{os.getuid()}'/'drops.json'
    path.parent.mkdir(parents=True)
    before=conftest.file_mtimes(tmp_path,attribute_external=False)
    path.write_text(json.dumps({'writer_pid':os.getpid(),'writer_started':psutil.Process().create_time(),'writer_at':time.time()}))
    assert conftest.real_state_changes(tmp_path,before,conftest.file_mtimes(tmp_path,attribute_external=False)) == [path.relative_to(tmp_path).as_posix()]
