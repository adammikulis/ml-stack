"""Native bubblewrap probe inputs, caching and process cleanup."""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import pairwise

import pytest

from ml_stack.sandbox import bubblewrap

pytestmark = pytest.mark.redteam


@pytest.fixture
def executable(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "bwrap; touch escaped"
    monkeypatch.setattr(bubblewrap.sys, "platform", "linux")
    monkeypatch.setattr(bubblewrap.shutil, "which", lambda name: str(path))
    bubblewrap._probe.cache_clear()
    yield path
    bubblewrap._probe.cache_clear()


def install(path, source):
    path.write_text(f"#!{sys.executable}\n" + source)
    path.chmod(0o700)


def test_hostile_binary_name_is_literal_and_probe_is_single_flight(executable, tmp_path):
    record = tmp_path / "arguments.jsonl"
    install(executable, "import json,sys,time\ntime.sleep(.2)\n" +
        f"with open({str(record)!r}, 'a') as out: out.write(json.dumps(sys.argv[1:]) + '\\n')\n" +
        "sys.stderr.write('bwrap: No permissions to create a new namespace')\nsys.exit(1)\n")
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: bubblewrap.Bubblewrap().available(), range(16)))
    assert all(not result.ok and 'namespace' in result.reason for result in results)
    rows = record.read_text().splitlines()
    assert len(rows) == 1
    argv = json.loads(rows[0])
    assert argv[:4] == ['--unshare-all', '--die-with-parent', '--new-session', '--clearenv']
    assert '--share-net' not in argv and '--bind' not in argv
    assert not any(left == '--ro-bind' and right == '/' for left, right in pairwise(argv))
    assert argv[-2:] == ['--', os.path.realpath('/usr/bin/true')]
    assert not (tmp_path / 'escaped').exists()
    install(executable, 'import sys\nsys.exit(0)\n')
    assert bubblewrap.Bubblewrap().available().ok


def test_control_characters_in_probe_binary_path_do_not_execute(executable, tmp_path, monkeypatch):
    bad = tmp_path / 'bwrap\nattack'
    marker = tmp_path / 'ran'
    install(bad, f"open({str(marker)!r}, 'w').write('executed')\n")
    monkeypatch.setattr(bubblewrap.shutil, 'which', lambda name: str(bad))
    assert not bubblewrap.Bubblewrap().available().ok
    assert not marker.exists()


def test_stalled_probe_is_killed_and_cached(executable):
    install(executable, 'import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n')
    started = time.monotonic()
    status = bubblewrap.Bubblewrap().available()
    assert not status.ok and 'timed out' in status.reason
    assert time.monotonic() - started < 6
    started = time.monotonic()
    assert bubblewrap.Bubblewrap().available() == status
    assert time.monotonic() - started < .5
