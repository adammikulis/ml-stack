"""scripts/testslots.py: a shared worker budget that makes test runs queue instead of piling up."""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "testslots.py"

CHILD = r'''
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("testslots", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["testslots"] = m; spec.loader.exec_module(m)
log, want, hold, label = sys.argv[2], int(sys.argv[3]), float(sys.argv[4]), sys.argv[5]
with m.lease(want, min(2, want), label=label, say=lambda s: None) as g:
    t0 = time.time()
    time.sleep(hold)
    with open(log, "a") as f:
        f.write(json.dumps({"label": label, "workers": g.workers, "start": t0, "end": time.time()}) + "\n")
'''


def _env(tmp_path, budget):
    inherited = {name: value for name, value in os.environ.items() if name not in {"DEV_TEST_REMOTE_LEASE", "DEV_TEST_LEASE"}}
    return {**inherited, "DEV_TEST_SLOTS_DIR": str(tmp_path / "slots"), "DEV_TEST_BUDGET": str(budget), "DEV_TEST_WAIT_S": "60"}


def _spawn(tmp_path, budget, want, hold, label):
    return subprocess.Popen([sys.executable, "-c", CHILD, str(SCRIPT), str(tmp_path / "log.jsonl"), str(want), str(hold), label],
                            env=_env(tmp_path, budget))


def _records(tmp_path):
    p = tmp_path / "log.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def test_concurrent_grants_never_exceed_the_budget_and_all_runs_finish(tmp_path):
    budget = 8
    procs = []
    for i in range(6):
        procs.append(_spawn(tmp_path, budget, 4, 0.8, f"run{i}"))
        time.sleep(0.05)
    assert all(p.wait(timeout=60) == 0 for p in procs)
    recs = _records(tmp_path)
    assert len(recs) == 6
    for r in recs:                                           # at every run's start, the workers of all live runs fit
        live = sum(o["workers"] for o in recs if o["start"] <= r["start"] < o["end"])
        assert live <= budget, (live, recs)
    assert max(r["end"] for r in recs) - min(r["start"] for r in recs) > 1.4     # they really queued (6x4 workers > 8)


def test_runs_are_granted_in_arrival_order(tmp_path):
    first = _spawn(tmp_path, 4, 4, 0.8, "first")
    time.sleep(0.3)
    second = _spawn(tmp_path, 4, 4, 0.3, "second")
    time.sleep(0.2)
    third = _spawn(tmp_path, 4, 4, 0.3, "third")
    for p in (first, second, third):
        assert p.wait(timeout=60) == 0
    order = [r["label"] for r in sorted(_records(tmp_path), key=lambda r: r["start"])]
    assert order == ["first", "second", "third"]


def test_a_killed_holder_frees_its_workers(tmp_path):
    holder = _spawn(tmp_path, 4, 4, 60, "doomed")
    deadline = time.time() + 20
    while time.time() < deadline and not list((tmp_path / "slots").glob("*.slot")):
        time.sleep(0.1)
    time.sleep(0.5)
    waiter = _spawn(tmp_path, 4, 4, 0.1, "waiter")
    time.sleep(1.0)
    assert waiter.poll() is None                              # still waiting behind the live holder
    holder.send_signal(signal.SIGKILL)
    assert waiter.wait(timeout=30) == 0
    assert [r["label"] for r in _records(tmp_path)] == ["waiter"]


def test_status_lists_running_and_waiting(tmp_path):
    holder = _spawn(tmp_path, 4, 4, 3, "holder")
    time.sleep(1.0)
    waiter = _spawn(tmp_path, 4, 4, 0.1, "waiter")
    time.sleep(1.0)
    out = subprocess.run([sys.executable, str(SCRIPT), "status"], env=_env(tmp_path, 4), capture_output=True, text=True, timeout=30).stdout
    assert "running" in out and "holder" in out and "waiting" in out and "waiter" in out
    for p in (holder, waiter):
        assert p.wait(timeout=60) == 0


def test_a_run_gets_fewer_workers_when_the_budget_is_partly_used_but_never_below_its_minimum(tmp_path):
    a = _spawn(tmp_path, 6, 4, 1.5, "a")
    time.sleep(0.6)
    b = _spawn(tmp_path, 6, 4, 0.2, "b")                       # 2 free: runs now with 2 (its minimum is min(2, 4))
    assert b.wait(timeout=30) == 0 and a.wait(timeout=30) == 0
    got = {r["label"]: r["workers"] for r in _records(tmp_path)}
    assert got == {"a": 4, "b": 2}


def test_disabled_environment_cannot_bypass_the_budget(tmp_path):
    environment = {**_env(tmp_path, 1), "DEV_TEST_SLOTS": "off"}
    result = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "7", "--min", "1", "--",
                             sys.executable, "-c", "import os; print(os.environ['DEV_TEST_WORKERS'])"],
                            env=environment, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert result.stdout.strip() == "1"


def test_the_run_command_queues_and_hands_the_granted_workers_to_the_command(tmp_path):
    out = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "3", "--min", "2", "--label", "t", "--",
                          sys.executable, "-c", "import os; print(os.environ['DEV_TEST_WORKERS'])"],
                         env=_env(tmp_path, 8), capture_output=True, text=True, timeout=60)
    assert out.returncode == 0 and out.stdout.strip() == "3"
    bad = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "1", "--", sys.executable, "-c", "raise SystemExit(7)"],
                         env=_env(tmp_path, 8), capture_output=True, text=True, timeout=60)
    assert bad.returncode == 7                                   # the command's exit status is passed through


def _load():
    spec = importlib.util.spec_from_file_location("testslots", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["testslots"] = module
    spec.loader.exec_module(module)
    return module


def test_the_cap_follows_the_load_average(monkeypatch):
    monkeypatch.setenv("DEV_TEST_RESERVED_CORES", "1")
    m = _load()
    assert m.cap_for(None, cores=16) == 15
    assert m.cap_for(2.0, cores=16) == 15
    assert m.cap_for(12.0, cores=16) == 15
    assert m.cap_for(31.9, cores=16) == 15
    assert m.cap_for(32.1, cores=16) == 7
    assert m.cap_for(500.0, cores=4) == 1
    assert m.cap_for(0.1, cores=4) == 3
    assert m.base_budget(1) == 1


def test_a_pinned_budget_ignores_the_load(monkeypatch):
    m = _load()
    monkeypatch.setenv("DEV_TEST_BUDGET", "5")
    assert m.budget() == 5


HEAVY_CHILD = r'''
import importlib.util, json, sys, time
spec = importlib.util.spec_from_file_location("testslots", sys.argv[1]); m = importlib.util.module_from_spec(spec); sys.modules["testslots"] = m; spec.loader.exec_module(m)
log = sys.argv[2]
with m.heavy_lane("t", say=lambda s: None):
    t0 = time.time(); time.sleep(float(sys.argv[3]))
    with open(log, "a") as f:
        f.write(json.dumps({"start": t0, "end": time.time()}) + "\n")
'''


def test_heavy_lanes_limit_concurrent_heavy_work_across_processes(tmp_path):
    env = {**_env(tmp_path, 8), "DEV_TEST_HEAVY_LANES": "2"}
    procs = [subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "0.6"],
                              env=env) for _ in range(6)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    recs = [json.loads(x) for x in (tmp_path / "lanes.jsonl").read_text().splitlines()]
    assert len(recs) == 6
    for r in recs:
        assert sum(1 for o in recs if o["start"] <= r["start"] + 1e-3 < o["end"]) <= 2, recs


def test_a_killed_heavy_test_frees_its_lane(tmp_path):
    env = {**_env(tmp_path, 8), "DEV_TEST_HEAVY_LANES": "1"}
    holder = subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "60"],
                              env=env)
    time.sleep(1.0)
    waiter = subprocess.Popen([sys.executable, "-c", HEAVY_CHILD, str(SCRIPT), str(tmp_path / "lanes.jsonl"), "0.1"],
                              env=env)
    time.sleep(1.0)
    assert waiter.poll() is None
    holder.send_signal(signal.SIGKILL)
    assert waiter.wait(timeout=30) == 0
    holder.wait(timeout=10)


def _grants(tmp_path):
    return {r["label"]: r["workers"] for r in _records(tmp_path)}


def test_an_auto_run_on_an_idle_machine_gets_the_whole_budget(tmp_path):
    """`want=0` is auto: a lone run is not capped by a default; the load-aware budget is the only limit."""
    assert _spawn(tmp_path, 12, 0, 0.1, "alone").wait(timeout=60) == 0
    assert _grants(tmp_path) == {"alone": 12}


def test_an_auto_run_takes_only_what_a_busy_machine_can_spare(tmp_path):
    holder = _spawn(tmp_path, 14, 8, 1.5, "holder")
    time.sleep(0.5)
    auto = _spawn(tmp_path, 14, 0, 0.1, "auto")
    assert holder.wait(timeout=60) == 0 and auto.wait(timeout=60) == 0
    got = _grants(tmp_path)
    assert got["holder"] == 8 and got["auto"] == 6, got      # 14 - 8 in use: neither the whole budget nor a fixed default


def test_a_fixed_want_is_still_a_ceiling_not_a_target(tmp_path):
    assert _spawn(tmp_path, 12, 4, 0.1, "fixed").wait(timeout=60) == 0
    assert _grants(tmp_path) == {"fixed": 4}


def test_queued_auto_runs_share_instead_of_the_first_taking_everything(tmp_path):
    blocker = _spawn(tmp_path, 10, 10, 1.2, "blocker")
    time.sleep(0.4)
    a = _spawn(tmp_path, 10, 0, 0.4, "a")
    time.sleep(0.1)
    b = _spawn(tmp_path, 10, 0, 0.4, "b")
    assert all(p.wait(timeout=60) == 0 for p in (blocker, a, b))
    got = _grants(tmp_path)
    assert got["a"] == 5 and got["b"] == 5, got               # both waiting when the blocker ended: half each


def test_nested_live_lease_is_refused_instead_of_queued(tmp_path, monkeypatch):
    module = _load()
    monkeypatch.setenv("DEV_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("DEV_TEST_BUDGET", "1")
    with module.lease(1, 1, say=lambda message: None):
        result = subprocess.run([sys.executable, str(SCRIPT), "run", "--want", "1", "--", sys.executable, "-c", "pass"],
                                capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "nested CPU run" in result.stderr


def test_a_minimum_cannot_override_the_budget(tmp_path, monkeypatch):
    import pytest
    module = _load()
    monkeypatch.setenv("DEV_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("DEV_TEST_BUDGET", "1")
    with pytest.raises(ValueError, match="minimum exceeds"), module.lease(2, 2):
        pass


ELASTIC_TEST = '''
import json, os, time
from pathlib import Path
import pytest

@pytest.fixture(autouse=True)
def record(request):
    start = time.time()
    time.sleep(.05)
    yield
    time.sleep(.05)
    with Path(os.environ["TEST_LOG"]).open("a") as stream:
        stream.write(json.dumps({"name": request.node.name, "start": start, "end": time.time()}) + "\\n")

def test_tail():
    Path(os.environ["TAIL_READY"]).touch()
    time.sleep(6)

def test_fast_a():
    time.sleep(.1)

def test_fast_b():
    time.sleep(.1)
'''


def _elastic(tmp_path, test_file, workers):
    environment = {**_env(tmp_path, 2), "PYTHONPATH": str(ROOT / "scripts"),
                   "TEST_LOG": str(tmp_path / "elastic.jsonl"), "TAIL_READY": str(tmp_path / "tail-ready")}
    command = [sys.executable, str(SCRIPT), "pytest", "--want", str(workers), "--", sys.executable,
               "-m", "pytest", "-q", "-n", "{workers}", "-p", "testslots_pytest", str(test_file)]
    return subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_idle_tail_workers_release_capacity_for_another_suite(tmp_path):
    first_file = tmp_path / "test_first.py"
    first_file.write_text(ELASTIC_TEST)
    second_file = tmp_path / "test_second.py"
    second_file.write_text(ELASTIC_TEST.replace("def test_tail():", "def unused_tail():"))
    first = _elastic(tmp_path, first_file, 2)
    deadline = time.monotonic() + 20
    while not (tmp_path / "tail-ready").exists():
        assert first.poll() is None, first.communicate()
        assert time.monotonic() < deadline
        time.sleep(.05)
    second = _elastic(tmp_path, second_file, 1)
    second_output = second.communicate(timeout=20)
    assert second.returncode == 0, second_output
    assert first.poll() is None, "The second suite must finish while the long tail remains active"
    first_output = first.communicate(timeout=20)
    assert first.returncode == 0, first_output
    records = [json.loads(line) for line in (tmp_path / "elastic.jsonl").read_text().splitlines()]
    assert len(records) == 5
    for record in records:
        assert sum(other["start"] <= record["start"] < other["end"] for other in records) <= 2, records


def test_remote_nested_command_is_refused_without_deadlocking(tmp_path):
    test_file = tmp_path / "test_nested.py"
    test_file.write_text(f'''import subprocess, sys

def test_nested():
    result = subprocess.run([sys.executable, {str(SCRIPT)!r}, "run", "--want", "1", "--", sys.executable, "-c", "pass"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode != 0
    assert "nested CPU run" in result.stderr
''')
    process = _elastic(tmp_path, test_file, 1)
    output = process.communicate(timeout=20)
    assert process.returncode == 0, output


def _rpc_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    import testslots_rpc
    return testslots_rpc


def _rpc_socket(server, token):
    import socket
    connection = socket.create_connection(server.server_address, timeout=5)
    stream = connection.makefile("rwb")
    stream.write(json.dumps({"token": token, "operation": "acquire", "label": "rpc proof"}).encode() + b"\n")
    stream.flush()
    return connection, stream


def test_invalid_rpc_token_cannot_acquire_capacity(tmp_path, monkeypatch):
    rpc = _rpc_module()
    monkeypatch.setenv("DEV_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("DEV_TEST_BUDGET", "1")
    server = rpc.Admission(False)
    try:
        server.admitted.set()
        server.collected.set()
        connection, stream = _rpc_socket(server, "invalid")
        with connection, stream:
            assert "invalid admission token" in json.loads(stream.readline())["error"]
        assert rpc.testslots.status()["in_use"] == 0
    finally:
        server.finish()


def test_disconnected_rpc_waiter_releases_its_queue_entry(tmp_path, monkeypatch):
    rpc = _rpc_module()
    monkeypatch.setenv("DEV_TEST_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("DEV_TEST_BUDGET", "1")
    server = rpc.Admission(False)
    holder = waiter = None
    holder_stream = waiter_stream = None
    try:
        server.admitted.set()
        server.collected.set()
        holder, holder_stream = _rpc_socket(server, server.token)
        assert "lease" in json.loads(holder_stream.readline())
        waiter, waiter_stream = _rpc_socket(server, server.token)
        deadline = time.monotonic() + 5
        while not rpc.testslots.status()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(.01)
        waiter_stream.close()
        waiter.close()
        while rpc.testslots.status()["waiting"]:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert rpc.testslots.status()["in_use"] == 1
    finally:
        for resource in (waiter_stream, waiter, holder_stream, holder):
            if resource is not None:
                resource.close()
        server.finish()


def test_short_tests_do_not_wait_for_a_polling_tail_between_batches(tmp_path):
    test_file = tmp_path / "test_short.py"
    test_file.write_text('''import json, os, time
from pathlib import Path
import pytest

@pytest.mark.parametrize("case", range(80))
def test_short(case):
    start = time.time()
    time.sleep(.005)
    with Path(os.environ["TEST_LOG"]).open("a") as stream:
        stream.write(json.dumps({"start": start, "end": time.time()}) + "\\n")
''')
    process = _elastic(tmp_path, test_file, 2)
    output = process.communicate(timeout=30)
    assert process.returncode == 0, output
    assert "testslots: waiting" not in output[1], output[1]
    records = [json.loads(line) for line in (tmp_path / "elastic.jsonl").read_text().splitlines()]
    assert len(records) == 80
    duration = max(record["end"] for record in records) - min(record["start"] for record in records)
    measurement = {"tests": 80, "workers": 2, "duration_seconds": duration}
    print(json.dumps(measurement))
    assert duration < 3, duration
    for record in records:
        assert sum(other["start"] <= record["start"] < other["end"] for other in records) <= 2


def test_backfill_uses_idle_capacity_and_is_bounded(tmp_path):
    holder = _spawn(tmp_path, 3, 2, 2, "holder")
    time.sleep(.2)
    older = _spawn(tmp_path, 3, 2, .1, "older")
    time.sleep(.1)
    first = _spawn(tmp_path, 3, 1, .1, "first")
    assert first.wait(timeout=10) == 0
    second = _spawn(tmp_path, 3, 1, .1, "second")
    assert second.wait(timeout=10) == 0
    third = _spawn(tmp_path, 3, 1, .1, "third")
    time.sleep(.2)
    assert third.poll() is None
    assert all(process.wait(timeout=10) == 0 for process in (holder, older, third))
    records = {record["label"]: record for record in _records(tmp_path)}
    assert records["first"]["start"] < records["holder"]["end"]
    assert records["second"]["start"] < records["holder"]["end"]
    assert records["third"]["start"] >= records["older"]["start"]


def test_reserved_capacity_is_configurable_and_owner_budget_overrides_it(monkeypatch):
    module = _load()
    monkeypatch.setenv("DEV_TEST_RESERVED_CORES", "4")
    assert module.base_budget(16) == 12
    monkeypatch.setenv("DEV_TEST_BUDGET", "22")
    assert module.budget() == 22


def test_worker_configuration_is_admitted_across_concurrent_suites(tmp_path):
    configuration = '''import json, os, time
from pathlib import Path
start = time.time()
time.sleep(.15)
with Path(os.environ["TEST_LOG"]).open("a") as stream:
    stream.write(json.dumps({"start": start, "end": time.time()}) + "\\n")
'''
    files = []
    for name in ("first", "second"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "conftest.py").write_text(configuration)
        test_file = directory / "test_work.py"
        test_file.write_text("def test_work():\n    pass\n")
        files.append(test_file)
    processes = [_elastic(tmp_path, test_file, 2) for test_file in files]
    for process in processes:
        output = process.communicate(timeout=30)
        assert process.returncode == 0, output
    records = [json.loads(line) for line in (tmp_path / "elastic.jsonl").read_text().splitlines()]
    assert len(records) == 6
    for record in records:
        assert sum(other["start"] <= record["start"] < other["end"] for other in records) <= 2, records


def test_live_incomplete_records_do_not_crash_admission(tmp_path):
    import fcntl

    directory = tmp_path / 'slots'
    directory.mkdir()
    descriptors = []
    try:
        for index, document in enumerate(('{}', '[]', '{"minimum":', '{"pid":true,"minimum":1}')):
            path = directory / f'0-{index}.slot'
            descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
            descriptors.append(descriptor)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            path.write_text(document)
        processes = [_spawn(tmp_path, 2, 1, .02, f'valid-{index}') for index in range(12)]
        assert all(process.wait(timeout=10) == 0 for process in processes)
        records = _records(tmp_path)
        assert len(records) == 12
        for record in records:
            assert sum(other['workers'] for other in records
                       if other['start'] <= record['start'] < other['end']) <= 2
    finally:
        for descriptor in descriptors:
            os.close(descriptor)
        for process in locals().get('processes', []):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)


def test_waiter_republishes_its_unlinked_record_and_keeps_fifo_position(tmp_path):
    module = _load()
    holder = _spawn(tmp_path, 1, 1, 60, 'holder')
    waiter = third = None
    try:
        deadline = time.monotonic() + 5
        path = None
        while path is None:
            for candidate in (tmp_path / 'slots').glob('*.slot'):
                if json.loads(candidate.read_text()).get('granted') == 1:
                    path = candidate
            assert time.monotonic() < deadline
            time.sleep(.01)
        waiter = _spawn(tmp_path, 1, 1, .02, 'waiter')
        path = None
        while path is None:
            for candidate in (tmp_path / 'slots').glob('*.slot'):
                if json.loads(candidate.read_text()).get('label') == 'waiter':
                    path = candidate
            assert time.monotonic() < deadline
            time.sleep(.01)
        original = path.stat().st_ino
        path.unlink()
        while not path.exists():
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert path.stat().st_ino != original and module._alive(path)
        third = _spawn(tmp_path, 1, 1, .02, 'third')
        holder.kill()
        assert waiter.wait(timeout=5) == 0 and third.wait(timeout=5) == 0
        assert [record['label'] for record in _records(tmp_path)] == ['waiter', 'third']
    finally:
        for process in (holder, waiter, third):
            if process is not None:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=5)


def test_concurrent_record_publication_is_complete_and_already_locked(tmp_path, monkeypatch):
    import concurrent.futures
    import threading

    module = _load()
    monkeypatch.setenv('DEV_TEST_SLOTS_DIR', str(tmp_path / 'slots'))
    monkeypatch.setenv('DEV_TEST_BUDGET', '3')
    monkeypatch.delenv('DEV_TEST_REMOTE_LEASE', raising=False)
    monkeypatch.delenv('DEV_TEST_LEASE', raising=False)
    (tmp_path / 'slots').mkdir()
    original = module.json.dump
    finished = threading.Event()
    samples = []

    def slow_dump(document, stream):
        payload = json.dumps(document)
        middle = len(payload) // 2
        stream.write(payload[:middle])
        stream.flush()
        time.sleep(.005)
        stream.write(payload[middle:])

    monkeypatch.setattr(module.json, 'dump', slow_dump)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=13) as executor:
            observer = executor.submit(_observe_publication, module, tmp_path, finished, samples)
            futures = [executor.submit(_concurrent_lease, module, index) for index in range(12)]
            try:
                for future in futures:
                    future.result(timeout=10)
            finally:
                finished.set()
            observer.result(timeout=5)
        assert samples and all(module._valid_record(sample) for sample in samples)
        assert module.status()['in_use'] == 0
    finally:
        finished.set()
        monkeypatch.setattr(module.json, 'dump', original)


def _observe_publication(module, tmp_path, finished, samples):
    while not finished.is_set():
        with module._mutex(tmp_path / 'slots'):
            for path in (tmp_path / 'slots').glob('*.slot'):
                document = json.loads(path.read_text())
                samples.append(document)
                assert module._valid_record(document) and module._alive(path)
        time.sleep(.001)


def _concurrent_lease(module, index):
    token = module.EXPORT_LEASE.set(False)
    try:
        with module.lease(1, label=f'concurrent-{index}', say=lambda message: None):
            time.sleep(.01)
    finally:
        module.EXPORT_LEASE.reset(token)


def test_rpc_admission_survives_a_live_incomplete_record(tmp_path, monkeypatch):
    import concurrent.futures
    import fcntl

    rpc = _rpc_module()
    monkeypatch.setenv('DEV_TEST_SLOTS_DIR', str(tmp_path / 'slots'))
    monkeypatch.setenv('DEV_TEST_BUDGET', '2')
    directory = rpc.testslots.slots_dir()
    damaged = directory / '0-incomplete.slot'
    descriptor = os.open(damaged, os.O_RDWR | os.O_CREAT, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    damaged.write_text('{}')
    server = rpc.Admission(False)
    try:
        server.admitted.set()
        server.collected.set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_consume_rpc_permit, server) for _ in range(24)]
            replies = [future.result(timeout=10) for future in futures]
        assert len(set(replies)) == 24
        deadline = time.monotonic() + 5
        while rpc.testslots.status()['in_use']:
            assert time.monotonic() < deadline
            time.sleep(.01)
        assert not rpc.testslots.status()['waiting']
    finally:
        server.finish()
        os.close(descriptor)


def _consume_rpc_permit(server):
    connection, stream = _rpc_socket(server, server.token)
    with connection, stream:
        reply = json.loads(stream.readline())
        assert 'lease' in reply, reply
        stream.write(b'{}\n')
        stream.flush()
        return reply['lease']


def test_wait_notices_are_delayed_and_time_gated(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import pytest

    module = _load()
    monkeypatch.setenv('DEV_TEST_SLOTS_DIR', str(tmp_path / 'slots'))
    monkeypatch.setenv('DEV_TEST_BUDGET', '1')
    monkeypatch.delenv('DEV_TEST_REMOTE_LEASE', raising=False)
    monkeypatch.delenv('DEV_TEST_LEASE', raising=False)
    clock = [0.0]
    notices = []

    def sleep(seconds):
        clock[0] += 1
        if clock[0] == 26:
            raise RuntimeError('Finished simulated wait')

    monkeypatch.setattr(module, 'time', SimpleNamespace(
        time=lambda: 100, time_ns=lambda: 100, monotonic=lambda: clock[0], sleep=sleep))
    monkeypatch.setattr(module, 'budget', lambda: 0)
    with pytest.raises(RuntimeError, match='Finished simulated wait'), module.lease(
            1, say=lambda message: notices.append((clock[0], message))):
        pass
    assert [stamp for stamp, _ in notices] == [1, 21]
    assert not list((tmp_path / 'slots').glob('*.slot'))
