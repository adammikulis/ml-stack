"""A server stops when the process that started it exits, is signalled, or is killed outright."""

import contextlib
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import psutil
import pytest

REPO = Path(__file__).resolve().parent.parent

HOST = """
import os, signal, subprocess, sys, time
sys.path.insert(0, {src!r})
{before}
from ml_stack.serve import exit_guard
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"],
                         start_new_session=True)
exit_guard.protect(child.pid)
{after}
print(child.pid, flush=True)
{finish}
"""


def _host(tmp_path, *, before="", after="", finish="time.sleep(120)"):
    script = tmp_path / "host.py"
    script.write_text(HOST.format(src=str(REPO / "src"), before=textwrap.dedent(before),
                                  after=textwrap.dedent(after), finish=finish))
    host = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    return host, int(host.stdout.readline())


def _gone(pid, *, within=15.0):
    deadline = time.monotonic() + within
    while True:
        try:
            if psutil.Process(pid).status() == psutil.STATUS_ZOMBIE:
                return True
        except psutil.NoSuchProcess:
            return True
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


@pytest.fixture
def hosts():
    started = []
    yield started
    for host, child in started:
        for pid in (host.pid, child):
            with contextlib.suppress(psutil.NoSuchProcess):
                psutil.Process(pid).kill()
        host.wait(timeout=10)


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_server_stops_when_the_host_exits(tmp_path, hosts):
    host, child = _host(tmp_path, finish="pass")
    hosts.append((host, child))
    host.wait(timeout=20)
    assert _gone(child, within=0.2), "stopped by the host itself, not by the watchdog"


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_server_stops_when_the_host_is_terminated(tmp_path, hosts):
    host, child = _host(tmp_path)
    hosts.append((host, child))
    host.send_signal(signal.SIGTERM)
    host.wait(timeout=20)
    assert host.returncode == -signal.SIGTERM, "the host still dies of the signal"
    assert _gone(child, within=0.2), "stopped by the handler, not by the watchdog"


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_server_stops_when_the_host_is_killed_outright(tmp_path, hosts):
    host, child = _host(tmp_path)
    hosts.append((host, child))
    host.kill()
    host.wait(timeout=20)
    assert _gone(child, within=20.0), "nothing but the watchdog can see a SIGKILL"


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_the_handler_the_host_already_had_still_runs(tmp_path, hosts):
    mark = tmp_path / "mark"
    host, child = _host(tmp_path, before=f"""
        def mine(number, frame):
            open({str(mark)!r}, "w").write("ran")
            raise SystemExit(7)
        signal.signal(signal.SIGTERM, mine)
        """)
    hosts.append((host, child))
    host.send_signal(signal.SIGTERM)
    host.wait(timeout=20)
    assert mark.read_text() == "ran" and host.returncode == 7
    assert _gone(child)


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_released_server_outlives_the_host(tmp_path, hosts):
    host, child = _host(tmp_path, after="exit_guard.release(child.pid)", finish="pass")
    hosts.append((host, child))
    host.wait(timeout=20)
    time.sleep(3.0)
    assert psutil.Process(child).is_running(), "released, so it was never the host's to stop"


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_pid_that_is_no_longer_the_server_is_left_alone(tmp_path):
    victim = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        from ml_stack.serve import exit_guard

        exit_guard._guarded[victim.pid] = time.time() - 3600
        assert exit_guard.stop_guarded() == []
        assert victim.poll() is None
    finally:
        exit_guard._guarded.pop(victim.pid, None)
        victim.kill()


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_importing_serve_installs_no_handler_and_no_exit_hook():
    program = ("import signal, atexit, sys, logging, concurrent.futures, multiprocessing\n"
               f"sys.path.insert(0, {str(REPO / 'src')!r})\n"
               "before = [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)]\n"
               "hooks = atexit._ncallbacks()\n"
               "import ml_stack.serve\n"
               "from ml_stack.serve import ServerManager\n"
               "ServerManager()\n"
               "assert [signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)] == before\n"
               "assert atexit._ncallbacks() == hooks, 'registered at import'\n")
    done = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
def test_a_forked_child_does_not_stop_the_parents_servers(tmp_path, hosts):
    host, child = _host(tmp_path, finish=(
        "pid = os.fork()\nif pid == 0:\n    sys.exit(0)\nos.waitpid(pid, 0)\ntime.sleep(120)"))
    hosts.append((host, child))
    time.sleep(2.0)
    assert psutil.Process(child).is_running()


MANAGER_HOST = """
import sys, time
sys.path.insert(0, {src!r})
from pathlib import Path
from ml_stack.serve import LlamaServerBackend, ServerManager, ServerSpec, free_port
from ml_stack.testing import fake_llama_binary
root = Path({root!r})
model = root / "model.gguf"
model.write_bytes(b"GGUF" + b"\\x00" * 64)
manager = ServerManager(backend=LlamaServerBackend(binary=fake_llama_binary(root)),
                        state_file=root / "servers.json", stop_on_exit={stop})
info = manager.lease(ServerSpec(model=model, port=free_port()), roam=False, timeout=30.0,
                     check_flags=False, preflight=False, warmup_request=False)
print(info.pid, flush=True)
time.sleep(120)
"""


@pytest.mark.skipif(sys.platform == "win32", reason="signals are POSIX")
@pytest.mark.slow
@pytest.mark.parametrize("stop, outcome", [(True, "stopped"), (False, "left running")])
def test_a_managers_server_follows_the_host_unless_told_not_to(tmp_path, hosts, stop, outcome):
    script = tmp_path / "manager_host.py"
    script.write_text(MANAGER_HOST.format(src=str(REPO / "src"), root=str(tmp_path), stop=stop))
    host = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    server = int(host.stdout.readline())
    hosts.append((host, server))
    host.kill()
    host.wait(timeout=20)
    if stop:
        assert _gone(server, within=20.0)
    else:
        time.sleep(3.0)
        assert psutil.Process(server).is_running(), outcome
