"""What Windows gets instead, proved against a faked platform on whatever runs the tests.

Nothing here runs a real Windows call: this machine is a Mac, and ``msvcrt``, ``schtasks``,
``netsh``, ``icacls`` and ``CTRL_BREAK_EVENT`` exist only there. What each test proves is
that the *branch* is taken -- the right module asked, the right argv built, the right
Popen keyword chosen -- against a fake that stands in for the Windows side. The one place a
fake is more than a recorder is the lock: the stand-in ``msvcrt.locking`` is built on
``flock``, so the Windows code path is exercised with real cross-process exclusion, and
a second process really is refused. Whether ``LockFile`` itself behaves the way the fake
does is the first thing a Windows machine will tell Adam (README, "On Windows").
"""

from __future__ import annotations

import errno
import os
import platform
import signal
import socket
import subprocess
import sys
import textwrap
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "src"


@pytest.fixture
def windows(monkeypatch):
    """``platform.system()`` says Windows; everything that reads it at call time follows."""
    monkeypatch.setattr(platform, "system", lambda: "Windows")


def _ok(returncode: int = 0, stdout: str = "", stderr: str = "") -> types.SimpleNamespace:
    return types.SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)


# -- the lock ------------------------------------------------------------------------
# A stand-in msvcrt whose `locking` is flock underneath: LK_NBLCK takes LOCK_EX|LOCK_NB and
# refuses with EACCES the way LockFile does; LK_UNLCK releases. Installed in this process
# by the fixture and in a child by the same source, so both sides run the Windows branch.
FAKE_MSVCRT = textwrap.dedent("""
    import errno, fcntl, sys, types
    _m = types.ModuleType("msvcrt")
    _m.LK_UNLCK, _m.LK_LOCK, _m.LK_NBLCK = 0, 1, 2
    _m.calls = []
    def _locking(fd, mode, nbytes):
        _m.calls.append((mode, nbytes))
        if mode == _m.LK_NBLCK:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise OSError(errno.EACCES, "Permission denied") from None
        elif mode == _m.LK_UNLCK:
            fcntl.flock(fd, fcntl.LOCK_UN)
        else:
            raise AssertionError(f"blocking mode {mode} must never be used")
    _m.locking = _locking
    sys.modules["msvcrt"] = _m
    sys.platform = "win32"
""")


@pytest.fixture
def win_lock(monkeypatch):
    """`ml_stack.lock` believing it is on Windows, with the flock-backed msvcrt."""
    # monkeypatch first, so the original platform is recorded before the fake source
    # assigns sys.platform itself -- recorded after, the "original" would be win32 and
    # every later test on this worker would inherit it (ssl went looking for the Windows
    # certificate store the first time this was got wrong).
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "msvcrt", None)
    scope: dict = {}
    exec(FAKE_MSVCRT, scope)            # noqa: S102 - our own source, above
    fake = scope["_m"]
    monkeypatch.setitem(sys.modules, "msvcrt", fake)
    from ml_stack.lock import Busy, only_one
    return only_one, Busy, fake


class TestTheLockOnWindows:
    def test_the_module_imports_where_fcntl_does_not_exist(self):
        """Windows has no fcntl. An import-time `import fcntl` would fail before anything
        else in the bench could run, which is exactly how it was written."""
        done = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.modules['fcntl'] = None; import ml_stack.lock; "
             "print('imported')"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)})
        assert done.returncode == 0, done.stderr
        assert "imported" in done.stdout

    def test_it_asks_msvcrt_not_fcntl(self, win_lock, tmp_path):
        only_one, _, fake = win_lock
        with only_one(tmp_path / "l"):
            assert (fake.LK_NBLCK, 1) in fake.calls, "took the lock through msvcrt.locking"
        assert fake.calls[-1] == (fake.LK_UNLCK, 1), "and released it the same way"

    def test_a_second_holder_is_refused_rather_than_allowed_to_overlap(
            self, win_lock, tmp_path):
        only_one, Busy, _ = win_lock
        with only_one(tmp_path / "l", wait=False):
            with pytest.raises(Busy) as why:
                with only_one(tmp_path / "l", wait=False):
                    raise AssertionError("two runs held the same lock at once")
        assert str(os.getpid()) in str(why.value), "says who has it, for a stalled machine"

    def test_the_lock_is_released_when_the_block_ends(self, win_lock, tmp_path):
        only_one, _, _ = win_lock
        with only_one(tmp_path / "l"):
            pass
        with only_one(tmp_path / "l", wait=False):
            pass

    def test_it_is_released_even_when_the_run_raises(self, win_lock, tmp_path):
        only_one, _, _ = win_lock
        with pytest.raises(ValueError):
            with only_one(tmp_path / "l"):
                raise ValueError("a run that failed still has to let the next one in")
        with only_one(tmp_path / "l", wait=False):
            pass

    def test_waiting_is_announced_rather_than_silent(self, win_lock, tmp_path):
        """A second *process*, also on the Windows branch, holds it; this one waits and
        says so. Real exclusion, because the fake msvcrt is flock underneath."""
        only_one, _, _ = win_lock
        said = []
        other = subprocess.Popen(
            [sys.executable, "-c", FAKE_MSVCRT + textwrap.dedent(f"""
                import time
                from ml_stack.lock import only_one
                with only_one({str(tmp_path / 'l')!r}):
                    print("held", flush=True)
                    time.sleep(1.5)
            """)], stdout=subprocess.PIPE, text=True,
            env={**os.environ, "PYTHONPATH": str(SRC)})
        try:
            assert other.stdout.readline().strip() == "held"
            with only_one(tmp_path / "l", timeout=10, announce=said.append):
                pass
        finally:
            other.wait(timeout=10)
        assert said and "waiting for" in said[0]
        assert str(other.pid) in said[0], "the holder's pid was read back across processes"

    def test_a_bounded_wait_gives_up_and_says_so(self, win_lock, tmp_path):
        only_one, Busy, _ = win_lock
        with only_one(tmp_path / "l"):
            with pytest.raises(Busy, match="still held"):
                with only_one(tmp_path / "l", timeout=0.2, announce=lambda _: None):
                    pass

    def test_the_pid_is_written_at_the_front_where_a_person_can_read_it(
            self, win_lock, tmp_path):
        """The locked byte is far past the pid text: a LockFile region is mandatory, so a
        pid that shared a byte with the lock could not be read by the process waiting."""
        from ml_stack import lock as lock_module

        only_one, _, _ = win_lock
        with only_one(tmp_path / "l") as held:
            assert held.read_text() == f"pid {os.getpid()}"
        assert held.read_text() == "", "cleared on release, so a stale pid never lingers"
        assert lock_module._LOCKED_BYTE > 64


# -- starting and stopping a job -------------------------------------------------------
class TestProcessGroups:
    def test_windows_gets_a_new_process_group_not_a_session(self, windows):
        from ml_stack.platform import CREATE_NEW_PROCESS_GROUP, process_group_kwargs

        kwargs = process_group_kwargs()
        assert kwargs == {"creationflags": CREATE_NEW_PROCESS_GROUP}
        assert CREATE_NEW_PROCESS_GROUP == 0x200, "Win32's own value, the same everywhere"

    def test_posix_keeps_its_session(self):
        from ml_stack.platform import process_group_kwargs

        assert process_group_kwargs() == {"start_new_session": True}

    def test_a_windows_job_is_asked_to_stop_with_ctrl_break(self, windows):
        from ml_stack.platform import CTRL_BREAK_EVENT, stop_gently

        sent: list[int] = []
        proc = types.SimpleNamespace(send_signal=sent.append,
                                     terminate=lambda: sent.append(-1))
        assert stop_gently(proc) == "CTRL_BREAK_EVENT"
        assert sent == [CTRL_BREAK_EVENT]
        assert CTRL_BREAK_EVENT == 1

    def test_a_job_with_no_console_is_terminated_and_that_is_said(self, windows):
        """GenerateConsoleCtrlEvent fails when the daemon has no console (a Scheduled Task
        with no window). The job still has to stop; what it got is reported, not hidden."""
        from ml_stack.platform import stop_gently

        terminated: list[bool] = []

        def refuse(_signal):
            raise OSError(errno.EINVAL, "The handle is invalid")

        proc = types.SimpleNamespace(send_signal=refuse,
                                     terminate=lambda: terminated.append(True))
        assert stop_gently(proc) == "TerminateProcess"
        assert terminated == [True]

    def test_posix_still_sends_sigterm(self):
        from ml_stack.platform import stop_gently

        sent: list[int] = []
        proc = types.SimpleNamespace(send_signal=sent.append, terminate=lambda: None)
        assert stop_gently(proc) == "SIGTERM"
        assert sent == [signal.SIGTERM]


class _FakeProc:
    """A Popen that runs until it is signalled, recording what it was sent."""

    pid = 4242

    def __init__(self) -> None:
        self._done = threading.Event()
        self.signals: list[int] = []

    def wait(self) -> int:
        self._done.wait()
        return 0

    def poll(self) -> int | None:
        return 0 if self._done.is_set() else None

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)
        self._done.set()

    def terminate(self) -> None:
        self.signals.append(-1)
        self._done.set()

    def kill(self) -> None:
        self._done.set()


def _await(predicate, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


class TestTheDaemonRunsAJobTheWindowsWay:
    def test_a_job_is_started_in_its_own_group_and_stopped_with_ctrl_break(
            self, windows, monkeypatch, tmp_path):
        from ml_stack.fleet import daemon as daemon_module
        from ml_stack.platform import CREATE_NEW_PROCESS_GROUP, CTRL_BREAK_EVENT

        started: list[dict] = []
        procs: list[_FakeProc] = []

        def fake_popen(argv, **kwargs):
            started.append(kwargs)
            procs.append(_FakeProc())
            return procs[-1]

        monkeypatch.setattr(daemon_module.subprocess, "Popen", fake_popen)
        runner = daemon_module.JobRunner(tmp_path / "traind")
        try:
            job = runner.submit("checkpointing-loop", ["a-job"], str(tmp_path))
            assert _await(lambda: runner.jobs[job.id].state == "running")
            assert started[0].get("creationflags") == CREATE_NEW_PROCESS_GROUP
            assert "start_new_session" not in started[0], \
                "silently ignored on Windows, and it would hide that the group is missing"
            runner.stop(job.id, grace_s=2.0)
            assert procs[0].signals == [CTRL_BREAK_EVENT]
            assert runner.jobs[job.id].state == "stopped"
        finally:
            runner.shutdown()


# -- shutting the daemon down cleanly ----------------------------------------------------
class TestQuitSignals:
    def test_windows_hooks_sigbreak_as_well_as_sigterm(self, windows, monkeypatch):
        from ml_stack.platform import quit_signals

        monkeypatch.setattr(signal, "SIGBREAK", 21, raising=False)
        assert quit_signals() == [signal.SIGTERM, 21]

    def test_posix_hooks_only_sigterm(self):
        from ml_stack.platform import quit_signals

        assert quit_signals() == [signal.SIGTERM]

    def test_the_handler_is_installed_from_the_main_thread_and_not_from_a_worker(self):
        from ml_stack.platform import on_quit

        before = signal.getsignal(signal.SIGTERM)
        try:
            hooked = on_quit(lambda *_: None)
            assert hooked == [signal.SIGTERM]
            from_worker: list = []
            t = threading.Thread(target=lambda: from_worker.append(on_quit(lambda *_: None)))
            t.start()
            t.join()
            assert from_worker == [[]], "a worker thread cannot set handlers; it must not raise"
        finally:
            signal.signal(signal.SIGTERM, before)


# -- a file only this user may read ----------------------------------------------------
class TestPrivateFile:
    def test_windows_cuts_the_acl_to_the_owner(self, windows, monkeypatch, tmp_path):
        from ml_stack import platform as platform_module

        ran: list[list[str]] = []
        monkeypatch.setattr(platform_module.subprocess, "run",
                            lambda argv, **k: ran.append(list(argv)) or _ok())
        monkeypatch.setenv("USERNAME", "fixture-user")
        target = tmp_path / "cluster.json"
        target.write_text("[]")

        platform_module.private_file(target)

        assert ran == [["icacls", str(target), "/inheritance:r", "/grant:r",
                        "fixture-user:F"]]

    def test_posix_is_chmod_600(self, tmp_path):
        from ml_stack.platform import private_file

        target = tmp_path / "cluster.json"
        target.write_text("[]")
        private_file(target)
        assert oct(target.stat().st_mode)[-3:] == "600"

    def test_the_cluster_key_goes_through_it(self, windows, monkeypatch, tmp_path):
        """`discovery` used to chmod directly, which on Windows protects nothing."""
        from ml_stack import platform as platform_module
        from ml_stack.fleet.discovery import create_cluster_key

        ran: list[list[str]] = []
        monkeypatch.setattr(platform_module.subprocess, "run",
                            lambda argv, **k: ran.append(list(argv)) or _ok())
        monkeypatch.setenv("USERNAME", "fixture-user")
        create_cluster_key(tmp_path / "cluster.key")
        assert any(argv[0] == "icacls" and argv[1].endswith("cluster.json") for argv in ran)


# -- starting at logon ---------------------------------------------------------------
@pytest.fixture
def win_autostart(monkeypatch, tmp_path):
    """`autostart` on Windows with no task registered yet and a recording schtasks."""
    from ml_stack.fleet import autostart

    ran: list[list[str]] = []
    refuse: dict[str, int] = {}

    def fake_run(argv, *a, **k):
        argv = list(argv)
        ran.append(argv)
        code = refuse.get(argv[1], 0) if argv and argv[0] == "schtasks" else 0
        return _ok(code, stderr="ERROR: Access is denied." if code else "")

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(autostart, "_windows_startup", lambda: tmp_path / "startup.cmd")
    monkeypatch.setattr(autostart, "_executable",
                        lambda: ["C:\\Tools\\ml stack\\ml-stack-traind.exe"])
    monkeypatch.setattr(autostart, "_windows_task_exists", lambda: False)
    monkeypatch.setattr(autostart, "_windows_login_task_exists", lambda: False)
    monkeypatch.setattr(subprocess, "run", fake_run)
    return autostart, ran, refuse


class TestAutostartOnWindows:
    def test_logon_is_a_scheduled_task_that_runs_a_logging_wrapper(
            self, win_autostart, tmp_path):
        auto, ran, _ = win_autostart

        done = auto.install("login", slots=2, labels=("prep",), log_dir=tmp_path)

        assert done.installed
        create = next(c for c in ran if c[:2] == ["schtasks", "/Create"])
        assert create[create.index("/TN") + 1] == auto.LOGIN_TASK == "com.ml-stack.traind.login"
        assert create[create.index("/SC") + 1] == "ONLOGON"
        assert create[create.index("/RL") + 1] == "LIMITED", "no administrator needed"
        wrapper = Path(create[create.index("/TR") + 1].strip('"'))
        assert wrapper == done.path == tmp_path / "ml-stack-traind.cmd"
        body = wrapper.read_text()
        assert '"C:\\Tools\\ml stack\\ml-stack-traind.exe" --slots 2 --label prep' in body
        assert str(tmp_path / "traind.log") in body, "a task's /TR cannot redirect; the wrapper does"
        assert ["schtasks", "/Run", "/TN", auto.LOGIN_TASK] in ran, "started now, not at the next logon"
        assert not (tmp_path / "startup.cmd").exists()

    def test_when_schtasks_refuses_the_startup_folder_is_the_fallback(
            self, win_autostart, tmp_path):
        auto, ran, refuse = win_autostart
        refuse["/Create"] = 1

        done = auto.install("login", log_dir=tmp_path)

        assert done.installed
        assert done.path == tmp_path / "startup.cmd"
        assert "Startup folder" in done.note and "Access is denied" in done.note
        assert "ml-stack-traind.exe" in done.path.read_text()

    def test_changing_the_answer_ends_and_deletes_the_logon_task(
            self, win_autostart, monkeypatch, tmp_path):
        auto, ran, _ = win_autostart
        monkeypatch.setattr(auto, "_windows_login_task_exists", lambda: True)

        auto.install("manual", log_dir=tmp_path)

        assert ["schtasks", "/End", "/TN", auto.LOGIN_TASK] in ran
        assert ["schtasks", "/Delete", "/F", "/TN", auto.LOGIN_TASK] in ran

    def test_status_reads_the_logon_task_as_login(self, win_autostart, monkeypatch):
        auto, _, _ = win_autostart
        monkeypatch.setattr(auto, "_windows_login_task_exists", lambda: True)
        assert auto.status()["mode"] == "login"
        assert auto.LOGIN_TASK in auto.status()["paths"]

    def test_the_boot_task_still_outranks_it(self, win_autostart, monkeypatch):
        auto, _, _ = win_autostart
        monkeypatch.setattr(auto, "_windows_login_task_exists", lambda: True)
        monkeypatch.setattr(auto, "_windows_task_exists", lambda: True)
        assert auto.status()["mode"] == "boot"


class TestTraindPersist:
    def test_persist_installs_at_login_with_the_flags_given_and_does_not_serve(
            self, monkeypatch, capsys, tmp_path):
        from ml_stack.fleet import autostart
        from ml_stack.fleet import daemon as daemon_module

        asked: list[dict] = []

        def fake_install(mode, **kwargs):
            asked.append({"mode": mode, **kwargs})
            return autostart.Autostart(mode, installed=True, path=tmp_path / "t.cmd",
                                       note="scheduled task runs it at logon")

        monkeypatch.setattr(autostart, "install", fake_install)
        monkeypatch.setattr(daemon_module, "serve_forever",
                            lambda *a, **k: pytest.fail("--persist must not serve"))

        code = daemon_module.main(["--persist", "--slots", "2", "--label", "prep",
                                   "--report", "ml_stack.fleet.daemon:stdlib_device_report"])

        assert code == 0
        assert asked == [{"mode": "login", "slots": 2, "labels": ("prep",),
                          "report": "ml_stack.fleet.daemon:stdlib_device_report"}]
        out = capsys.readouterr().out
        assert "installed to start at login" in out and "t.cmd" in out

    def test_a_refused_install_says_what_to_run_and_fails(self, monkeypatch, capsys):
        from ml_stack.fleet import autostart
        from ml_stack.fleet import daemon as daemon_module

        monkeypatch.setattr(autostart, "install", lambda mode, **k: autostart.Autostart(
            mode, installed=False, command="schtasks /Create ...", note="no permission"))
        assert daemon_module.main(["--persist"]) == 2
        err = capsys.readouterr().err
        assert "schtasks /Create" in err and "no permission" in err


# -- what the firewall has to let through -------------------------------------------
class TestDiscoveryAndTheFirewall:
    def test_beacons_are_udp_multicast_and_broadcast_on_a_fixed_port(self):
        from ml_stack.fleet import discovery

        assert discovery.DEFAULT_GROUP == "239.255.77.70"
        assert discovery.DEFAULT_PORT == 8771 and discovery.DEFAULT_HTTP_PORT == 8770
        assert discovery._destinations(discovery.DEFAULT_GROUP, 8771) == [
            ("239.255.77.70", 8771), ("255.255.255.255", 8771), ("127.0.0.1", 8771)]

    def test_the_two_inbound_rules_name_the_two_ports(self):
        from ml_stack.fleet.discovery import windows_firewall_line, windows_firewall_rules

        rules = dict(windows_firewall_rules())
        assert rules["ml-stack traind"].endswith("protocol=TCP localport=8770")
        assert rules["ml-stack discovery"].endswith("protocol=UDP localport=8771")
        line = windows_firewall_line()
        assert line.count("netsh advfirewall firewall add rule") == 2 and " && " in line
        assert "\n" not in line, "one line, for one administrator's prompt"

    def test_the_daemon_announces_itself_the_moment_it_starts(self, monkeypatch, tmp_path):
        """The loop's first beacon is interval_s away. A daemon that has just come up must
        appear in the next `ml-stack-peers ls`, not the one after."""
        from ml_stack.fleet import discovery
        from ml_stack.fleet.discovery import Advertiser, Beacon, _verify

        discovery.create_cluster_key(tmp_path / "cluster.key")
        key = discovery.load_cluster_key(tmp_path / "cluster.key")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as ear:
            ear.bind(("127.0.0.1", 0))
            ear.settimeout(3.0)
            heard_on = ear.getsockname()[1]
            # Unicast to the listener alone, so this is deterministic on any interface.
            monkeypatch.setattr(discovery, "_destinations",
                                lambda group, port: [("127.0.0.1", heard_on)])
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.bind(("", 0))
                serve_on = s.getsockname()[1]
            adv = Advertiser(Beacon(name="fixture-box", port=8770), key,
                             port=serve_on, interval_s=60.0).start()
            try:
                raw, _ = ear.recvfrom(65535)
            finally:
                adv.stop()
        msg = _verify(key, raw, kind="beacon")
        assert msg is not None and msg["beacon"]["name"] == "fixture-box"

    def test_setup_prints_the_netsh_line_until_both_rules_exist(self, windows, monkeypatch):
        from ml_stack import setup

        monkeypatch.setattr(setup, "_sysctl", lambda key: "")
        monkeypatch.setattr(setup, "_arches", lambda binary, known=None: set())
        monkeypatch.setattr(setup.subprocess, "run",
                            lambda argv, **k: _ok(1, stdout="\nNo rules match the specified criteria.\n"))

        found = [f for f in setup.look() if f.name == "firewall"]

        assert len(found) == 1
        finding = found[0]
        assert not finding.good and finding.root
        assert "ml-stack traind" in finding.said and "ml-stack discovery" in finding.said
        assert "protocol=TCP localport=8770" in finding.fix
        assert "protocol=UDP localport=8771" in finding.fix
        assert "administrator" in finding.note

    def test_setup_is_satisfied_once_netsh_finds_them(self, windows, monkeypatch):
        from ml_stack import setup

        monkeypatch.setattr(setup, "_sysctl", lambda key: "")
        monkeypatch.setattr(setup, "_arches", lambda binary, known=None: set())
        monkeypatch.setattr(setup.subprocess, "run",
                            lambda argv, **k: _ok(0, stdout="Rule Name: ml-stack traind\n"))

        finding = next(f for f in setup.look() if f.name == "firewall")
        assert finding.good and not finding.fix

    def test_no_firewall_finding_anywhere_else(self, monkeypatch):
        from ml_stack import setup

        monkeypatch.setattr(setup, "_sysctl", lambda key: "")
        monkeypatch.setattr(setup, "_arches", lambda binary, known=None: set())
        assert not [f for f in setup.look() if f.name == "firewall"]

    def test_peers_init_says_how_to_copy_the_key_in_powershell(self, capsys, tmp_path):
        from ml_stack.fleet.peers import main

        assert main(["--cluster-key", str(tmp_path / "cluster.key"), "init"]) == 0
        out = capsys.readouterr().out
        assert "chmod 600" in out and "Set-Content" in out
