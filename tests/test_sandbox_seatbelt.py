"""The Seatbelt backend against the real ``sandbox-exec``: what a confined process can read,
write, connect to and start."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack import sandbox
from ml_stack.sandbox import Limits, Net, SandboxViolation, run
from tests.sandbox_kit import CONNECT, policy

pytest_plugins = ["tests.sandbox_kit"]

SECRET = "the-secret-contents"


def decoy(tmp_path, *parts: str):
    path = tmp_path.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(SECRET)
    return path


def test_a_read_inside_the_allow_list_works(tmp_path, seatbelt):
    inside = decoy(tmp_path, "project", "notes.txt")
    result = run(["/bin/cat", str(inside)], policy(read=[inside.parent]))
    assert (result.returncode, result.stdout) == (0, SECRET)
    assert result.sandboxed and result.backend == "seatbelt"


def test_a_read_outside_the_allow_list_fails_and_names_the_path(tmp_path, seatbelt):
    decoy(tmp_path, "project", "notes.txt")
    outside = decoy(tmp_path, "home", ".ssh", "id_rsa")
    result = run(["/bin/cat", str(outside)], policy(read=[tmp_path / "project"]))
    assert result.returncode != 0
    assert SECRET not in result.stdout + result.stderr
    assert any(d["operation"] == "file-read-data" and d["target"] == str(outside)
               for d in result.denials)


def test_a_write_outside_the_allow_list_fails_and_inside_works(tmp_path, seatbelt):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pol = policy(read=[elsewhere], write=[scratch])
    bad = run(["/bin/sh", "-c", f"echo x > {elsewhere}/out.txt"], pol)
    good = run(["/bin/sh", "-c", f"echo x > {scratch}/out.txt"], pol)
    assert bad.returncode != 0 and not (elsewhere / "out.txt").exists()
    assert any(d["operation"].startswith("file-write") for d in bad.denials)
    assert good.returncode == 0 and (scratch / "out.txt").read_text() == "x\n"


def test_a_denied_directory_cannot_be_listed(tmp_path, seatbelt):
    decoy(tmp_path, "home", ".aws", "credentials")
    result = run(["/bin/ls", str(tmp_path / "home" / ".aws")], policy())
    assert result.returncode != 0 and "credentials" not in result.stdout


def test_the_network_is_refused_unless_the_policy_allows_it(tmp_path, seatbelt, listener):
    code = CONNECT.format(port=listener.port)
    py = sys.executable
    refused = run([py, "-c", code], policy(python=True, net=Net.deny()))
    loopback = run([py, "-c", code], policy(python=True, net=Net.loopback()))
    named = run([py, "-c", code], policy(python=True, net=Net.only(listener.port)))
    other = run([py, "-c", code], policy(python=True, net=Net.only(listener.port + 1)))
    assert refused.stdout.strip() == "refused 1"
    assert loopback.stdout.strip() == "connected pong"
    assert named.stdout.strip() == "connected pong"
    assert other.stdout.strip() == "refused 1"
    assert listener.accepted == 2


def test_a_child_of_the_command_is_inside_the_same_sandbox(tmp_path, seatbelt):
    outside = decoy(tmp_path, "home", "secret.txt")
    result = run(["/bin/sh", "-c", f"/bin/sh -c '/bin/cat {outside}'; echo rc=$?"], policy())
    assert "rc=1" in result.stdout and SECRET not in result.stdout


def test_a_program_outside_the_exec_list_is_not_started(tmp_path, seatbelt):
    pol = policy(read=[tmp_path])
    pol = sandbox.Policy("narrow", read=pol.read, exec=("/bin/sh",), env=dict(pol.env))
    result = run(["/bin/sh", "-c", "/bin/echo hi; echo rc=$?"], pol)
    assert "hi" not in result.stdout
    assert any(d["operation"] == "process-exec*" and d["target"].startswith("/bin/")
               for d in result.denials)


def test_a_confined_process_cannot_signal_another_process(tmp_path, seatbelt):
    bystander = subprocess.Popen(["/bin/sleep", "30"], start_new_session=True)
    try:
        result = run(["/bin/sh", "-c", f"kill -9 {bystander.pid}; echo rc=$?"], policy())
        assert "rc=0" not in result.stdout and bystander.poll() is None
    finally:
        os.killpg(bystander.pid, signal.SIGKILL)
        bystander.wait()


def test_a_symlink_inside_an_allowed_directory_does_not_lead_out(tmp_path, seatbelt):
    outside = decoy(tmp_path, "home", "key")
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    (allowed / "link").symlink_to(outside)
    result = run(["/bin/cat", str(allowed / "link")], policy(read=[allowed]))
    assert result.returncode != 0 and SECRET not in result.stdout


def test_a_symlinked_or_climbing_allow_list_entry_is_refused(tmp_path, seatbelt):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "alias").symlink_to(real)
    for bad in (tmp_path / "alias", f"{real}/../real", "relative/path", f"{real}/", ""):
        with pytest.raises(sandbox.PolicyError):
            run(["/bin/echo", "x"], policy(read=[bad]))
    with pytest.raises(sandbox.PolicyError):
        run(["/bin/echo", "x"], policy(read=[tmp_path / "does-not-exist"]))
    with pytest.raises(sandbox.PolicyError):
        run(["/bin/echo", "x"], policy(read=[f"{real}\nfile"]))


def test_the_environment_is_exactly_what_the_policy_lists(tmp_path, seatbelt, monkeypatch):
    monkeypatch.setenv("ML_STACK_TEST_TOKEN", "leaked-value")
    result = run(["/usr/bin/env"], policy(env={"LISTED": "yes"}))
    lines = sorted(result.stdout.split())
    assert "LISTED=yes" in lines and "PATH=/usr/bin:/bin" in lines
    assert not any("ML_STACK_TEST_TOKEN" in line or "leaked-value" in line for line in lines)
    assert all(line.split("=")[0] in {"LISTED", "PATH", "PWD", "SHLVL", "_", "__CF_USER_TEXT_ENCODING"}
               for line in lines)


@pytest.mark.parametrize("name", [
    'x") (allow file-read* (regex #".*")) ;',
    "x\\\") (allow default) (",
    "x)(allow file-write* (regex #\".\"))",
    "semi;colon (paren) 'quote' \"dq\"",
])
def test_a_hostile_directory_name_cannot_add_rules(tmp_path, seatbelt, name):
    hostile = tmp_path / name
    hostile.mkdir()
    (hostile / "ok.txt").write_text("fine")
    outside = decoy(tmp_path, "elsewhere", "secret.txt")
    inside = run(["/bin/cat", str(hostile / "ok.txt")], policy(read=[hostile]))
    assert inside.returncode == 0 and inside.stdout == "fine"
    stolen = run(["/bin/cat", str(outside)], policy(read=[hostile]))
    assert stolen.returncode != 0 and SECRET not in stolen.stdout
    wrote = run(["/bin/sh", "-c", f"echo x > {outside}.new"], policy(read=[hostile]))
    assert wrote.returncode != 0 and not Path(f"{outside}.new").exists()


def test_a_wall_clock_limit_kills_the_group_and_only_that_group(tmp_path, seatbelt):
    bystander = subprocess.Popen(["/bin/sleep", "60"], start_new_session=True)
    try:
        pol = policy(limits=Limits(wall_seconds=1.0))
        result = run(["/bin/sh", "-c", "/bin/sleep 60 & echo $!; wait"], pol)
        grandchild = int(result.stdout.split()[0])
        assert result.timed_out and result.returncode != 0 and result.seconds < 10
        time.sleep(0.3)
        with pytest.raises(ProcessLookupError):
            os.kill(grandchild, 0)
        assert bystander.poll() is None
    finally:
        os.killpg(bystander.pid, signal.SIGKILL)
        bystander.wait()


def test_a_cpu_limit_stops_a_busy_loop(tmp_path, seatbelt):
    pol = policy(python=True, limits=Limits(wall_seconds=30, cpu_seconds=1))
    result = run([sys.executable, "-c", "while True: pass"], pol)
    assert result.returncode == -signal.SIGXCPU and not result.timed_out and result.seconds < 15


def test_a_file_size_limit_stops_a_large_write(tmp_path, seatbelt):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    pol = policy(python=True, write=[scratch], limits=Limits(wall_seconds=30, file_bytes=100_000))
    code = f"open({str(scratch / 'big')!r}, 'wb').write(b'x' * 5_000_000)"
    result = run([sys.executable, "-c", code], pol)
    assert result.returncode != 0
    assert (scratch / "big").stat().st_size <= 100_000


def test_output_over_the_cap_is_cut_and_the_command_stopped(tmp_path, seatbelt):
    pol = policy(limits=Limits(wall_seconds=30, output_bytes=10_000))
    result = run(["/usr/bin/yes"], pol)
    assert result.truncated and len(result.stdout) == 10_000 and result.returncode != 0


def test_a_denial_is_reported_as_an_event_and_as_a_clear_error(tmp_path, seatbelt):
    outside = decoy(tmp_path, "home", ".ssh", "id_rsa")
    events: list[tuple[str, dict]] = []
    result = run(["/bin/cat", str(outside)], policy(),
                 on_event=lambda name, fields: events.append((name, fields)))
    denied = [fields for name, fields in events if name == "sandbox.denied"]
    assert len(denied) == 1 and denied[0]["first"]["target"] == str(outside)
    with pytest.raises(SandboxViolation, match=r"file-read-data .*id_rsa"):
        result.check()


def test_a_command_that_is_not_refused_has_no_denials(tmp_path, seatbelt):
    inside = decoy(tmp_path, "p", "f")
    result = run(["/bin/cat", str(inside)], policy(read=[inside.parent]), diagnose="always")
    assert result.denials == [] and result.check() is result


def test_the_deprecation_is_logged_once_per_process(tmp_path, seatbelt, caplog):
    from ml_stack.sandbox import seatbelt as module

    module._WARNED.clear()
    caplog.set_level("WARNING", logger="ml_stack.sandbox")
    for _ in range(3):
        run(["/bin/echo", "x"], policy(), diagnose="never")
    said = [r for r in caplog.records if "deprecated" in r.getMessage()]
    assert len(said) == 1 and "sandbox-exec" in said[0].getMessage()


def test_a_missing_binary_makes_the_backend_unavailable(monkeypatch, seatbelt):
    from ml_stack.sandbox import seatbelt as module

    monkeypatch.setattr(module, "BINARY", "/nonexistent/sandbox-exec")
    state = module.Seatbelt().available()
    assert not state.ok and "/nonexistent/sandbox-exec" in state.reason
    with pytest.raises(sandbox.SandboxUnavailable):
        run(["/bin/echo", "x"], policy())
