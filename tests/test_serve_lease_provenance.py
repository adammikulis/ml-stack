"""A lease records why it exists and who took it, and every view of leases shows it.

The broker is the real one over a real fake-llama server; every holder is a real process whose
pid, parents, working directory and secrets-in-its-command-line are read back from the machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml_stack import hub
from ml_stack.serve import cli, holding, lease_cli, lease_history, ops, provenance
from ml_stack.serve.backend import LlamaServerBackend
from ml_stack.serve.broker import Ask, Broker
from ml_stack.serve.leases import lease_file
from ml_stack.serve.manager import ServerManager
from ml_stack.serve.process import kill_process_tree
from ml_stack.testing.fakes import fake_llama_binary

MIB = 1024 * 1024
SLEEP = "import time; time.sleep(120)"


@pytest.fixture
def broker(tmp_path):
    manager = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                            state_file=tmp_path / "servers.json")
    made = Broker(manager, idle_s=3600.0, room=lambda: None, scan=lambda: [])
    yield made
    for held in list(made.servers.values()):
        if held.pid and held.ours:
            kill_process_tree(held.pid)


@pytest.fixture
def model(tmp_path):
    path = tmp_path / f"first-M1-{tmp_path.name}.gguf"
    path.write_bytes(b"GGUF" + b"\x00" * 64)
    return str(path)


@pytest.fixture
def procs():
    started: list[subprocess.Popen] = []

    def one(*argv: str, cwd: Path | None = None) -> subprocess.Popen:
        proc = subprocess.Popen([sys.executable, "-c", SLEEP, *argv], cwd=cwd)
        started.append(proc)
        return proc

    yield one
    for proc in started:
        proc.kill()
        proc.wait(timeout=10)


def asking(model: str, pid: int, reason: str = "", requester: str = "") -> Ask:
    claim = provenance.asked(reason, requester) if reason or requester else {}
    return Ask(purpose="chat", models=(model,), pid=pid, label=f"pid {pid}",
               spec={"context": 512}, claim=claim)


def holder_of(broker: Broker, lease: str) -> dict:
    return next(h for s in broker.snapshot()["servers"] for h in s["holders"] if h["lease"] == lease)


def test_a_lease_taken_with_a_reason_shows_the_reason_in_the_snapshot_and_the_listing(
        broker, model, procs, monkeypatch, capsys):
    holder = procs()
    grant = broker.lease(asking(model, holder.pid, "reviewing the parser change", "coder-1"), timeout=30)

    one = holder_of(broker, grant.lease)
    assert (one["reason"], one["requester"], one["pid"]) == (
        "reviewing the parser change", "coder-1", holder.pid)
    monkeypatch.setattr(lease_cli, "snapshot", broker.snapshot)
    assert cli.main(["leases"]) == 0
    listed = capsys.readouterr().out
    assert "reviewing the parser change" in listed and "coder-1" in listed


def test_a_lease_taken_with_no_reason_says_so(broker, model, procs):
    holder = procs()
    grant = broker.lease(asking(model, holder.pid), timeout=30)

    assert holder_of(broker, grant.lease)["reason"] == provenance.NO_REASON
    assert provenance.NO_REASON in "\n".join(provenance.lines(holder_of(broker, grant.lease)))


def test_a_second_holder_of_a_server_is_listed_with_its_own_reason(broker, model, procs):
    first, second = procs(), procs()
    one = broker.lease(asking(model, first.pid, "extraction run"), timeout=30)
    two = broker.lease(asking(model, second.pid, "gym rollout", "gym"), timeout=30)

    assert two.shared and two.port == one.port
    held = next(s for s in broker.snapshot()["servers"] if s["port"] == one.port)
    assert {(h["pid"], h["reason"]) for h in held["holders"]} == {
        (first.pid, "extraction run"), (second.pid, "gym rollout")}
    assert {h["requester"] for h in held["holders"] if h["pid"] == second.pid} == {"gym"}


def test_the_process_chain_directory_and_branch_are_read_off_the_real_process(
        broker, model, tmp_path, procs):
    repo = tmp_path / "work"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "feat-lease-why", str(repo)], check=True)
    holder = procs(cwd=repo)
    grant = broker.lease(asking(model, holder.pid, "x"), timeout=30)

    one = holder_of(broker, grant.lease)
    assert Path(one["cwd"]).resolve() == repo.resolve()
    assert (Path(one["worktree"]).resolve(), one["branch"]) == (repo.resolve(), "feat-lease-why")
    assert str(holder.pid) in one["chain"][0] and SLEEP[:12] in one["chain"][0]
    assert any(str(Path(sys.executable).name) in line for line in one["chain"][1:]) or len(one["chain"]) > 1
    assert one["pid_started"] and abs(one["pid_started"] - time.time()) < 3600


def test_a_worktree_checkout_reads_its_own_branch(broker, model, tmp_path, procs):
    main = tmp_path / "main"
    subprocess.run(["git", "init", "-q", "-b", "trunk", str(main)], check=True)
    subprocess.run(["git", "-C", str(main), "-c", "user.name=t", "-c", "user.email=t@example.org",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    linked = tmp_path / "linked"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", "-b", "side", str(linked)],
                   check=True)
    holder = procs(cwd=linked)
    grant = broker.lease(asking(model, holder.pid, "x"), timeout=30)

    assert holder_of(broker, grant.lease)["branch"] == "side"


def test_a_secret_in_a_parents_command_line_is_masked(broker, model, tmp_path):
    parent = subprocess.Popen(
        [sys.executable, "-c",
         "import subprocess, sys, time\n"
         "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
         "print(child.pid, flush=True)\n"
         "time.sleep(120)\n",
         "--api-key", "hunter2hunter2hunter2", "password=swordfish-swordfish",
         "--token=ghp_abcdefghijklmnopqrstuvwxyz0123456789"],
        stdout=subprocess.PIPE, text=True)
    try:
        child = int(parent.stdout.readline())
        grant = broker.lease(asking(model, child, "x"), timeout=30)
        chain = " ".join(holder_of(broker, grant.lease)["chain"])
    finally:
        parent.kill()
        parent.wait(timeout=10)
        kill_process_tree(child)
    assert str(parent.pid) in chain
    for secret in ("hunter2hunter2hunter2", "swordfish", "ghp_abcdefghijklmnop"):
        assert secret not in chain, chain


def test_an_ended_lease_leaves_a_row_in_the_history(broker, model, procs):
    holder = procs()
    grant = broker.lease(asking(model, holder.pid, "benchmark sweep", "bench"), timeout=30)
    assert lease_history.rows(broker.history_file) == []

    assert broker.release(grant.lease)

    (row,) = lease_history.rows(broker.history_file)
    assert (row["reason"], row["requester"], row["model"], row["lease"]) == (
        "benchmark sweep", "bench", Path(model).name, grant.lease)
    assert row["ended"] >= row["taken"] and row["duration_s"] >= 0
    assert lease_history.rows(broker.history_file, model="nothing-like-it") == []
    assert lease_history.rows(broker.history_file, since=row["ended"] + 10) == []


def test_a_holder_that_dies_leaves_a_row_when_the_broker_reaps_it(broker, model, procs):
    holder = procs()
    grant = broker.lease(asking(model, holder.pid, "short job"), timeout=30)
    holder.kill()
    holder.wait(timeout=10)

    broker.reap()

    assert [r["lease"] for r in lease_history.rows(broker.history_file)] == [grant.lease]


def test_the_history_command_lists_ended_leases(broker, model, procs, monkeypatch, capsys):
    holder = procs()
    broker.release(broker.lease(asking(model, holder.pid, "listing check", "tester"), timeout=30).lease)
    monkeypatch.setattr(lease_cli, "history_file", lambda: broker.history_file)

    assert cli.main(["history", "--model", "first-M1", "--since", "1d"]) == 0
    out = capsys.readouterr().out
    assert "listing check" in out and "tester" in out
    assert cli.main(["history", "--model", "no-such-model"]) == 0
    assert "no ended lease" in capsys.readouterr().out
    assert cli.main(["history", "--since", "not-a-time"]) == 2


class _Machine:
    def __init__(self) -> None:
        self.free = 65536 * MIB


@pytest.fixture
def machine(tmp_path, monkeypatch):
    memory = _Machine()
    made: dict[str, ServerManager] = {}

    def manager_for(binary_arg="", build=""):
        if "m" not in made:
            m = ServerManager(LlamaServerBackend(binary=fake_llama_binary(tmp_path)),
                              state_file=lease_file())
            m._broker = Broker(m, room=lambda: memory.free, scan=lambda: [])
            made["m"] = m
        return made["m"]

    monkeypatch.setattr(ops, "manager_for", manager_for)
    monkeypatch.setattr(hub, "room", lambda: 65536 * MIB)
    monkeypatch.setattr(hub, "total_memory", lambda: 64 * 1024 * MIB)
    yield manager_for
    for hold in holding.holds():
        holding.down([hold], manager=manager_for(), wait_s=20)
    if "m" in made:
        made["m"].stop_all()


def test_up_for_names_the_reason_and_a_second_up_is_listed_as_another_user(
        tmp_path, machine, capsys, monkeypatch):
    path = tmp_path / "Qwen3.8-27B-UD-Q4_K_XL.gguf"
    with path.open("wb") as out:
        out.write(b"GGUF" + b"\x00" * 64)
        out.truncate(600 * MIB)
    monkeypatch.delenv(provenance.ENV_FOR, raising=False)

    def up(*extra: str) -> dict:
        assert cli.main(["up", str(path), "--binary", "x", "--no-profile", "--no-mtp",
                         "--context", "4096", "--json", *extra]) == 0
        return json.loads(capsys.readouterr().out)

    first = up("--for", "workspace coding view")
    cli.main(["status", "--port", str(first["port"])])
    shown = capsys.readouterr().out
    assert "workspace coding view" in shown and "why " in shown

    again = up("--for", "gym run for the sensor task")
    assert again["adopted"] and again["lease"] == first["lease"]
    cli.main(["status", "--port", str(first["port"])])
    shown = capsys.readouterr().out
    assert "also used by" in shown and "gym run for the sensor task" in shown

    assert cli.main(["down", path.name]) == 0
    capsys.readouterr()
    cli.main(["history"])
    assert "workspace coding view" in capsys.readouterr().out


def test_the_environment_supplies_the_reason_when_none_is_given(monkeypatch):
    monkeypatch.setenv(provenance.ENV_FOR, "from the environment")
    assert provenance.asked()["reason"] == "from the environment"
    assert provenance.asked("explicit")["reason"] == "from the environment (explicit)"
    monkeypatch.delenv(provenance.ENV_FOR)
    assert provenance.asked()["reason"] == ""


def test_the_workspace_label_names_the_requester(monkeypatch):
    monkeypatch.setenv(provenance.ENV_LABEL, "coder-7")
    assert provenance.asked("x")["requester"] == "coder-7"
    monkeypatch.delenv(provenance.ENV_LABEL)
    assert provenance.asked("x")["requester"] == provenance.program()


def test_every_lease_the_source_takes_gives_a_reason():
    import ast
    import re

    root = Path(__file__).resolve().parents[1] / "src" / "ml_stack"
    missing = []
    for path in sorted(root.rglob("*.py")):
        if path.name in ("manager.py", "broker.py", "broker_wire.py", "fakes.py"):
            continue
        text = path.read_text()
        imports_serve = re.search(r"from ml_stack\.serve(?:\.manager)? import [^\n]*\bserve\b", text)
        for node in ast.walk(ast.parse(text, str(path))):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            leases = (isinstance(func, ast.Attribute) and func.attr == "lease"
                      and isinstance(func.value, ast.Name) and func.value.id in ("broker_wire", "manager"))
            serves = (bool(imports_serve) and isinstance(func, ast.Name) and func.id in ("serve", "serve_fn")
                      ) or ast.unparse(func) == "ml_stack.serve.serve"
            if (leases or serves) and "reason" not in {k.arg for k in node.keywords}:
                missing.append(f"{path.relative_to(root)}:{node.lineno}")
    assert not missing, "a lease with no reason= at " + ", ".join(missing)
