"""``ml-stack-serve up`` is a lease from the broker, held until ``down``.

Everything here runs the real command against a real broker (in this process, as the suite
does everywhere) and a real fake llama-server binary: nothing is mocked between the command
and the process it starts. Only the machine's memory is made up, the one thing a test cannot
make short by really filling it.
"""

from __future__ import annotations

import json
import os
import socket
import time

import pytest

from ml_stack import hub
from ml_stack.serve import cli, holding, ops
from ml_stack.serve.backend import LlamaServerBackend
from ml_stack.serve.broker import Ask, Broker
from ml_stack.serve.leases import lease_file
from ml_stack.serve.manager import ServerManager
from ml_stack.testing.fakes import fake_llama_binary

MIB = 1024 * 1024
Q4_K_XL = "Qwen3.8-27B-UD-Q4_K_XL.gguf"


class Machine:
    """The memory this machine reports, changed by the test."""

    def __init__(self, free: int, budget: int) -> None:
        self.free, self.budget = free, budget


@pytest.fixture
def machine(tmp_path, monkeypatch):
    binary = fake_llama_binary(tmp_path)
    memory = Machine(free=65536 * MIB, budget=65536 * MIB)
    made: dict[str, ServerManager] = {}

    def manager_for(binary_arg="", build=""):
        if "m" not in made:
            m = ServerManager(LlamaServerBackend(binary=binary), state_file=lease_file())
            m._broker = Broker(m, room=lambda: memory.free)
            made["m"] = m
        return made["m"]

    monkeypatch.setattr(ops, "manager_for", manager_for)
    monkeypatch.setattr(hub, "room", lambda: memory.budget)
    monkeypatch.setattr(hub, "total_memory", lambda: 64 * 1024 * MIB)
    yield memory
    for hold in holding.holds():
        holding.down([hold], manager=manager_for(), wait_s=20)
    if "m" in made:
        made["m"].stop_all()


def gguf(tmp_path, name, size_mib=600):
    path = tmp_path / name
    with path.open("wb") as out:
        out.write(b"GGUF" + b"\x00" * 64)
        out.truncate(size_mib * MIB)
    return str(path)


def up(capsys, *argv):
    code = cli.main(["up", *argv, "--binary", "x", "--no-profile", "--no-mtp"])
    out = capsys.readouterr()
    return code, out.out, out.err


def test_up_takes_a_lease_shows_in_status_adopts_on_a_second_up_and_releases_on_down(
        tmp_path, machine, capsys):
    model = gguf(tmp_path, Q4_K_XL)
    code, out, err = up(capsys, model, "--context", "4096", "--json")
    first = json.loads(out)
    assert code == 0 and first["status"] == "ready" and not first["adopted"], (out, err)
    assert "no measured profile" not in out

    cli.main(["status", "--json", "--port", str(first["port"])])
    leases = json.loads(capsys.readouterr().out)["leases"]
    assert [(h["id"], h["port"], h["context"]) for h in leases] == [
        (first["lease"], first["port"], 4096)]

    code, out, _ = up(capsys, model, "--context", "4096", "--json")
    again = json.loads(out)
    assert code == 0 and again["adopted"] and again["lease"] == first["lease"]
    assert again["port"] == first["port"] and len(holding.holds()) == 1

    assert cli.main(["down", Q4_K_XL]) == 0
    assert "released lease" in capsys.readouterr().out
    assert holding.holds() == []
    assert not _listening(first["port"]), "the server must stop with its only lease"


def test_a_port_held_by_a_stranger_is_never_taken_the_broker_picks_another(
        tmp_path, machine, capsys):
    stranger = socket.socket()
    stranger.bind(("127.0.0.1", 0))
    stranger.listen()
    try:
        asked = stranger.getsockname()[1]
        code, out, err = up(capsys, gguf(tmp_path, "a.gguf"), "--port", str(asked), "--json")
        got = json.loads(out)
        assert code == 0 and got["port"] != asked, (out, err)
        assert "only a request" in err
    finally:
        stranger.close()


def test_a_shape_that_cannot_fit_is_refused_with_the_reason_and_the_human_command(
        tmp_path, machine, capsys):
    machine.budget = 256 * MIB
    code, out, err = up(capsys, gguf(tmp_path, Q4_K_XL), "--context", "256k")
    text = out + err
    assert code == 2 and holding.holds() == []
    assert "refused:" in text and "this machine lets a model use 256" in text
    assert "ml-stack-serve memory --for" in text and "--apply" in text
    assert "never by an agent" in text


def test_a_lease_queues_when_memory_is_short_and_starts_when_it_frees(
        tmp_path, machine, capsys):
    first = gguf(tmp_path, "first.gguf", 900)
    second = gguf(tmp_path, "second.gguf", 900)
    code, up_first_out, _ = up(capsys, first, "--json")
    assert code == 0
    machine.free = 1200 * MIB  # the second fits by its file size, not by its estimate (KV, runtime)

    code, out, err = up(capsys, second, "--no-wait")
    assert code == 0 and "queued" in (out + err)
    cli.main(["status", "--port", str(json.loads(up_first_out)["port"])])
    assert "queued" in capsys.readouterr().out
    time.sleep(2.0)  # long enough for a broker that would admit it to have started it
    assert [h.status for h in holding.holds() if "second" in h.model] == ["queued"]

    assert cli.main(["down", "first"]) == 0
    machine.free = 65536 * MIB
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if [h.status for h in holding.holds() if "second" in h.model] == ["ready"]:
            break
        time.sleep(0.2)
    assert [h.status for h in holding.holds() if "second" in h.model] == ["ready"]


def test_down_leaves_a_server_another_lease_still_holds(tmp_path, machine, capsys):
    model = gguf(tmp_path, "shared.gguf")
    held = json.loads(up(capsys, model, "--json")[1])
    manager = ops.manager_for()
    other = manager.broker.lease(
        Ask(purpose=f"serve:{holding.holds()[0].shape}", models=(model,), pid=os.getppid(), label="another client"),
        timeout=5)
    assert other.port == held["port"]
    assert cli.main(["down", "shared"]) == 0
    assert "still used by" in capsys.readouterr().out
    assert _listening(held["port"]), "the other lease keeps the server up"
    manager.broker.release(other.lease)


def test_the_idle_timeout_releases_the_lease(tmp_path, machine, capsys, monkeypatch):
    from ml_stack.serve import reclaim

    monkeypatch.setattr(reclaim, "busy_now", lambda base_url, **k: False)
    assert up(capsys, gguf(tmp_path, "idle.gguf"), "--idle", "1s", "--json")[0] == 0
    deadline = time.monotonic() + 20
    while holding.holds() and time.monotonic() < deadline:
        time.sleep(0.2)
    assert holding.holds() == []


def test_there_is_no_direct_start_left_on_the_command_line(capsys):
    """The flags that skipped the broker are gone, not ignored."""
    for gone in ("--anyway", "--timeout"):
        with pytest.raises(SystemExit):
            cli.main(["up", "m.gguf", gone, "1"])
        capsys.readouterr()
    assert cli.main(["up", "m.gguf", "--escalate"]) == 2
    assert "escalate" in capsys.readouterr().err


def _listening(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) == 0
