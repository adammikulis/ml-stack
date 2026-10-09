"""Lease commands use isolated broker discovery, private records, and fake native servers."""

from __future__ import annotations

import json
import os
import socket
import time

import pytest

from poolhouse import hub
from poolhouse.serve import cli, holding, ops
from poolhouse.serve.backend import LlamaServerBackend
from poolhouse.serve.broker import Ask, Broker
from poolhouse.serve.leases import lease_file
from poolhouse.serve.manager import ServerManager
from poolhouse.testing.fakes import fake_llama_binary

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
            m._broker = Broker(m, room=lambda: memory.free, scan=lambda: [])
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
    assert "poolhouse-serve memory --for" in text and "--apply" in text
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
    from poolhouse.serve import reclaim

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


def test_two_simultaneous_ups_for_one_shape_start_one_holder(tmp_path, machine, capsys):
    import threading

    model = gguf(tmp_path, "race.gguf")
    results = []

    def one():
        results.append(cli.main(["up", model, "--binary", "x", "--no-profile", "--no-mtp",
                                 "--context", "4096", "--json"]))

    threads = [threading.Thread(target=one) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    out = capsys.readouterr().out
    assert results == [0, 0], out
    assert len(holding.holds()) == 1, "the second up adopts the first's lease"
    assert out.count('"adopted": true') == 1 and out.count('"adopted": false') == 1


def test_a_larger_context_is_never_served_by_a_smaller_server(tmp_path, machine):
    """Sharing is by the shape that matters, not by model: a 256K ask next to a 32K server
    for the same file gets its own server (or waits), never the 32K one."""
    from poolhouse.serve.broker import Ask

    model = gguf(tmp_path, "ctx.gguf")
    manager = ops.manager_for()
    broker = manager.broker
    small = broker.lease(Ask(purpose="chat", models=(model,), pid=os.getppid(),
                             spec={"context": 4096}), timeout=30)
    same = broker.lease(Ask(purpose="chat", models=(model,), pid=os.getppid(),
                            spec={"context": 2048}), timeout=30)
    assert same.port == small.port and same.shared, "a smaller ask is served by the bigger"
    with pytest.raises(Exception, match="4,096 tokens of context, 262,144 asked"):
        broker.lease(Ask(purpose="chat", models=(model,), pid=os.getpid(),
                         spec={"context": 262144}), timeout=2)
    for grant in (small, same):
        broker.release(grant.lease)
    big = broker.lease(Ask(purpose="chat", models=(model,), pid=os.getppid(),
                           spec={"context": 8192}), timeout=30)
    assert big.port != small.port or not big.shared
    broker.release(big.lease)


def test_status_shows_each_leases_context(tmp_path, machine, capsys):
    port = json.loads(up(capsys, gguf(tmp_path, "shown.gguf"), "--context", "8192",
                         "--json")[1])["port"]
    cli.main(["status", "--port", str(port)])
    assert "context 8,192" in capsys.readouterr().out
