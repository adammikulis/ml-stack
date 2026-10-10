"""The public API against a real node on a loopback socket: the board, the pool, the lease table; and the fakes
for models and serving. Every test runs under a temporary home and a node that never listens on the network."""

from __future__ import annotations

import struct

import pytest
from node_kit import BOARD
from test_serve_up_is_a_lease import (  # noqa: F401 - the fake-binary machine of the serve tests
    Q4_K_XL,
    gguf,
    machine,
)

import poolhouse as ph
from poolhouse import node_launch
from poolhouse.testing.fakehub import fake_hub

pytest_plugins = ["node_kit"]

BIG = b"GGUF" + struct.pack("<IQQ", 3, 0, 0) + bytes(range(251)) * 64
REPO = "maker/thing-GGUF"


@pytest.fixture
def mine(workspace_node, monkeypatch):
    """This device's node is the test's node: the API finds it the way any caller does."""
    monkeypatch.setattr(node_launch, "default_state", lambda: workspace_node.state)
    monkeypatch.setenv("POOLHOUSE_BOARD", BOARD)
    return workspace_node


def test_a_board_round_trip_between_two_sessions_through_ph_board(mine, monkeypatch):
    first, second = mine.member("one"), mine.member("two")
    monkeypatch.setenv("POOLHOUSE_WORKSPACE_AGENT", first.name)
    sender = ph.board.connect()
    assert sender.name == first.name and sender.id == BOARD
    ref = sender.send(second.name, "the second device is in", kind="status", subject="pool")
    answer = ph.board(agent=second.name)
    got = answer.inbox()
    assert [(m.sender, m.kind, m.subject, m.body, m.foreign) for m in got] == [
        (first.name, "status", "pool", "the second device is in", False)]
    reply = answer.send(first.name, "welcome", kind="answer", reply_to=ref)
    assert [m.body for m in answer.thread(ref)] == ["the second device is in", "welcome"]
    assert [m.body for m in sender.dm(second.name)] == ["the second device is in", "welcome"]
    assert reply and answer.inbox(ack=True) == got
    assert answer.inbox() == [] and answer.wait(0.1) == []
    assert [m.body for m in sender.wait(2)] == ["welcome"]
    with pytest.raises(ValueError, match="kind is one of"):
        sender.send(second.name, "x", kind="shout")


def test_claims_conflict_notes_announcements_and_agents(mine):
    one, two = mine.member("one"), mine.member("two")
    mine_, other = ph.board.connect(agent=one.name), ph.board.connect(agent=two.name)
    held = mine_.claim("branch", "0.3dev-parser", ttl_s=600)
    assert (held.kind, held.key, held.owner) == ("branch", "0.3dev-parser", one.name)
    with pytest.raises(ph.Conflict):
        other.claim("branch", "0.3dev-parser")
    assert [c.key for c in other.claims("branch")] == ["0.3dev-parser"]
    assert [c.owner for c in mine_.renew(900)] == [one.name]
    assert mine_.release("branch", "0.3dev-parser") and other.claim("branch", "0.3dev-parser").owner == two.name
    mine_.add_note("decision", "Parser", "Use the recursive-descent one.", tags=("parser",))
    assert [n.title for n in other.notes("recursive")] == ["Parser"]
    assert mine_.announce("milestone", "parser landed")
    with pytest.raises(ValueError, match="announcement"):
        mine_.announce("shouting", "x")
    assert {a.name for a in other.agents()} >= {one.name, two.name}
    assert mine_.links() == [] and isinstance(mine_.projects(), list)
    with pytest.raises(ph.Denied):
        ph.board.connect(agent="nobody-registered")


def test_a_pool_of_one_lists_this_device_and_capacity(mine):
    pool = ph.pool.status()
    assert pool.policy in ("open", "secure") and pool.id and pool.listening == ""
    assert [m.this_device for m in ph.pool.members()] == [True] and ph.pool.policy() == pool.policy
    cap = ph.pool.capacity()
    assert cap.cpu_cores >= 1 and cap.cpu_slots >= 1 and cap.memory_bytes > 0 and cap.leases_held == 0
    with pytest.raises(ph.Error, match="agreed=True"):
        ph.pool.listen("open")
    with pytest.raises(ph.Error, match="open, secure"):
        ph.pool.set_policy("whenever")
    with pytest.raises(ph.Error, match="no member"):
        ph.pool.remove("ghost")


def test_the_policy_changes_and_a_member_that_is_not_there_cannot_be_joined_by_a_wrong_code(mine):
    assert ph.pool.set_policy("secure") == "secure" and ph.pool.policy() == "secure"
    assert ph.pool.set_policy("open") == "open"
    with pytest.raises(ph.Error):
        ph.pool.join("127.0.0.1", "")


def test_a_lease_is_acquired_shows_in_the_table_counts_in_capacity_and_is_released(mine):
    mine.member("holder")
    lease = ph.leases.acquire([ph.leases.cpu_slots(1), ph.leases.memory_mb(64)], ttl_s=120)
    assert lease.state == "held" and [r["type"] for r in lease.resources] == ["cpu_slots", "memory_mb"]
    assert [x.id for x in ph.leases.view()] == [lease.id] and ph.pool.capacity().cpu_slots_leased == 1
    assert ph.leases.renew(lease.id, 300).id == lease.id and ph.leases.wait(lease.id, 1).state == "held"
    assert ph.leases.release(lease.id) and ph.leases.view() == []
    taken = ph.leases.acquire([ph.leases.gpu(), ph.leases.model_slot("m.gguf", context=512), ph.leases.claim("port", "9999")])
    assert taken.state == "held" and len(ph.leases.view()) == 1
    assert ph.leases.release(taken.id)


def test_models_pull_into_the_hub_cache_and_path_finds_them_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    from poolhouse import http

    monkeypatch.setattr(http, "check", lambda url: url)
    assert ph.models.cache_dir() == tmp_path / "hub" and ph.models.path(REPO, "thing-Q4_K_M.gguf") is None
    with fake_hub({REPO: {"thing-Q4_K_M.gguf": BIG}}) as served:
        served.point(monkeypatch)
        got = ph.models.pull(REPO, "thing-Q4_K_M.gguf")
        assert got.read_bytes() == BIG and got.parent.parent.parent.name == "models--maker--thing-GGUF"
        assert ph.models.path(REPO, "thing-Q4_K_M.gguf") == got
        assert ph.hub.fetch(f"hf:{REPO}/thing-Q4_K_M.gguf") == got
        assert ph.hub.hub_cache() == tmp_path / "hub" and ph.hub.located("thing-Q4_K_M.gguf") == got
        with pytest.raises(ph.Error):
            ph.models.pull(REPO, "missing.gguf")
    with pytest.raises(ph.Error, match="owner/name"):
        ph.models.pull("not a repo", "x.gguf")


def test_serve_up_status_and_down_under_the_broker_lease(tmp_path, machine):  # noqa: F811
    model = gguf(tmp_path, Q4_K_XL)
    served = ph.serve.up(model, context=4096, reason="the api test")
    assert served.state == "ready" and served.base_url.startswith("http://127.0.0.1:") and served.context == 4096
    assert [s.lease for s in ph.serve.status()] == [served.lease]
    assert ph.serve.up(model, context=4096).adopted
    assert ph.serve.down(Q4_K_XL) == [served.lease] and ph.serve.status() == []
    assert ph.serve.down("nothing-by-this-name") == []
    machine.budget = 256 * 1024 * 1024
    with pytest.raises(ph.Error, match="refused"):
        ph.serve.up(model, context=262144)


def test_the_client_is_the_one_the_docs_name(tmp_path):
    assert not ph.client.is_healthy("http://127.0.0.1:9", timeout=0.2)
    assert issubclass(ph.client.ServerUnreachable, ph.client.ServerError)
    assert ph.client.Request is not None and ph.client.Client("http://127.0.0.1:9") is not None
    assert ph.hub.discover([tmp_path]) == [] and ph.hub.ModelInfo is not None


def test_remote_tests_refuse_what_is_off_and_name_what_is_wrong(mine, tmp_path):
    assert ph.test.devices() == []
    with pytest.raises(ph.Error, match="one of fast"):
        ph.test.run("everything", on="x")
    with pytest.raises(ph.Error, match=r"tests/NAME\.py"):
        ph.test.run("all", on="x", files=("../evil.py",))
    with pytest.raises(ph.Error, match="remote tests are off"):
        ph.test.run("fast", on="x")


def test_the_returned_types_are_the_documented_dataclasses(mine):
    one = mine.member("typed")
    registered = ph.board.register("script")
    assert isinstance(registered, ph.board.Board) and ph.board.register("script").name == registered.name
    registered.send(registered.name, "to myself", kind="note")
    assert isinstance(registered.inbox()[0], ph.board.Message)
    registered.add_note("fact", "T", "body")
    assert isinstance(registered.notes()[0], ph.board.Note)
    assert isinstance(registered.claim("port", "9876"), ph.board.Claim)
    assert all(isinstance(a, ph.board.Agent) for a in registered.agents()) and one.name
    pool = ph.pool.status()
    assert isinstance(pool, ph.pool.Pool) and isinstance(pool.members[0], ph.pool.Member)
    assert isinstance(ph.pool.capacity(), ph.pool.Capacity)
    lease = ph.leases.acquire([ph.leases.cpu_slots(1)])
    assert isinstance(lease, ph.leases.LeaseRecord) and ph.leases.release(lease.id)
    assert {ph.test.Device.__name__, ph.test.Result.__name__, ph.serve.Served.__name__} == {"Device", "Result", "Served"}


def test_adding_a_device_and_syncing_need_the_network_a_person_turned_on(mine):
    with pytest.raises(ph.Error, match="network"):
        ph.pool.add_device()
    with pytest.raises(ph.Error, match="network"):
        ph.pool.sync()
    with pytest.raises(ph.Error, match="network"):
        ph.pool.join("127.0.0.1", "a-code")
