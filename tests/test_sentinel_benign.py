"""Sentinel on benign traces: real documents, steady peers, ordinary file churn. Nothing here
may be quarantined or watched."""

from __future__ import annotations

import os
import random
from pathlib import Path

import pytest

from ml_stack.sentinel import Mode, Sentinel, State
from ml_stack.sentinel.honey import Honey
from ml_stack.sentinel.rates import PeerLimits, PeerWatch

DOCS = Path(__file__).resolve().parent.parent / "docs"
TOOLS = ["models_find", "serve_status", "read_file", "web_search", "bench_run", "jobs_status"]


@pytest.fixture
def node(tmp_path):
    made = Sentinel(tmp_path / "s", mode=Mode.ENFORCE, roots=[tmp_path])
    made.honey = Honey(tmp_path / "home", state=tmp_path / "s" / "honey.json")
    made.honey.plant()
    return made


def test_a_day_of_ordinary_use_quarantines_nothing_even_in_enforce_mode(node, tmp_path):
    rng = random.Random(11)
    paragraphs = [p for doc in sorted(DOCS.glob("*.md")) for p in doc.read_text().split("\n\n")
                  if len(p.split()) >= 8]
    models = tmp_path / "models"
    models.mkdir()
    files = []
    for i in range(6):
        path = models / f"m{i}.gguf"
        path.write_bytes(rng.randbytes(40_000))
        node.manifest.pin(path, "model" if i % 2 else "binary")
        files.append(path)

    clock = [0.0]
    node.peers = PeerWatch(PeerLimits(), clock=lambda: clock[0])
    handled = screened = calls = 0
    for step in range(3000):
        clock[0] += 1.0
        text = rng.choice(paragraphs)
        session = f"s{step % 7}"
        assert node.screen(text, "tool:web_search", session=session).text == text
        screened += 1
        assert node.screen_call(rng.choice(TOOLS), {"path": f"/data/f{step}.txt"},
                                session=session, caller=f"agent-{step % 3}") == ""
        calls += 1
        peer = f"10.0.0.{step % 5}"
        outcome = "bad_sig" if step % 997 == 0 else "ok"
        node.handle_all([node.peers.note(peer, outcome)])
        handled += 1
        if step % 250 == 0:
            victim = rng.choice(files)
            victim.write_bytes(victim.read_bytes())
            os.utime(victim, (clock[0] + 9, clock[0] + 9))
            assert node.scan(deep=step % 500 == 0) == []
    held = node.store.records(state=State.QUARANTINED)
    watched = node.store.records(state=State.WATCH)
    print(f"benign day: {screened} screened, {calls} tool calls, {handled} peer requests, "
          f"{len(held)} quarantined, {len(watched)} watched")
    assert held == []
    assert watched == []
    assert node.status()["log_ok"] is True


def test_peers_that_retry_after_a_slow_start_are_only_watched(tmp_path):
    clock = [0.0]
    watch = PeerWatch(PeerLimits(), clock=lambda: clock[0])
    found = []
    for _ in range(4):
        clock[0] += 2
        found.append(watch.note("10.0.0.9", "bad_sig"))
    for _ in range(200):
        clock[0] += 1
        found.append(watch.note("10.0.0.9", "ok"))
    assert not any(f and f.confidence == "high" for f in found)
