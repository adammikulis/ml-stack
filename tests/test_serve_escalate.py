"""Growing a server's seats without losing a live conversation.

`FakeBackend` stands in for the relaunch -- it never binds a socket, so the same real
llama-server (`FakeLlamaServer`, on a real port) answers before and after, and every call
the manager made is in its ``requests`` log to check. ``fit.json`` is pointed at
``tmp_path`` by the autouse fixture in ``conftest.py``, so a "grow fits" test writes its
own record there rather than reading a real machine's.
"""

from __future__ import annotations

import json
import os

import pytest
from ml_stack import home
from ml_stack.serve.backend import ServerFailed, ServerSpec
from ml_stack.serve.manager import EscalationRefused, ServerManager
from ml_stack.testing.fakes import FakeBackend, FakeLlamaServer, Served

MODEL = "quince-2b.gguf"
SUMMARY = "a summary"


def _slot(id_slot: int, n_ctx: int, tokens: int = 0, prompt: str = "") -> dict:
    row = {"id": id_slot, "n_ctx": n_ctx, "is_processing": False}
    if tokens:
        row["id_task"] = id_slot
        row["n_prompt_tokens"] = tokens
        if prompt:
            row["prompt"] = prompt
    return row


@pytest.fixture
def serving():
    """A llama-server on a real socket holding the slots the test hands it."""
    started: list[FakeLlamaServer] = []

    def start(slots: list[dict], *, chat_reply: str = SUMMARY) -> FakeLlamaServer:
        def said(body: dict) -> str:
            if "messages" in body:
                return chat_reply
            # a real seed-the-cache prefill asks for one token; a summary asks for more
            return SUMMARY if int(body.get("n_predict") or 0) > 1 else ""

        fake = FakeLlamaServer(Served(model=MODEL, context=32768, slots=len(slots),
                                      answer=said))
        fake.slots = list(slots)
        started.append(fake)
        return fake

    yield start
    for fake in started:
        fake.close()


def write_fit(model: str, *, per_token: int, per_seq: int = 0, room: int = 10**12) -> None:
    """One measured record, under the state root the autouse fixture points at."""
    path = home.state("fit.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"model": model, "weights": 0, "room": room, "per_token": per_token,
          "per_seq": per_seq, "compute": 0, "cache_type": "f16"}
    with open(path, "w") as f:
        json.dump([row], f)


class TestGrow:
    def test_grows_when_the_fit_record_says_it_fits(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=5000)]
        instance = serving(slots)
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")
        info = manager.escalate(current, add_seats=1, room=10**9)

        assert info.pid == 90001
        assert backend.started and backend.started[0].parallel == 2
        assert backend.started[0].context == 65536 * 2, "each seat keeps its size"
        assert backend.started[0].slot_save_path == "slots"
        assert instance.saved == [(0, instance.saved[0][1])]
        assert instance.restored == instance.saved, "restored from the file just saved"

    def test_events_fire_in_order(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=5000)]
        instance = serving(slots)
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[str] = []
        manager.escalate(current, add_seats=1, room=10**9,
                        on_event=lambda e: seen.append(e["event"]))

        assert seen == ["escalating", "saving", "stopping", "loading", "ready",
                        "restoring", "done"]

    def test_a_caller_with_no_handler_is_unaffected(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]
        instance = serving(slots)
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1, room=10**9)
        assert info.port == instance.port


class TestSplit:
    def test_splits_when_growing_does_not_fit_but_every_slot_is_short_enough(
            self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)  # nothing fits by growing
        slots = [_slot(0, 65536, tokens=100), _slot(1, 65536, tokens=200)]
        instance = serving(slots)
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1, room=1)

        assert backend.started[0].parallel == 3
        assert backend.started[0].context == 131072, "the total is unchanged, only split finer"
        assert sorted(instance.saved) == instance.saved and len(instance.saved) == 2
        assert sorted(instance.restored) == sorted(instance.saved)
        assert not instance.sent_to("/completion"), "nothing needed summarising"

    def test_no_fit_record_falls_back_to_split_rather_than_claiming_growth_fits(
            self, serving, tmp_path):
        """No measurement means no claim that the extra room is there."""
        slots = [_slot(0, 65536, tokens=100)]
        instance = serving(slots)
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1)

        assert backend.started[0].parallel == 2
        assert backend.started[0].context == 65536, "split, not grown, with nothing measured"


class TestSummarize:
    def test_summarizes_a_slot_too_long_to_split_then_restores_the_summary(
            self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)  # growing never fits
        slots = [_slot(0, 65536, tokens=60000)]  # would not fit a 32768-token half
        instance = serving(slots, chat_reply="short summary")
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[dict] = []
        info = manager.escalate(current, add_seats=1, room=1, on_event=seen.append)

        assert backend.started[0].parallel == 2
        assert instance.saved == [(0, instance.saved[0][1])], "the full cache is still saved"
        assert not instance.restored, "a summary was re-seeded instead of the saved cache"
        sent = instance.sent_to("/completion")
        assert sent and sent[0]["prompt"] == "short summary"
        events = [e["event"] for e in seen]
        assert events == ["escalating", "saving", "summarizing", "summarized", "stopping",
                          "loading", "ready", "restoring", "done"]

    def test_a_slots_debug_prompt_is_summarised_as_a_cached_continuation_not_cold(
            self, serving, tmp_path):
        """A bare instruction gives a slot's cache nothing to read -- driven for real,
        this is the difference between a useless summary and a real one. With the
        prompt text `/slots` carries under LLAMA_SERVER_SLOTS_DEBUG, the request is a
        continuation of it, not a fresh question with no conversation attached."""
        write_fit(MODEL, per_token=10, per_seq=0, room=1)
        slots = [_slot(0, 65536, tokens=60000, prompt="the garden plan so far")]
        instance = serving(slots)
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[dict] = []
        manager.escalate(current, add_seats=1, room=1, on_event=seen.append)

        from ml_stack.serve.manager import SUMMARY_SUFFIX

        prompts_sent = [c["prompt"] for c in instance.sent_to("/completion")
                        if "prompt" in c]
        assert any(p == "the garden plan so far" + SUMMARY_SUFFIX for p in prompts_sent), (
            "the summary request must be the slot's own prompt plus the ask, so the "
            "shared prefix is a cache hit rather than the model being asked cold")

    def test_refuses_rather_than_truncates_when_summarising_fails(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)
        slots = [_slot(0, 65536, tokens=60000)]

        instance = serving(slots)
        instance.refuse["/v1/chat/completions"] = 500
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        with pytest.raises(EscalationRefused, match="cache is kept at"):
            manager.escalate(current, add_seats=1, room=1)
        assert instance.saved, ("the cache was saved before the failed summary, "
                                "and is not lost")


class TestPreconditions:
    def test_refuses_without_a_slot_save_path(self, serving, tmp_path):
        instance = serving([_slot(0, 65536)])
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1)

        with pytest.raises(ServerFailed, match="slot-save-path"):
            manager.escalate(current, add_seats=1)

    def test_a_restore_failure_is_surfaced_not_swallowed(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]

        instance = serving(slots)
        instance.refuse["/slots/0?action=restore"] = 500
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        with pytest.raises(ServerFailed, match="could not restore slot 0"):
            manager.escalate(current, add_seats=1, room=10**9)

        assert manager._recorded_pid(instance.port) is not None, (
            "the relaunched server is recorded even though a restore failed -- a "
            "healthy process this manager forgot it started is the bug the pending "
            "lease exists to rule out")


class TestLeaseEscalatesOnItsOwn:
    def test_lease_escalates_instead_of_refusing_when_asked_to(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]
        instance = serving(slots)
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2,
                            slot_save_path="slots")
        info = manager.lease(wanted, escalate=True)

        assert backend.started and backend.started[0].parallel == 2

    def test_lease_without_escalate_still_refuses(self, serving, tmp_path):
        slots = [_slot(0, 65536, tokens=100)]
        instance = serving(slots)
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2)
        with pytest.raises(ServerFailed, match="slots: asked for 2, serving 1"):
            manager.lease(wanted, roam=False)

    def test_escalate_true_fills_in_a_default_save_path(self, serving, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 32768, tokens=10)]
        instance = serving(slots)
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=2)
        manager.lease(wanted, escalate=True)

        assert backend.started[0].slot_save_path
