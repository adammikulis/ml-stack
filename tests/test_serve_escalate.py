"""Growing a server's seats without losing a live conversation.

A fake ``ServerBackend`` stands in for the relaunch -- it never binds a socket, so the
same real HTTP server (from the ``server`` fixture) answers before and after, and every
call the manager made is in its ``requests`` log to check. ``fit.json`` is pointed at
``tmp_path`` by the autouse fixture in ``conftest.py``, so a "grow fits" test writes its
own record there rather than reading a real machine's.
"""

from __future__ import annotations

import json
import os
import time
import urllib.parse
from dataclasses import replace

import pytest
from conftest import json_reply
from ml_stack.serve.backend import ServerBackend, ServerFailed, ServerInfo, ServerSpec
from ml_stack.serve.manager import EscalationRefused, ServerManager

MODEL = "quince-2b.gguf"


class FakeBackend(ServerBackend):
    """Records every spec it is asked to start; answers instantly, no process."""

    name = "fake"

    def __init__(self) -> None:
        self.started: list[ServerSpec] = []

    def command(self, spec: ServerSpec) -> list[str]:
        return ["fake"]

    def start(self, spec: ServerSpec, *, lease, timeout: float = 300.0,
              **starting) -> ServerInfo:
        self.started.append(spec)
        return ServerInfo(base_url=f"http://127.0.0.1:{spec.port}", port=spec.port,
                          pid=90000 + len(self.started), backend="fake",
                          load_s=1.5, warmup_s=0.5)


def _slot(id_slot: int, n_ctx: int, tokens: int = 0, prompt: str = "") -> dict:
    row = {"id": id_slot, "n_ctx": n_ctx, "is_processing": False}
    if tokens:
        row["id_task"] = id_slot
        row["n_prompt_tokens"] = tokens
        if prompt:
            row["prompt"] = prompt
    return row


def fake_server(*, slots: list[dict], model: str = MODEL, n_ctx: int = 32768,
               chat_reply: str = "a summary", completion_reply: str = "a summary",
               saves: list, restores: list, completions: list):
    """A handler standing in for a running llama-server, recording save/restore/completion
    calls into the lists given."""

    def handle(method: str, path: str, body: bytes):
        parsed = urllib.parse.urlparse(path)
        parts = [p for p in parsed.path.split("/") if p]
        query = dict(urllib.parse.parse_qsl(parsed.query))

        if parsed.path in ("/health", "/v1/models") and method == "GET":
            if parsed.path == "/health":
                return json_reply({"status": "ok"})
            return json_reply({"data": [{"id": model}]})
        if parsed.path == "/props":
            return json_reply({"model_path": model, "total_slots": len(slots),
                               "default_generation_settings": {"n_ctx": n_ctx}})
        if parsed.path == "/slots" and method == "GET":
            return json_reply(slots)
        if len(parts) == 2 and parts[0] == "slots" and method == "POST":
            id_slot = int(parts[1])
            action = query.get("action", "")
            payload = json.loads(body or b"{}")
            if action == "save":
                saves.append((id_slot, payload.get("filename")))
                return json_reply({"id_slot": id_slot, "filename": payload.get("filename")})
            if action == "restore":
                restores.append((id_slot, payload.get("filename")))
                return json_reply({"id_slot": id_slot, "filename": payload.get("filename")})
            return json_reply({"error": "unknown action"}, status=400)
        if parsed.path == "/v1/chat/completions":
            return json_reply({"model": model, "choices": [
                {"message": {"role": "assistant", "content": chat_reply},
                 "finish_reason": "stop"}]})
        if parsed.path == "/completion":
            payload = json.loads(body or b"{}")
            completions.append(payload)
            # a real seed-the-cache prefill asks for one token; a summary asks for more
            said = completion_reply if int(payload.get("n_predict") or 0) > 1 else ""
            return json_reply({"content": said})
        return json_reply({"error": f"no route for {method} {path}"}, status=404)

    return handle


def write_fit(model: str, *, per_token: int, per_seq: int = 0, room: int = 10**12) -> None:
    """One measured record, at the path the autouse fixture already pointed
    ``$MLSTACK_FIT_FILE`` at."""
    path = os.environ["MLSTACK_FIT_FILE"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = {"model": model, "weights": 0, "room": room, "per_token": per_token,
          "per_seq": per_seq, "compute": 0, "cache_type": "f16"}
    with open(path, "w") as f:
        json.dump([row], f)


class TestGrow:
    def test_grows_when_the_fit_record_says_it_fits(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=5000)]
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")
        info = manager.escalate(current, add_seats=1, room=10**9)

        assert info.pid == 90001
        assert backend.started and backend.started[0].parallel == 2
        assert backend.started[0].context == 65536 * 2, "each seat keeps its size"
        assert backend.started[0].slot_save_path == "slots"
        assert saves == [(0, saves[0][1])]
        assert restores == [(0, saves[0][1])], "restored from the file just saved"

    def test_events_fire_in_order(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=5000)]
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions))
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[str] = []
        manager.escalate(current, add_seats=1, room=10**9,
                        on_event=lambda e: seen.append(e["event"]))

        assert seen == ["escalating", "saving", "stopping", "loading", "ready",
                        "restoring", "done"]

    def test_a_caller_with_no_handler_is_unaffected(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]
        instance = server(fake_server(slots=slots, saves=[], restores=[], completions=[]))
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1, room=10**9)
        assert info.port == instance.port


class TestSplit:
    def test_splits_when_growing_does_not_fit_but_every_slot_is_short_enough(
            self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)  # nothing fits by growing
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=100), _slot(1, 65536, tokens=200)]
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1, room=1)

        assert backend.started[0].parallel == 3
        assert backend.started[0].context == 131072, "the total is unchanged, only split finer"
        assert sorted(saves) == [(0, saves[0][1]), (1, saves[1][1])]
        assert sorted(restores) == sorted(saves)
        assert not completions, "nothing needed summarising"

    def test_no_fit_record_falls_back_to_split_rather_than_claiming_growth_fits(
            self, server, tmp_path):
        """No measurement means no claim that the extra room is there."""
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=100)]
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        info = manager.escalate(current, add_seats=1)

        assert backend.started[0].parallel == 2
        assert backend.started[0].context == 65536, "split, not grown, with nothing measured"


class TestSummarize:
    def test_summarizes_a_slot_too_long_to_split_then_restores_the_summary(
            self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)  # growing never fits
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=60000)]  # would not fit a 32768-token half
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions, chat_reply="short summary"))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[dict] = []
        info = manager.escalate(current, add_seats=1, room=1, on_event=seen.append)

        assert backend.started[0].parallel == 2
        assert saves == [(0, saves[0][1])], "the full cache is still saved"
        assert not restores, "a summary was re-seeded instead of the saved cache"
        assert completions and completions[0]["prompt"] == "short summary"
        events = [e["event"] for e in seen]
        assert events == ["escalating", "saving", "summarizing", "summarized", "stopping",
                          "loading", "ready", "restoring", "done"]

    def test_a_slots_debug_prompt_is_summarised_as_a_cached_continuation_not_cold(
            self, server, tmp_path):
        """A bare instruction gives a slot's cache nothing to read -- driven for real,
        this is the difference between a useless summary and a real one. With the
        prompt text `/slots` carries under LLAMA_SERVER_SLOTS_DEBUG, the request is a
        continuation of it, not a fresh question with no conversation attached."""
        write_fit(MODEL, per_token=10, per_seq=0, room=1)
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=60000, prompt="the garden plan so far")]
        instance = server(fake_server(slots=slots, saves=saves, restores=restores,
                                      completions=completions))
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        seen: list[dict] = []
        manager.escalate(current, add_seats=1, room=1, on_event=seen.append)

        from ml_stack.serve.manager import SUMMARY_SUFFIX

        prompts_sent = [c["prompt"] for c in completions if "prompt" in c]
        assert any(p == "the garden plan so far" + SUMMARY_SUFFIX for p in prompts_sent), (
            "the summary request must be the slot's own prompt plus the ask, so the "
            "shared prefix is a cache hit rather than the model being asked cold")

    def test_refuses_rather_than_truncates_when_summarising_fails(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=1)
        saves, restores, completions = [], [], []
        slots = [_slot(0, 65536, tokens=60000)]

        def broken_chat(method, path, body):
            if path == "/v1/chat/completions":
                return json_reply({"error": "model exploded"}, status=500)
            return fake_server(slots=slots, saves=saves, restores=restores,
                               completions=completions)(method, path, body)

        instance = server(broken_chat)
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1,
                             slot_save_path="slots")

        with pytest.raises(EscalationRefused, match="cache is kept at"):
            manager.escalate(current, add_seats=1, room=1)
        assert saves, "the cache was saved before the failed summary, and is not lost"


class TestPreconditions:
    def test_refuses_without_a_slot_save_path(self, server, tmp_path):
        instance = server(fake_server(slots=[_slot(0, 65536)], saves=[], restores=[],
                                      completions=[]))
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")
        current = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=1)

        with pytest.raises(ServerFailed, match="slot-save-path"):
            manager.escalate(current, add_seats=1)

    def test_a_restore_failure_is_surfaced_not_swallowed(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]

        def handle(method, path, body):
            parsed = urllib.parse.urlparse(path)
            if parsed.path.startswith("/slots/") and "action=restore" in path:
                return json_reply({"error": "disk full"}, status=500)
            return fake_server(slots=slots, saves=[], restores=[], completions=[])(
                method, path, body)

        instance = server(handle)
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
    def test_lease_escalates_instead_of_refusing_when_asked_to(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 65536, tokens=100)]
        instance = server(fake_server(slots=slots, saves=[], restores=[], completions=[]))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2,
                            slot_save_path="slots")
        info = manager.lease(wanted, escalate=True)

        assert backend.started and backend.started[0].parallel == 2

    def test_lease_without_escalate_still_refuses(self, server, tmp_path):
        slots = [_slot(0, 65536, tokens=100)]
        instance = server(fake_server(slots=slots, saves=[], restores=[], completions=[]))
        manager = ServerManager(FakeBackend(), state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=131072, parallel=2)
        with pytest.raises(ServerFailed, match="slots: asked for 2, serving 1"):
            manager.lease(wanted, roam=False)

    def test_escalate_true_fills_in_a_default_save_path(self, server, tmp_path):
        write_fit(MODEL, per_token=10, per_seq=0, room=10**9)
        slots = [_slot(0, 32768, tokens=10)]
        instance = server(fake_server(slots=slots, saves=[], restores=[], completions=[]))
        backend = FakeBackend()
        manager = ServerManager(backend, state_file=tmp_path / "servers.json")

        wanted = ServerSpec(model=MODEL, port=instance.port, context=65536, parallel=2)
        manager.lease(wanted, escalate=True)

        assert backend.started[0].slot_save_path
