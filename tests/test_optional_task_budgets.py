"""Optional task limits, saved defaults and independent cancellation."""

import pytest

from ml_stack.workspace import localagent as la, localloop, localtools


def test_task_caps_default_unlimited_and_legacy_profiles_migrate():
    from dataclasses import asdict
    for profile, legacy in (("chat", {"rounds": 12, "calls": 30, "steps": 24, "seconds": 600.0}),
                            ("coding", {"rounds": 60, "calls": 150, "steps": 120, "seconds": 3600.0})):
        agent = la.Agent("worker", "model", profile=profile, extra={"task_caps": legacy})
        assert all(value is None for value in asdict(localloop.caps_of(agent)).values())
        agent.extra["task_caps"] = {**legacy, "seconds": 7200}
        assert localloop.caps_of(agent).seconds == 7200
    assert localloop.checked_caps({"rounds": 1000, "calls": 10000, "steps": 5000, "seconds": 200000}).calls == 10000
    for bad in ({"calls": True}, {"seconds": float("inf")}, {"steps": 1.5}, {"rounds": 0}):
        with pytest.raises(ValueError):
            localloop.checked_caps(bad)


def test_unlimited_guard_survives_time_and_calls_but_honors_cancel(monkeypatch):
    stopped = [False]
    from types import SimpleNamespace
    client = SimpleNamespace(chat=lambda *args, **kwargs: kwargs)
    guarded = localtools.Guarded(client, effort="off", limits=localtools.Limits(None, None),
                                  stop=lambda: stopped[0])
    monkeypatch.setattr(localtools.time, "monotonic", lambda: 1e12)
    for _ in range(100):
        assert guarded.chat([]) == {"think": False}
    stopped[0] = True
    with pytest.raises(localtools.TaskStopped, match="person"):
        guarded.chat([])

def test_context_sizes_parse_k_and_profiles_carry_their_caps():
    from ml_stack.workspace import localprofile as lp
    assert lp.parse_ctx("256k") == lp.parse_ctx("256K") == 262144
    assert lp.parse_ctx("32768") == 32768 and lp.parse_ctx("") == lp.CODING.ctx == lp.CHAT.ctx == 0
    for bad in ("big", "1", "-5k", "256 k"):
        with pytest.raises(ValueError):
            lp.parse_ctx(bad)
    assert lp.CODING.rounds is lp.CHAT.rounds is lp.CODING.seconds is lp.CHAT.seconds is None
    assert localloop.caps_of(la.Agent(name="a", model="m", profile="coding")).calls == lp.CODING.calls



def test_saved_explicit_old_default_values_remain_finite(tmp_path):
    from ml_stack.workspace.service import Workspace
    ws = Workspace(tmp_path)
    agent = la.Agent("worker", "model", max_output_tokens=8192,
                     extra={"explicit_output_limit": True, "explicit_task_limits": True,
                            "task_caps": {"rounds": 12, "calls": 30, "steps": 24, "seconds": 600.0}})
    la.save(ws, agent)
    saved = la.load(ws, agent.name)
    assert saved.max_output_tokens == 8192
    assert localloop.caps_of(saved).seconds == 600
    agent.extra.pop("explicit_output_limit")
    agent.extra.pop("explicit_task_limits")
    la.save(ws, agent)
    saved = la.load(ws, agent.name)
    assert saved.max_output_tokens is None
    assert localloop.caps_of(saved).seconds is None


@pytest.mark.parametrize("cause", ["cancel", "wall"])
def test_guarded_none_transport_interrupts_before_response_headers(cause):
    import threading

    from test_http_cancel import pending_headers

    from ml_stack.client import Client, Transport, families
    stopped = threading.Event()
    with pending_headers() as (url, received, closed):
        client = Client(url.removesuffix("/v1/chat/completions"), family=families.GENERIC,
                        transport=Transport(timeout=None))
        guarded = localtools.Guarded(client, effort="off",
            limits=localtools.Limits(0.2 if cause == "wall" else None, None), stop=stopped.is_set)
        failures = []
        def run():
            try:
                guarded.chat([{"role": "user", "content": "wait"}])
            except localtools.TaskStopped as error:
                failures.append(str(error))
        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        assert received.wait(2)
        if cause == "cancel":
            stopped.set()
        worker.join(1)
        assert not worker.is_alive()
        assert closed.wait(1)
        assert failures and ("person" if cause == "cancel" else "limit") in failures[0]


def test_optional_wall_limits_preserve_recovery_bounds_and_counter_schema(tmp_path):
    import json

    from ml_stack.workspace import task_caps, task_schema
    spec = {"title": "Inspect queue", "acceptance": ["Report queue state"]}
    for wall in (None, 200000):
        assert task_schema.task_spec({**spec, "limits": {"max_wall_s": wall}})["limits"]["max_wall_s"] == wall
    with pytest.raises(ValueError, match="max_retries"):
        task_schema.task_spec({**spec, "limits": {"max_retries": None}})
    path = tmp_path / "counter.json"
    for value in ([], {"version": 1, "calls": 0}, {"version": True, "calls": 0, "limit": None}):
        path.write_text(json.dumps(value))
        with pytest.raises(ValueError):
            task_caps.admit(path)
