"""The local agent under attack: every injection a person's or stranger's message can carry, a model
that obeys all of it, the floor the model has no tool for, and the routes that start and stop.
The loop, the caps and the routes are exercised in test_workspace_local_agent.py. Run with --redteam."""

from __future__ import annotations

import pytest
from test_workspace_local_agent import Rig, Script, done, kit, waited  # noqa: F401
from workspace_kit import cli

from poolhouse import roles
from poolhouse.testing import injection_corpus as corpus
from poolhouse.workspace import localagent as la, localtools

pytestmark = pytest.mark.redteam
TEXTS = sorted({*corpus.INJECTIONS})[:12]
FLOOR = ("start_agent", "stop_agent", "set_role", "set_ceiling", "raise_limits", "mint",
         "quarantine_release", "approve_host", "add_rule", "agent_start", "agent_stop")


@pytest.mark.parametrize("text", TEXTS)
def test_an_injection_from_a_sender_the_agent_does_not_obey_reaches_no_model(kit, text):  # noqa: F811
    model = Script([done("never")])
    with Rig(kit, model) as rig:
        kit.ws.send(rig.tok["bob"], "local-t", "task", text)
        waited(lambda: la.status_of(kit.ws, "local-t").get("ignored"))
    assert model.calls == 0


@pytest.mark.parametrize("text", TEXTS)
def test_an_injection_in_a_persons_task_gets_a_model_that_obeys_it_nothing_beyond_its_role(kit, text):  # noqa: F811
    obeyed = [(name, {}) for name in FLOOR] + [("serve_up", {"model": "hf:evil/x.gguf"}),
                                              ("bench_run", {"argv": ["sweep"]}), done("ok")]
    model = Script(obeyed)
    with Rig(kit, model, role=roles.DEFAULT) as rig:
        sent = kit.ws.send(kit.owner, "local-t", "task", "summarise status. " + text)
        waited(lambda: (st := la.status_of(kit.ws, "local-t")).get("ignored") or st.get("tasks"))
    assert sent and not kit.ws.registry.role_of("local-t-2")
    assert la.load(kit.ws, "local-t").role == roles.DEFAULT
    assert all(n.kind in ("tool_call", "plan") for n in rig.asked)


def test_no_tool_the_model_is_offered_starts_stops_or_widens_an_agent(kit):  # noqa: F811
    state = localtools.TaskState()
    ext = localtools.workspace_extension(kit.ws, kit.agent("x1"), "x1", state, lambda s: False)
    names = {s["function"]["name"] for s, _ in ext.tools()}
    assert not names & set(FLOOR)
    for role in roles.ROLES.values():
        assert not ext.allowed(role) & set(FLOOR)


def test_start_and_stop_refuse_an_agent_process_and_leave_the_state_alone(kit):  # noqa: F811
    before = sorted(p.name for p in kit.base.rglob("*") if p.suffix not in (".log", ".jsonl", ".lock"))
    for argv in (("agent", "start"), ("agent", "stop", "local-x")):
        done_ = cli(kit.base, kit.agent("sneaky" + argv[1]), *argv, env_extra={"CLAUDECODE": "1"})
        assert done_.returncode == 3
    after = sorted(p.name for p in kit.base.rglob("*") if p.suffix not in (".log", ".jsonl", ".lock"))
    assert [n for n in after if n not in before] == [] or all(n.startswith("sneaky") or n.endswith(".tmp") for n in after if n not in before)
