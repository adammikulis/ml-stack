"""The native worker receives one complete bounded task rather than an inbox preview."""
import threading

from workspace_kit import Kit, clean_env

from poolhouse.workspace import localagent as la, localloop


def test_worker_receives_constraints_after_a_long_task(monkeypatch, tmp_path):
    kit = Kit(clean_env(monkeypatch, tmp_path))
    parent = kit.agent("lead")
    child = kit.ws.delegate(parent, "worker")
    agent = la.Agent("worker", "local.gguf", identity=child["id"], profile="coding", orders_from=("lead",))
    la.save(kit.ws, agent)
    payload = "Read the source. " * 300 + "REQUIRED: no unbrokered tests"
    kit.ws.send(parent, child["id"], "task", payload)
    stopped = threading.Event()
    seen = []

    def execute(_, row, why, stop):
        seen.append(row["text"])
        stopped.set()
        return "answer", "done", 1

    settings = localloop.Settings(cancel=stopped, serve=lambda _: localloop.Held(None, {}), execute=execute)
    assert localloop.run(kit.ws, agent.name, settings) == 0
    assert len(seen) == 1 and "REQUIRED: no unbrokered tests" in seen[0]
