"""Live CPU policy updates and persistent native world ownership."""

import json
import queue

import pytest

from ml_stack.gym import simulation
from ml_stack.gym.live_learning import learn_rollout


@pytest.mark.slow
def test_online_ppo_changes_weights_and_frozen_mode_preserves_them(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("rware")
    torch.set_num_threads(1)
    monkeypatch.setattr(simulation, "artifact_root", lambda: tmp_path)
    live = simulation.Simulation({"id": "live", "environment": "warehouse", "seed": 11,
        "controller": "ppo", "config": {"simulation_mode": "world", "learning_mode": "online",
                                         "task_horizon": 3, "world": {"seed": 11}}})
    try:
        original = [p.detach().clone() for p in live.policy.policy.parameters()]
        actors = tuple(map(id, live.env.native.agents))
        live.command("speed", {"speed": 60})
        live.command("play", {})
        assert learn_rollout(live, queue.Queue(), queue.Queue(), lambda *_: None)
        learned = [p.detach().clone() for p in live.policy.policy.parameters()]
        assert any(not torch.equal(a, b) for a, b in zip(original, learned, strict=True))
        assert live.state["policy_version"] == 1
        assert live.state["optimizer_updates"] == 4
        assert live.state["training_timesteps"] == 64
        assert live.state["info"]["world_steps"] == 64
        assert actors == tuple(map(id, live.env.native.agents))
        assert live.state["info"]["world_reset_count"] == 1
        assert live.state["episode_id"] > 1
        assert (live.path / "policy.zip").is_file()
        live.command("learning", {"mode": "frozen"})
        for _ in range(6):
            live.step()
        assert all(torch.equal(a, b) for a, b in zip(learned, live.policy.policy.parameters(), strict=True))
        assert live.state["info"]["world_steps"] == 70
        records = [json.loads(line) for line in (live.path / "trajectory.jsonl").read_text().splitlines()]
        assert len(records) == 70
        assert records[0]["decision"]["state"]["native_observation"] == records[0]["transition"]["observation"]
        assert any(row["transition"]["truncated"] for row in records)
        assert records[-1]["learning_mode"] == "frozen"
    finally:
        live.env.close()


@pytest.mark.slow
def test_online_pause_and_freeze_interrupt_before_optimizer_update(tmp_path, monkeypatch):
    pytest.importorskip("stable_baselines3")
    pytest.importorskip("rware")
    monkeypatch.setattr(simulation, "artifact_root", lambda: tmp_path)
    live = simulation.Simulation({"id": "interrupt", "environment": "warehouse", "seed": 2,
        "controller": "ppo", "config": {"simulation_mode": "world", "learning_mode": "online"}})
    try:
        commands = queue.Queue()
        commands.put(("pause", {}))
        commands.put(("learning", {"mode": "frozen"}))
        live.command("play", {})
        assert learn_rollout(live, commands, queue.Queue(), lambda *_: None)
        assert live.state["info"]["world_steps"] == 1
        assert not live.running
        assert live.state["optimizer_updates"] == 0
        assert live.state["policy_version"] == 0
        live.command("play", {})
        live.step()
        assert live.state["info"]["world_steps"] == 2
    finally:
        live.env.close()
