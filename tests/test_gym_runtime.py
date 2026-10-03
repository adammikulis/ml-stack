"""Gym session lifecycle and observation/action alignment."""

import json
import queue
import time
from types import SimpleNamespace

import pytest

from ml_stack.gym import adapters, catalog, simulation as runtime
from ml_stack.gym.cli import argument_parser
from ml_stack.gym.runtime import SessionManager


@pytest.fixture
def simulation(monkeypatch, tmp_path):
    class Environment:
        np_random = SimpleNamespace(integers=lambda size: size - 1)

        def reset(self, seed=None):
            self.value = seed
            return [seed], {}

        def step(self, action):
            self.value += action
            return [self.value], 2., False, self.value >= 5, {"native": True}

    monkeypatch.setattr(runtime, "artifact_root", lambda: tmp_path)
    monkeypatch.setattr(runtime, "make_environment", lambda *_: Environment())
    monkeypatch.setattr(runtime, "actions", lambda *_: (["hold", "move"], [0, 1]))
    monkeypatch.setattr(runtime, "render_state", lambda *_: ({"actual": True}, None))
    return runtime.Simulation({"id": "episode", "environment": "warehouse", "config": {},
                               "seed": 2, "controller": "manual"})


def test_transition_matches_controller_input_and_native_output(simulation):
    simulation.command("action", {"action": 1})
    simulation.step()
    state = simulation.state
    assert state["decision"]["observation"] == [2]
    assert state["observation"] == [3]
    assert state["action"] == 1
    assert state["transition"] == {"observation": [2], "action": 1, "reward": 2.,
                                    "next_observation": [3], "terminated": False,
                                    "truncated": False, "episode_id": 1, "sequence": 2}
    recording = json.loads((simulation.path / "trajectory.jsonl").read_text())
    assert recording["transition"] == state["transition"]


def test_reset_starts_an_episode_and_preserves_sequence(simulation):
    simulation.step()
    previous = simulation.state["sequence"]
    simulation.command("reset", {"seed": 9})
    assert simulation.state["sequence"] == previous + 1
    assert simulation.state["episode_id"] == 2
    assert simulation.state["observation"] == [9]
    assert simulation.state["decision"] is None


def test_completed_episode_cannot_advance(simulation):
    simulation.command("action", {"action": 1})
    for _ in range(3):
        simulation.step()
    assert simulation.state["truncated"]
    sequence = simulation.state["sequence"]
    simulation.step()
    assert simulation.state["sequence"] == sequence
    with pytest.raises(ValueError, match="Reset"):
        simulation.command("play", {})


def test_invalid_action_does_not_replace_manual_action(simulation):
    simulation.command("action", {"action": 1})
    with pytest.raises(ValueError, match="Invalid manual"):
        simulation.command("action", {"action": 5})
    assert simulation.manual == 1


def test_catalogue_missing_dependencies_and_unknown_environment(monkeypatch):
    monkeypatch.delenv("ML_STACK_GYM_PYTHON", raising=False)
    monkeypatch.setattr(catalog, "find_spec", lambda _: None)
    monkeypatch.setattr(catalog.shutil, "which", lambda _: None)
    entries = catalog.catalogue()
    assert {entry["id"] for entry in entries} == {"car", "warehouse", "traffic", "traffic-driving"}
    assert all(not entry["available"] for entry in entries)
    with pytest.raises(RuntimeError, match="gym-driving"):
        catalog.require("car")
    with pytest.raises(ValueError, match="Unknown"):
        catalog.require("invented")


def test_publish_replaces_old_snapshot():
    updates = queue.Queue(maxsize=1)
    updates.put({"sequence": 1})
    runtime.publish(updates, {"sequence": 2})
    assert updates.get_nowait() == {"sequence": 2}


def test_manual_action_bounds_and_unknown_controller():
    choices = ["hold", "move"], [0, 1], None
    with pytest.raises(ValueError, match="outside"):
        runtime.select_action("manual", [1], choices, -1)
    with pytest.raises(ValueError, match="Unknown controller"):
        runtime.select_action("invented", [1], choices, 0)


def test_decider_abstention_holds_with_probability_telemetry():
    decision = SimpleNamespace(scores={"hold": .51, "move": .49}, model="model", backend="pointer",
                               abstained=True, choice="hold", latency_ms=3.)
    policy = SimpleNamespace(decide=lambda *_: decision)
    with pytest.raises(runtime.DecisionHeld) as held:
        runtime.select_action("decider", [2], (["hold", "move"], [0, 1], None), 0, policy)
    assert held.value.decision["observation"] == [2]
    assert held.value.decision["probabilities"] == decision.scores


def test_decision_controller_uses_cpu(simulation, monkeypatch):
    requests = []
    monkeypatch.setattr(runtime, "PointerDecider", lambda **kwargs: requests.append(kwargs))
    simulation.command("controller", {"controller": "decider"})
    assert requests == [{"device": "cpu"}]


def test_cli_exposes_native_configuration_and_resume():
    args = argument_parser().parse_args(["train", "warehouse", "--timesteps", "512",
                                         "--checkpoint", "policy.zip", "--config", '{"max_steps": 10}'])
    assert args.timesteps == 512
    assert args.checkpoint == "policy.zip"
    assert json.loads(args.config) == {"max_steps": 10}


def test_json_value_converts_nested_native_values():
    value = SimpleNamespace(tolist=lambda: [1, 2])
    assert adapters.json_value({"data": (value,)}) == {"data": [[1, 2]]}


def test_warehouse_native_adapter():
    pytest.importorskip("rware")
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.env_checker import check_env
    env = adapters.make_environment("warehouse", {"max_steps": 3})
    try:
        check_env(env, warn=False)
        observation, _ = env.reset(seed=2)
        assert env.observation_space.contains(observation)
        names, native = adapters.actions("warehouse", env)
        assert len(names) == len(native) == 25
        _, reward, _, _, _ = env.step(native[0])
        assert isinstance(reward, float)
        geometry, frame = adapters.render_state("warehouse", env)
        assert len(geometry["robots"]) == 2
        assert frame is None
    finally:
        env.close()


@pytest.mark.slow
def test_external_worker_lifecycle_and_recording():
    pytest.importorskip("rware")
    manager = SessionManager()
    identifier = manager.create("warehouse", {"max_steps": 3})["id"]
    try:
        deadline = time.monotonic() + 10
        snapshot = manager.get(identifier)
        while snapshot["status"] == "starting" and time.monotonic() < deadline:
            time.sleep(.05)
            snapshot = manager.get(identifier)
        assert snapshot["status"] == "paused", snapshot
        sequence = snapshot["sequence"]
        manager.control(identifier, "step")
        while snapshot["sequence"] == sequence and time.monotonic() < deadline:
            time.sleep(.05)
            snapshot = manager.get(identifier)
        assert snapshot["transition"]["sequence"] == sequence + 1
        assert snapshot["decision"]["observation"] == snapshot["transition"]["observation"]
    finally:
        assert manager.close(identifier)["status"] == "closed"
    assert identifier not in manager.sessions
