"""Native simulation workers and recorded controller telemetry."""

import base64
import json
import logging
import queue
import time
from pathlib import Path

from ml_stack.decide.pins import STRANDS
from ml_stack.decide.pointer import PointerDecider
from ml_stack.decide.sources import local_source
from ml_stack.gym.adapters import actions, json_value, make_environment, render_state
from ml_stack.gym.observations import decision_state
from ml_stack.gym.paths import artifact_root
from ml_stack.gym.provenance import native_provenance
from ml_stack.gym.training import load_policy


class DecisionHeld(RuntimeError):
    """A controller decision held without advancing the simulator."""

    def __init__(self, decision):
        super().__init__("Decision model abstained; simulation paused")
        self.decision = decision


def decision_controller(checkpoint=None):
    """Load a default or verified trained pointer decider on the CPU."""
    if not checkpoint:
        return PointerDecider(device="cpu")
    source = Path(checkpoint).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("Decision checkpoint must be a local trained model directory")
    local_source(source, download=False)
    return PointerDecider(source, device="cpu")


def select_action(controller, observation, choices, manual, policy=None):
    """Select an action and record its decision inputs."""
    names, native_actions, rng, *named = choices
    began = time.perf_counter()
    detail = {"observation": json_value(observation), "probabilities": None}
    detail["state"] = named[0] if named else detail["observation"]
    if controller == "manual":
        selected = manual
    elif controller == "random":
        selected = int(rng.integers(len(names)))
    elif controller == "ppo":
        if policy is None:
            raise ValueError("PPO controller requires a checkpoint")
        action, _ = policy.predict(observation, deterministic=True)
        native = tuple(map(int, action)) if hasattr(action, "shape") and action.shape else int(action)
        selected = native_actions.index(native)
    elif controller == "decider":
        if policy is None:
            raise ValueError("Decision controller is not loaded")
        answer = policy.decide("Choose the next safe environment action", detail["state"], names)
        detail.update(probabilities=dict(answer.scores), model=answer.model, backend=answer.backend,
                      abstained=answer.abstained)
        if answer.abstained:
            detail.update(choice=answer.choice, latency_ms=answer.latency_ms)
            raise DecisionHeld(detail)
        selected = names.index(answer.choice)
    else:
        raise ValueError(f"Unknown controller: {controller}")
    if not 0 <= selected < len(names):
        raise ValueError("Action is outside the environment action space")
    detail.update(choice=names[selected], latency_ms=(time.perf_counter() - began) * 1000)
    return native_actions[selected], detail


class Simulation:
    """Advance one native environment and capture its controller events."""

    def __init__(self, settings):
        self.environment = settings["environment"]
        self.seed = settings["seed"]
        self.controller = settings["controller"]
        if self.environment == "car":
            settings["config"].setdefault("steering_magnitude", .35)
        self.running, self.speed, self.manual, self.policy = False, 10., 3 if self.environment == "car" else 0, None
        decision_checkpoint = settings["config"].get("decision_checkpoint")
        if self.controller == "decider":
            self.policy = decision_controller(decision_checkpoint)
            settings["model"] = str(decision_checkpoint or STRANDS)
            settings["device"] = "cpu"
        self.settings = settings
        self.path = artifact_root() / settings["id"]
        self.path.mkdir(parents=True, exist_ok=True)
        config = dict(settings["config"])
        self.continuous = config.pop("continuous", False)
        self.new_scenario = config.pop("new_scenario", True)
        if not isinstance(self.continuous, bool) or not isinstance(self.new_scenario, bool):
            raise ValueError("continuous and new_scenario must be booleans")
        self.record_frames = bool(config.pop("record_frames", False))
        checkpoint = config.pop("checkpoint", None)
        config.pop("decision_checkpoint", None)
        if self.environment == "car":
            config.setdefault("render_preview", True)
        self.env = make_environment(self.environment, config)
        settings.update(native_provenance(self.environment, self.env))
        settings.setdefault("model", checkpoint if self.controller == "ppo" else None)
        settings.setdefault("device", "cpu")
        (self.path / "manifest.json").write_text(json.dumps(settings))
        if self.controller == "ppo":
            if not checkpoint:
                self.env.close()
                raise ValueError("PPO controller requires a checkpoint in the session configuration")
            self.policy = load_policy(checkpoint, self.env)
        self.names, self.native_actions = actions(self.environment, self.env)
        self.state = {"id": settings["id"], "environment": self.environment, "status": "starting",
                      "sequence": 0, "episode_id": 0, "controller": self.controller, "observation": None,
                      "action": None, "decision": None, "reward": 0., "terminated": False,
                      "truncated": False, "info": {}, "frame": None, "error": None,
                      "actions": self.names, "config": dict(settings["config"]), "seed": self.seed,
                      "model": settings["model"], "device": settings["device"],
                      "manual_action": self.manual, "speed": self.speed,
                      "continuous": self.continuous, "new_scenario": self.new_scenario,
                      "episode_steps": 0, "episode_reward": 0., "last_episode": None,
                      "trajectory_path": str(self.path / "trajectory.jsonl")}
        self.reset({})

    def reset(self, payload):
        self.running = False
        seed = int(payload.get("seed", self.seed))
        self.observation, info = self.env.reset(seed=seed)
        self.seed = seed
        self.settings["seed"] = seed
        self.state["seed"] = seed
        (self.path / "manifest.json").write_text(json.dumps(self.settings))
        self.state["episode_id"] += 1
        self.state["sequence"] += 1
        self.state.update(status="paused", observation=json_value(self.observation), info=json_value(info),
                          reward=0., action=None, decision=None, terminated=False, truncated=False, error=None,
                          episode_steps=0, episode_reward=0.)
        self.state.pop("transition", None)
        geometry, frame = render_state(self.environment, self.env)
        self.state["info"]["render"] = geometry
        self.state["frame"] = frame

    def step(self):
        if self.state["terminated"] or self.state["truncated"]:
            if self.continuous:
                self.next_episode()
                return
            self.running = False
            self.state["status"] = "completed"
            return
        named = decision_state(self.environment, self.env, self.observation)
        choices = self.names, self.native_actions, self.env.np_random, named
        action, decision = select_action(self.controller, self.observation, choices, self.manual, self.policy)
        previous = json_value(self.observation)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.state.update(sequence=self.state["sequence"] + 1, observation=json_value(self.observation),
                          action=json_value(action), decision=decision, reward=float(reward),
                          terminated=bool(terminated), truncated=bool(truncated),
                          info=json_value(info), error=None)
        self.state["episode_steps"] += 1
        self.state["episode_reward"] += float(reward)
        self.state["transition"] = {"observation": previous, "action": json_value(action),
                                    "reward": float(reward), "next_observation": json_value(self.observation),
                                    "terminated": bool(terminated), "truncated": bool(truncated),
                                    "episode_id": self.state["episode_id"], "sequence": self.state["sequence"]}
        geometry, frame = render_state(self.environment, self.env)
        self.state["info"]["render"], self.state["frame"] = geometry, frame
        if terminated or truncated:
            self.running = self.running and self.continuous
            self.state["status"] = "completed"
            self.state["last_episode"] = {key: self.state[key] for key in
                                          ("episode_id", "seed", "sequence", "episode_steps", "episode_reward", "terminated", "truncated")}
        recorded = {**self.state, "frame": None}
        if self.record_frames and frame:
            frames = self.path / "frames"
            frames.mkdir(exist_ok=True)
            target = frames / f"{self.state['episode_id']}-{self.state['sequence']}.png"
            target.write_bytes(base64.b64decode(frame))
            recorded["frame_path"] = str(target)
        with (self.path / "trajectory.jsonl").open("a") as handle:
            handle.write(json.dumps(json_value(recorded)) + "\n")

    def next_episode(self):
        """Reset a finished native episode without changing its controller."""
        seed = self.seed + 1 if self.new_scenario else self.seed
        if self.new_scenario and self.environment == "car":
            config = self.env.unwrapped.config
            seed = config["start_seed"] + (seed - config["start_seed"]) % config["num_scenarios"]
        running = self.running
        self.reset({"seed": seed})
        self.running = running
        self.state["status"] = "running" if running else "paused"

    def command(self, command, payload):
        if command == "play":
            if (self.state["terminated"] or self.state["truncated"]) and not self.continuous:
                raise ValueError("Reset the completed episode before playing")
            self.running = True
            self.state["status"] = "running"
        elif command == "pause":
            self.running = False
            self.state["status"] = "paused"
        elif command == "continuous":
            enabled, new = payload["enabled"], payload.get("new_scenario", self.new_scenario)
            if not isinstance(enabled, bool) or not isinstance(new, bool):
                raise ValueError("enabled and new_scenario must be booleans")
            self.continuous, self.new_scenario = enabled, new
            self.state.update(continuous=enabled, new_scenario=new)
            self.settings["config"].update(continuous=enabled, new_scenario=new)
            self.state["config"] = dict(self.settings["config"])
            (self.path / "manifest.json").write_text(json.dumps(self.settings))
        elif command == "speed":
            self.speed = max(0.1, min(60., float(payload["speed"])))
            self.state["speed"] = self.speed
        elif command == "action":
            selected = int(payload["action"])
            if not 0 <= selected < len(self.names):
                raise ValueError("Invalid manual action")
            self.manual = selected
            self.state["manual_action"] = selected
        elif command == "controller":
            controller = payload["controller"]
            if controller not in {"manual", "random", "ppo", "decider"}:
                raise ValueError("Unknown controller")
            if controller == "ppo":
                self.policy = load_policy(payload["checkpoint"], self.env)
            elif controller == "decider":
                self.policy = decision_controller(payload.get("decision_checkpoint"))
            self.controller = controller
            self.state["controller"] = controller
            self.settings.update(controller=controller, device="cpu",
                                 model=payload.get("decision_checkpoint", STRANDS) if controller == "decider"
                                 else payload.get("checkpoint", ""))
            for field in ("checkpoint", "decision_checkpoint"):
                if field in payload:
                    self.settings["config"][field] = payload[field]
            self.state["config"] = dict(self.settings["config"])
            self.state.update(model=self.settings["model"], device=self.settings["device"])
            (self.path / "manifest.json").write_text(json.dumps(self.settings))
        elif command == "reset":
            self.reset(payload)
        elif command == "step":
            self.step()


def worker(settings, commands, updates):
    simulation = None
    try:
        simulation = Simulation(settings)
        publish(updates, simulation.state)
        while True:
            try:
                command, payload = commands.get(timeout=1 / simulation.speed if simulation.running else .2)
            except queue.Empty:
                if not simulation.running:
                    continue
                command, payload = "step", {}
            if command == "close":
                break
            try:
                simulation.command(command, payload)
            except (RuntimeError, ValueError, OSError, KeyError, TypeError, ImportError, AssertionError) as exc:
                logging.exception("Simulation command failed")
                simulation.running = False
                simulation.state.update(status="paused", error=str(exc))
                if isinstance(exc, DecisionHeld):
                    simulation.state.update(decision=exc.decision, action=None)
            publish(updates, simulation.state)
    except (RuntimeError, ValueError, OSError, KeyError, TypeError, ImportError, AssertionError) as exc:
        logging.exception("Simulation worker failed")
        publish(updates, {"id": settings["id"], "environment": settings["environment"],
                         "status": "error", "sequence": 0, "error": str(exc)})
    finally:
        if simulation is not None:
            simulation.env.close()


def publish(updates, state):
    """Keep a bounded stream of the latest simulation states."""
    try:
        updates.put_nowait(json_value(state))
    except queue.Full:
        try:
            updates.get(timeout=.05)
        except queue.Empty:
            return
        updates.put_nowait(json_value(state))
