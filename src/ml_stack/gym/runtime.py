"""Isolated simulation sessions and recorded decision telemetry."""

import atexit
import json
import logging
import os
import queue
import threading
import time
import uuid

from ml_stack.decide.pins import STRANDS
from ml_stack.decide.pointer import PointerDecider
from ml_stack.gym.adapters import actions, json_value, make_environment, render_state
from ml_stack.gym.catalog import require
from ml_stack.gym.paths import artifact_root
from ml_stack.gym.training import load_policy
from ml_stack.gym.transport import Process


class DecisionHeld(RuntimeError):
    """A controller decision held without advancing the simulator."""

    def __init__(self, decision):
        super().__init__("Decision model abstained; simulation paused")
        self.decision = decision


def select_action(controller, observation, choices, manual, policy=None):
    """Select an action and record its decision inputs."""
    names, native_actions, rng = choices
    began = time.perf_counter()
    detail = {"observation": json_value(observation), "probabilities": None}
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
        answer = policy.decide("Choose the next safe environment action", detail["observation"], names)
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
        self.running, self.speed, self.manual, self.policy = False, 10., 3 if self.environment == "car" else 0, None
        if self.controller == "decider":
            self.policy = PointerDecider(device="cpu")
            settings["model"] = STRANDS
            settings["device"] = "cpu"
        self.path = artifact_root() / settings["id"]
        self.path.mkdir(parents=True, exist_ok=True)
        (self.path / "manifest.json").write_text(json.dumps(settings))
        config = dict(settings["config"])
        checkpoint = config.pop("checkpoint", None)
        if self.environment == "car":
            config.setdefault("render_preview", True)
        self.env = make_environment(self.environment, config)
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
                      "actions": self.names, "trajectory_path": str(self.path / "trajectory.jsonl")}
        self.reset({})

    def reset(self, payload):
        self.running = False
        self.observation, info = self.env.reset(seed=int(payload.get("seed", self.seed)))
        self.state["episode_id"] += 1
        self.state["sequence"] += 1
        self.state.update(status="paused", observation=json_value(self.observation), info=json_value(info),
                          reward=0., action=None, decision=None, terminated=False, truncated=False, error=None)
        self.state.pop("transition", None)
        geometry, frame = render_state(self.environment, self.env)
        self.state["info"]["render"] = geometry
        self.state["frame"] = frame

    def step(self):
        if self.state["terminated"] or self.state["truncated"]:
            self.running = False
            self.state["status"] = "completed"
            return
        choices = self.names, self.native_actions, self.env.np_random
        action, decision = select_action(self.controller, self.observation, choices, self.manual, self.policy)
        previous = json_value(self.observation)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.state.update(sequence=self.state["sequence"] + 1, observation=json_value(self.observation),
                          action=json_value(action), decision=decision, reward=float(reward),
                          terminated=bool(terminated), truncated=bool(truncated),
                          info=json_value(info), error=None)
        self.state["transition"] = {"observation": previous, "action": json_value(action),
                                    "reward": float(reward), "next_observation": json_value(self.observation),
                                    "terminated": bool(terminated), "truncated": bool(truncated),
                                    "episode_id": self.state["episode_id"], "sequence": self.state["sequence"]}
        geometry, frame = render_state(self.environment, self.env)
        self.state["info"]["render"], self.state["frame"] = geometry, frame
        if terminated or truncated:
            self.running = False
            self.state["status"] = "completed"
        with (self.path / "trajectory.jsonl").open("a") as handle:
            handle.write(json.dumps(json_value(self.state)) + "\n")

    def command(self, command, payload):
        if command == "play":
            if self.state["terminated"] or self.state["truncated"]:
                raise ValueError("Reset the completed episode before playing")
            self.running = True
            self.state["status"] = "running"
        elif command == "pause":
            self.running = False
            self.state["status"] = "paused"
        elif command == "speed":
            self.speed = max(0.1, min(60., float(payload["speed"])))
        elif command == "action":
            selected = int(payload["action"])
            if not 0 <= selected < len(self.names):
                raise ValueError("Invalid manual action")
            self.manual = selected
        elif command == "controller":
            controller = payload["controller"]
            if controller not in {"manual", "random", "ppo", "decider"}:
                raise ValueError("Unknown controller")
            if controller == "ppo":
                self.policy = load_policy(payload["checkpoint"], self.env)
            elif controller == "decider":
                self.policy = PointerDecider(device="cpu")
            self.controller = controller
            self.state["controller"] = controller
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


class SessionManager:
    """Manage worker processes and their latest snapshots."""

    def __init__(self):
        self.sessions = {}
        self.lock = threading.RLock()

    def configure(self, python):
        """Select the managed environment's simulator interpreter."""
        os.environ["ML_STACK_GYM_PYTHON"] = str(python)

    def create(self, environment, config=None, controller="manual", seed=0):
        require(environment)
        if controller not in {"manual", "random", "decider", "ppo"}:
            raise ValueError("Choose manual, random, decider, or PPO at session creation")
        if controller == "ppo" and not (config or {}).get("checkpoint"):
            raise ValueError("PPO controller requires a checkpoint in the session configuration")
        identifier = uuid.uuid4().hex
        updates = queue.Queue(maxsize=2)
        path = artifact_root() / identifier
        path.mkdir(parents=True)
        log = (path / "worker.log").open("w")
        process = Process({"environment": environment, "config": dict(config or {}),
                           "id": identifier, "seed": seed, "controller": controller}, updates, log)
        process.start()
        with self.lock:
            self.sessions[identifier] = {"process": process, "commands": process.commands, "updates": updates,
                                             "snapshot": {"id": identifier, "environment": environment, "status": "starting", "sequence": 0}}
        return self.get(identifier)

    def get(self, identifier):
        with self.lock:
            session = self.sessions[identifier]
            while True:
                try:
                    session["snapshot"] = session["updates"].get_nowait()
                except queue.Empty:
                    break
            if not session["process"].is_alive() and session["snapshot"]["status"] not in ("error", "closed"):
                session["snapshot"].update(status="error", error="Simulation worker exited")
            return dict(session["snapshot"])

    def control(self, identifier, command, payload=None):
        if command not in {"play", "pause", "step", "reset", "speed", "action", "controller"}:
            raise ValueError(f"Unknown simulation command: {command}")
        with self.lock:
            session = self.sessions[identifier]
            if not session["process"].is_alive():
                raise RuntimeError("Simulation worker exited; create a new session")
            session["commands"].put((command, payload or {}))
        return self.get(identifier)

    def close(self, identifier):
        with self.lock:
            session = self.sessions.pop(identifier)
        if session["process"].is_alive():
            session["commands"].put(("close", {}))
        session["process"].join(timeout=3)
        if session["process"].is_alive():
            session["process"].terminate()
            session["process"].join(timeout=2)
        session["commands"].close()
        session["process"].log.close()
        return {"id": identifier, "status": "closed"}

    def close_all(self):
        for identifier in list(self.sessions):
            self.close(identifier)


manager = SessionManager()
atexit.register(manager.close_all)
