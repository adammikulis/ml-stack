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
from ml_stack.files import write_json
from ml_stack.gym.adapters import actions, json_value, make_environment, render_state
from ml_stack.gym.live_learning import create_policy, learn_rollout
from ml_stack.gym.observations import decision_state
from ml_stack.gym.paths import artifact_root
from ml_stack.gym.provenance import native_provenance


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
        self.world = settings["config"].get("simulation_mode", "episode") == "world"
        self.learning_mode = settings["config"].get("learning_mode", "frozen")
        self.control_revision = 0
        if self.environment == "car":
            settings["config"].setdefault("steering_magnitude", .35)
        self.running, self.speed, self.manual, self.policy = False, 10., 3 if self.environment == "car" else 0, None
        decision_checkpoint = settings["config"].get("decision_checkpoint")
        if self.controller == "decider":
            self.policy = decision_controller(decision_checkpoint)
            settings["model"] = str(decision_checkpoint or STRANDS)
            settings["device"] = "cpu"
        settings["version"] = 1
        self.settings = settings
        self.path = artifact_root() / settings["id"]
        self.path.mkdir(parents=True, exist_ok=True)
        config = dict(settings["config"])
        self.record_frames = bool(config.pop("record_frames", False))
        checkpoint = config.pop("checkpoint", None)
        config.pop("decision_checkpoint", None)
        if self.environment == "car":
            config.setdefault("render_preview", True)
        self.env = make_environment(self.environment, config, self.seed)
        self.observation, initial_info = self.env.reset(seed=self.seed)
        settings.update(native_provenance(self.environment, self.env))
        settings.setdefault("model", checkpoint if self.controller == "ppo" else
                            "MetaDrive IDM" if self.controller == "native-idm" else None)
        settings.setdefault("device", "cpu")
        write_json(self.path / "manifest.json", settings)
        if self.controller == "ppo":
            if not checkpoint and not (self.world and self.learning_mode == "online"):
                self.env.close()
                raise ValueError("PPO controller requires a checkpoint in the session configuration")
            self.policy = create_policy(self, checkpoint)
        self.names, self.native_actions = actions(self.environment, self.env)
        self.state = {"id": settings["id"], "environment": self.environment, "status": "starting",
                      "sequence": 0, "episode_id": 0, "controller": self.controller, "observation": None,
                      "action": None, "decision": None, "reward": 0., "terminated": False,
                      "truncated": False, "info": {}, "frame": None, "error": None,
                      "actions": self.names, "config": dict(settings["config"]), "seed": self.seed,
                      "model": settings["model"], "device": settings["device"],
                      "manual_action": self.manual, "speed": self.speed,
                      "simulation_mode": "world" if self.world else "episode", "learning_mode": self.learning_mode,
                      "policy_version": 0, "training_timesteps": 0, "optimizer_updates": 0,
                      "trajectory_path": str(self.path / "trajectory.jsonl")}
        self.reset({}, initial_info)

    def reset(self, payload, initial_info=None):
        self.control_revision += 1
        self.running = False
        seed = int(payload.get("seed", self.seed))
        if self.world and seed != self.seed:
            raise ValueError("Create a new world to change its seed")
        if initial_info is None:
            self.observation, info = self.env.reset(seed=seed)
        else:
            info = initial_info
        self.seed = seed
        self.settings["seed"] = seed
        self.state["seed"] = seed
        write_json(self.path / "manifest.json", self.settings)
        self.state["episode_id"] += 1
        self.state["sequence"] += 1
        self.state.update(status="paused", observation=json_value(self.observation), info=json_value(info),
                          reward=0., action=None, decision=None, terminated=False, truncated=False, error=None)
        self.state.pop("transition", None)
        self.state["agent_id"] = info.get("ego_actor_id")
        geometry, frame = render_state(self.environment, self.env)
        self.state["info"]["render"] = geometry
        self.state["frame"] = frame

    def step(self):
        if self.state["terminated"] or self.state["truncated"]:
            self.running = False
            self.state["status"] = "completed"
            return
        named = decision_state(self.environment, self.env, self.observation)
        choices = self.names, self.native_actions, self.env.np_random, named
        if self.controller == "native-idm":
            action, decision = None, {"state": named, "choice": "Native IDM driving", "probabilities": None,
                                      "model": "MetaDrive IDM", "backend": "MetaDrive", "latency_ms": 0.}
        else:
            action, decision = select_action(self.controller, self.observation, choices, self.manual, self.policy)
        self.advance(action, decision)

    def advance(self, action, decision):
        """Record one native step from the supplied controller action."""
        previous = json_value(self.observation)
        if self.controller == "ppo" and self.policy is not None:
            self.state.update(training_timesteps=self.policy.num_timesteps,
                              optimizer_updates=self.policy._n_updates)
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
        self.state["applied_native_control"] = info.get("applied_native_control")
        self.state["agent_id"] = info.get("ego_actor_id")
        if terminated or truncated:
            if self.world:
                self.observation, task_info = self.env.reset()
                self.state.update(observation=json_value(self.observation), terminated=False, truncated=False,
                                  episode_id=self.state["episode_id"] + 1)
                self.state["info"].update(task_info)
            else:
                self.running = False
                self.state["status"] = "completed"
        recorded = {**self.state, "frame": None}
        if self.record_frames and frame:
            frames = self.path / "frames"
            frames.mkdir(exist_ok=True)
            target = frames / f"{self.state['episode_id']}-{self.state['sequence']}.png"
            target.write_bytes(base64.b64decode(frame))
            recorded["frame_path"] = str(target)
        with (self.path / "trajectory.jsonl").open("a") as handle:
            handle.write(json.dumps(json_value(recorded)) + "\n")

    def change_controller(self, payload):
        controller = payload["controller"]
        if controller not in {"manual", "random", "ppo", "decider", "native-idm"}:
            raise ValueError("Unknown controller")
        if controller == "native-idm" and not (self.world and self.environment == "car"):
            raise ValueError("Native IDM requires a persistent car world")
        if controller == "ppo":
            mode = payload.get("learning_mode", self.learning_mode)
            if mode not in {"online", "frozen"} or (mode == "online" and not self.world):
                raise ValueError("Online PPO requires a persistent world")
            policy = create_policy(self, payload.get("checkpoint"), mode)
            self.policy = policy
            self.learning_mode = mode
            self.state["learning_mode"] = mode
            self.settings["config"]["learning_mode"] = mode
        elif controller == "decider":
            self.policy = decision_controller(payload.get("decision_checkpoint"))
        self.controller = controller
        if controller != "ppo":
            self.learning_mode = "frozen"
            self.state["learning_mode"] = "frozen"
            self.settings["config"]["learning_mode"] = "frozen"
        self.state["controller"] = controller
        self.settings.update(controller=controller, device="cpu",
                             model=payload.get("decision_checkpoint", STRANDS) if controller == "decider"
                             else "MetaDrive IDM" if controller == "native-idm"
                             else payload.get("checkpoint", ""))
        for field in ("checkpoint", "decision_checkpoint"):
            if field in payload:
                self.settings["config"][field] = payload[field]
        self.state["config"] = dict(self.settings["config"])
        self.state.update(model=self.settings["model"], device=self.settings["device"])
        write_json(self.path / "manifest.json", self.settings)

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
            self.state["speed"] = self.speed
        elif command == "action":
            selected = int(payload["action"])
            if not 0 <= selected < len(self.names):
                raise ValueError("Invalid manual action")
            self.manual = selected
            self.state["manual_action"] = selected
        elif command == "controller":
            self.change_controller(payload)
        elif command == "learning":
            mode = payload["mode"]
            if mode not in {"online", "frozen"}:
                raise ValueError("Learning mode must be online or frozen")
            if mode == "online" and not (self.world and self.controller == "ppo"):
                raise ValueError("Online learning requires PPO in a persistent world")
            self.learning_mode = mode
            self.state["learning_mode"] = mode
            self.settings["config"]["learning_mode"] = mode
            self.state["config"]["learning_mode"] = mode
            write_json(self.path / "manifest.json", self.settings)
        elif command == "agent":
            if not self.world or not hasattr(self.env, "select_agent"):
                raise ValueError("This environment does not expose individually controllable agents")
            self.observation, info = self.env.select_agent(payload["agent_id"])
            self.control_revision += 1
            self.state.update(observation=json_value(self.observation), decision=None, action=None)
            self.state["reward"] = None
            self.state["agent_id"] = info.get("ego_actor_id")
            geometry, frame = render_state(self.environment, self.env)
            self.state["info"].update(info, render=geometry)
            self.state["frame"] = frame
        elif command == "reset":
            self.reset(payload)
        elif command == "step":
            if self.controller == "ppo" and self.learning_mode == "online":
                raise ValueError("Switch to frozen mode before single stepping PPO")
            self.step()
        else:
            raise ValueError(payload.get("error", f"Unknown simulation command: {command}"))


def worker(settings, commands, updates):
    simulation = None
    try:
        simulation = Simulation.__new__(Simulation)
        simulation.__init__(settings)
        publish(updates, simulation.state)
        while True:
            if simulation.running and simulation.controller == "ppo" and simulation.learning_mode == "online":
                if not learn_rollout(simulation, commands, updates, publish):
                    break
                continue
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
    except (RuntimeError, ValueError, OSError, KeyError, TypeError, ImportError, AssertionError, AttributeError) as exc:
        logging.exception("Simulation worker failed")
        publish(updates, {"id": settings["id"], "environment": settings["environment"],
                         "status": "error", "sequence": 0, "error": str(exc)})
    finally:
        if simulation is not None and hasattr(simulation, "env"):
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
