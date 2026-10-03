"""Installed-interpreter simulation session supervision."""

import atexit
import os
import queue
import threading
import uuid

from ml_stack.gym.catalog import require
from ml_stack.gym.paths import artifact_root
from ml_stack.gym.transport import Process


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
        cfg = config or {}
        world = cfg.get("simulation_mode") == "world"
        if cfg.get("simulation_mode", "episode") not in {"world", "episode"}:
            raise ValueError("Simulation mode must be world or episode")
        if cfg.get("learning_mode", "frozen") not in {"online", "frozen"}:
            raise ValueError("Learning mode must be online or frozen")
        online = cfg.get("learning_mode", "frozen") == "online"
        if controller not in {"manual", "random", "decider", "ppo", "native-idm"}:
            raise ValueError("Unknown simulation controller")
        if controller == "native-idm" and not (world and environment == "car"):
            raise ValueError("Native IDM is available in the persistent car world")
        if online and not (world and controller == "ppo"):
            raise ValueError("Online learning requires PPO in a persistent world")
        if controller == "ppo" and not online and not cfg.get("checkpoint"):
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

    def list(self):
        """Return current live session snapshots."""
        with self.lock:
            return [self.get(identifier) for identifier in self.sessions]

    def control(self, identifier, command, payload=None):
        if command not in {"play", "pause", "step", "reset", "speed", "action", "controller", "learning", "agent"}:
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
