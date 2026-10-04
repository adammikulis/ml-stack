"""Person-started coding turns on maintained launchers, with native session resumes."""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue
import shutil
import signal
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import coding, platform, roles
from ml_stack.fleet import conversation_graph as graph
from ml_stack.fleet.conversation_types import safe
from ml_stack.fleet.conversations import Conversations
from ml_stack.serve.process import kill_process_tree
from ml_stack.workspace import coding_events, localstart, project
from ml_stack.workspace.harness_seat import Seat
from ml_stack.workspace.service import Workspace

MAX_EVENT = 1048576


@dataclass
class Turn:
    conversation: str
    state: str = "starting"
    events: deque[dict] = field(default_factory=lambda: deque(maxlen=1024))
    sequence: int = 0
    process: Any = None
    cancelled: threading.Event = field(default_factory=threading.Event)
    changed: threading.Condition = field(default_factory=threading.Condition)
    session: str = ""
    text: str = ""
    streamed: bool = False
    error: str = ""
    output: Any = None

    def emit(self, payload: dict) -> None:
        if self.output is not None:
            self.output.put(payload)
            return
        with self.changed:
            self.sequence += 1
            self.events.append({"sequence": self.sequence, **payload})
            self.changed.notify_all()

    def view(self) -> dict:
        return {"state": self.state, "session": self.session, "sequence": self.sequence, "error": self.error}

    def cancel(self) -> None:
        self.cancelled.set()
        process = self.process
        if process is None:
            return
        if isinstance(process, multiprocessing.process.BaseProcess):
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                kill_process_tree(process.pid, grace_s=1)
                process.kill()
                process.join(timeout=1)
        else:
            kill_process_tree(process.pid, grace_s=2)


class Manager:
    def __init__(self, conversations) -> None:
        self.store = conversations
        self.turns: dict[str, Turn] = {}
        self.lock = threading.Lock()

    def start(self, cid: str, prompt: str) -> Turn:
        if not safe(cid) or not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 200000:
            raise ValueError("a turn needs an existing conversation and a nonempty message under 200000 characters")
        conversation = self.store.get(cid)
        if conversation is None or conversation.settings["mode"] != "coding":
            raise ValueError("select a Coding conversation first")
        settings = conversation.settings
        if settings["harness"] not in coding.HARNESSES:
            raise ValueError("unsupported coding harness")
        roles.get(settings["role"] or roles.DEFAULT)
        folder = Path(settings["project"]).expanduser().resolve()
        if not settings["project"] or not folder.is_dir():
            raise ValueError("choose an existing project directory")
        root = self.store.root.resolve()
        if folder == root or folder in root.parents or root in folder.parents:
            raise ValueError("the project and saved conversation directory must not overlap")
        with self.lock:
            previous = self.turns.get(cid)
            if previous and previous.state in ("starting", "running", "cancelling"):
                raise ValueError("this conversation already has a running turn")
            if sum(turn.state in ("starting", "running", "cancelling") for turn in self.turns.values()) >= 4:
                raise ValueError("four coding turns are already active")
            if len(self.turns) >= 128:
                for key, old in list(self.turns.items()):
                    if old.state not in ("starting", "running", "cancelling"):
                        del self.turns[key]
                        break
            turn = Turn(cid)
            self.turns[cid] = turn
            self.store.append(cid, "user", prompt)
            context = multiprocessing.get_context("spawn")
            messages = context.Queue(maxsize=128)
            process = context.Process(target=worker, args=(str(self.store.root), turn.conversation, conversation, prompt, messages), daemon=True)
            turn.process = process
            process.start()
            threading.Thread(target=self._collect, args=(turn, messages), daemon=True).start()
        return turn

    def _collect(self, turn, messages) -> None:
        while True:
            try:
                payload = messages.get(timeout=.2)
            except queue.Empty:
                if turn.process.is_alive():
                    continue
                if turn.state not in ("completed", "cancelled", "failed"):
                    turn.state = "cancelled" if turn.cancelled.is_set() else "failed"
                    turn.error = "" if turn.cancelled.is_set() else "The coding worker ended before completing the turn"
                    turn.emit({"state": turn.state, "error": turn.error, "done": True})
                break
            if "state" in payload:
                turn.state = payload["state"]
            if "session" in payload:
                turn.session = str(uuid.UUID(payload["session"]))
            if "error" in payload:
                turn.error = payload["error"]
            turn.emit(payload)
        turn.process.join(timeout=1)
        messages.close()

    def status(self, cid: str) -> dict:
        turn = self.turns.get(cid)
        if turn:
            return turn.view()
        with graph.opened(self.store.root) as store:
            saved = store.get_doc(f"coding:{cid}", {})
        return {"state": "idle", "session": saved.get("session", ""), "sequence": 0, "previous_state": saved.get("state", "")}

    def cancel(self, cid: str) -> dict:
        turn = self.turns.get(cid)
        if turn and turn.state in ("starting", "running"):
            turn.state = "cancelling"
            threading.Thread(target=turn.cancel, daemon=True).start()
        return self.status(cid)

    def _seat(self, name, folder, parent, say):
        ws = Workspace()
        localstart._mint(ws, name, project.describe(str(folder)))
        return Seat(name, minted=True, base=ws.base)

    def _run(self, turn: Turn, conversation, prompt: str) -> None:
        settings = conversation.settings
        fingerprint = hashlib.sha256(json.dumps({"model": conversation.model, **settings}, sort_keys=True).encode()).hexdigest()[:24]
        home = self.store.root / "harness" / conversation.id / fingerprint
        home.mkdir(parents=True, exist_ok=True, mode=0o700)
        with graph.opened(self.store.root) as store:
            previous = store.get_doc(f"coding:{conversation.id}", {})
        turn.session = previous.get("session", "") if previous.get("fingerprint") == fingerprint else ""
        if not turn.session and conversation.messages:
            prompt = "Previous conversation:\n" + "\n\n".join(
                f"{message.role}: {message.content}" for message in conversation.messages) + "\n\nNew request:\n" + prompt
        try:
            if turn.cancelled.is_set():
                return
            turn.emit({"state": "starting"})
            harness = settings["harness"]
            if harness == "codex":
                args = ["exec", "--json", "--color", "never", "--skip-git-repo-check", "-"]
                if turn.session:
                    args = ["exec", "resume", "--json", "--skip-git-repo-check", turn.session, "-"]
            else:
                args = ["--print", "--verbose", "--output-format", "stream-json", "--include-partial-messages"]
                if turn.session:
                    args += ["--resume", turn.session]
            result = coding.launch_coding_agent(conversation.model, settings["role"] or roles.DEFAULT,
                settings["project"], harness=harness, context=settings["context"], draft=settings.get("draft", "auto"), name=f"chat-{conversation.id}",
                harness_args=args, seat_factory=self._seat, say=lambda text: turn.emit({"status": text}),
                **{f"run_{harness}": lambda command, env: self._process(turn, command, env, (home, harness, prompt, settings["project"]))})
            if result and not turn.cancelled.is_set():
                raise RuntimeError(turn.error or f"{harness} exited with status {result}")
        except (OSError, RuntimeError, ValueError, SystemExit) as error:
            if not turn.cancelled.is_set():
                turn.error = f"The coding launcher stopped with status {error.code}" if isinstance(error, SystemExit) else str(error)
                turn.emit({"error": turn.error})
        finally:
            if turn.text:
                self.store.append(conversation.id, "assistant", turn.text)
            turn.state = "cancelled" if turn.cancelled.is_set() else "failed" if turn.error else "completed"
            with graph.opened(self.store.root) as store, store.transaction():
                store.put_doc(f"coding:{conversation.id}", {"version": 1, "session": turn.session,
                    "fingerprint": fingerprint, "state": turn.state, "updated": time.time()})
            turn.emit({"state": turn.state, "done": True})

    def _event(self, turn, row, harness) -> None:
        payload = coding_events.event(row, harness)
        if payload.get("session"):
            turn.session = str(uuid.UUID(payload["session"]))
        if payload.get("error"):
            turn.error = str(payload["error"])
        if "delta" in payload:
            turn.streamed = True
            turn.text += payload["delta"]
        elif "text" in payload:
            text = payload.pop("text")
            if not turn.streamed:
                turn.text += ("\n\n" if turn.text else "") + text
                payload["delta"] = ("\n\n" if turn.text != text else "") + text
            turn.streamed = False
        if payload:
            turn.emit(payload)

    def _process(self, turn, command, environment, context) -> int:
        home, harness, prompt, folder = context
        environment = dict(environment)
        if harness == "codex":
            temporary = Path(environment["CODEX_HOME"])
            for path in home.iterdir():
                destination = temporary / path.name
                if path.is_dir():
                    shutil.copytree(path, destination, dirs_exist_ok=True)
                elif not destination.exists():
                    shutil.copy2(path, destination)
        else:
            environment["CLAUDE_CONFIG_DIR"] = str(home)
            temporary = home
        if turn.cancelled.is_set():
            return 0
        with (home / "stderr.log").open("w") as errors:
            process = platform.start_process(command, env=environment, cwd=folder, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=errors, text=True, bufsize=1)
            turn.process = process
            turn.state = "running"
            turn.emit({"state": turn.state})
            try:
                process.stdin.write(prompt)
                process.stdin.close()
                while line := process.stdout.readline(MAX_EVENT + 1):
                    if len(line) > MAX_EVENT:
                        raise RuntimeError("the harness emitted an oversized event")
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    self._event(turn, row, harness)
                return process.wait()
            finally:
                if process.poll() is None:
                    kill_process_tree(process.pid, grace_s=2)
                    process.wait()
                turn.process = None
                if harness == "codex":
                    for path in temporary.iterdir():
                        if path.name in ("config.toml", "AGENTS.md", "auth.json"):
                            continue
                        destination = home / path.name
                        if path.is_dir():
                            shutil.copytree(path, destination, dirs_exist_ok=True)
                        else:
                            shutil.copy2(path, destination)


def worker(root, cid, conversation, prompt, output) -> None:
    turn = Turn(cid, output=output)
    def stop(signum, frame):
        turn.cancelled.set()
        raise InterruptedError("The coding turn was cancelled")
    signal.signal(signal.SIGTERM, stop)
    Manager(Conversations(root))._run(turn, conversation, prompt)
