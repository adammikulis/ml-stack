"""Dataset browsing and monitored workspace jobs."""

from __future__ import annotations

import importlib.metadata
import json
import shlex
from pathlib import Path

from ml_stack.contracts import recipes
from ml_stack.files import write_json
from ml_stack.gym.catalog import ENVIRONMENTS
from ml_stack.gym.transport import interpreter

from .files import safe_relpath
from .gym_interpreters import configure_interpreters
from .jobs import DaemonError
from .request_fields import field, object_body

PURPOSES = {
    "graph": "Serve the graph exploration workspace.", "store": "Inspect and manage graph stores.",
    "world": "Build and inspect world models.", "ingest": "Ingest source documents into a graph.",
    "speech": "Transcribe audio and synthesize speech.", "security": "Inspect security configuration and network policy.",
    "audit": "Audit redaction and data handling.", "credentials": "Manage credentials for model providers.",
    "doctor": "Check installation, dependencies and machine readiness.", "setup": "Configure the local installation.",
    "workspace": "Coordinate working directories and project tasks.", "jobs": "Inspect and manage queued work.",
    "suite": "Run training and evaluation suites.", "agent": "Run model-driven agent tasks.",
    "claude": "Run the coding-agent bridge.",
    "mcp": "Expose tools through the Model Context Protocol.", "serve": "Serve models and manage inference.",
    "draft": "Manage speculative decoding draft heads.", "chat": "Converse with an agent, plan tasks and approve model-driven operations.",
    "memory": "Inspect, add, confirm, link, forget and export agent memory facts.",
    "surface": "Inspect retrieval and extraction surfaces.", "walk": "Traverse and inspect graph data.",
    "decide": "Ask closed questions, calibrate and evaluate decision models.",
    "peers": "Discover peers and transfer files.", "fleet": "Coordinate machines and shared workloads.",
    "bench": "Measure model accuracy, throughput and resource use.", "models": "Discover, download and inspect models.",
    "gym": "Run, train and evaluate simulator policies.", "train-run": "Train a supervised recipe.",
    "train-tools": "Prepare and train tool-calling models.", "train-decider": "Fine-tune decision models.",
    "traind": "Start the local daemon and job queue.", "help": "Explore installed ml-stack commands.",
}


def commands() -> list[dict[str, str]]:
    """Installed ml-stack commands and their entry points."""
    entries = importlib.metadata.entry_points(group="console_scripts")
    return [{"name": e.name, "entry": e.value,
             "description": PURPOSES.get(e.name.removeprefix("ml-stack-"), "Inspect options and run this specialist workflow.")} for e in sorted(entries, key=lambda e: e.name)
            if e.name.startswith("ml-stack-")]


class WorkspaceRoutes:
    """Authenticated data and job operations."""

    def route(self) -> bool:
        if not self.path.startswith("/ui/workspace/"):
            return super().route()
        try:
            return self._workspace()
        except (DaemonError, ValueError, OSError) as exc:
            self.send(400, {"error": str(exc)})
            return True

    def _workspace(self) -> bool:
        runner = self.ui.runner
        if self.path == "/ui/workspace/recipes" and self.method == "GET":
            self.send(200, {"recipes": recipes()})
            return True
        if self.path == "/ui/workspace/commands" and self.method == "GET":
            self.send(200, {"commands": commands()})
            return True
        if runner is None:
            self.send(501, {"error": "Start ml-stack-traind to use datasets and monitored jobs."})
            return True
        root = runner.files_root
        if self.path == "/ui/workspace/files" and self.method == "GET":
            return self._files(root)
        if self.path == "/ui/workspace/file" and self.method == "GET":
            path = safe_relpath(root, self.asked("path"))
            with path.open("rb") as source:
                raw = source.read(1_000_001)
            self.send(200, {"path": self.asked("path"), "text": raw[:1_000_000].decode("utf-8", "replace"),
                            "truncated": len(raw) > 1_000_000})
            return True
        if self.path == "/ui/workspace/file" and self.method == "POST":
            req = object_body(self)
            path = safe_relpath(root, field(req, "path", str, ""))
            content = field(req, "text", str, "")
            if len(content.encode()) > 10_000_000:
                raise ValueError("Dataset upload limit is 10 MB; use ml-stack-peers for larger files.")
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as target:
                target.write(content)
            self.send(201, {"path": str(path.relative_to(root))})
            return True
        if self.path == "/ui/workspace/jobs" and self.method == "GET":
            self.send(200, {"jobs": runner.snapshot(), "capacity": runner.status()})
            return True
        if self.path == "/ui/workspace/jobs" and self.method == "POST":
            return self._submit(root)
        prefix = "/ui/workspace/jobs/"
        if self.path.startswith(prefix):
            parts = self.path[len(prefix):].split("/")
            job = runner.jobs.get(parts[0])
            if job is None:
                self.send(404, {"error": "No such job."})
                return True
            if self.method == "POST" and parts[1:] == ["stop"]:
                self.send(200, runner.stop(job.id).public())
                return True
            if self.method == "GET":
                return self._job_detail(runner, job)
        self.send(405, {"error": "Unsupported workspace route or method."})
        return True

    def _files(self, root: Path) -> bool:
        rel = self.asked("path")
        path = safe_relpath(root, rel) if rel else root.resolve()
        rows = []
        if path.exists():
            for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name)):
                resolved = child.resolve()
                if not resolved.is_relative_to(root.resolve()):
                    continue
                rows.append({"name": child.name, "path": str(child.relative_to(root)),
                             "directory": child.is_dir(), "bytes": child.stat().st_size})
        self.send(200, {"path": rel, "root": str(root), "files": rows})
        return True

    def _submit(self, root: Path) -> bool:
        req = object_body(self)
        command = field(req, "command", str, "")
        if command not in {entry["name"] for entry in commands()}:
            raise ValueError("Choose an installed ml-stack command.")
        args = req.get("args", [])
        if not isinstance(args, list) or any(not isinstance(a, str) for a in args):
            raise ValueError("Arguments must be a list of strings.")
        if "--detach" in args:
            raise ValueError("Workspace jobs are monitored directly; remove --detach.")
        argv = [command, *args]
        environment = None
        if command == "ml-stack-gym" and len(args) >= 2 and args[0] in {"run", "train", "evaluate"}:
            if args[1] not in ENVIRONMENTS:
                raise ValueError("Choose a supported Gym environment.")
            configure_interpreters(self.ui)
            argv = [interpreter(args[1]), "-m", "ml_stack.gym.cli", *args]
            environment = {"ML_STACK_GYM_PYTHON": "", "ML_STACK_GYM_PYTHONS": "{}",
                           "ML_STACK_GYM_FILES_ROOT": str(root)}
        metadata = field(req, "metadata", dict, {})
        name = field(req, "name", str, command)
        if field(req, "preview", bool, False):
            self.send(200, {"argv": argv, "command": shlex.join(argv)})
            return True
        root.mkdir(parents=True, exist_ok=True)
        job = self.ui.runner.submit(name, argv, str(root), env=environment)
        if isinstance(metadata, dict):
            write_json(self.ui.runner.job_dir(job.id) / "workspace.json", {**metadata, "version": 1})
        self.send(202, job.public())
        return True

    def _job_detail(self, runner, job) -> bool:
        log = runner.log_path(job.id)
        text = ""
        if log.exists():
            with log.open("rb") as source:
                source.seek(max(0, log.stat().st_size - 100_000))
                text = source.read().decode("utf-8", "replace")
        metrics = runner.job_dir(job.id) / "metrics.jsonl"
        rows = []
        if metrics.exists():
            with metrics.open() as source:
                for line in source:
                    try:
                        rows.append(json.loads(line))
                    except ValueError:
                        continue
            rows = rows[-200:]
        metadata = runner.job_dir(job.id) / "workspace.json"
        self.send(200, {"job": job.public(), "log": text,
                        "metrics": rows, "metadata": json.loads(metadata.read_text()) if metadata.exists() else {}})
        return True
