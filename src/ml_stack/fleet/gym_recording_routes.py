"""Persisted Gym recordings, reviewed labels and dataset export."""

from __future__ import annotations

import base64
import json
from pathlib import Path

from ml_stack.gym.paths import artifact_root
from ml_stack.gym.recordings import export_reviewed

from .files import safe_relpath
from .jobs import DaemonError
from .request_fields import field, object_body


class GymRecordingRoutes:
    """Review authoritative episode transitions."""

    def route(self) -> bool:
        prefix = "/ui/gym/recordings"
        if not self.path.startswith(prefix):
            return super().route()
        try:
            root = artifact_root()
            tail = self.path[len(prefix):].strip("/").split("/")
            if tail == [""] and self.method == "GET":
                files = sorted((p for p in root.glob("*/trajectory.jsonl")
                                if p.resolve().is_relative_to(root.resolve())),
                               key=lambda p: p.stat().st_mtime, reverse=True)
                self.send(200, {"recordings": [{"id": p.parent.name, "bytes": p.stat().st_size,
                    "modified": p.stat().st_mtime} for p in files[:200]]})
                return True
            path = safe_relpath(root, tail[0])
            trajectory = safe_relpath(root, tail[0] + "/trajectory.jsonl")
            safe_relpath(root, tail[0] + "/reviews.jsonl")
            if not trajectory.exists():
                self.send(404, {"error": "No such recorded episode."})
                return True
            if self.method == "GET" and tail[1:] in ([], ["frame"]):
                return self._recording_read(path, trajectory, tail[1:])
            if self.method == "POST" and tail[1:] == ["review"]:
                return self._recording_review(path, trajectory)
            if self.method == "POST" and tail[1:] == ["export"]:
                return self._recording_export(path, trajectory)
        except (ImportError, DaemonError, ValueError, OSError, KeyError) as exc:
            self.send(400, {"error": str(exc)})
            return True
        self.send(405, {"error": "Unsupported recording route or method."})
        return True

    def _recording_read(self, path, trajectory, rest) -> bool:
        rows = []
        offset = max(0, int(self.asked("offset", "0")))
        reviews = path / "reviews.jsonl"
        labels = {}
        if reviews.exists():
            for line in reviews.read_text().splitlines():
                item = json.loads(line)
                labels[(item["episode_id"], item["sequence"])] = item["label"]
        with trajectory.open() as source:
            for index, line in enumerate(source):
                row = json.loads(line)
                if rest == ["frame"]:
                    if str(row["sequence"]) == self.asked("sequence"):
                        saved = row.get("frame_path")
                        if saved:
                            frame = Path(saved).resolve()
                            if not frame.is_relative_to(path.resolve()):
                                raise ValueError("Recorded camera path escapes its session directory.")
                            if frame.exists():
                                row["frame"] = base64.b64encode(frame.read_bytes()).decode()
                        self.send(200, row)
                        return True
                    continue
                if index < offset:
                    continue
                transition = row.get("transition", {})
                key = transition.get("episode_id"), transition.get("sequence")
                rows.append({"sequence": row["sequence"], "episode_id": key[0],
                    "environment": row["environment"], "actions": row["actions"],
                    "action": row["action"], "reward": row["reward"], "label": labels.get(key),
                    "terminated": row["terminated"], "truncated": row["truncated"]})
                if len(rows) >= 100:
                    break
        if rest == ["frame"]:
            self.send(404, {"error": "No such recorded transition."})
            return True
        self.send(200, {"id": path.name, "steps": rows, "offset": offset, "next": offset + len(rows)})
        return True

    def _recording_review(self, path, trajectory) -> bool:
        req = object_body(self)
        key = field(req, "episode_id", int), field(req, "sequence", int)
        label = field(req, "label", str)
        found = False
        with trajectory.open() as source:
            for line in source:
                row = json.loads(line)
                transition = row["transition"]
                if key == (transition["episode_id"], transition["sequence"]):
                    if label not in row["actions"]:
                        raise ValueError("Choose a native action label from this transition.")
                    found = True
                    break
        if not found:
            raise ValueError("No such transition in this recording.")
        reviews = path / "reviews.jsonl"
        existing = [json.loads(line) for line in reviews.read_text().splitlines()] if reviews.exists() else []
        existing = [item for item in existing if (item["episode_id"], item["sequence"]) != key]
        existing.append({"episode_id": key[0], "sequence": key[1], "label": label})
        reviews.write_text("".join(json.dumps(item) + "\n" for item in existing))
        self.send(200, {"reviewed": len(existing), "label": label})
        return True

    def _recording_export(self, path, trajectory) -> bool:
        runner = self.ui.runner
        if runner is None:
            raise ValueError("Start the daemon job queue to export training datasets.")
        rel = field(object_body(self), "path", str, f"datasets/gym-{path.name}.jsonl")
        output = safe_relpath(runner.files_root, rel)
        if output.exists():
            raise ValueError("Choose a new dataset filename.")
        output.parent.mkdir(parents=True, exist_ok=True)
        count = export_reviewed(trajectory, path / "reviews.jsonl", output)
        self.send(201, {"path": rel, "cases": count, "recording": path.name})
        return True
