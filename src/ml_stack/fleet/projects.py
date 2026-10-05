"""Explicit project publication and canonical workspace authority."""

from __future__ import annotations

import hashlib
import threading
import urllib.parse
from dataclasses import asdict, dataclass
from pathlib import Path

from ml_stack.files import read_json, write_json, writing
from ml_stack.home import DEFAULT_NAME
from ml_stack.net import git

from . import project_source as source


def bootstrap() -> bytes:
    """Return the fixed source verifier and Git helpers for authenticated bootstrapping."""
    helpers = Path(git.__file__).read_text(encoding="utf-8")
    module = Path(source.__file__).read_text(encoding="utf-8")
    module = module.replace("from ml_stack.home import DEFAULT_NAME", f"DEFAULT_NAME = {DEFAULT_NAME!r}")
    module = module.replace("from ml_stack.net import git", "git = _project_git")
    prefix = ("from types import SimpleNamespace\n"
              "_project_git_scope = {'__name__': 'ml_stack.net.git'}\n"
              f"exec({helpers!r}, _project_git_scope)\n"
              "_project_git = SimpleNamespace(**_project_git_scope)\n")
    return (prefix + f"exec({module!r}, globals())\n").encode()


def identity(root: Path) -> str:
    """Return the project identity from checkout metadata or its Git origin."""
    attached = read_json(root / ".ml-stack-project.json", {})
    if attached.get("kind") == "project-checkout":
        return source.project_id(attached["project_id"])
    try:
        origin = git.run(["remote", "get-url", "origin"], cwd=root).stdout.strip()
        parts = urllib.parse.urlsplit(origin)
        if parts.scheme:
            origin = (parts.hostname or "") + parts.path
        elif ":" in origin:
            origin = origin.split("@")[-1].replace(":", "/", 1)
        origin = origin.removesuffix(".git").rstrip("/")
    except git.GitFailed:
        origin = git.run(["rev-list", "--max-parents=0", "HEAD"], cwd=root).stdout.strip()
    if not origin:
        raise source.ProjectError("Project has no Git identity")
    return hashlib.sha256(origin.encode()).hexdigest()[:32]


@dataclass
class Project:
    id: str
    name: str
    root: str
    authority_machine: str
    shared: bool = False
    source_hash: str = ""
    archive_sha256: str = ""
    size_bytes: int = 0
    files: int = 0
    excluded_files: int = 0

    def public(self) -> dict:
        result = asdict(self)
        result.pop("root")
        return result


class ProjectRegistry:
    """Registered local projects and immutable source bundles."""

    def __init__(self, root: Path, machine: str, candidates: tuple[Path, ...] = ()):
        self.root, self.machine = root, machine
        self.path = root / "projects.json"
        self.lock = threading.RLock()
        self._candidates: dict[str, Path] = {}
        self._projects: dict[str, Project] = {}
        stored = read_json(self.path, {})
        for row in stored.get("projects", []):
            project = Project(**row)
            source.project_id(project.id)
            self._projects[project.id] = project
        for path in candidates:
            path = path.resolve()
            try:
                git.run(["rev-parse", "--show-toplevel"], cwd=path)
                identifier = hashlib.sha256(str(path).encode()).hexdigest()[:32]
                self._candidates[identifier] = path
            except (OSError, git.GitFailed):
                continue

    def candidates(self) -> list[dict]:
        return [{"id": identifier, "name": path.name} for identifier, path in self._candidates.items()]

    def get(self, identifier: str) -> Project:
        source.project_id(identifier)
        with self.lock:
            project = self._projects.get(identifier)
            if project is None or not project.shared:
                raise source.ProjectError("Project is not shared here")
            return project

    def list(self) -> list[dict]:
        with self.lock:
            return [project.public() for project in self._projects.values() if project.shared]

    def _save(self) -> None:
        write_json(self.path, {"projects": [asdict(p) for p in self._projects.values()]})

    def share(self, candidate: str, name: str = "") -> Project:
        with self.lock:
            root = self._candidates.get(candidate)
            if root is None:
                raise source.ProjectError("Choose an available local project")
            attached = read_json(root / ".ml-stack-project.json", {})
            authority = attached.get("authority", {}).get("machine", "")
            if authority and authority != self.machine:
                raise source.ProjectError("This checkout uses another device's project authority")
            identifier = identity(root)
            manifest, packed = source.build(root, identifier)
            source.verify(packed, identifier, manifest["source_hash"])
            project = Project(identifier, name.strip() or root.name, str(root), self.machine,
                              True, manifest["source_hash"], hashlib.sha256(packed).hexdigest(),
                              manifest["size_bytes"], len(manifest["files"]), manifest["excluded_files"])
            target = self._bundle(project)
            with writing(target) as temporary:
                temporary.write_bytes(packed)
            self._projects[identifier] = project
            self._save()
            return project

    def unshare(self, identifier: str) -> None:
        with self.lock:
            self.get(identifier).shared = False
            self._save()

    def _bundle(self, project: Project) -> Path:
        return self.root / "project-bundles" / project.id / (project.source_hash + ".zip")

    def snapshot(self, identifier: str, revision: str) -> bytes:
        with self.lock:
            project = self.get(identifier)
            if revision != project.source_hash:
                raise source.ProjectError("Project source revision is no longer published")
            data = self._bundle(project).read_bytes()
            if len(data) > source.MAX_ARCHIVE or hashlib.sha256(data).hexdigest() != project.archive_sha256:
                raise source.ProjectError("Published project bundle failed verification")
            return data

    def workspace_base(self, identifier: str) -> Path:
        project = self.get(identifier)
        if project.authority_machine != self.machine:
            raise source.ProjectError("Project workspace belongs to another device")
        return self.root / "shared-workspaces" / project.id

    def catalogue(self) -> dict:
        code = bootstrap()
        return {"projects": self.list(), "machine": self.machine,
                "capabilities": ["project-source", "canonical-workspace"],
                "bootstrap_sha256": hashlib.sha256(code).hexdigest()}


def answer(handler, registry: ProjectRegistry | None, parsed) -> bool:
    """Answer authenticated source catalogue and immutable bundle requests."""
    prefix = "/workspace/v1/projects"
    if not parsed.path.startswith(prefix) or registry is None:
        return False
    pieces = parsed.path.removeprefix(prefix).strip("/").split("/")
    try:
        if parsed.path == prefix:
            handler._send(200, registry.catalogue())
        elif pieces == ["bootstrap"]:
            handler._send(200, {}, raw=bootstrap(), content_type="text/x-python")
        elif len(pieces) == 2 and pieces[1] == "snapshot":
            revision = urllib.parse.parse_qs(parsed.query).get("hash", [""])[0]
            handler._send(200, {}, raw=registry.snapshot(pieces[0], revision), content_type="application/zip")
        else:
            return False
    except (source.ProjectError, OSError) as exc:
        handler._send(400, {"error": str(exc)})
    return True
