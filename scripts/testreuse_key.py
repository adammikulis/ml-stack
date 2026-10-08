"""The reuse key of one test file: its lookup key, and the manifest of everything a run touched."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import affected

NEVER_MARKS = frozenset({"heavy", "live_api", "live_net", "redteam", "gpu", "model"})
NEVER_WORDS = re.compile(r"\b(?:Lease|llama[-_]server|ml-stack-serve)\b")
SPAWNS = re.compile(r"\b(?:subprocess|Popen|os\.system|os\.exec\w*|os\.spawn\w*|multiprocessing|pexpect)\b")
ENV_NAME = re.compile(r"""["']([A-Z][A-Z0-9_]{3,})["']""")
ENV_FIXED = ("PATH", "CI", "TZ", "LANG", "LC_ALL", "CLAUDECODE", "ML_STACK_NONINTERACTIVE",
             "ML_STACK_LIVE_API", "ML_STACK_LIVE_NET", "ML_STACK_NOTIFY", "DEV_TEST_SLOTS")
ENV_SKIP = frozenset({"PYTHONPATH", "DEV_TEST_PYTEST_TOKEN", "DEV_TEST_PYTEST_ENDPOINT",
                      "DEV_TEST_WORKERS", "DEV_TEST_SLOTS_DIR", "DEV_TEST_REMOTE_BROKER",
                      "DEV_TEST_LEASE", "DEV_TEST_REMOTE_LEASE", "DEV_TEST_REUSE_DIR",
                      "DEV_TEST_REUSE_CANARY", "DEV_TEST_REUSE_RECORD", "DEV_TEST_REUSE_ROOT",
                      "DEV_TEST_JOB", "DEV_TEST_AGENT", "ML_STACK_SHIM_LOG"})
CONFIG = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "tests/heavy-modules.txt")
TREE_DIRS = ("src", "scripts", "tests")
STDLIB = frozenset(sys.stdlib_module_names)
_hashes: dict[tuple[str, int, int], str] = {}
_texts: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class Lookup:
    """A test file's lookup key, the labelled digests it is made of, and why it can never be reused."""

    file: str
    key: str
    parts: dict[str, str]
    barred: str = ""


def sha(data: bytes | str) -> str:
    """The hex sha256 of ``data``."""
    return hashlib.sha256(data.encode() if isinstance(data, str) else data).hexdigest()


def _stamp(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path), stat.st_mtime_ns, stat.st_size)


def file_sha(path: Path) -> str:
    """The sha256 of a file's bytes, memoised on size and modification time."""
    marker = _stamp(path)
    if marker not in _hashes:
        _hashes[marker] = sha(path.read_bytes())
    return _hashes[marker]


def text_of(path: Path) -> str:
    """A file's text, memoised on size and modification time."""
    marker = _stamp(path)
    if marker not in _texts:
        _texts[marker] = path.read_text(encoding="utf-8", errors="replace")
    return _texts[marker]


def dir_sha(path: Path) -> str:
    """A digest of the names in a directory, or ``missing``."""
    try:
        return sha("\n".join(sorted(os.listdir(path))))
    except OSError:
        return "missing"


def tree_digest(root: Path) -> str:
    """One digest of every code file under the repository's code directories."""
    rows = []
    for top in TREE_DIRS:
        for path in sorted((root / top).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and not (
                    top == "tests" and path.name.startswith("test_")):
                rows.append(f"{path.relative_to(root).as_posix()} {file_sha(path)}")
    return sha("\n".join(rows))


def marks_in(source: str) -> set[str]:
    """Every ``pytest.mark.<name>`` named anywhere in ``source``."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {"unparsable"}
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute) and node.value.attr == "mark"}


def import_names(source: str, module: str, is_package: bool) -> tuple[set[str], bool]:
    """The dotted names ``source`` imports, and whether it imports a module chosen at run time."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set(), True
    names, dynamic = set(), False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = affected.absolute(node, module, is_package)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call) and ast.unparse(node.func).endswith(("import_module", "__import__")):
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                names.add(node.args[0].value)
            else:
                dynamic = True
    return names, dynamic


class Closures:
    """Static import closures over one checkout, built from import statements only."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.known = affected.index(root)
        self.by_path = {path: name for name, path in self.known.items()}
        self.edges: dict[str, tuple[set[str], bool]] = {}
        self._tree = ""

    def deps(self, name: str) -> tuple[set[str], bool]:
        """The first-party modules ``name`` imports, and whether it imports one by a computed name."""
        if name not in self.edges:
            rel = self.known[name]
            names, dynamic = import_names(text_of(self.root / rel), name, rel.endswith("__init__.py"))
            found: set[str] = set()
            for wanted in names:
                hit = affected.resolve(wanted, self.known)
                if hit:
                    found |= affected.parents(hit, self.known)
            found.discard(name)
            self.edges[name] = (found, dynamic)
        return self.edges[name]

    def closure(self, rel: str) -> tuple[list[str], bool]:
        """``rel`` with the first-party files it imports at any depth, and any run-time import."""
        start = self.by_path.get(rel)
        if start is None:
            return [rel], False
        seen, todo, dynamic = {start}, [start], False
        while todo:
            name = todo.pop()
            found, here = self.deps(name)
            dynamic = dynamic or here
            for dep in found - seen:
                seen.add(dep)
                todo.append(dep)
        return sorted({self.known[n] for n in seen} | {rel}), dynamic

    def tree(self) -> str:
        """The digest of the whole code tree."""
        self._tree = self._tree or tree_digest(self.root)
        return self._tree


def conftests(root: Path, rel: str) -> list[str]:
    """The ``conftest.py`` files that apply to ``rel``, outermost first."""
    found, folder = [], (root / rel).parent
    while True:
        if (folder / "conftest.py").is_file():
            found.append((folder / "conftest.py").relative_to(root).as_posix())
        if folder == root or folder == folder.parent:
            return found[::-1]
        folder = folder.parent


def plugin_pins() -> list[str]:
    """``name==version`` of every installed distribution that registers a pytest plugin."""
    return sorted({f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                   if any(point.group == "pytest11" for point in d.entry_points)})


def pins_for(names: set[str]) -> list[str]:
    """``name==version`` of the installed distributions that provide the top-level ``names``."""
    owners = metadata.packages_distributions()
    return sorted({f"{owner}=={metadata.version(owner)}" for name in names for owner in owners.get(name, ())})


def env_digest(root: Path, name: str) -> str:
    """A digest of one environment variable's value with the checkout path masked and, for PATH, temporary entries dropped."""
    value = os.environ.get(name)
    if value is not None and name == "PATH":
        value = os.pathsep.join(p for p in value.split(os.pathsep) if not p.startswith(tempfile.gettempdir()))
    return "<unset>" if value is None else sha(value.replace(str(root), "<root>"))


def runtime_pin(root: Path) -> str:
    """The installed ml-stack distribution and how it was installed, or empty."""
    try:
        dist = metadata.distribution("ml-stack")
    except metadata.PackageNotFoundError:
        return ""
    return f"{dist.version} {dist.read_text('direct_url.json') or ''}".replace(str(root), "<root>")


def argument_digest(arguments: list[str]) -> str:
    """The pytest arguments that can change a result: no selectors, worker count or output paths."""
    kept, skip = [], False
    for word in arguments:
        if skip:
            skip = False
        elif word in ("-n", "--numprocesses", "-p"):
            skip = True
        elif word == "-q" or word.startswith(("--junitxml", "tests/", "-n")):
            continue
        else:
            kept.append(word)
    return sha(json.dumps(kept))


def lookup(root: Path, rel: str, arguments: list[str]) -> Lookup:
    """The lookup key of test file ``rel`` under ``arguments``; ``barred`` says why it is never reused."""
    texts = {p: text_of(root / p) for p in [rel, *conftests(root, rel)]}
    listed = root / "tests/heavy-modules.txt"
    barred = ""
    if marks_in(texts[rel]) & NEVER_MARKS:
        barred = "carries a heavy, live, redteam or model marker"
    elif listed.is_file() and Path(rel).name in listed.read_text().split():
        barred = "listed in tests/heavy-modules.txt"
    elif NEVER_WORDS.search(texts[rel]):
        barred = "uses a model or GPU lease"
    names = set(ENV_FIXED).union(*(set(ENV_NAME.findall(t)) for t in texts.values())) - ENV_SKIP
    parts = {
        "files": sha("\n".join(f"{p} {file_sha(root / p)}" for p in sorted({
            *texts, *(c for c in CONFIG if (root / c).is_file())}))),
        "interpreter": sha(f"{sys.version} {platform.platform()} {platform.machine()} "
                           f"{platform.python_implementation()}"),
        "plugins": sha("\n".join(plugin_pins())),
        "arguments": argument_digest(arguments),
        "environment": sha("\n".join(f"{n}={env_digest(root, n)}" for n in sorted(names))),
        "runtime": sha(runtime_pin(root)),
    }
    return Lookup(rel, sha(json.dumps(parts, sort_keys=True)), parts, barred)


def build_manifest(root: Path, rel: str, reads: set[str], dirs: set[str], closures: Closures) -> dict:
    """What a run of ``rel`` depends on: the files and directories it touched, the packages they
    import, the environment their modules name, and the whole code tree when it spawns processes."""
    static, _ = closures.closure(rel)
    files = sorted({*reads, *static, *conftests(root, rel)} - {rel})
    files = [p for p in files if (root / p).is_file()]
    externals: set[str] = set()
    envs: set[str] = set()
    for path in files:
        if path.endswith(".py"):
            names, _ = import_names(text_of(root / path), "", False)
            externals |= {n.split(".")[0] for n in names}
            envs |= set(ENV_NAME.findall(text_of(root / path)))
    first_party = {k.split(".")[0] for k in closures.known}
    spawning = any(SPAWNS.search(text_of(root / p)) for p in static
                   if p.startswith("tests/") and not p.endswith("conftest.py"))
    return {"files": {p: file_sha(root / p) for p in files},
            "dirs": {d: dir_sha(root / d) for d in sorted(dirs)},
            "dists": pins_for({n for n in externals if n not in STDLIB and n not in first_party}),
            "env": {n: env_digest(root, n) for n in sorted(envs - ENV_SKIP)},
            "tree": closures.tree() if spawning else ""}


def manifest_holds(root: Path, manifest: dict, closures: Closures) -> str:
    """Empty when every recorded input still matches the checkout, else the first that does not."""
    for path, digest in manifest["files"].items():
        try:
            if file_sha(root / path) != digest:
                return f"changed: {path}"
        except OSError:
            return f"missing: {path}"
    for path, digest in manifest["dirs"].items():
        if dir_sha(root / path) != digest:
            return f"listing changed: {path}"
    for name, digest in manifest["env"].items():
        if env_digest(root, name) != digest:
            return f"environment changed: {name}"
    if manifest["tree"] and closures.tree() != manifest["tree"]:
        return "code tree changed"
    wanted = {pin.split("==")[0] for pin in manifest["dists"]}
    installed = {f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions()
                 if d.metadata["Name"] in wanted}
    if set(manifest["dists"]) != installed:
        return "installed distributions changed"
    return ""


def manifest_digest(manifest: dict) -> str:
    """A digest of a manifest."""
    return sha(json.dumps(manifest, sort_keys=True))
