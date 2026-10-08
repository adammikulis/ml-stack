"""The reuse key of one test file: its lookup key, and the manifest of everything a run touched."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import affected

NEVER_MARKS = frozenset({"heavy", "live_api", "live_net", "redteam", "gpu", "model"})
NEVER_WORDS = re.compile(r"\b(?:Lease|llama[-_]server|ml-stack-serve)\b")
SPAWNS = re.compile(r"\b(?:subprocess|Popen|os\.system|os\.exec\w*|os\.spawn\w*|os\.fork\w*|multiprocessing|pexpect|"
                    r"pty|create_subprocess_\w+|ProcessPoolExecutor|runpytest_subprocess|playwright|"
                    r"concurrent\.futures\.process)\b")
DYNAMIC = re.compile(r"\b(?:importlib\.util|runpy|pkgutil|entry_points|importorskip|__import__|import_module)\b")
UNSEEN = re.compile(r"\b(?:DirEntry|scandir|fstat|utime|chown|setxattr|mmap|sqlite3|ladybug)\b")
SNAPSHOT_SKIP = frozenset({".git", ".venv", "venv", "node_modules"})
RUNNER_PLUGINS = frozenset({"testslots_pytest", "testreuse_plugin"})
VALUE_OPTIONS = frozenset({"-m", "-k", "-c", "-o", "-W", "-p", "-n", "--rootdir", "--confcutdir", "--ignore",
                           "--ignore-glob", "--deselect", "--override-ini", "--maxfail", "--tb", "--basetemp",
                           "--import-mode", "--dist", "--numprocesses", "--junitxml", "--durations",
                           "--durations-min", "--log-level", "--color", "--assert", "--capture",
                           "--cov", "--timeout"})
DROPPED_OPTIONS = frozenset({"-n", "--numprocesses", "--junitxml"})
RUNNER_FILES = ("scripts/testreuse_plugin.py", "scripts/testreuse_key.py", "scripts/testreuse_store.py",
                "scripts/testreuse_run.py", "scripts/testslots_pytest.py", "scripts/testslots_rpc.py")
ENV_NAME = re.compile(r"""["']([A-Z][A-Z0-9_]{3,})["']""")
ENV_FIXED = ("PATH", "CI", "TZ", "LANG", "LC_ALL", "CLAUDECODE", "ML_STACK_NONINTERACTIVE",
             "ML_STACK_LIVE_API", "ML_STACK_LIVE_NET", "ML_STACK_NOTIFY", "DEV_TEST_SLOTS")
ENV_SKIP = frozenset({"PYTHONPATH", "DEV_TEST_PYTEST_TOKEN", "DEV_TEST_PYTEST_ENDPOINT",
                      "DEV_TEST_WORKERS", "DEV_TEST_SLOTS_DIR", "DEV_TEST_REMOTE_BROKER",
                      "DEV_TEST_LEASE", "DEV_TEST_REMOTE_LEASE", "DEV_TEST_REUSE_DIR",
                      "DEV_TEST_REUSE_CANARY", "DEV_TEST_REUSE_RECORD", "DEV_TEST_REUSE_ROOT",
                      "DEV_TEST_JOB", "DEV_TEST_AGENT", "ML_STACK_SHIM_LOG"})
CONFIG = ("pyproject.toml", "pytest.ini", "tox.ini", "setup.cfg", "tests/heavy-modules.txt", *RUNNER_FILES)
TREE_DIRS = ("src", "scripts", "tests", "docs", "packaging")
TREE_FILES = ("budgets.json", "AGENTS.md", "CLAUDE.md", "pyproject.toml", "HANDOFF.md")
IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", ".pytest_cache", ".testmondata", ".ruff_cache"})
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
    """A digest of the names in a directory without caches, or ``missing``."""
    try:
        return sha("\n".join(sorted(p.name for p in path.iterdir()
                                   if p.name not in IGNORED_NAMES and p.suffix != ".pyc")))
    except OSError:
        return "missing"


def tree_digest(root: Path) -> str:
    """One digest of the code, documents and packaging in the repository, and of its HEAD commit."""
    rows = []
    for top in TREE_DIRS:
        for path in sorted((root / top).rglob("*")):
            if path.is_file() and not IGNORED_NAMES.intersection(path.parts) and not (
                    top == "tests" and path.name.startswith("test_")):
                rows.append(f"{path.relative_to(root).as_posix()} {file_sha(path)}")
    rows += [f"{name} {file_sha(root / name)}" for name in TREE_FILES if (root / name).is_file()]
    head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return sha("\n".join([*rows, f"HEAD {head.stdout.strip() if head.returncode == 0 else ''}"]))


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

    def closure(self, rels: list[str], modules: tuple[str, ...] = ()) -> tuple[list[str], bool]:
        """The files in ``rels`` and the first-party files they and ``modules`` import at any depth, and
        whether any of them imports a module by a computed name or by machinery the closure cannot follow."""
        seen, todo, dynamic = set(), [], False
        for rel in rels:
            start = self.by_path.get(rel)
            dynamic = dynamic or bool(DYNAMIC.search(text_of(self.root / rel)))
            if start and start not in seen:
                seen.add(start)
                todo.append(start)
        for name in modules:
            hit = affected.resolve(name, self.known)
            for module in affected.parents(hit, self.known) if hit else ():
                if module not in seen:
                    seen.add(module)
                    todo.append(module)
        while todo:
            name = todo.pop()
            found, here = self.deps(name)
            dynamic = dynamic or here or bool(DYNAMIC.search(text_of(self.root / self.known[name])))
            for dep in found - seen:
                seen.add(dep)
                todo.append(dep)
        return sorted({self.known[n] for n in seen} | set(rels)), dynamic

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


def selector(word: str, root: Path) -> bool:
    """Whether a bare word names tests (a node, a ``.py`` file or an existing path) rather than being an option value."""
    return "::" in word or word.endswith(".py") or (root / word).exists()


def argument_digest(arguments: list[str], root: Path) -> str:
    """The pytest arguments that can change a result: no selectors, worker count, output paths or the
    runner's own plugins; the values of options (``-c``, ``--rootdir``, ``--ignore``, ``--log-cli-level``) stay."""
    words = list(arguments)
    if "pytest" in words:
        words = words[words.index("pytest") + 1:]
    kept, index = [], 0
    while index < len(words):
        word = words[index]
        index += 1
        if word in VALUE_OPTIONS and index < len(words):
            value = words[index]
            index += 1
            if word not in DROPPED_OPTIONS and not (word == "-p" and value in RUNNER_PLUGINS):
                kept += [word, value]
        elif word == "-q" or word.startswith(("--junitxml=", "--numprocesses=")) or (
                word.startswith("-n") and word[2:].isdigit()):
            continue
        elif word.startswith("-") or not selector(word, root):
            kept.append(word)
    return sha(json.dumps(kept))


_executed: dict[str, str] = {}


def executed_identity(arguments: list[str]) -> str:
    """The Python version and pytest version of the interpreter a command runs, asked of that interpreter."""
    exe = arguments[0] if arguments and Path(arguments[0]).is_file() and os.access(arguments[0], os.X_OK) \
        else sys.executable
    if exe not in _executed:
        done = subprocess.run([exe, "-c", "import sys, pytest; print(sys.version); print(pytest.__version__)"],
                              capture_output=True, text=True, check=False)
        _executed[exe] = done.stdout if done.returncode == 0 else f"unknown {sys.version}"
    return _executed[exe]


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
        "arguments": argument_digest(arguments, root),
        "pytest": sha(executed_identity(arguments)),
        "environment": sha("\n".join(f"{n}={env_digest(root, n)}" for n in sorted(names))),
        "runtime": sha(runtime_pin(root)),
    }
    return Lookup(rel, sha(json.dumps(parts, sort_keys=True)), parts, barred)


def plugin_modules(root: Path, rel: str, arguments: list[str]) -> tuple[str, ...]:
    """Modules the run loads besides imports: ``-p`` names and ``pytest_plugins`` of the applicable conftests."""
    names = [arguments[i + 1] for i, w in enumerate(arguments[:-1]) if w == "-p" and arguments[i + 1] not in RUNNER_PLUGINS
             and ":" not in arguments[i + 1]]
    for conftest in conftests(root, rel):
        try:
            tree = ast.parse(text_of(root / conftest))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in node.targets):
                names += [c.value for c in ast.walk(node.value) if isinstance(c, ast.Constant) and isinstance(c.value, str)]
    return tuple(names)


def build_manifest(root: Path, rel: str, seen: dict, closures: Closures, arguments: tuple[str, ...] = ()) -> dict:
    """What a run of ``rel`` depends on: the files, probes and directory listings the run recorded, the
    closure of the file, its conftests and plugins, the packages they import, the environment their
    modules name, and the whole tree when anything in the closure spawns processes or imports by machinery
    the closure cannot follow."""
    members, dynamic = closures.closure([rel, *conftests(root, rel)], plugin_modules(root, rel, list(arguments)))
    probed = set(seen.get("stats", ()))
    files = sorted(({*seen.get("reads", ()), *members, *(p for p in probed if (root / p).is_file())}) - {rel})
    files = [p for p in files if (root / p).is_file()]
    externals: set[str] = set()
    envs: set[str] = set()
    for path in files:
        if path.endswith(".py"):
            names, _ = import_names(text_of(root / path), "", False)
            externals |= {n.split(".")[0] for n in names}
            envs |= set(ENV_NAME.findall(text_of(root / path)))
    first_party = {k.split(".")[0] for k in closures.known}
    spawning = any(SPAWNS.search(text_of(root / p)) for p in members if p.endswith(".py"))
    unseen = next((m.group(0) for p in members if p.endswith(".py")
                   for m in [UNSEEN.search(text_of(root / p))] if m), "")
    return {"files": {p: file_sha(root / p) for p in files},
            "dirs": {d: dir_sha(root / d) for d in sorted({*seen.get("dirs", ()), *(p for p in probed if (root / p).is_dir())})},
            "absent": sorted(p for p in probed if not (root / p).exists()),
            "links": dict(seen.get("links", {})),
            "unobserved": unseen,
            "dists": pins_for({n for n in externals if n not in STDLIB and n not in first_party}),
            "env": {n: env_digest(root, n) for n in sorted(envs - ENV_SKIP)},
            "tree": closures.tree() if spawning or dynamic else ""}


def snapshot(root: Path) -> dict[str, tuple[int, int]]:
    """Modification time and size of every file under ``root`` except caches and environments, ignored files included."""
    found: dict[str, tuple[int, int]] = {}
    for folder, names, files in os.walk(root):
        names[:] = [n for n in names if n not in IGNORED_NAMES and n not in SNAPSHOT_SKIP]
        for name in files:
            if name.endswith(".pyc") or name.startswith(".testmondata") or name == ".DS_Store":
                continue
            path = Path(folder) / name
            try:
                stat = path.lstat()
            except OSError:
                continue
            found[path.relative_to(root).as_posix()] = (stat.st_mtime_ns, stat.st_size)
    return found


def manifest_holds(root: Path, manifest: dict, closures: Closures) -> str:
    """Empty when every recorded input still matches the checkout, else the first that does not."""
    for path, digest in manifest["files"].items():
        try:
            if file_sha(root / path) != digest:
                return f"changed: {path}"
        except OSError:
            return f"missing: {path}"
    for path in manifest.get("absent", ()):
        if (root / path).exists():
            return f"appeared: {path}"
    for path, target in manifest.get("links", {}).items():
        try:
            if (root / path).readlink().as_posix() != target:
                return f"link changed: {path}"
        except OSError:
            return f"link changed: {path}"
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
