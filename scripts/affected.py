"""Which test files a set of changed paths can reach, from the import graph.

A module's tests are the test files whose imports lead to it, directly or through other
modules, and the test files that name it in a string (``import_module``, ``-m`` arguments,
``monkeypatch.setattr("ml_stack.x.y", ...)``). A path nothing here can place is reported as
unmapped and the caller runs everything.
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import affected_rules as rules

PACKAGE_ROOTS = (("src", "ml_stack"), ("scripts", "gates"))
SCRIPT_TABLE = re.compile(r'''\[["']project["']\]\[["']scripts["']\]''')
DOTTED = re.compile(r"\b((?:ml_stack|gates|tests)(?:\.\w+)+|ml_stack|gates)\b")
INERT = {"CHANGELOG.md", "HANDOFF.md", "README.md", "LICENSE", "NOTICE", "CLAUDE.md",
         "AGENTS.md", "release-please-config.json", "version.txt", ".gitignore"}
EVERYTHING = {"pyproject.toml", "budgets.json", "tests/conftest.py", "tests/known-fixtures.txt"}
DEPTH = 1
GUARDS = ("test_layers.py", "test_wiring.py", "test_conftest_guard.py", "test_isolation_guard.py",
          "test_one_python.py", "test_suite.py")


@dataclass
class Selection:
    """The test files to run, why each was chosen, and the paths that forced a full run."""

    files: set[str] = field(default_factory=set)
    why: dict[str, list[str]] = field(default_factory=dict)
    unmapped: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)
    reasons: dict[str, str] = field(default_factory=dict)

    def unmap(self, rel: str, because: str) -> None:
        """Record a path no rule can place, so the caller runs everything, and say why."""
        self.unmapped.append(rel)
        self.reasons[rel] = because

    def add(self, test: str, because: str) -> None:
        self.files.add(test)
        self.why.setdefault(test, []).append(because)


def module_of(rel: str) -> str | None:
    """The dotted name of a repo-relative .py path under a package root, else None."""
    path = Path(rel)
    for base, package in PACKAGE_ROOTS:
        if path.parts[:1] == (base,) and path.suffix == ".py":
            parts = list(path.with_suffix("").parts[1:])
            if parts and parts[-1] == "__init__":
                parts.pop()
            if parts and (base == "scripts" or parts[0] == package):
                return ".".join(parts)
    if path.parts[:1] == ("tests",) and path.suffix == ".py" and len(path.parts) == 2:
        return f"tests.{path.stem}"
    return None


def index(root: Path) -> dict[str, str]:
    """Every importable module in the repo, dotted name to repo-relative path."""
    out: dict[str, str] = {}
    for base, _ in PACKAGE_ROOTS:
        for path in sorted((root / base).rglob("*.py")):
            rel = path.relative_to(root).as_posix()
            name = module_of(rel)
            if name:
                out[name] = rel
    for path in sorted((root / "tests").glob("*.py")):
        rel = path.relative_to(root).as_posix()
        out[module_of(rel) or path.stem] = rel
    return out


def resolve(name: str, known: dict[str, str]) -> str | None:
    """The longest module in ``known`` that is ``name`` or a package prefix of it."""
    for candidate in (name, f"tests.{name}"):
        parts = candidate.split(".")
        while parts:
            if ".".join(parts) in known:
                return ".".join(parts)
            parts.pop()
    return None


def parents(name: str, known: dict[str, str]) -> set[str]:
    """``name`` and every package above it that importing it runs."""
    parts = name.split(".")
    return {".".join(parts[:i]) for i in range(1, len(parts) + 1) if ".".join(parts[:i]) in known}


def absolute(node: ast.ImportFrom, module: str, is_package: bool) -> str:
    """The dotted base a ``from`` import names, with relative levels resolved."""
    if not node.level:
        return node.module or ""
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
    return ".".join([*base, *([node.module] if node.module else [])])


def references(source: str, module: str, is_package: bool, known: dict[str, str]) -> set[str]:
    """The modules in ``known`` that ``source`` imports or names in a string."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    wanted: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            wanted.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = absolute(node, module, is_package)
            wanted.add(base)
            wanted.update(f"{base}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            wanted.update(m.group(1) for m in DOTTED.finditer(node.value))
    found: set[str] = set()
    for name in wanted:
        hit = resolve(name, known)
        if hit:
            found |= parents(hit, known)
    found.discard(module)
    return found


def graph(root: Path, known: dict[str, str]) -> dict[str, set[str]]:
    """Module to the modules it depends on, over the whole index."""
    edges: dict[str, set[str]] = {}
    for name, rel in known.items():
        if name == "tests.conftest":
            edges[name] = set()
            continue
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        edges[name] = references(text, name, rel.endswith("__init__.py"), known)
    return edges


def dependants(edges: dict[str, set[str]]) -> dict[str, set[str]]:
    """The reverse of ``edges``: module to the modules that depend on it."""
    out: dict[str, set[str]] = {name: set() for name in edges}
    for name, deps in edges.items():
        for dep in deps:
            out.setdefault(dep, set()).add(name)
    return out


def reach(start: set[str], reverse: dict[str, set[str]], depth: int) -> set[str]:
    """``start`` and what depends on it within ``depth`` steps; a test module ends a path."""
    seen, frontier = set(start), set(start)
    for _ in range(depth):
        step = {y for x in frontier for y in reverse.get(x, ()) if y not in seen}
        seen |= step
        frontier = {y for y in step if not y.startswith("tests.")}
    return seen


def test_files(root: Path) -> list[str]:
    """Every collected test file, repo-relative."""
    return sorted(p.relative_to(root).as_posix() for p in (root / "tests").glob("test_*.py"))


def dynamic_tests(root: Path, tests: list[str]) -> list[str]:
    """Test files that import a module chosen at run time or read the console-script table."""
    out = []
    for rel in tests:
        text = (root / rel).read_text(encoding="utf-8", errors="replace")
        if SCRIPT_TABLE.search(text) or any(
                isinstance(n, ast.Call) and ast.unparse(n.func).endswith(("import_module", "__import__"))
                and n.args and not isinstance(n.args[0], ast.Constant)
                for n in ast.walk(ast.parse(text))):
            out.append(rel)
    return out


def mentioning(root: Path, rel: str, tests: list[str]) -> list[str]:
    """Test files whose text contains the changed path or its base name."""
    needles = {rel, Path(rel).name}
    return [t for t in tests if any(n in (root / t).read_text(encoding="utf-8", errors="replace")
                                    for n in needles)]


def package_dir_modules(rel: str, known: dict[str, str]) -> set[str]:
    """Modules that live beside a non-Python file under a package root."""
    folder = Path(rel).parent.as_posix() + "/"
    return {name for name, path in known.items() if path.startswith(folder)}


def add_script_tests(out: Selection, root: Path, rel: str, reverse: dict[str, set[str]],
                     tests: list[str]) -> None:
    """Select the tests that import, run or name the script ``rel``, through helpers too."""
    files = rules.corpus(root)
    why, modules, shared = rules.script_tests(files, rel)
    if shared:
        out.unmap(rel, shared)
        return
    direct = {n for n in why if not why[n].endswith("enumerates scripts/")}
    for test, reason in why.items():
        out.add(test, f"{rel}: {reason}")
    names = {m for m in (module_of(x) for x in modules) if m}
    for dependant in sorted(reach(names, reverse, DEPTH)):
        if dependant.startswith("tests.") and f"tests/{dependant[6:]}.py" in tests:
            out.add(f"tests/{dependant[6:]}.py", f"{rel} -> a source module that names it")
    if not (direct or names or rel.endswith(".py")):
        out.unmap(rel, "no test imports, runs or names it, so nothing says which tests it affects")


def select(root: Path, changed: list[str], deleted: frozenset[str] = frozenset(),
           depth: int = DEPTH, base: str = "0.2dev") -> Selection:
    """The test files within ``depth`` imports of ``changed``; ``unmapped`` non-empty means run all."""
    out = Selection()
    known = index(root)
    tests = test_files(root)
    reverse = dependants(graph(root, known))
    for rel in changed:
        name = Path(rel).name
        module = module_of(rel)
        script = rel.startswith("scripts/") and not rel.startswith("scripts/gates/")
        if rel in deleted and (module or script):
            out.unmap(rel, "a deleted module: anything may have used it")
        elif rel == "budgets.json":
            if rules.budgets_lowered(root, base):
                for test, why in rules.budget_tests(rules.corpus(root)).items():
                    out.add(test, f"{rel} only fell: {why}")
            else:
                out.unmap(rel, "a budget rose or the file changed in shape: every gate's baseline moved")
        elif rel in EVERYTHING:
            out.unmap(rel, "shared by every test")
        elif name in INERT or ((rel.endswith(".md") or rel.startswith(".github/")) and not mentioning(root, rel, tests)):
            out.ignored.append(rel)               # no test names it, so it cannot change a result (CI config, CODEOWNERS ...)
        elif rel in rules.GENERATED:
            checks = rules.generated_tests(rules.corpus(root), rel)
            for test, why in checks.items():
                out.add(test, f"{rel} is generated: {why}")
            if not checks:
                out.ignored.append(rel)           # only the gate (--check) reads it
        elif script:
            add_script_tests(out, root, rel, reverse, tests)
            if module:
                for dependant in sorted(reach({module}, reverse, depth)):
                    if dependant.startswith("tests.") and f"tests/{dependant[6:]}.py" in tests:
                        out.add(f"tests/{dependant[6:]}.py", f"{rel} -> {module}")
        elif module:
            hit = reach({module}, reverse, depth)
            for dependant in sorted(hit):
                if dependant.startswith("tests.") and f"tests/{dependant[6:]}.py" in tests:
                    out.add(f"tests/{dependant[6:]}.py", f"{rel} -> {module}")
        elif rel.startswith(("src/ml_stack/", "scripts/gates/")):
            near = package_dir_modules(rel, known)
            for dependant in sorted(reach(near, reverse, depth)):
                if dependant.startswith("tests.") and f"tests/{dependant[6:]}.py" in tests:
                    out.add(f"tests/{dependant[6:]}.py", f"{rel} -> its package")
        else:
            said = mentioning(root, rel, tests)
            for t in said:
                out.add(t, f"{rel} named in the test")
            if not said:
                out.unmap(rel, "not source, a test or prose, and no test names it")
    if any(Path(c).parts[:2] == ("src", "ml_stack") for c in changed):
        for rel in dynamic_tests(root, tests):
            out.add(rel, "imports a module chosen at run time")
    if out.files:
        for guard in GUARDS:
            if f"tests/{guard}" in tests:
                out.add(f"tests/{guard}", "tree-wide guard")
    return out


def lines(root: Path, *args: str) -> list[str]:
    """Output lines of a git command run in ``root``."""
    done = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
    if done.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {done.stderr.strip()}")
    return [x for x in done.stdout.splitlines() if x]


def changed_since(root: Path, base: str) -> tuple[list[str], frozenset[str]]:
    """Paths changed against the merge-base with ``base``, working tree and untracked included.

    The second value is the subset that no longer exists.
    """
    mb = lines(root, "merge-base", base, "HEAD")[0]
    paths = set(lines(root, "diff", "--name-only", "--no-renames", mb))
    paths |= set(lines(root, "ls-files", "--others", "--exclude-standard"))
    gone = frozenset(p for p in paths if not (root / p).exists())
    return sorted(paths), gone
