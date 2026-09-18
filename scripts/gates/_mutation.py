"""Semantic mutations of one function, the tests that name it, and a tree to run them in.

Holds the mutation operators, the sampler, the test picker, and the copied working tree a
mutant is written into. Nothing here writes to the repository it reads.
"""

from __future__ import annotations

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from random import Random

SOURCE_ROOT = "src/ml_stack"
TEST_ROOT = "tests"
COPY_SKIP = {".git", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build",
             "node_modules", ".venv", "venv"}
NEVER_RUN = ("tests/test_budgets.py", "tests/test_gates_mutation.py",
             "tests/test_asking_is_the_same_asking.py")
FLIP = {ast.Lt: ast.GtE, ast.GtE: ast.Lt, ast.Gt: ast.LtE, ast.LtE: ast.Gt,
        ast.Eq: ast.NotEq, ast.NotEq: ast.Eq, ast.Is: ast.IsNot, ast.IsNot: ast.Is,
        ast.In: ast.NotIn, ast.NotIn: ast.In}
KINDS = ("flip-comparison", "swap-boolop", "negate-condition", "skip-branch",
         "constant-return", "empty-body")


@dataclass(frozen=True)
class Target:
    """One function a mutation can be applied to."""

    path: str
    qualname: str
    lineno: int

    def name(self) -> str:
        return f"{self.path}::{self.qualname}"


@dataclass(frozen=True)
class Mutant:
    """One mutation of one function, with the whole mutated file."""

    target: Target
    kind: str
    index: int
    lineno: int
    source: str

    def name(self) -> str:
        return f"{self.kind}:{self.index}"


@dataclass(frozen=True)
class Outcome:
    """What the tests did about one mutant."""

    mutant: Mutant
    tests: tuple[str, ...]
    survived: bool
    note: str


def functions(tree: ast.Module) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    """Every function in the module, by dotted qualified name."""
    out: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def walk(body: list[ast.stmt], prefix: str) -> None:
        for node in body:
            if isinstance(node, ast.ClassDef):
                walk(node.body, f"{prefix}{node.name}.")
            elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                out.append((f"{prefix}{node.name}", node))

    walk(tree.body, "")
    return out


def _guarded(node: ast.If) -> bool:
    """True for an ``if`` whose branch is not taken at run time anyway."""
    text = ast.unparse(node.test)
    return "TYPE_CHECKING" in text or "__name__" in text


def _sites(func: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every (kind, node) a mutation operator applies to, in a stable order."""
    found: list[tuple[str, ast.AST]] = [("empty-body", func)]
    for node in ast.walk(func):
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and type(node.ops[0]) in FLIP:
            found.append(("flip-comparison", node))
        elif isinstance(node, ast.BoolOp):
            found.append(("swap-boolop", node))
        elif isinstance(node, ast.If) and not _guarded(node):
            found.append(("negate-condition", node))
            found.append(("skip-branch", node))
        elif isinstance(node, ast.Return) and node.value is not None:
            found.append(("constant-return", node))
    return found


def _apply(kind: str, node: ast.AST) -> None:
    """Change the node in place so the function means something else."""
    if kind == "flip-comparison":
        node.ops = [FLIP[type(node.ops[0])]()]
    elif kind == "swap-boolop":
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    elif kind == "negate-condition":
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
    elif kind == "skip-branch":
        node.test = ast.Constant(value=False)
    elif kind == "constant-return":
        was = node.value
        flipped = not was.value if isinstance(was, ast.Constant) and was.value in (True, False) \
            else True
        node.value = ast.Constant(value=flipped)
    elif kind == "empty-body":
        node.body = [ast.Return(value=ast.Constant(value=None))]


def _splice(source: str, func: ast.AST, replacement: str) -> str:
    """The file with the function's own lines replaced by ``replacement``."""
    lines = source.splitlines(keepends=True)
    indent = " " * func.col_offset
    body = "".join(indent + line + "\n" for line in replacement.splitlines())
    return "".join(lines[:func.lineno - 1]) + body + "".join(lines[func.end_lineno:])


def mutants(source: str, target: Target) -> list[Mutant]:
    """Every mutant of one function, dropping the ones that change no source."""
    out: list[Mutant] = []
    original = ast.parse(source)
    named = dict(functions(original))
    if target.qualname not in named:
        return out
    before = ast.unparse(named[target.qualname])
    for index, (kind, _) in enumerate(_sites(named[target.qualname])):
        fresh = dict(functions(ast.parse(source)))[target.qualname]
        site = _sites(fresh)[index][1]
        at = getattr(site, "lineno", target.lineno)
        _apply(kind, site)
        fresh.decorator_list = []
        rendered = ast.unparse(fresh)
        mutated = _splice(source, named[target.qualname], rendered)
        if rendered == before or not _parses(mutated):
            continue
        out.append(Mutant(target, kind, index, at, mutated))
    return out


def _parses(source: str) -> bool:
    try:
        ast.parse(source)
    except SyntaxError:
        return False
    return True


def candidates(root: Path) -> list[Target]:
    """Every function under src/ml_stack that at least one mutation operator reaches."""
    out: list[Target] = []
    base = root / SOURCE_ROOT
    for path in sorted(base.rglob("*.py")):
        if any(part in COPY_SKIP for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        where = path.relative_to(root).as_posix()
        for qualname, node in functions(tree):
            if len(_sites(node)) > 1:
                out.append(Target(where, qualname, node.lineno))
    return out


def sample(targets: list[Target], count: int, seed: str) -> list[Target]:
    """``count`` targets chosen the same way every time for the same seed."""
    picked = Random(seed).sample(targets, min(count, len(targets)))  # noqa: S311
    return sorted(picked, key=lambda t: (t.path, t.lineno))


def dotted(path: str) -> str:
    """The module path a source file is imported as."""
    return path.removeprefix("src/").removesuffix(".py").removesuffix("/__init__") \
        .replace("/", ".")


def covering_tests(root: Path, target: Target, limit: int) -> list[str]:
    """The test files that import the module, best first.

    A file that does not name the module cannot be said to cover it, and running one that
    does not would report a survivor the tests never had a chance at.
    """
    module = dotted(target.path)
    leaf = target.qualname.split(".")[-1]
    scored: list[tuple[int, str]] = []
    for path in sorted((root / TEST_ROOT).glob("test_*.py")):
        where = path.relative_to(root).as_posix()
        if where in NEVER_RUN:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        named = text.count(module)
        if not named:
            continue
        score = 4 * named
        if len(leaf) > 3 and not leaf.startswith("__"):
            score += 3 * len(re.findall(rf"\b{re.escape(leaf)}\b", text))
        scored.append((score, where))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [where for _, where in scored[:limit]]


class Tree:
    """A copy of the working tree that mutants are written into and tests run in."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.where = Path(tempfile.mkdtemp(prefix="ml-stack-mutation-"))
        self._copy()

    def _tracked(self) -> list[str] | None:
        done = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"],
                              cwd=self.root, capture_output=True, text=True, check=False)
        if done.returncode != 0:
            return None
        return [line for line in done.stdout.splitlines() if line]

    def _copy(self) -> None:
        names = self._tracked()
        if names is None:
            shutil.copytree(self.root, self.where, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(*COPY_SKIP))
            return
        for name in names:
            source = self.root / name
            if not source.is_file():
                continue
            target = self.where / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

    def imports_the_copy(self) -> bool:
        """True when a test run in the copy reads ml_stack out of the copy first.

        ``ml_stack`` is a namespace package, so every path holding one is merged; only the
        first of them answers for a module that exists in all of them.
        """
        done = self.python("-c", "import ml_stack, sys; sys.stdout.write(ml_stack.__path__[0])")
        return done.stdout.startswith(str(self.where))

    def python(self, *args: str, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
        """Run the interpreter in the copy with the copy's src first on the path."""
        return subprocess.run(
            [sys.executable, *args], cwd=self.where, capture_output=True, text=True,
            check=False, timeout=timeout,
            env={"PYTHONPATH": str(self.where / "src"), "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                 "HOME": os.environ.get("HOME", tempfile.gettempdir()),
                 "PYTHONDONTWRITEBYTECODE": "1", "ML_STACK_MUTATION": "1"})

    def write(self, relpath: str, source: str) -> None:
        (self.where / relpath).write_text(source, encoding="utf-8")

    def restore(self, relpath: str) -> None:
        shutil.copyfile(self.root / relpath, self.where / relpath)

    def pytest(self, tests: tuple[str, ...], timeout: float) -> tuple[bool, str]:
        """(every test passed, the last lines of output); a timeout counts as a failure."""
        try:
            done = self.python("-m", "pytest", *tests, "-q", "-x", "--no-header",
                               "-p", "no:cacheprovider", "-p", "no:randomly",
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"timed out after {timeout:.0f}s"
        tail = "\n".join((done.stdout + done.stderr).strip().splitlines()[-3:])
        return done.returncode == 0, tail

    def close(self) -> None:
        shutil.rmtree(self.where, ignore_errors=True)


def head(root: Path) -> str:
    """The commit the seed is taken from, or 'unversioned'."""
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                          capture_output=True, text=True, check=False)
    return done.stdout.strip() or "unversioned"


def pick(found: list[Mutant], count: int, seed: str) -> list[Mutant]:
    """``count`` mutants of one function, spread across kinds, stable for the same seed."""
    by_kind: dict[str, list[Mutant]] = {}
    for mutant in found:
        by_kind.setdefault(mutant.kind, []).append(mutant)
    chooser = Random(seed)  # noqa: S311
    kinds = sorted(by_kind)
    chooser.shuffle(kinds)
    for kind in kinds:
        chooser.shuffle(by_kind[kind])
    out: list[Mutant] = []
    while len(out) < count and any(by_kind[kind] for kind in kinds):
        for kind in kinds:
            if by_kind[kind] and len(out) < count:
                out.append(by_kind[kind].pop())
    return out


def campaign(tree: Tree, root: Path, targets: list[Target], shape: dict) -> list[Outcome]:
    """Mutate each target and run the tests that name it; one Outcome per mutant run."""
    say = shape["say"]
    timeout = shape["timeout"]
    outcomes: list[Outcome] = []
    baselines: dict[tuple[str, ...], tuple[bool, str]] = {}
    for target in targets:
        tests = shape.get("only") or tuple(covering_tests(root, target, shape["tests"]))
        if not tests:
            say(f"  {target.name()}: no test file names it; not measured")
            continue
        if tests not in baselines:
            baselines[tests] = tree.pytest(tests, timeout)
        green, note = baselines[tests]
        if not green:
            say(f"  {target.name()}: {' '.join(tests)} red before mutation -- {note}")
            continue
        source = (root / target.path).read_text(encoding="utf-8")
        for mutant in pick(mutants(source, target), shape["mutations"], target.name()):
            tree.write(target.path, mutant.source)
            try:
                survived, tail = tree.pytest(tests, timeout)
            finally:
                tree.restore(target.path)
            outcomes.append(Outcome(mutant, tests, survived, tail))
            say(f"  {target.name()} {mutant.name()}: "
                f"{'SURVIVED' if survived else 'caught'} ({' '.join(tests)})")
    return outcomes
