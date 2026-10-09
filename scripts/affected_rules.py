"""The narrow cases of ``affected.select``: a changed script, a lowered budget, a generated file.

A script is reached by every test file that imports it, names ``scripts/<name>`` in a string or a
command line, builds its path from the bare name, or reaches it through another script or a
tests helper that does. Anything these rules cannot place stays with the caller, which runs
everything; where a rule is unsure it selects more.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

SKIP = {"__pycache__", ".git", "node_modules"}
LIMIT = 600_000
LISTERS = ("glob(", "iterdir(", "listdir(", "scandir(", "os.walk(", "ls-files")
GENERATED = {"docs/redteam/coverage.json": "scripts/redteam_coverage.py",
             "docs/commands.md": "scripts/reference"}
BUDGETS = "budgets.json"


@dataclass
class Source:
    """One scanned file: its text, the names it imports and the string constants it holds."""

    rel: str
    text: str
    segments: set[str] = field(default_factory=set)
    consts: set[str] = field(default_factory=set)
    blob: str = ""
    parsed: bool = False

    def in_scripts_context(self) -> bool:
        """True when a bare name in this file can be a script: it lives in scripts/, holds the
        word ``scripts`` in a string or imports something named for scripts."""
        return (self.rel.startswith("scripts/") or "scripts" in self.blob.lower()
                or any("script" in seg.lower() for seg in self.segments))

    @property
    def lists_scripts(self) -> bool:
        """True when the file enumerates a directory and mentions ``scripts``."""
        return self.parsed and "scripts" in self.blob and any(x in self.text for x in LISTERS)


def parse(rel: str, text: str) -> Source:
    """The imports and string constants of ``text`` when it is Python, else the text alone."""
    src = Source(rel, text)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return src
    src.parsed = True
    prose = {id(n.body[0].value) for n in ast.walk(tree) if isinstance(
        n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body
        and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant)}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                src.segments.update(alias.name.split("."))
        elif isinstance(node, ast.ImportFrom):
            src.segments.update((node.module or "").split("."))
            src.segments.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in prose:
            src.consts.add(node.value)
    src.blob = "\n".join(sorted(src.consts))
    return src


def corpus(root: Path) -> dict[str, Source]:
    """Every script, tests Python file and source module, scanned."""
    out: dict[str, Source] = {}
    for top, pattern in (("scripts", "*"), ("tests", "*.py"), ("src", "*.py")):
        for path in sorted((root / top).rglob(pattern)):
            if not path.is_file() or SKIP & set(path.parts) or path.stat().st_size > LIMIT:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            rel = path.relative_to(root).as_posix()
            out[rel] = parse(rel, text)
    return out


@dataclass
class Target:
    """How other files name one file: by path, by import or by a bare name in a string."""

    rel: str
    path: re.Pattern
    shell: re.Pattern
    names: set[str]
    stem: str | None


def target(rel: str) -> Target:
    """The patterns that find references to ``rel``."""
    path = Path(rel)
    name = path.name
    stem = path.stem if path.suffix == ".py" else None
    inner = path.with_suffix("").as_posix() if stem else rel
    return Target(
        rel,
        re.compile(re.escape(inner) + (r"(?:\.py)?" if stem else "") + r"(?![\w.-])"),
        re.compile(r"(?<![\w.-])(?:\./|\$\{?\w+\}?/|/)" + re.escape(name) + r"(?![\w.-])"),
        {name, *([stem] if stem else [])},
        stem)


def refers(src: Source, tgt: Target) -> str:
    """Why ``src`` names the target, or an empty string."""
    if src.rel == tgt.rel:
        return ""
    if tgt.stem and tgt.stem in src.segments and src.parsed:
        return f"imports {tgt.stem}"
    if tgt.path.search(src.blob if src.parsed else src.text):
        return f"names {tgt.rel}"
    if any(c.endswith("/" + n) for c in src.consts for n in tgt.names):
        return f"names a path ending in {Path(tgt.rel).name}"
    if src.consts & tgt.names and src.in_scripts_context():
        return f"names {sorted(src.consts & tgt.names)[0]!r} in a string"
    if not src.parsed and tgt.shell.search(src.text):
        return f"runs {Path(tgt.rel).name}"
    return ""


def is_test(rel: str) -> bool:
    """True for a collected test file."""
    return rel.startswith("tests/test_") and rel.count("/") == 1 and rel.endswith(".py")


PLUGINS = re.compile(r"""["']-p["'],\s*["'](\w+)["']""")


def infrastructure(files: dict[str, Source]) -> dict[str, str]:
    """Files that run in every test session: conftest, the package marker and the ``-p`` plugins."""
    out = {"tests/conftest.py": "tests/conftest.py", "tests/__init__.py": "tests/__init__.py"}
    runner = files.get("scripts/test")
    for name in PLUGINS.findall(runner.text) if runner else ():
        out[f"scripts/{name}.py"] = f"the {name} plugin that scripts/test loads"
    return out


def reaching(files: dict[str, Source], rel: str) -> tuple[dict[str, str], set[str], str]:
    """Test files that reach ``rel`` through scripts and helpers, with why, the src modules on
    the way and, when a file every test session runs depends on ``rel``, which one."""
    why: dict[str, str] = {}
    modules: set[str] = set()
    infra = infrastructure(files)
    shared = f"{infra[rel]} runs in every test session" if rel in infra else ""
    seen, queue = {rel}, [(rel, True)]
    while queue:
        current, pure = queue.pop()
        tgt = target(current)
        for name, src in files.items():
            reason = refers(src, tgt)
            if not reason:
                continue
            if name in infra:
                if pure:    # its code runs in every session; a name only held by a helper may never run
                    shared = shared or f"{infra[name]} {reason} (every test session loads it)"
            elif is_test(name):
                why.setdefault(name, f"{name} {reason}" + ("" if current == rel else f" (via {current})"))
            elif name.startswith("src/"):
                modules.add(name)
            elif name not in seen:
                seen.add(name)
                queue.append((name, pure and reason.startswith("imports")))
    return why, modules, shared


def listers(files: dict[str, Source]) -> list[str]:
    """Test files that enumerate directories while mentioning ``scripts``."""
    return sorted(n for n, s in files.items() if is_test(n) and s.lists_scripts)


def script_tests(files: dict[str, Source], rel: str) -> tuple[dict[str, str], set[str], str]:
    """Test files that exercise the script ``rel``, with why, the src modules that name it and
    the session-wide file that loads it, if any."""
    why, modules, shared = reaching(files, rel)
    for name in listers(files):
        why.setdefault(name, f"{name} enumerates scripts/")
    return why, modules, shared


def lowered_only(old: object, new: object) -> bool:
    """True when ``new`` differs from ``old`` only by integers that did not rise."""
    if isinstance(old, dict) and isinstance(new, dict):
        return old.keys() == new.keys() and all(lowered_only(old[k], new[k]) for k in old)
    ints = isinstance(old, int) and isinstance(new, int) and not isinstance(old, bool) \
        and not isinstance(new, bool)
    return ints and new <= old  # type: ignore[operator]


def budgets_lowered(root: Path, base: str) -> bool:
    """True when budgets.json against its merge-base with ``base`` only fell; unsure is False."""
    try:
        mb = subprocess.run(["git", "merge-base", base, "HEAD"], cwd=root, capture_output=True,
                            text=True, check=True).stdout.strip()
        old = subprocess.run(["git", "show", f"{mb}:{BUDGETS}"], cwd=root, capture_output=True,
                             text=True, check=True).stdout
        return lowered_only(json.loads(old), json.loads((root / BUDGETS).read_text(encoding="utf-8")))
    except (subprocess.CalledProcessError, OSError, ValueError):
        return False


def budget_tests(files: dict[str, Source]) -> dict[str, str]:
    """Test files that check or read budgets.json, directly or through the scripts that read it."""
    why = {n: f"{n} checks the budgets" for n in files if is_test(n) and Path(n).name.startswith(
        ("test_budget", "test_optional_task_budget"))}
    for name, src in files.items():
        if BUDGETS in src.text and not name.startswith("src/"):
            if is_test(name):
                why.setdefault(name, f"{name} reads {BUDGETS}")
            elif not name.startswith("tests/"):
                why.update({k: f"{v} (reads {BUDGETS})" for k, v in reaching(files, name)[0].items()})
    return why


def generated_tests(files: dict[str, Source], rel: str) -> dict[str, str]:
    """Test files that check the generated ``rel`` or its generator; the gate checks the rest."""
    why = {n: f"{n} names {rel}" for n, s in files.items()
           if is_test(n) and Path(rel).name in s.text}
    why.update(reaching(files, GENERATED[rel])[0])
    return why
