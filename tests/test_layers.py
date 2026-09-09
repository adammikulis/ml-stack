"""Which package may import which.

The layers are core, model, graph, machine, tools. A package may import a package in a
lower layer, and one in its own layer as long as the other does not import it back.
``KNOWN`` lists the edges that break that today; it may only shrink.
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "src" / "ml_stack"

LAYERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("core", ("backend", "command", "contracts", "data", "entities", "files", "geo",
              "home", "http", "jobs", "jsonl", "lock", "log", "media", "paths",
              "platform", "redact", "scrape", "records", "telemetry", "ui", "units")),
    ("model", ("client", "gguf", "hub", "speech", "vision")),
    ("graph", ("graph", "ingest", "sources", "world")),
    ("machine", ("fleet", "serve", "setup")),
    ("tools", ("bench", "claude", "cli", "do", "draft", "harness", "mcp", "surface",
               "testing", "train", "walk", "web")),
)

RANK = {package: height for height, (_, packages) in enumerate(LAYERS)
        for package in packages}

KNOWN = {
    ("fleet", "bench"),
    ("fleet", "serve"),
    ("fleet", "setup"),
    ("gguf", "serve"),
    ("graph", "serve"),
    ("graph", "train"),
    ("hub", "serve"),
    ("ingest", "bench"),
    ("ingest", "serve"),
    ("ingest", "web"),
    ("serve", "bench"),
    ("serve", "fleet"),
    ("serve", "setup"),
    ("setup", "bench"),
    ("setup", "fleet"),
    ("setup", "serve"),
    ("sources", "web"),
    ("sources", "world"),
    ("world", "sources"),
}


def _package_of_file(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    return parts[0][:-3] if len(parts) == 1 else parts[0]


def _package_of_module(name: str) -> str | None:
    parts = name.split(".")
    return parts[1] if len(parts) > 1 else None


def _imports() -> dict[tuple[str, str], set[str]]:
    """Every ``ml_stack`` import in the tree, keyed by (importer, imported) package."""
    found: dict[tuple[str, str], set[str]] = {}
    for path in sorted(ROOT.rglob("*.py")):
        source = _package_of_file(path)
        where = str(path.relative_to(REPO))
        for node in ast.walk(ast.parse(path.read_text(), str(path))):
            named: list[str] = []
            if isinstance(node, ast.Import):
                named = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and not node.level:
                module = node.module or ""
                named = [module]
                if module == "ml_stack":
                    named += [f"ml_stack.{alias.name}" for alias in node.names]
            for name in named:
                if name != "ml_stack" and not name.startswith("ml_stack."):
                    continue
                target = _package_of_module(name)
                if target and target != source:
                    found.setdefault((source, target), set()).add(where)
    return found


def _packages() -> set[str]:
    return {_package_of_file(path) for path in ROOT.rglob("*.py")}


def _violations(edges: dict[tuple[str, str], set[str]]) -> set[tuple[str, str]]:
    """Edges that point up a layer, or across one layer in both directions."""
    out: set[tuple[str, str]] = set()
    for source, target in edges:
        if RANK[source] < RANK[target]:
            out.add((source, target))
        elif RANK[source] == RANK[target] and (target, source) in edges:
            out.add((source, target))
    return out


def test_every_package_is_placed_in_a_layer() -> None:
    unplaced = sorted(_packages() - set(RANK))
    assert not unplaced, f"packages with no layer: {unplaced}"


def test_no_import_reaches_above_its_layer() -> None:
    edges = _imports()
    new = sorted(_violations(edges) - KNOWN)
    detail = [f"{a} -> {b}  {sorted(edges[(a, b)])}" for a, b in new]
    assert not new, "layer violations not in KNOWN:\n" + "\n".join(detail)


def test_known_holds_nothing_already_fixed() -> None:
    stale = sorted(KNOWN - _violations(_imports()))
    assert not stale, f"no longer violations, delete from KNOWN: {stale}"


def _loaded(module: str) -> set[str]:
    """The ``ml_stack`` modules a fresh interpreter loads when it imports ``module``."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src")] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    code = (f"import sys, {module}; "
            "print('\\n'.join(m for m in sys.modules if m.startswith('ml_stack')))")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          env=env, timeout=120)
    assert done.returncode == 0, done.stderr
    return set(done.stdout.split())


def test_a_client_does_not_load_the_fleet() -> None:
    unwanted = {"ml_stack.fleet.daemon", "ml_stack.fleet.discovery"}
    for module in ("ml_stack.client", "ml_stack.hub", "ml_stack.serve.cli"):
        pulled = _loaded(module) & unwanted
        assert not pulled, f"{module} loads {sorted(pulled)}"
