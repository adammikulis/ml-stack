"""The public API (docs/api.md): the exact surface, its signatures, its docs, and that importing it costs nothing."""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

import poolhouse as ph
from poolhouse import api

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "poolhouse"
DOC = (REPO / "docs" / "api.md").read_text(encoding="utf-8")
SNAPSHOT = Path(__file__).parent / "api-signatures.json"
HEAVY = ("poolhouse.node_launch", "poolhouse.serve", "poolhouse.hub", "poolhouse.board", "poolhouse.client",
         "poolhouse.fleet", "poolhouse.testfarm", "poolhouse.workspace")


def surface() -> list[tuple[str, str, object]]:
    """Every public name as (namespace, name, object)."""
    return [(space, name, getattr(importlib.import_module(api.NAMESPACES[space]), name))
            for space, names in api.SURFACE.items() for name in names]


def test_the_facade_exports_exactly_the_documented_namespaces_and_errors() -> None:
    assert set(ph.__all__) == {"__version__", *api.ERRORS, *api.SURFACE}
    assert set(api.SURFACE) == set(api.NAMESPACES)
    assert all(isinstance(getattr(ph, name), type) and issubclass(getattr(ph, name), ph.Error) for name in api.ERRORS
               if name != "Error") and ph.Error is not None


def test_every_public_name_is_documented_with_its_namespace_and_none_is_missing() -> None:
    for space, names in api.SURFACE.items():
        for name in names:
            assert re.search(rf"`(ph\.)?{space}\.{name}\b", DOC), f"docs/api.md does not document {space}.{name}"
    for planned in api.PLANNED:
        assert planned in DOC, f"docs/api.md must list {planned} as planned"
        space, _, name = planned.partition(".")
        assert name.split("(")[0] not in api.SURFACE.get(space, ()), f"{planned} is planned but listed as public"


def test_every_public_name_is_exercised_by_a_test() -> None:
    text = "\n".join(p.read_text(encoding="utf-8") for p in (Path(__file__).parent).glob("test_api*.py"))
    missing = [f"{s}.{n}" for s, names in api.SURFACE.items() for n in names if not re.search(rf"\b{s}\.{n}\b|\b{n}\b", text)]
    assert not missing, f"no test in tests/test_api*.py mentions: {missing}"


def test_every_public_callable_has_a_docstring_and_type_hints() -> None:
    for space, name, obj in surface():
        if space in ("client", "hub") and not inspect.isfunction(obj):
            assert inspect.getdoc(obj), f"{space}.{name} has no docstring"
            continue
        assert inspect.getdoc(obj), f"{space}.{name} has no docstring"
        if inspect.isfunction(obj):
            hints = inspect.signature(obj)
            assert hints.return_annotation is not inspect.Signature.empty, f"{space}.{name} has no return type"
            bare = [p for p in hints.parameters.values() if p.annotation is inspect.Parameter.empty and p.name != "self"]
            assert not bare, f"{space}.{name} has untyped parameters {[p.name for p in bare]}"
        if inspect.isclass(obj) and space in ("pool", "board", "leases", "test", "serve"):
            for member, fn in inspect.getmembers(obj, inspect.isfunction):
                if not member.startswith("_"):
                    assert inspect.getdoc(fn), f"{space}.{name}.{member} has no docstring"


def signatures() -> dict[str, str]:
    out = {}
    for space, name, obj in surface():
        if inspect.isclass(obj):
            out[f"{space}.{name}"] = str(inspect.signature(obj))
            for member, fn in inspect.getmembers(obj, inspect.isfunction):
                if space not in ("client", "hub") and not member.startswith("_") and fn.__qualname__.startswith(obj.__name__):
                    out[f"{space}.{name}.{member}"] = str(inspect.signature(fn))
        elif callable(obj):
            out[f"{space}.{name}"] = str(inspect.signature(obj))
    return out


def test_a_public_signature_changes_only_deliberately() -> None:
    """The snapshot is the promise: a change here is a change of the API (docs/api.md, "Versions"). Update the
    file by running this test with POOLHOUSE_API_SNAPSHOT=write, and say why in the commit."""
    now = signatures()
    if os.environ.get("POOLHOUSE_API_SNAPSHOT") == "write":
        SNAPSHOT.write_text(json.dumps(now, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    promised = json.loads(SNAPSHOT.read_text(encoding="utf-8"))
    removed = sorted(set(promised) - set(now))
    changed = sorted(k for k in promised if k in now and promised[k] != now[k])
    assert not removed and not changed, f"public API changed: removed {removed}, changed {changed}"
    assert set(now) == set(promised), f"public names added without the snapshot: {sorted(set(now) - set(promised))}"


def test_importing_poolhouse_loads_nothing_heavy_and_is_fast() -> None:
    code = ("import sys, time; t = time.perf_counter(); import poolhouse; ms = (time.perf_counter() - t) * 1000; "
            "print(ms, sorted(m for m in sys.modules if m.startswith('poolhouse')))")
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    began = time.perf_counter()
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60, check=True)
    ms, _, loaded = done.stdout.partition(" ")
    assert float(ms) < 100, f"import poolhouse took {ms} ms"
    assert (time.perf_counter() - began) < 5
    modules = json.loads(loaded.replace("'", '"'))
    assert not [m for m in modules if m.startswith(HEAVY)], f"import poolhouse loaded {modules}"


def test_a_namespace_loads_only_when_first_used() -> None:
    code = "import sys, poolhouse as ph; a = 'poolhouse.api.pool' in sys.modules; ph.pool; print(a, 'poolhouse.api.pool' in sys.modules)"
    done = subprocess.run([sys.executable, "-c", code], env={**os.environ, "PYTHONPATH": str(REPO / "src")},
                          capture_output=True, text=True, timeout=60, check=True)
    assert done.stdout.split() == ["False", "True"]


def test_no_code_inside_poolhouse_imports_through_the_facade() -> None:
    """`import poolhouse as ph` is for other repos; inside the package an import names its module."""
    spaces = set(api.NAMESPACES)
    offenders = []
    for path in SRC.rglob("*.py"):
        if path == SRC / "__init__.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                offenders += [f"{path}: import {a.name} as {a.asname}" for a in node.names if a.name == "poolhouse" and a.asname]
            elif isinstance(node, ast.ImportFrom) and node.module == "poolhouse" and not node.level:
                offenders += [f"{path}: from poolhouse import {a.name}" for a in node.names if a.name in {"Error", *api.ERRORS, "__version__"}]
            elif isinstance(node, ast.ImportFrom) and node.module == "poolhouse.api" and path.parts[-2] != "api":
                offenders.append(f"{path}: from poolhouse.api import ...")
    assert not offenders, offenders
    assert spaces  # the facade is the one place that names them


def test_only_the_facade_and_the_api_package_know_the_api_package() -> None:
    users = {p.relative_to(SRC).as_posix() for p in SRC.rglob("*.py") if "poolhouse.api" in p.read_text(encoding="utf-8")}
    assert users <= {"__init__.py", "api/__init__.py", "api/pool.py", "api/leases.py", "api/remote_tests.py", "api/models.py",
                     "api/_node.py"}, users


def test_the_internal_errors_are_public_errors() -> None:
    from poolhouse.board import client
    from poolhouse.node_launch import NodeUnavailable
    from poolhouse.serve.holding import Refusal
    from poolhouse.testfarm.client import ShardError

    assert issubclass(client.Conflict, ph.Conflict) and issubclass(client.Denied, ph.Denied)
    assert issubclass(NodeUnavailable, ph.NotRunning) and issubclass(Refusal, ph.Error) and issubclass(ShardError, ph.Error)
    assert ph.board.Board is not None


def test_help_leads_with_pooling_devices() -> None:
    first = inspect.getdoc(ph).splitlines()[0]
    assert first.startswith("Poolhouse: pool every device's compute")
    assert inspect.getdoc(ph).index("ph.pool") < inspect.getdoc(ph).index("ph.board") < inspect.getdoc(ph).index("ph.serve")


@pytest.mark.parametrize("old", ["ml_stack", "ml-stack"])
def test_the_public_docs_use_the_new_name(old: str) -> None:
    for page in ("api.md", "api-migration.md"):
        text = (REPO / "docs" / page).read_text(encoding="utf-8")
        if page == "api-migration.md":
            text = text.replace("ml_stack", "").replace("ml-stack", "")
        assert old not in text, f"docs/{page} mentions {old}"
