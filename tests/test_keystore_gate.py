"""Nothing but `poolhouse/keystore.py` reaches the operating system's keystore.

The scan reads every module under ``src/poolhouse`` and flags an import of `keyring`, a
`import_module` of it, a call to one of the backend's three methods, and a command-line
keystore tool run by subprocess. A module that wants a key asks `poolhouse.keystore`, which
counts, limits and audits every call.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent / "src" / "poolhouse"
ALLOWED = {"keystore.py": "the one module that talks to the keystore"}
METHODS = {"get_password", "set_password", "delete_password", "get_credential"}
LOADERS = {"import_module", "__import__"}
TOOLS = {"security", "secret-tool", "cmdkey", "kwallet-query", "gnome-keyring-daemon"}
KEYSTORE_VERBS = {"find-generic-password", "add-generic-password", "delete-generic-password",
                  "find-internet-password", "lookup", "store", "clear"}


def is_keyring(name: str) -> bool:
    return name == "keyring" or name.startswith("keyring.")


def findings(path: Path) -> list[str]:
    found: list[str] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found += [f"import {a.name}" for a in node.names if is_keyring(a.name)]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level and is_keyring(node.module):
            found.append(f"from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in METHODS:
                found.append(f"call {name}")
            if name in LOADERS and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str) and is_keyring(node.args[0].value):
                found.append(f"{name} of keyring")
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                words = [e.value for e in node.args[0].elts if isinstance(e, ast.Constant)
                         and isinstance(e.value, str)]
                if words and words[0] in TOOLS and KEYSTORE_VERBS & set(words):
                    found.append(f"runs {words[0]}")
    return found


def modules() -> list[tuple[str, Path]]:
    return [(p.relative_to(ROOT).as_posix(), p) for p in sorted(ROOT.rglob("*.py"))]


def test_no_module_but_the_keystore_reaches_the_os_keystore():
    bad = [f"{rel}: {item}" for rel, path in modules() if rel not in ALLOWED for item in findings(path)]
    assert not bad, "OS keystore access outside poolhouse.keystore:\n  " + "\n  ".join(bad)


def test_the_keystore_module_is_where_the_access_lives():
    assert all((ROOT / rel).exists() for rel in ALLOWED)
    assert findings(ROOT / "keystore.py")


@pytest.mark.parametrize("source", [
    "import keyring\n",
    "import keyring.errors\n",
    "from keyring import get_password\n",
    "from keyring.errors import KeyringError\n",
    "import importlib\nimportlib.import_module('keyring')\n",
    "ring.set_password('s', 'a', 'v')\n",
    "backend.get_password('s', 'a')\n",
    "backend.delete_password('s', 'a')\n",
    "import subprocess\nsubprocess.run(['security', 'find-generic-password', '-s', 'x'])\n",
    "import subprocess\nsubprocess.run(['secret-tool', 'lookup', 'service', 'x'])\n",
])
def test_the_scan_recognises_each_way_of_reaching_the_keystore(tmp_path, source):
    path = tmp_path / "m.py"
    path.write_text(source)
    assert findings(path), source


@pytest.mark.parametrize("source", [
    "from poolhouse import keystore\nkeystore.default().subkey('memory', 'a')\n",
    "from . import keyring_notes\n",
    "import subprocess\nsubprocess.run(['security', 'help'])\n",
    "import json\njson.loads('{}')\n",
])
def test_the_scan_leaves_alone_a_module_that_asks_the_keystore_module(tmp_path, source):
    path = tmp_path / "m.py"
    path.write_text(source)
    assert findings(path) == []
