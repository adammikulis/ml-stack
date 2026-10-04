"""A model server started, or a grant to start one made, outside the broker's own modules."""

from __future__ import annotations

import ast
from pathlib import Path

from . import Finding
from ._util import calls, exempt, parse, python_files, rel

NAME = "server-starts"
OWNER = "ml_stack.serve.broker"
HARD = True
ROOTS = ("src/ml_stack",)
GRANTORS = ("src/ml_stack/serve/grant.py", "src/ml_stack/serve/broker.py")
"""Where `broker_grant` is defined and where it is opened."""
LEASERS = ("src/ml_stack/serve/backend.py", "src/ml_stack/serve/manager.py")
"""Where the `Lease` a backend launches on is defined and made."""
SPAWNERS = (
    "src/ml_stack/serve/backend.py",
    "src/ml_stack/serve/build_platform.py",
    "src/ml_stack/serve/build_source.py",
    "src/ml_stack/serve/llamacpp_compile.py",
    "src/ml_stack/setup.py",
)
"""The modules that may both start a process and name llama-server, each for its own reason."""
SPAWN = {"subprocess.Popen", "subprocess.run", "subprocess.call", "subprocess.check_call",
         "subprocess.check_output", "os.execv", "os.execve", "os.execvp", "os.posix_spawn"}


def describe() -> str:
    return ("A model server started, or the grant to start one opened, outside the broker; "
            "`ml-stack-serve up` asks the broker for a lease.")


def find(root: Path) -> list[Finding]:
    out = []
    for path in python_files(root, ROOTS):
        where = rel(path, root)
        tree = parse(path)
        if tree is None:
            continue
        if not exempt(where, GRANTORS):
            out += [Finding(where, n.lineno, "broker_grant")
                    for n in ast.walk(tree)
                    if (isinstance(n, ast.Name) and n.id == "broker_grant")
                    or (isinstance(n, ast.Attribute) and n.attr == "broker_grant")]
        found = calls(tree)
        if not exempt(where, LEASERS):
            out += [Finding(where, n.lineno, "Lease(")
                    for n, name in found if name.rsplit(".", 1)[-1] == "Lease"]
        names_server = any(isinstance(n, ast.Constant) and n.value in ("llama-server", "llama_server")
                           for n in ast.walk(tree))
        if names_server and not exempt(where, SPAWNERS):
            out += [Finding(where, n.lineno, f"{name} beside llama-server")
                    for n, name in found if name in SPAWN]
    return out
