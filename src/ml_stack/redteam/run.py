"""One red-team run: stand up the model and the daemon, run the scenarios, write the report."""

from __future__ import annotations

import asyncio
import importlib
import platform
import subprocess
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from ml_stack.hub import located
from ml_stack.redteam import daemon, pyrit_bridge
from ml_stack.redteam.lab import lab, own_home
from ml_stack.redteam.report import Report
from ml_stack.redteam.scenarios import NAMES, Options
from ml_stack.redteam.stub import StubModel
from ml_stack.serve import serve

__all__ = ["DEFAULT_MODEL", "Plan", "execute"]

DEFAULT_MODEL = "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
PACKAGE = "ml_stack.redteam.scenarios"


@dataclass(frozen=True, slots=True)
class Plan:
    """What to run: the scenarios, the model ("stub" for the scripted stand-in, a name or path
    of an installed GGUF otherwise) and how much of each scenario."""

    scenarios: tuple[str, ...] = NAMES
    model: str = DEFAULT_MODEL
    options: Options = field(default_factory=Options)
    stub_mode: str = "gullible"


def version() -> str:
    """The installed ml-stack version, or ``unknown``."""
    try:
        return metadata.version("ml-stack")
    except metadata.PackageNotFoundError:
        return "unknown"


def git_ref() -> str:
    """This checkout's short commit, with ``+dirty`` when files have changed."""
    here = Path(__file__).resolve().parent
    run = ["git", "-C", str(here)]
    head = subprocess.run([*run, "rev-parse", "--short", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    dirty = subprocess.run([*run, "status", "--porcelain", "--", str(here)],
                           capture_output=True, text=True, check=False).stdout.strip()
    return (head + ("+dirty" if dirty else "")) if head else "unknown"


@contextmanager
def _model(plan: Plan, parallel: int = 2) -> Iterator[tuple[str, str, int | None]]:
    """``(base url, model name, port)`` of the model under attack, torn down afterwards."""
    if plan.model == "stub":
        stub = StubModel(plan.stub_mode)
        try:
            yield stub.base_url, f"stub-{plan.stub_mode}", stub.port
        finally:
            stub.close()
        return
    path = located(plan.model)
    if path is None:
        raise SystemExit(f"{plan.model} is not an installed model; nothing is downloaded")
    with serve(path, context=8192 * parallel, parallel=parallel, roam=True,
               reason=f"red-team run on {path.name}") as info:
        yield info.base_url, path.name, info.port


async def _scenarios(plan: Plan, one: Any, report: Report) -> None:
    for name in plan.scenarios:
        started = time.monotonic()
        module = importlib.import_module(f"{PACKAGE}.{name}")
        await module.run(one, report, plan.options)
        report.meta.setdefault("seconds", {})[name] = round(time.monotonic() - started, 1)


def execute(plan: Plan, command: str = "") -> Report:
    """Run ``plan`` and return its report."""
    if not pyrit_bridge.available():
        raise SystemExit("PyRIT is not installed: pip install 'ml-stack[redteam]'")
    report = Report({
        "date": datetime.now(UTC).strftime("%Y-%m-%d"), "command": command,
        "ml_stack": f"{version()} @ {git_ref()}", "pyrit": pyrit_bridge.version(),
        "python": platform.python_version(), "platform": platform.platform(terse=True),
        "scenarios": ",".join(plan.scenarios)})
    with _model(plan) as (url, name, port), tempfile.TemporaryDirectory(prefix="redteam-d-") as root:
        report.meta["model"] = name
        # the daemon builds its sentinel when it starts, so the home is swapped before it, not only inside lab()
        with own_home(Path(root) / "machine-state"), daemon.running(Path(root), port, name) as served, lab(
                model_url=url, model_name=name, served=served) as one:
            asyncio.run(_scenarios(plan, one, report))
    return report
