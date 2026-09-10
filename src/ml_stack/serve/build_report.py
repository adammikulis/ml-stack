"""What is built: the current binary and its architectures against master's own source,
every named build kept beside it, and rolling ``current`` back to an earlier one."""

from __future__ import annotations

import json
from pathlib import Path

import ml_stack.setup as setup_module
from ml_stack.log import say, warn
from ml_stack.serve.binary import find_binary
from ml_stack.serve.build_paths import builds_dir, current_link, named_dir, src_dir
from ml_stack.serve.build_platform import arches_from_source, server_name, version_of
from ml_stack.serve.build_verify import point_current

__all__ = ["cmd_list", "do_rollback", "manifest_of", "report"]


def do_rollback() -> int:
    entries: list[tuple[str, Path]] = []
    for manifest in sorted(builds_dir().glob("*/BUILD.json")):
        try:
            info = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        entries.append((str(info.get("built_at", "")), manifest.parent))
    entries.sort()

    link = current_link()
    current = link.resolve() if link.is_symlink() or link.exists() else None
    for _, build_dir in reversed(entries):
        if build_dir != current:
            point_current(build_dir)
            say(f"current -> {build_dir}")
            return 0
    warn("no earlier build to roll back to")
    return 2


def report(args) -> int:
    target: Path | None = None
    link = current_link()
    if link.is_symlink() or link.exists():
        target = link / server_name()
    else:
        found = find_binary("llama-server")
        target = Path(found) if found else None
    if target is None or not target.is_file():
        say("no llama-server build found")
        return 1

    build_json = target.parent / "BUILD.json"
    if build_json.is_file():
        try:
            info = json.loads(build_json.read_text())
        except (OSError, ValueError):
            info = {}
        say(f"{target}")
        say(f"  commit {info.get('commit', '?')} ({info.get('source', '?')}), "
            f"built {info.get('built_at', '?')}, {info.get('version', '?')}")
    else:
        say(f"{target}  {version_of(target) or 'version unknown'} (not a managed build)")

    source = src_dir()
    if source.is_dir():
        master = arches_from_source(source)
        if not master:
            say(f"could not read architecture names out of {source} -- "
                "src/llama-arch.cpp may have moved or renamed its table")
        else:
            mine = setup_module._arches(str(target), known=master)
            lacking = master - mine
            if lacking:
                say("master reads architectures this build lacks: "
                    + ", ".join(sorted(lacking)))
            else:
                say("reads every architecture master's own source does")
    else:
        say(f"no source checkout at {source} to compare architectures against "
            "-- ml-stack-serve build --from source clones one")
    return 0


def _named_builds() -> list[tuple[str, Path]]:
    """Every named build's link, sorted by name. Not every entry is trustworthy on its own
    -- a link is only ever written once ``verify_and_switch`` has already run for it -- but
    a link that no longer resolves (its build directory was removed by hand) is skipped
    rather than reported as a build that is not there."""
    named = named_dir()
    if not named.is_dir():
        return []
    out = []
    for link in sorted(named.iterdir()):
        if (link.is_symlink() or link.is_dir()) and (link / server_name()).exists():
            out.append((link.name, link))
    return out


def manifest_of(build_dir: Path) -> dict:
    manifest = build_dir / "BUILD.json"
    if not manifest.is_file():
        return {}
    try:
        return json.loads(manifest.read_text())
    except (OSError, ValueError):
        return {}


def cmd_list() -> int:
    def _line(label: str, build_dir: Path) -> str:
        info = manifest_of(build_dir)
        age = setup_module._age(str(info.get("built_at", ""))) or "?"
        repo = info.get("repo", "ggml-org/llama.cpp")
        return f"{label:14} {info.get('commit', '?'):12} {age:>4} old  {repo}"

    link = current_link()
    if link.is_symlink() or link.exists():
        say(_line("current", link))
    else:
        say("current        not built yet -- ml-stack-serve build")

    named = _named_builds()
    for name, link in named:
        say(_line(name, link))
    if not named:
        say("no named builds -- ml-stack-serve build --repo OWNER/REPO --ref REF --name NAME")
    return 0
