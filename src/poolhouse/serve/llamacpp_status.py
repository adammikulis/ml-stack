"""``llama-cpp status``: which llama-server is in use, where it came from, and whether upstream is ahead."""

from __future__ import annotations

import os
import re
from pathlib import Path

from poolhouse import net
from poolhouse.serve import llamacpp_state
from poolhouse.serve.binary import BinaryNotFound, find_binary, manifest_of
from poolhouse.serve.build_paths import builds_dir
from poolhouse.serve.build_platform import version_of
from poolhouse.serve.llamacpp_upstream import Upstream, latest, tag_number

__all__ = ["gather", "origin_of"]

_NUMBERS = re.compile(r"build\s+(\d+)(?:,\s*commit\s+([0-9a-f]+))?")


def origin_of(binary: Path) -> str:
    """Where ``binary`` came from: an environment override, a managed build or a package manager."""
    real = str(binary)
    for key in ("LLAMA_CPP_SERVER", "LLAMA_CPP_DIR"):
        value = os.environ.get(key)
        if value and real.startswith(os.path.realpath(value)):
            return f"env override (${key})"
    try:
        binary.relative_to(os.path.realpath(builds_dir()))
    except ValueError:
        pass
    else:
        info = manifest_of(binary)
        return "managed build (" + str(info.get("source") or "unknown") + ")"
    if "/homebrew/" in real.lower() or "/Cellar/" in real:
        return "homebrew"
    return "PATH"


def _numbers(line: str, info: dict) -> tuple[int | None, str]:
    found = _NUMBERS.search(line)
    number = info.get("build") if isinstance(info.get("build"), int) else (
        int(found.group(1)) if found else None)
    commit = str(info.get("commit") or (found.group(2) if found and found.group(2) else ""))
    return number, commit


def gather(upstream: Upstream | None = None, pipeline: net.Pipeline | None = None,
           *, check: bool = True) -> dict:
    """The facts `status` prints. ``upstream_checked`` is false when the one upstream check
    could not be made, with the reason in ``upstream_note``."""
    out: dict = {"binary": "", "origin": "", "version": "", "build": None, "commit": "",
                 "pinned": llamacpp_state.load()["pinned"], "problem": "",
                 "newer": "unknown (offline)", "upstream": {}, "upstream_note": ""}
    try:
        found = find_binary("llama-server")
    except BinaryNotFound as exc:
        out["problem"] = str(exc)
        found = None
    if found is not None:
        info = manifest_of(found)
        out["binary"], out["origin"] = str(found), origin_of(found)
        out["version"] = str(info.get("version") or version_of(found))
        out["build"], out["commit"] = _numbers(out["version"], info)
        if info.get("smoke"):
            out["smoke"] = bool(info["smoke"].get("passed"))
    if not check:
        return out
    try:
        newest = latest(upstream, pipeline)
    except net.NeedsApproval as exc:
        out["newer"], out["upstream_note"] = "unknown (source host not approved)", str(exc)
        return out
    out["upstream"] = {"tag": newest.tag, "stable": newest.stable, "master": newest.master}
    if not newest.known:
        out["upstream_note"] = newest.reason
        return out
    mine, commit = out["build"], str(out["commit"])
    if commit and newest.master:
        behind = not newest.master.startswith(commit[:9]) and not commit.startswith(newest.master[:9])
        newer_tag = tag_number(newest.tag)
        released = newer_tag is not None and mine is not None and newer_tag > mine
        out["newer"] = "yes" if behind else "no"
        out["newer_release"] = bool(released)
    elif mine is not None and (tag := tag_number(newest.tag)) is not None:
        out["newer"] = "yes" if tag > mine else "no"
    return out
