"""Owned console and existing hook entrypoints for selected immutable runtimes."""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import tomllib
from pathlib import Path

from ml_stack import runtime, windows_private
from ml_stack.files import writing

MODULES = frozenset({"ml_stack.harnesshook", "ml_stack.profilehook", "ml_stack.workspace.cli"})


def owned(path: Path) -> None:
    """Verify an existing launcher or configuration belongs to the current account."""
    runtime.plain(path)
    info = path.stat()
    if (not stat.S_ISREG(info.st_mode) or (os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o022))
            or (os.name == "nt" and windows_private.problem(path))):
        raise OSError("runtime entrypoint must be an owned regular file")


def install(directory: Path, chosen: runtime.Runtime | None = None) -> list[Path]:
    """Replace the selected distribution's owned console entrypoints atomically."""
    chosen = runtime.verify(chosen) if chosen is not None else runtime.selected()
    if chosen is None:
        raise OSError("select an immutable runtime before installing its launchers")
    runtime.confine_tests(directory)
    runtime.plain(directory)
    if (not directory.is_dir() or (os.name != "nt" and
            (directory.stat().st_uid != os.getuid() or directory.stat().st_mode & 0o022))):
        raise OSError("launcher directory must belong to the current account")
    script = ("import json;from importlib.metadata import distribution;"
              "print(json.dumps({p.name:p.value for p in distribution('ml-stack').entry_points "
              "if p.group=='console_scripts'}))")
    done = subprocess.run([str(chosen.python), "-I", "-c", script], capture_output=True,
                          text=True, timeout=30, env=runtime.environment())
    if done.returncode or len(done.stdout) > 65536:
        raise OSError("selected runtime console metadata is unavailable")
    entries = json.loads(done.stdout)
    if not isinstance(entries, dict) or not 1 <= len(entries) <= 256:
        raise ValueError("invalid runtime console metadata")
    targets = []
    for name, value in entries.items():
        if (not isinstance(name, str) or not re.fullmatch(r"ml-stack(?:-[a-z0-9-]+)?", name)
                or not isinstance(value, str)
                or not re.fullmatch(r"ml_stack(?:\.[a-z_][a-z0-9_]*)+:[a-z_][a-z0-9_]*", value)):
            raise ValueError("runtime entrypoint must belong to ml-stack")
        target = directory / (name + ".exe" if os.name == "nt" else name)
        if os.name == "nt":
            owned(chosen.python.parent / target.name)
        if target.exists() or target.is_symlink():
            owned(target)
        targets.append((target, value.split(":")))
    for target, (module, function) in targets:
        if os.name == "nt":
            with writing(target) as temporary:
                shutil.copyfile(chosen.python.parent / target.name, temporary)
                runtime.protect(temporary)
        else:
            runtime.write_launcher(target, module, function, chosen)
    return [target for target, _ in targets]


def hook_command(command: str, python: Path) -> str:
    """Replace the interpreter of an existing maintained Python module hook."""
    if len(command) > 16384:
        raise ValueError("hook command exceeds its size limit")
    words = shlex.split(command)
    path = words[0] if words and words[0].startswith("PATH=") else ""
    offset = bool(path)
    rest = words[offset:]
    if (len(rest) < 4 or rest[1] != "-m" or rest[2] not in MODULES
            or not Path(rest[0]).is_absolute() or Path(rest[0]).name not in {"python", "python3", "python.exe"}):
        return command
    if any(re.search(r"[$`;|&<>\n]", word) for word in rest):
        return command
    if path and path != f"PATH={Path(rest[0]).parent}:$PATH":
        return command
    prefix = f"PATH={shlex.quote(str(python.parent))}:$PATH " if path else ""
    return prefix + shlex.join([str(python), *rest[1:]])


def claude_hooks(settings: Path) -> int:
    """Update only existing maintained command hooks in owned Claude settings."""
    owned(settings)
    identity = settings.stat()
    if settings.stat().st_size > 1_048_576:
        raise ValueError("hook settings exceed their size limit")
    chosen = runtime.selected()
    if chosen is None:
        raise OSError("select an immutable runtime before updating hooks")
    original = settings.read_text()
    document = json.loads(original)
    changed = 0
    for groups in document.get("hooks", {}).values():
        for group in groups:
            for hook in group.get("hooks", []):
                if hook.get("type") != "command" or not isinstance(hook.get("command"), str):
                    continue
                replacement = hook_command(hook["command"], chosen.python)
                if replacement != hook["command"]:
                    hook["command"] = replacement
                    changed += 1
    if changed:
        _replace(settings, json.dumps(document, indent=2) + "\n", original, identity)
    return changed


def codex_hooks(settings: Path) -> dict[str, object]:
    """Update existing single-line command hook values without changing trusted hashes."""
    owned(settings)
    identity = settings.stat()
    if settings.stat().st_size > 1_048_576:
        raise ValueError("hook settings exceed their size limit")
    chosen = runtime.selected()
    if chosen is None:
        raise OSError("select an immutable runtime before updating hooks")
    text = settings.read_text()
    tomllib.loads(text)
    changed = 0
    pattern = re.compile(r'''(?m)^(\s*command\s*=\s*)("(?:[^"\\\n]|\\.)*"|'[^'\n]*')([ \t]*(?:\#[^\n]*)?)$''')

    def replace(match):
        nonlocal changed
        headings = list(re.finditer(r"(?m)^\s*\[\[?[^\n]+\]\]?\s*(?:\#[^\n]*)?$", text[:match.start()]))
        if not headings or "hooks" not in tomllib.loads(headings[-1][0] + "\n_marker=1"):
            return match[0]
        command = tomllib.loads("command=" + match[2])["command"]
        replacement = hook_command(command, chosen.python)
        if command == replacement:
            return match[0]
        changed += 1
        return match[1] + json.dumps(replacement) + match[3]

    replacement = pattern.sub(replace, text)
    tomllib.loads(replacement)
    if changed:
        _replace(settings, replacement, text, identity)
    return {"changed": changed, "pending_trust": bool(changed),
            "detail": "Host trust revalidation is required; saved trusted hashes are unchanged." if changed else ""}


def _replace(settings: Path, replacement: str, original: str, identity: os.stat_result) -> None:
    with writing(settings) as temporary:
        temporary.write_text(replacement)
        runtime.protect(temporary)
        owned(settings)
        current = settings.stat()
        if ((current.st_dev, current.st_ino) != (identity.st_dev, identity.st_ino)
                or settings.read_text() != original):
            raise OSError("hook settings changed during runtime cutover")
