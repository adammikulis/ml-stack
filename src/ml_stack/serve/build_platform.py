"""What this machine can build llama.cpp with, and the small facts every build step needs:
the server binary's name, whether a compiler or Vulkan is here, cmake's flags, a flat
install copied into place, and the architectures a checkout's own source reads."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ml_stack.http import ServerError, request_bytes
from ml_stack.serve.binary import child_env, is_windows

__all__ = ["arches_at", "arches_from_source", "can_build_from_source", "cmake_flags", "copy_flat",
           "now_iso", "server_name", "version_of", "vulkan_available"]

# `LLM_ARCH_QWEN4EXP, "qwen4exp"` -- every architecture name llama.cpp's own source reads,
# read straight out of the enum-to-string table rather than kept as a separate list.
_ARCH_NAME = re.compile(r'LLM_ARCH_\w+,\s*"([a-z0-9_]+)"')


def server_name() -> str:
    return "llama-server.exe" if is_windows() else "llama-server"


def vulkan_available() -> bool:
    return bool(os.environ.get("VULKAN_SDK")) or shutil.which("vulkaninfo") is not None


def can_build_from_source() -> bool:
    """Whether this machine has a compiler at all -- the source path needs one, the release
    path does not, and a machine with neither should not be told to try compiling."""
    if shutil.which("cmake") is None:
        return False
    if platform.system() == "Windows":
        return any(shutil.which(c) for c in ("cl", "gcc", "clang"))
    return any(shutil.which(c) for c in ("cc", "gcc", "clang"))


def cmake_flags() -> list[str]:
    """Which GPU backend to build, decided the same way on every platform: a CUDA compiler
    beats a Vulkan SDK beats neither. Metal is unconditional on macOS -- every Mac since the
    architecture this reads GGUF for has one."""
    if platform.system() == "Darwin":
        return ["-DGGML_METAL=ON", "-DLLAMA_CURL=ON"]
    if shutil.which("nvcc"):
        return ["-DGGML_CUDA=ON"]
    if vulkan_available():
        return ["-DGGML_VULKAN=ON"]
    return []


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


LIB_GLOBS = ("lib*.dylib", "lib*.so", "*.dll")


def copy_flat(src_dir: Path, dest: Path) -> list[str]:
    """The server binary and every library beside it, flattened into ``dest`` -- the same
    shape as the hand-built ``~/.local/llama-next`` this replaces, so `child_env` finds the
    libraries by putting the binary's own directory on PATH."""
    copied: list[str] = []
    seen: set[str] = set()
    for pattern in (server_name(), *LIB_GLOBS):
        for item in sorted(src_dir.glob(pattern)):
            if item.name in seen:
                continue
            seen.add(item.name)
            target = dest / item.name
            if target.exists() or target.is_symlink():
                target.unlink()
            if item.is_symlink():
                target.symlink_to(os.readlink(item))
            else:
                shutil.copy2(item, target)
            copied.append(item.name)
    return copied


def version_of(binary: Path) -> str:
    try:
        done = subprocess.run([str(binary), "--version"], capture_output=True, text=True,
                              timeout=20, env=child_env(binary))
    except (OSError, subprocess.SubprocessError):
        return ""
    text = ((done.stdout or "") + (done.stderr or "")).strip()
    return text.splitlines()[0] if text else ""


def arches_from_source(source: Path) -> set[str]:
    """Every architecture name master's own source reads, for ``--check`` to compare against."""
    arch_file = source / "src" / "llama-arch.cpp"
    if not arch_file.is_file():
        return set()
    text = arch_file.read_text(encoding="utf-8", errors="replace")
    return set(_ARCH_NAME.findall(text))


def arches_at(ref: str, *, repo: str = "ggml-org/llama.cpp") -> set[str]:
    """Every architecture name ``repo``'s source reads at ``ref``; empty when it cannot be read."""
    url = f"https://raw.githubusercontent.com/{repo}/{ref}/src/llama-arch.cpp"
    try:
        reply = request_bytes(url, timeout=30.0)
    except (ServerError, OSError, ValueError):
        return set()
    return set(_ARCH_NAME.findall(reply.body.decode("utf-8", errors="replace")))
