"""Finding llama.cpp's converter and quantiser."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from ml_stack import home

CONVERTER_NAME = "convert_hf_to_gguf.py"


def llama_cpp_src() -> Path:
    """Where `ensure_converter` clones llama.cpp."""
    return home.cache("llama.cpp-src")


def source_dirs() -> tuple[Path, ...]:
    """Every directory a llama.cpp checkout is looked for in, in order."""
    account = home.user_home()
    return (
        llama_cpp_src(),
        account / ".local" / "opt" / "llama.cpp-src",
        account / "llama.cpp",
        account / ".unsloth" / "llama.cpp",
        Path("/opt/homebrew/share/llama.cpp"),
        Path("/usr/local/share/llama.cpp"),
    )

QUANTIZE_NAMES = ("llama-quantize", "llama-quantize.exe")


class ToolNotFound(RuntimeError):
    """A llama.cpp tool could not be located."""


def find_converter(explicit: str | Path | None = None) -> Path | None:
    """Locate ``convert_hf_to_gguf.py``. ``None`` if it is nowhere."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()

    for key in ("LLAMA_CPP_ROOT", "LLAMA_CPP_DIR"):
        if root := os.environ.get(key):
            candidate = Path(root).expanduser() / CONVERTER_NAME
            if candidate.is_file():
                return candidate.resolve()

    for directory in source_dirs():
        candidate = directory / CONVERTER_NAME
        if candidate.is_file():
            return candidate.resolve()

    return None


def require_converter(explicit: str | Path | None = None) -> Path:
    """``find_converter`` or raise, naming every directory tried and how to fix it."""
    found = find_converter(explicit)
    if found is not None:
        return found
    raise ToolNotFound(
        f"{CONVERTER_NAME} not found. Looked at $LLAMA_CPP_ROOT, $LLAMA_CPP_DIR and:\n"
        + "\n".join(f"  {d}" for d in source_dirs())
        + "\n\nIt ships only with the llama.cpp *source*, not with a release binary or a "
        "brew install. Either clone it:\n"
        f"  git clone --depth 1 https://github.com/ggml-org/llama.cpp {llama_cpp_src()}\n"
        "or call ensure_converter(), which does exactly that."
    )


def ensure_converter(*, ref: str = "master") -> Path:
    """Locate the converter, shallow-cloning llama.cpp into the cache if it is absent."""
    if found := find_converter():
        return found

    git = shutil.which("git")
    if git is None:
        raise ToolNotFound("git is not on PATH, so llama.cpp cannot be fetched")

    import subprocess

    source = llama_cpp_src()
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [git, "clone", "--depth", "1", "--branch", ref,
         "https://github.com/ggml-org/llama.cpp", str(source)],
        check=True,
    )
    return require_converter()


def find_quantize(explicit: str | Path | None = None) -> Path | None:
    """Locate ``llama-quantize``. Unlike the converter, this one ships as a binary."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return path.resolve()

    for key in ("LLAMA_CPP_ROOT", "LLAMA_CPP_DIR"):
        if root := os.environ.get(key):
            base = Path(root).expanduser()
            for sub in ("", "build/bin", "bin"):
                for name in QUANTIZE_NAMES:
                    candidate = base / sub / name if sub else base / name
                    if candidate.is_file():
                        return candidate.resolve()

    for name in QUANTIZE_NAMES:
        if found := shutil.which(name):
            return Path(found).resolve()

    for directory in source_dirs():
        for sub in ("build/bin", "bin", ""):
            for name in QUANTIZE_NAMES:
                candidate = directory / sub / name if sub else directory / name
                if candidate.is_file():
                    return candidate.resolve()

    return None


def require_quantize(explicit: str | Path | None = None) -> Path:
    found = find_quantize(explicit)
    if found is not None:
        return found
    raise ToolNotFound(
        "llama-quantize not found on PATH, in $LLAMA_CPP_ROOT/$LLAMA_CPP_DIR, or in "
        f"{', '.join(str(d) for d in source_dirs())}.\nOn macOS: brew install llama.cpp"
    )
