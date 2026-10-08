"""Select verified Python runtimes without replacing running processes' import files."""

from __future__ import annotations

import json
import os
import platform
import re
import stat
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from ml_stack import windows_private
from ml_stack.files import writing
from ml_stack.home import state
from ml_stack.platform import private_file

COMMIT = re.compile(r"[0-9a-f]{40}")


def directory() -> Path:
    """Return the machine's owned runtime installation directory."""
    return state("runtimes", identity()).absolute()


def identity() -> str:
    """Return this platform, architecture and interpreter ABI's runtime key."""
    value = f"{sys.platform}-{platform.machine().lower()}-{sys.implementation.cache_tag}"
    if not re.fullmatch(r"[a-z0-9_-]+", value):
        raise OSError("runtime interpreter identity is invalid")
    return value


@dataclass(frozen=True, slots=True)
class Runtime:
    prefix: Path
    commit: str
    version: str
    identity: str

    @property
    def python(self) -> Path:
        return self.prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def environment(source=None) -> dict[str, str]:
    """Remove interpreter paths and frozen loader state from a child environment."""
    return {key: value for key, value in (os.environ if source is None else source).items()
            if key not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "_MEIPASS2",
                           "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"}
            and not key.startswith("_PYI_")}


def installer_environment() -> dict[str, str]:
    """Remove pip destination selectors and disable inherited pip configuration."""
    values = environment()
    for key in ('PIP_TARGET', 'PIP_PREFIX', 'PIP_USER', 'PIP_ROOT', 'PIP_PYTHON', 'PYTHONUSERBASE'):
        values.pop(key, None)
    values['PIP_CONFIG_FILE'] = os.devnull
    return values


def plain(path: Path) -> None:
    """Refuse symbolic links in a runtime storage path."""
    if any(candidate.is_symlink() for candidate in (path, *path.parents)):
        raise OSError("runtime selection paths cannot be symbolic links")


def _owned(path: Path) -> None:
    plain(path)
    if os.name == "nt" and windows_private.problem(path):
        raise OSError("runtime selection must be private to its Windows account")
    info = path.stat()
    if os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o077):
        raise OSError("runtime selection must be private to its owner")


def protect(path: Path) -> None:
    """Apply account-private permissions to new owned runtime storage."""
    plain(path)
    if os.name == "nt":
        windows_private.restrict(path)
    elif path.is_dir():
        path.chmod(0o700)
    else:
        private_file(path)


def verify(runtime: Runtime) -> Runtime:
    """Verify the interpreter, imported package, distribution and source revision."""
    if runtime.identity != identity():
        raise OSError("runtime belongs to a different platform or interpreter")
    if not COMMIT.fullmatch(runtime.commit):
        raise ValueError("runtime requires a full source commit")
    root = directory()
    _owned(root)
    family = root / runtime.commit
    _owned(family)
    expected = runtime.prefix
    if (expected.parent != family or not re.fullmatch(r"[0-9a-f]{32}", expected.name)
            or expected.is_symlink()):
        raise ValueError("runtime must use its owned source revision directory")
    _owned(expected)
    config = expected / "pyvenv.cfg"
    if config.is_symlink() or config.stat().st_size > 16384:
        raise OSError("runtime virtual environment configuration is invalid")
    fields = dict(line.split("=", 1) for line in config.read_text().splitlines() if "=" in line)
    fields = {key.strip().lower(): value.strip().lower() for key, value in fields.items()}
    if fields.get("include-system-site-packages") != "false":
        raise OSError("runtime dependencies must be isolated from system site packages")
    script = (
        "import json,sys,platform;from pathlib import Path;from importlib.metadata import distribution;"
        "import ml_stack;d=distribution('ml-stack');p=Path(ml_stack.__file__).resolve();"
        "print(json.dumps([sys.prefix,sys.base_prefix,sys.executable,str(p),"
        "str(d.locate_file('ml_stack/__init__.py').resolve()),d.version,"
        "p.with_name('fleet').joinpath('built-from').read_text().strip(),"
        "f'{sys.platform}-{platform.machine().lower()}-{sys.implementation.cache_tag}']))"
    )
    done = subprocess.run([str(runtime.python), "-I", "-c", script], capture_output=True,
                          text=True, timeout=30, env=environment())
    if done.returncode:
        raise OSError("runtime interpreter verification failed")
    prefix, base, executable, imported, packaged, version, commit, identity_key = json.loads(done.stdout)
    if (Path(prefix) != runtime.prefix or prefix == base or Path(executable) != runtime.python
            or imported != packaged or not Path(imported).is_relative_to(runtime.prefix)
            or version != runtime.version or commit != runtime.commit or identity_key != runtime.identity):
        raise OSError("runtime imports, metadata and source revision do not agree")
    return runtime


def publish(runtime: Runtime) -> None:
    """Atomically select a verified installed revision."""
    verify(runtime)
    target = directory() / "selected.json"
    if target.is_symlink():
        raise OSError("runtime selection cannot replace a symbolic link")
    with writing(target) as temporary:
        protect(temporary)
        temporary.write_text(json.dumps({"format": 1, **asdict(runtime), "prefix": str(runtime.prefix)}), encoding="utf-8")
        protect(temporary)


def selected() -> Runtime | None:
    """Read and verify the selected machine runtime."""
    root = directory()
    target = root / "selected.json"
    if not target.exists() and not target.is_symlink():
        return None
    _owned(root)
    _owned(target)
    if target.stat().st_size > 4096:
        raise OSError("runtime selection exceeds its size limit")
    row = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(row, dict) or set(row) != {"format", "prefix", "commit", "version", "identity"}:
        raise ValueError("invalid runtime selection")
    if type(row["format"]) is not int or row["format"] != 1:
        raise ValueError("unknown runtime selection format")
    if any(not isinstance(row[key], str) for key in ("prefix", "commit", "version", "identity")):
        raise ValueError("invalid runtime selection fields")
    return verify(Runtime(Path(row["prefix"]), row["commit"], row["version"], row["identity"]))


def python() -> Path:
    """Return the selected interpreter, or the current nonfrozen interpreter."""
    chosen = selected()
    if chosen is not None:
        return chosen.python
    if getattr(sys, "frozen", False):
        raise OSError("prepare a verified standalone runtime before starting workers")
    return Path(sys.executable)


def forward(module: str, argv: list[str]) -> bool:
    """Replace a launcher with the selected runtime when its prefix differs."""
    chosen = selected()
    if chosen is None or (not getattr(sys, "frozen", False) and Path(sys.prefix) == chosen.prefix):
        return False
    os.execve(str(chosen.python), [str(chosen.python), "-I", "-m", module, *argv], environment())  # noqa: S606
    return True


def gateway(target: Path, module: str = "ml_stack.fleet.launch", function: str = "",
            chosen: Runtime | None = None) -> None:
    """Atomically write an owned standalone launcher for the selected interpreter."""
    chosen = verify(chosen) if chosen is not None else selected()
    if chosen is None:
        raise OSError("select an installed runtime before creating its launcher")
    if not re.fullmatch(r"ml_stack(?:\.[a-z_][a-z0-9_]*)+", module):
        raise ValueError("launcher module must belong to ml-stack")
    if function and not re.fullmatch(r"[a-z_][a-z0-9_]*", function):
        raise ValueError("launcher function must be an identifier")
    plain(target)
    if target.exists():
        info = target.lstat()
        if (not stat.S_ISREG(info.st_mode) or (os.name != "nt" and (info.st_uid != os.getuid() or info.st_mode & 0o022))
                or (os.name == "nt" and windows_private.problem(target))):
            raise OSError("runtime launcher must be an owned regular file")
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    arguments = (["-c", f"import sys;sys.argv[0]={target.name!r};from {module} import {function};sys.exit({function}())"]
                 if function else ["-m", module])
    text = ("#!/usr/bin/env python3\nimport os,sys\n"
            f"python = {str(chosen.python)!r}\n"
            "env = {k:v for k,v in os.environ.items() if k not in "
            "{'PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','_MEIPASS2','LD_LIBRARY_PATH','DYLD_LIBRARY_PATH'} "
            "and not k.startswith('_PYI_')}\n"
            f"os.execve(python,[python,'-I',*{arguments!r},*sys.argv[1:]],env)\n")
    with writing(target) as temporary:
        temporary.write_text(text, encoding="utf-8")
        protect(temporary)
        if os.name != "nt":
            temporary.chmod(0o700)
