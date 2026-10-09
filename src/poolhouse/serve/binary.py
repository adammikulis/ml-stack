"""Find ``llama-server``, and give it an environment it can actually load in."""

from __future__ import annotations

import logging
import os
import platform
import shutil
from pathlib import Path

from poolhouse import credentials, home
from poolhouse.credentials import child_environment
from poolhouse.serve import llamacpp_trust

logger = logging.getLogger(__name__)

SERVER_NAMES = ("llama-server", "llama-server.exe")


def managed_root() -> Path:
    """Where `poolhouse-serve build` installs what it builds or downloads."""
    return home.state("llama.cpp")


def managed_current() -> Path:
    """The managed build that is trusted right now."""
    return managed_root() / "current"


def managed_named() -> Path:
    """Where builds kept beside `current` are pointed at, one link per name."""
    return managed_root() / "named"


# The repository `poolhouse-serve build` builds by default. A BUILD.json naming any other
# `repo` is a fork, and a fork is the only kind of build that can load a draft head its
# repository says "does not work on mainline".
MAINLINE = "ggml-org/llama.cpp"

# Directories that are on PATH only in a login shell, so a subprocess never sees them.
_LOGIN_SHELL_DIRS = (
    home.user_home() / "bin",
    home.user_home() / ".local" / "bin",
    Path("/opt/homebrew/bin"),
    Path("/usr/local/bin"),
)


class BinaryNotFound(RuntimeError):
    """No ``llama-server`` could be located, and none could be fetched."""


class BinaryTampered(BinaryNotFound):
    """The managed build is not the one that was pinned, or sentinel holds it."""


def is_windows() -> bool:
    return platform.system() == "Windows"


def find_binary(
    name: str = "llama-server",
    *,
    explicit: str | Path | None = None,
    vendor_dir: Path | None = None,
    build: str | None = None,
) -> Path | None:
    """Locate a llama.cpp binary. ``None`` if it is nowhere to be found; raises
    `BinaryNotFound` when the build it names is not installed.

    ``build`` (or, absent that, ``$POOLHOUSE_LLAMA_BUILD``) names a build ``poolhouse-serve
    build --name NAME`` made and kept beside ``current`` rather than replacing it -- a fork
    whose fixes have not reached master. It outranks ``current`` but never an explicit path
    or ``$LLAMA_CPP_SERVER``, so a caller that names a build gets it even while ``current``
    stays mainline, and never another binary in its place.
    """
    candidates = _name_variants(name)

    if explicit:
        path = home.expand(explicit)
        if path.is_file():
            return path.resolve()
        logger.debug("explicit binary %s does not exist; falling through", path)

    env_key = "LLAMA_CPP_SERVER" if name == "llama-server" else None
    if env_key and (value := os.environ.get(env_key)):
        path = home.expand(value)
        if path.is_file():
            return path.resolve()

    if env_dir := os.environ.get("LLAMA_CPP_DIR"):
        for candidate in candidates:
            path = home.expand(env_dir) / candidate
            if path.is_file():
                return path.resolve()

    if name == "llama-server":
        named = build or os.environ.get("POOLHOUSE_LLAMA_BUILD")
        if named:
            for candidate in candidates:
                path = managed_named() / named / candidate
                if path.is_file():
                    return path.resolve()
            raise BinaryNotFound(_absent_build(named, name, "build=" if build
                                               else "$POOLHOUSE_LLAMA_BUILD"))

        # A verified `poolhouse-serve build` outranks a login shell's PATH and the stale
        # bottle a release lags behind -- but never an explicit path or $LLAMA_CPP_SERVER,
        # both handled above.
        for candidate in candidates:
            path = managed_current() / candidate
            real = Path(os.path.realpath(path))
            if path.is_file() or llamacpp_trust.held(real):
                if why := llamacpp_trust.problem(real):
                    raise BinaryTampered(why)
                return real

    for directory in (vendor_dir, home.cache()):
        if directory is None:
            continue
        for candidate in candidates:
            path = home.expand(directory) / candidate
            if path.is_file():
                return path.resolve()

    return machine_binary(name, candidates)


def _absent_build(named: str, name: str, source: str) -> str:
    """What `find_binary` says of a named build with no ``name`` under `managed_named`."""
    return (f"llama.cpp build {named!r} (named by {source}) is not installed: "
            f"{managed_named() / named} holds no {name}.\n"
            f"poolhouse-serve build --repo OWNER/REPO --ref REF --name {named}   builds it; "
            f"poolhouse-serve build --check lists the builds this machine holds.")


def machine_binary(name: str, candidates: tuple[str, ...]) -> Path | None:
    """``name`` as this machine installs it: a login shell's directories, then PATH."""
    for directory in _LOGIN_SHELL_DIRS:
        for candidate in candidates:
            path = directory / candidate
            if path.is_file():
                return path.resolve()

    if found := shutil.which(name):
        return Path(found).resolve()

    return None


def require_binary(name: str = "llama-server", **kwargs: object) -> Path:
    """``find_binary`` or raise with the search path spelled out."""
    found = find_binary(name, **kwargs)  # type: ignore[arg-type]
    if found is not None:
        return found
    raise BinaryNotFound(
        f"{name} not found. Looked at: $LLAMA_CPP_SERVER, $LLAMA_CPP_DIR, a named build "
        f"($POOLHOUSE_LLAMA_BUILD or build=) under {managed_named()}, "
        f"{managed_current()}, "
        f"a vendor dir, PATH, {home.cache()}, and "
        f"{', '.join(str(d) for d in _LOGIN_SHELL_DIRS)}.\n"
        f"poolhouse-serve build   builds llama.cpp's own master (or downloads the newest "
        f"release, on a machine with no compiler) -- usually what you want, since a "
        f"release lags master by an architecture or two.\n"
        f"On macOS: brew install llama.cpp"
    )


def manifest_of(binary: str | Path | None) -> dict:
    """The ``BUILD.json`` `poolhouse-serve build` wrote beside ``binary``, or ``{}``.

    Read beside the path as given and beside where it resolves to: `find_binary` resolves
    the `named/<name>` link into `builds/<name>-<commit>/`, and the manifest lives in the
    build directory, not at the link. A brew bottle, a release unpacked by hand and a
    binary on PATH have no manifest, and ``{}`` is the honest answer for those.
    """
    if not binary:
        return {}
    import json

    path = home.expand(binary)
    for where in (path.parent, path.resolve().parent):
        manifest = where / "BUILD.json"
        if not manifest.is_file():
            continue
        try:
            return json.loads(manifest.read_text())
        except (OSError, ValueError):
            return {}
    return {}


def borrows(binary: str | Path | None) -> bool:
    """Whether ``binary`` is a fork build -- one that can load a draft head that borrows.

    Measured for real (2026-09-01): every `mtp-` head under
    `unsloth/Qwen3.8-Flash-Next-GGUF/MTP/` fails on mainline llama.cpp master with
    `check_tensor_dims: tensor 'output_hc_norm.weight' not found`, because mainline loads a
    draft as a whole model and those heads carry only the head, borrowing the trunk's
    embeddings and output layer from the target. Only a fork's loader accepts that, so which
    binary is serving decides which head may be offered -- and this is the one place that
    decision is read off a binary.

    A fork is a build kept under ``managed_named()`` (`poolhouse-serve build --name NAME`), or
    one whose ``BUILD.json`` names a ``repo`` other than ``ggml-org/llama.cpp``. `current`,
    a brew bottle, a release, anything on PATH, and ``None`` are mainline.
    """
    if not binary:
        return False
    path = home.expand(binary)
    try:
        path.relative_to(managed_named())
        return True
    except ValueError:
        pass
    info = manifest_of(path)
    repo = str(info.get("repo") or MAINLINE).strip().lower()
    return bool(info.get("name")) or repo != MAINLINE


def named_builds(name: str = "llama-server") -> list[tuple[str, Path]]:
    """Every named build on this machine as ``(name, binary)``, sorted by name.

    Read off ``managed_named()`` when asked, not at import, so a caller that points it
    elsewhere (a test, a machine with a different home) sees that. A link that no longer
    resolves -- its build directory removed by hand -- is skipped rather than reported as a
    build that is not there.
    """
    named = managed_named()
    if not named.is_dir():
        return []
    out = []
    for link in sorted(named.iterdir()):
        for candidate in _name_variants(name):
            if (link.is_symlink() or link.is_dir()) and (link / candidate).is_file():
                out.append((link.name, link / candidate))
                break
    return out


def hub_environment() -> dict[str, str]:
    """The environment for a server that downloads its own weights: this process's less its
    secrets, plus the Hugging Face token the credentials resolve to."""
    token = credentials.get("HF_TOKEN")
    return child_environment({"HF_TOKEN": str(token)} if token else None)


def child_env(binary: Path | str, extra: dict[str, str] | None = None) -> dict[str, str]:
    """The environment to launch ``binary`` with, with its own directory on PATH and none of
    this process's tokens or keys; ``extra`` is what the child is meant to have."""
    env = child_environment()
    bindir = str(Path(binary).resolve().parent)
    env["PATH"] = bindir + os.pathsep + env.get("PATH", "")
    if extra:
        env.update(extra)
    return env


def _name_variants(name: str) -> tuple[str, ...]:
    if name.endswith(".exe"):
        return (name,)
    return (name, f"{name}.exe") if is_windows() else (name,)
