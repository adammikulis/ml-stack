"""Where model files live on each operating system: one table row per tool that keeps them.

A row names the tool, how its folder is laid out, the environment variables that move it
and its default folders per system. ``places`` turns the table into the folders this
machine has, in search order. ``verified`` is True where the layout was read off a real
install; the rest follow the tool's documentation and `docs/model-discovery.md` says which.
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from poolhouse import home as _home

EXTRA_ENV = "POOLHOUSE_MODEL_PATHS"
"""Extra model folders, separated by the platform's path separator."""

VOLUMES_ENV = "POOLHOUSE_SCAN_VOLUMES"
"""Set to 1 to search ``/Volumes/*/{models,Models}`` on macOS."""

HF_HOME = ".cache/huggingface"


@dataclass(frozen=True, slots=True)
class Spec:
    """One tool's model folders."""

    label: str
    layout: str
    verified: bool
    env: tuple[tuple[str, str], ...] = ()
    """``(variable, subfolder)``, most specific first: the first variable that is set, joined
    with its subfolder, replaces the default folders."""
    paths: Mapping[str, tuple[str, ...]] | tuple[str, ...] = ()
    """Templates per system (``Darwin``, ``Linux``, ``Windows``), or one tuple for all."""
    depth: int = 6


@dataclass(frozen=True, slots=True)
class Place:
    """A folder to search."""

    label: str
    layout: str
    path: Path
    verified: bool
    depth: int = 6


_LMSTUDIO = ("{home}/.lmstudio/models", "{home}/.cache/lm-studio/models")

SPECS: tuple[Spec, ...] = (
    Spec("poolhouse", "flat", True, (), ("{state}/models",)),
    Spec("huggingface", "hf", True,
         (("HF_HUB_CACHE", ""), ("HUGGINGFACE_HUB_CACHE", ""), ("HF_HOME", "hub"),
          ("TRANSFORMERS_CACHE", ""), ("XDG_CACHE_HOME", "huggingface/hub")),
         ("{home}/" + HF_HOME + "/hub",)),
    Spec("llama.cpp", "flat", True, (("LLAMA_CACHE", ""),), {
        "Darwin": ("{home}/Library/Caches/llama.cpp", "{home}/.cache/llama.cpp"),
        "Linux": ("{xdg_cache}/llama.cpp", "{home}/.cache/llama.cpp"),
        "Windows": ("{local}/llama.cpp", "{home}/.cache/llama.cpp"),
    }, depth=2),
    Spec("lmstudio", "flat", False, (), _LMSTUDIO),
    Spec("ollama", "ollama", True, (("OLLAMA_MODELS", ""),), {
        "Darwin": ("{home}/.ollama/models",),
        "Linux": ("{home}/.ollama/models", "/usr/share/ollama/.ollama/models"),
        "Windows": ("{home}/.ollama/models",),
    }),
    Spec("gpt4all", "flat", False, (), {
        "Darwin": ("{support}/nomic.ai/GPT4All",),
        "Linux": ("{home}/.local/share/nomic.ai/GPT4All",),
        "Windows": ("{local}/nomic.ai/GPT4All",),
    }, depth=2),
    Spec("jan", "flat", False, (), {
        "Darwin": ("{home}/jan/models", "{support}/Jan/data/models",
                   "{support}/Jan/data/llamacpp/models"),
        "Linux": ("{home}/jan/models", "{home}/.config/Jan/data/models",
                  "{home}/.config/Jan/data/llamacpp/models"),
        "Windows": ("{home}/jan/models", "{appdata}/Jan/data/models",
                    "{appdata}/Jan/data/llamacpp/models"),
    }, depth=3),
    Spec("modelscope", "flat", False, (("MODELSCOPE_CACHE", "hub"),),
         ("{home}/.cache/modelscope/hub",), depth=5),
    Spec("kagglehub", "flat", False, (("KAGGLEHUB_CACHE", "models"),),
         ("{home}/.cache/kagglehub/models",), depth=8),
    Spec("manual", "flat", True, (), {
        "Darwin": ("{home}/models", "{home}/Models", "/opt/models"),
        "Linux": ("{home}/models", "{home}/Models", "/opt/models", "/srv/models"),
        "Windows": ("{home}/models", "{home}/Models", "{local}/models"),
    }),
    Spec("downloads", "flat", True, (), ("{home}/Downloads",), depth=1),
)


def system() -> str:
    """``Darwin``, ``Linux`` or ``Windows``, read when asked."""
    return platform.system()


def _context(env: Mapping[str, str], home: Path, state: Path) -> dict[str, str]:
    cache = env.get("XDG_CACHE_HOME") or str(home / ".cache")
    return {
        "home": str(home),
        "state": str(state),
        "xdg_cache": cache,
        "local": env.get("LOCALAPPDATA") or str(home / "AppData" / "Local"),
        "appdata": env.get("APPDATA") or str(home / "AppData" / "Roaming"),
        "support": str(home / "Library" / "Application Support"),
    }


def _templates(spec: Spec, where: str) -> tuple[str, ...]:
    if isinstance(spec.paths, tuple):
        return spec.paths
    return spec.paths.get(where, ())


def _env_roots(spec: Spec, env: Mapping[str, str]) -> list[Path]:
    out: list[Path] = []
    for name, sub in spec.env:
        value = env.get(name)
        if value:
            out.append(Path(value).expanduser() / sub if sub else Path(value).expanduser())
    return out


def lmstudio_folders(home: Path) -> list[Path]:
    """The download folder LM Studio's settings file names, when it names one."""
    for name in (".lmstudio/settings.json", ".cache/lm-studio/settings.json"):
        try:
            data = json.loads((home / name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        folder = data.get("downloadsFolder") if isinstance(data, dict) else None
        if isinstance(folder, str) and folder:
            return [Path(folder).expanduser()]
    return []


def _volumes(env: Mapping[str, str], where: str, volumes: Path) -> list[Place]:
    if where != "Darwin" or env.get(VOLUMES_ENV) != "1":
        return []
    out: list[Place] = []
    try:
        mounts = sorted(p for p in volumes.iterdir() if p.is_dir() and not p.is_symlink())
    except OSError:
        return out
    for mount in mounts:
        out += [Place("volume", "flat", mount / name, False) for name in ("models", "Models")]
    return out


def places(*, env: Mapping[str, str] | None = None, where: str | None = None,
           home: Path | None = None, state: Path | None = None,
           volumes: Path = Path("/Volumes")) -> list[Place]:
    """Every folder worth searching on this machine, in search order, existing or not.

    ``env``, ``where`` (a system name), ``home`` and ``state`` default to the real ones; a
    test passes its own to imitate another operating system. A folder reached twice is
    listed once, at its first position.
    """
    env = os.environ if env is None else env
    where = where or system()
    home = home or _home.user_home()
    state = state or _home.home()
    ctx = _context(env, home, state)
    found: list[Place] = []
    for spec in SPECS:
        roots = _env_roots(spec, env)[:1] or [Path(t.format(**ctx))
                                              for t in _templates(spec, where)]
        if spec.label == "lmstudio":
            roots = lmstudio_folders(home) + roots
        found += [Place(spec.label, spec.layout, root, spec.verified, spec.depth)
                  for root in roots]
    for extra in (env.get(EXTRA_ENV) or "").split(os.pathsep):
        if extra.strip():
            found.append(Place("extra", "auto", Path(extra.strip()).expanduser(), True, 8))
    found += _volumes(env, where, volumes)
    seen: set[Path] = set()
    out: list[Place] = []
    for place in found:
        if place.path not in seen:
            seen.add(place.path)
            out.append(place)
    return out
