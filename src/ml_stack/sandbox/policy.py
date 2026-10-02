"""What a sandboxed process may touch: files, network, programs, environment, time and resources."""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import StrEnum

__all__ = ["AllowUnsandboxed", "Limits", "Net", "NetMode", "Policy", "PolicyError",
           "checked_path", "checked_paths"]

CONTROL = re.compile(r"[\x00-\x1f\x7f]")
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class PolicyError(ValueError):
    """A policy that names something the sandbox cannot hold to."""


def checked_path(value: str | os.PathLike[str], *, what: str = "path", link: bool = False) -> str:
    """``value`` as a string when it is absolute, exists, has no control characters or ``..``
    segment and no symlink on the way (``link`` lets the last part be one); otherwise
    `PolicyError`."""
    text = os.fspath(value)
    if not isinstance(text, str) or not text:
        raise PolicyError(f"{what}: an empty or non-text path")
    if CONTROL.search(text):
        raise PolicyError(f"{what}: {text!r} holds a control character")
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        raise PolicyError(f"{what}: {text!r} is not valid UTF-8") from None
    if not text.startswith("/"):
        raise PolicyError(f"{what}: {text!r} is not absolute")
    if ".." in text.split("/"):
        raise PolicyError(f"{what}: {text!r} climbs with ..")
    if text != "/" and text.endswith("/"):
        raise PolicyError(f"{what}: {text!r} ends with a slash")
    if "//" in text:
        raise PolicyError(f"{what}: {text!r} holds an empty segment")
    if "/./" in text or text.endswith("/."):
        raise PolicyError(f"{what}: {text!r} holds a . segment")
    if not os.path.lexists(text):
        raise PolicyError(f"{what}: {text!r} does not exist")
    parent, _, leaf = text.rpartition("/")
    real = f"{os.path.realpath(parent or '/').rstrip('/')}/{leaf}" if link else os.path.realpath(text)
    if real != text:
        raise PolicyError(f"{what}: {text!r} passes through a symlink and resolves to "
                          f"{real!r}; name the resolved path")
    return text


def checked_paths(values: Iterable[str | os.PathLike[str]], *, what: str) -> tuple[str, ...]:
    """`checked_path` of each, in order, without repeats."""
    return tuple(dict.fromkeys(checked_path(v, what=what) for v in values))


class NetMode(StrEnum):
    DENY = "deny"
    LOOPBACK = "loopback"
    PORTS = "ports"


@dataclass(frozen=True, slots=True)
class Net:
    """Network access: none, this machine's loopback, or loopback on the named ports only."""

    mode: NetMode = NetMode.DENY
    ports: tuple[int, ...] = ()

    @classmethod
    def deny(cls) -> Net:
        return cls()

    @classmethod
    def loopback(cls) -> Net:
        return cls(NetMode.LOOPBACK)

    @classmethod
    def only(cls, *ports: int) -> Net:
        return cls(NetMode.PORTS, tuple(sorted(set(ports))))

    def __post_init__(self) -> None:
        if self.mode == NetMode.PORTS and not self.ports:
            raise PolicyError("net: ports mode names no port")
        if self.mode != NetMode.PORTS and self.ports:
            raise PolicyError(f"net: {self.mode} mode takes no ports")
        for port in self.ports:
            if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise PolicyError(f"net: {port!r} is not a port")


@dataclass(frozen=True, slots=True)
class Limits:
    """Ceilings on one run. ``wall_seconds`` kills the process group; the others are rlimits
    the child starts with. ``None`` is no ceiling."""

    wall_seconds: float | None = 60.0
    cpu_seconds: int | None = None
    file_bytes: int | None = None
    open_files: int | None = None
    output_bytes: int | None = 1_000_000

    def __post_init__(self) -> None:
        for name in ("wall_seconds", "cpu_seconds", "file_bytes", "open_files", "output_bytes"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value <= 0):
                raise PolicyError(f"limits: {name} must be positive or None, not {value!r}")


@dataclass(frozen=True, slots=True)
class Policy:
    """A deny-by-default grant. ``read`` and ``write`` are the only trees the process can read
    and write (``write`` implies read), ``exec`` the only programs it can start, ``env`` the
    whole environment it sees, ``net`` its network. ``gpu`` adds what Metal needs and nothing
    else; ``cache`` is one directory the process may read and write besides ``write`` (shader
    and compile caches)."""

    name: str = "policy"
    read: tuple[str, ...] = ()
    write: tuple[str, ...] = ()
    exec: tuple[str, ...] = ()
    net: Net = field(default_factory=Net)
    env: Mapping[str, str] = field(default_factory=dict)
    limits: Limits = field(default_factory=Limits)
    gpu: bool = False
    cache: str = ""

    def validated(self) -> Policy:
        """This policy with every path checked, or `PolicyError`."""
        env = dict(self.env)
        for key, value in env.items():
            if not isinstance(key, str) or not NAME.fullmatch(key):
                raise PolicyError(f"env: {key!r} is not a variable name")
            if not isinstance(value, str) or "\x00" in value:
                raise PolicyError(f"env: the value of {key} is not text")
        return replace(
            self, read=checked_paths(self.read, what="read"),
            write=checked_paths(self.write, what="write"),
            exec=checked_paths(self.exec, what="exec"), env=env,
            cache=checked_path(self.cache, what="cache") if self.cache else "")

    def with_net(self, net: Net) -> Policy:
        return replace(self, net=net)

    def without_network(self) -> Policy:
        """This policy with the network taken away."""
        return replace(self, net=Net.deny())


@dataclass(frozen=True, slots=True)
class AllowUnsandboxed:
    """The named, logged choice to run with no sandbox when none is available. ``reason`` is
    recorded with every run."""

    reason: str

    def __post_init__(self) -> None:
        if not self.reason.strip():
            raise PolicyError("running unsandboxed needs a reason")
