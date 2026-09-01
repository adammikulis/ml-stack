"""Launching a model server, and the shape every backend presents."""

from __future__ import annotations

import logging
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from ml_stack.client import wait_for_health
from ml_stack.serve.binary import CACHE_ROOT, child_env, require_binary
from ml_stack.serve.ports import DEFAULT_HOST, port_is_free, reclaim_port

logger = logging.getLogger(__name__)

LOG_DIR = CACHE_ROOT / "logs"


class ServerFailed(RuntimeError):
    """The server never became healthy. Carries whatever it managed to say."""


@dataclass(frozen=True, slots=True)
class ServerSpec:
    """What to serve, and how."""

    model: str | Path
    port: int = 8080
    context: int = 4096
    n_gpu_layers: int | str = "auto"
    parallel: int = 1
    embedding: bool = False
    mmproj: str | Path | None = None
    flash_attn: bool = True
    jinja: bool = True
    # A small model of the same family, guessing ahead so the large one only has to agree.
    # Same two forms as `model`: a path, or hf:owner/repo[/file.gguf].
    draft: str | Path | None = None
    extra_args: tuple[str, ...] = ()

    @property
    def is_hf_ref(self) -> bool:
        """``hf:owner/repo/file.gguf`` -- let llama-server do the download and caching."""
        return isinstance(self.model, str) and self.model.startswith("hf:")

    @staticmethod
    def hf_parts(value: str | Path) -> tuple[str, str] | None:
        """``hf:owner/repo[/file]`` split into ``(repo, file)``, or None when it is a path."""
        if not (isinstance(value, str) and value.startswith("hf:")):
            return None
        parts = [p for p in str(value).partition("hf:")[2].split("/") if p]
        if len(parts) < 2:
            raise ServerFailed(
                f"malformed HF reference {value!r}; expected hf:owner/repo[/file.gguf]")
        return f"{parts[0]}/{parts[1]}", (parts[-1] if len(parts) > 2 else "")


@dataclass(frozen=True, slots=True)
class ServerInfo:
    base_url: str
    port: int
    pid: int | None
    backend: str
    adopted: bool = False
    """True when this server was already running and we did not start it."""
    log_path: Path | None = field(default=None, repr=False)


class ServerBackend(ABC):
    """One way of putting a model behind an HTTP endpoint."""

    name: str

    @abstractmethod
    def start(self, spec: ServerSpec, *, timeout: float = 300.0) -> ServerInfo:
        ...

    @abstractmethod
    def command(self, spec: ServerSpec) -> list[str]:
        """The argv this backend would run. Separate from ``start`` so it is testable"""


class LlamaServerBackend(ServerBackend):
    """llama.cpp's ``llama-server``."""

    name = "llama.cpp"

    def __init__(
        self,
        *,
        binary: str | Path | None = None,
        vendor_dir: Path | None = None,
        quiet: bool = True,
    ) -> None:
        self._explicit = binary
        self._vendor_dir = vendor_dir
        self.quiet = quiet

    @property
    def binary(self) -> Path:
        return require_binary("llama-server", explicit=self._explicit, vendor_dir=self._vendor_dir)

    def command(self, spec: ServerSpec) -> list[str]:
        """Build the argv."""
        argv = [str(self.binary), "--host", DEFAULT_HOST, "--port", str(spec.port)]

        if spec.is_hf_ref:
            repo, name = spec.hf_parts(spec.model)
            argv += ["--hf-repo", repo]
            if name:
                argv += ["--hf-file", name]
        else:
            argv += ["-m", str(spec.model)]

        argv += ["-c", str(spec.context)]

        if spec.n_gpu_layers == "auto":
            argv += ["-ngl", "99"]
        elif spec.n_gpu_layers is not None:
            argv += ["-ngl", str(spec.n_gpu_layers)]

        if spec.parallel > 1:
            argv += ["-np", str(spec.parallel)]
        if spec.embedding:
            argv += ["--embeddings", "--pooling", "mean"]
        if spec.mmproj:
            argv += ["--mmproj", str(spec.mmproj)]
        if spec.draft:
            # A draft guesses several tokens ahead and the large model checks them in one
            # pass, so an agreeing run costs about what one token used to. The flags differ
            # from the main model's: -hfd takes owner/repo[:quant], not a separate file.
            drafted = spec.hf_parts(spec.draft)
            if drafted is None:
                argv += ["-md", str(spec.draft)]
            else:
                repo, name = drafted
                argv += ["-hfd", f"{repo}:{name}" if name else repo]
        if spec.flash_attn:
            argv += ["-fa", "on"]
        if spec.jinja and not spec.embedding:
            argv += ["--jinja"]

        argv += list(spec.extra_args)
        return argv

    def start(self, spec: ServerSpec, *, timeout: float = 300.0) -> ServerInfo:
        """Launch and wait until healthy. Raises ``ServerFailed`` with the log tail."""
        if not spec.is_hf_ref:
            model = Path(spec.model)
            if not model.is_file():
                raise ServerFailed(f"no model file at {model}")

        if not port_is_free(spec.port):
            reclaim_port(spec.port)
            if not port_is_free(spec.port):
                raise ServerFailed(
                    f"port {spec.port} is held by a process that is not one of ours; "
                    "refusing to kill it"
                )

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"llama-server-{spec.port}.log"
        argv = self.command(spec)
        logger.info("starting: %s", " ".join(argv))

        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=child_env(self.binary),
        )

        base_url = f"http://{DEFAULT_HOST}:{spec.port}"
        healthy = wait_for_health(
            base_url,
            timeout=timeout,
            is_alive=lambda: process.poll() is None,
        )

        if not healthy:
            code = process.poll()
            process.terminate()
            log_handle.close()
            raise ServerFailed(
                f"llama-server did not become healthy on {base_url}"
                + (f" (exited {code})" if code is not None else f" within {timeout:.0f}s")
                + f"\n--- {log_path} ---\n"
                + tail(log_path)
            )

        return ServerInfo(
            base_url=base_url,
            port=spec.port,
            pid=process.pid,
            backend=self.name,
            adopted=False,
            log_path=log_path,
        )


def tail(path: Path, lines: int = 40) -> str:
    """The last ``lines`` of a log file, for an error message."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"(no log at {path})"
    return "\n".join(content[-lines:])
