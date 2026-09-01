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
    # How to guess ahead. "" leaves the server's own default, which is `none` -- passing a
    # draft model turns on draft-simple by itself, but the n-gram kinds must be asked for.
    #
    # The n-gram kinds need no second model at all: they propose tokens by looking up
    # sequences already seen in the prompt, which the large model then checks in one pass.
    # That suits work that copies from its context -- answering out of retrieved text,
    # transcribing, summarising -- and costs no weights and no memory, where a draft head
    # costs both. `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`.
    spec_type: str = ""
    spec_draft_max: int | None = None       # tokens guessed ahead (server default 3)
    spec_draft_min: int | None = None
    spec_ngram_min: int | None = None       # ngram-mod lookup floor (server default 48)
    spec_ngram_max: int | None = None
    spec_draft_ngl: int | None = None       # draft layers on the GPU; without it, the CPU
    # Where the n-gram table lives. The in-context kinds keep none: they look up sequences
    # already in the prompt, in memory, and touch no disk. `ngram-cache` is the exception --
    # `lookup_static` is read and never written, `lookup_dynamic` is updated as it
    # generates, so it carries what was learnt from one question into the next. That is
    # worth having when the same names keep coming back, which is what answering about one
    # community is.
    lookup_static: str | Path | None = None
    lookup_dynamic: str | Path | None = None
    # Where individual tensors live, as `pattern=buffer` -- the way to keep part of a model
    # off the GPU without keeping all of it off.
    #
    # This is what Qwen3.8-Flash-Next's N-gram Embedding wants, and it is a different thing
    # entirely from the n-gram *speculation* above. That is a decoding trick; this is
    # architecture: a 51B-parameter table looked up by the current token and the few before
    # it, adding capacity at almost no compute per token. Its lookups are known in advance,
    # so the table is meant to sit in host memory and be prefetched alongside the
    # computation rather than occupy the GPU permanently. Naming its tensors here is how
    # that is arranged.
    #
    # Find the pattern from the model rather than guessing: `gguf_dump` or llama-server's
    # own load log lists the tensor names.
    override_tensor: tuple[str, ...] = ()
    # MoE experts on the CPU: all of them, or the first N layers' worth. The same trade for
    # a different shape of model -- a 35B with 3B active fits a small machine this way.
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
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
            # `--mmproj` takes a file on disk, so an `hf:` reference has to become the URL
            # it stands for; `--mmproj-url` fetches it. Passing the reference itself is a
            # path that does not exist, and llama-server says so in a way that reads like
            # the projector is corrupt rather than misspelled.
            seeing = spec.hf_parts(spec.mmproj)
            if seeing is None:
                argv += ["--mmproj", str(spec.mmproj)]
            else:
                repo, name = seeing
                argv += ["--mmproj-url",
                         f"https://huggingface.co/{repo}/resolve/main/{name}"]
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
        if spec.spec_type:
            argv += ["--spec-type", str(spec.spec_type)]
        for flag, value in (("--spec-draft-n-max", spec.spec_draft_max),
                            ("--spec-draft-n-min", spec.spec_draft_min),
                            ("--spec-ngram-mod-n-min", spec.spec_ngram_min),
                            ("--spec-ngram-mod-n-max", spec.spec_ngram_max),
                            ("--spec-draft-ngl", spec.spec_draft_ngl)):
            if value is not None:
                argv += [flag, str(value)]
        for pattern in spec.override_tensor:
            argv += ["--override-tensor", str(pattern)]
        if spec.cpu_moe:
            argv += ["--cpu-moe"]
        if spec.n_cpu_moe is not None:
            argv += ["--n-cpu-moe", str(spec.n_cpu_moe)]
        if spec.lookup_static:
            argv += ["--lookup-cache-static", str(spec.lookup_static)]
        if spec.lookup_dynamic:
            argv += ["--lookup-cache-dynamic", str(spec.lookup_dynamic)]
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
