"""Launching a model server, and the shape every backend presents."""

from __future__ import annotations

import difflib
import logging
import math
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from pathlib import Path

from ml_stack.client import wait_for_health
from ml_stack.serve.binary import CACHE_ROOT, child_env, require_binary
from ml_stack.serve.ports import DEFAULT_HOST, port_is_free, reclaim_port

logger = logging.getLogger(__name__)

LOG_DIR = CACHE_ROOT / "logs"

# Where a slot's KV cache is saved when a lease asks to escalate seats without one named.
DEFAULT_SLOT_SAVE_PATH = CACHE_ROOT / "slots"


class ServerFailed(RuntimeError):
    """The server never became healthy. Carries whatever it managed to say."""


class UnknownFlag(ValueError):
    """The argv names a flag this build of llama-server does not have.

    Raised before the process is started, because the alternative is finding out at the
    far end of a 70 GB load: `--draft-max` became `--spec-draft-n-max` between releases,
    and a build that lacks a flag exits saying only "invalid argument". One line per flag,
    each naming the nearest flag the build does have.
    """


# The option strings a build accepts, keyed by (resolved path, mtime) so a rebuilt binary
# at the same path is read again and an unchanged one is never read twice.
_FLAGS: dict[tuple[str, float], frozenset[str]] = {}

# A flag at the start of a help line, or after a comma: llama-server prints
# `-c,    --ctx-size N`, and `-hfd, -hfrd, --hf-repo-draft REPO`. The first character after
# the dashes must be a letter, so `(default: -1)` reads as a number and not a flag.
_FLAG_IN_HELP = re.compile(r"(?:^|,)\s*(-{1,2}[A-Za-z][\w-]*)")


def flags_of(binary: str | Path, *, timeout: float = 20.0) -> frozenset[str]:
    """The option strings a llama-server build accepts, read out of its ``--help``.

    Empty means *unknown* -- the binary is missing, hung, or printed nothing that looks
    like usage -- and an unknown build is given no opinion, never "supports none". Reading
    help costs a fraction of a second where loading a model to find out costs minutes.
    """
    path = Path(binary)
    try:
        key = (str(path.resolve()), path.stat().st_mtime)
    except OSError:
        return frozenset()
    if key in _FLAGS:
        return _FLAGS[key]
    try:
        got = subprocess.run([str(path), "--help"], capture_output=True, text=True,
                             errors="replace", timeout=timeout, env=child_env(path))
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    found: set[str] = set()
    for line in (got.stdout + "\n" + got.stderr).splitlines():
        # A retired flag stays in the parser only to say so: llama.cpp 0.3.0 lists
        # `--draft, --draft-n, --draft-max N  the argument has been removed. use
        # --spec-draft-n-max`, and passing it is an error just the same. Not known.
        if "has been removed" in line:
            continue
        found.update(_FLAG_IN_HELP.findall(line))
    if not found:
        return frozenset()
    _FLAGS[key] = frozenset(found)
    return _FLAGS[key]


def unknown_flags(argv: list[str], known: frozenset[str] | set[str]) -> list[tuple[str, str]]:
    """The flags in ``argv`` that ``known`` lacks, each with the nearest flag it has.

    The nearest is "" when nothing is close. An empty ``known`` is an unknown build, and
    an unknown build gets no opinion: the result is ``[]``.
    """
    if not known:
        return []
    lacking: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in argv:
        if not (token.startswith("-") and len(token) > 1 and token.lstrip("-")[:1].isalpha()):
            continue
        if token in known or token in seen:
            continue
        seen.add(token)
        near = difflib.get_close_matches(token, sorted(known), n=1, cutoff=0.6)
        lacking.append((token, near[0] if near else ""))
    return lacking


def emitted_flags(backend: LlamaServerBackend) -> list[str]:
    """Every flag ``backend.command`` can emit, from specs with every field set.

    Several shapes are needed because they exclude each other: an embedding server drops
    ``--jinja`` for ``--embeddings``, an ``hf:`` reference swaps ``-m`` for ``--hf-repo``,
    and a ``--no-`` flag is only emitted where its positive form is not. Values are harmless placeholders; nothing here is run.
    """
    full = ServerSpec(
        model="model.gguf", parallel=2, mmproj="mmproj-model.gguf", draft="draft.gguf",
        spec_type="draft-simple", spec_draft_max=3, spec_draft_min=1, spec_ngram_min=48,
        spec_ngram_max=64, spec_draft_ngl=99, spec_draft_type_k="q8_0",
        spec_draft_type_v="q8_0", lookup_static="static.bin", lookup_dynamic="dynamic.bin",
        cache_reuse=256, warmup=False, context_per_slot=4096, override_tensor=("x=CPU",),
        cpu_moe=True, n_cpu_moe=1, kv_unified=True, cache_ram_mb=8192, cache_idle_slots=True,
        slot_prompt_similarity=0.5, slot_save_path="slots", chat_template_file="t.jinja",
        cache_type_k="q8_0",
        cache_type_v="q8_0", mlock=True, reasoning_budget=2048, rope_scaling="yarn",
        rope_scale=4.0, yarn_orig_ctx=32768, yarn_ext_factor=1.0, yarn_attn_factor=1.0,
        yarn_beta_fast=32.0, yarn_beta_slow=1.0)
    # the `--no-` forms are a third shape for the same reason: True and False exclude
    # each other on one spec
    shapes = (full,
              ServerSpec(model="model.gguf", kv_unified=False, cache_idle_slots=False,
                         mmap=False),
              ServerSpec(model="model.gguf", embedding=True),
              ServerSpec(model="hf:owner/repo/model.gguf", mmproj="hf:owner/repo/mmproj.gguf",
                         draft="hf:owner/repo"))
    flags: list[str] = []
    for shape in shapes:
        for token in backend.command(shape)[1:]:
            if token.startswith("-") and token.lstrip("-")[:1].isalpha() and token not in flags:
                flags.append(token)
    return flags


_CONTEXT_SUFFIX = {"": 1, "k": 1_000, "m": 1_000_000}


def parse_context(text: str | int) -> int:
    """A context size the way a person names it: ``32768``, ``256k``, ``1m``.

    ``k`` and ``m`` are the plain decimal multipliers a model card's own round figures
    use -- 1,000 and 1,000,000 -- not a file size's binary kibi/mebi, so ``1m`` is
    1,000,000 tokens, not 1,048,576.
    """
    if isinstance(text, int):
        return text
    raw = str(text).strip().lower()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([km]?)", raw)
    if not match:
        raise ValueError(f"not a context size: {text!r} (want e.g. 32768, 256k, 1m)")
    number, suffix = match.groups()
    return int(float(number) * _CONTEXT_SUFFIX[suffix])


def trained_context(model: str | Path) -> int:
    """The context this model trained at, off its own GGUF header.

    0 when that cannot be read -- an ``hf:`` reference not yet on disk, or a header that
    does not name one -- which is unknown, not zero: a caller asking whether a context is
    beyond it must treat 0 as no opinion rather than as a real training length.
    """
    if isinstance(model, str) and model.startswith("hf:"):
        return 0
    path = Path(model)
    if not path.is_file():
        return 0
    try:
        from ml_stack.serve.layout import layout

        return layout(path).context_length
    except Exception:  # noqa: BLE001 - an unreadable header names no trained length
        return 0


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
    # A draft that is a whole model keeps its **own KV cache**, at the same context as the
    # target, because it has to read the same prompt to predict against it. That is the real
    # cost of drafting with a model rather than a head -- on shared memory the two caches
    # compete for one pool -- and it is why llama.cpp lets the draft's cache be quantised
    # separately. A head like EAGLE3 reuses the target's own states and keeps almost nothing.
    spec_draft_type_k: str = ""
    spec_draft_type_v: str = ""
    # Where the n-gram table lives. The in-context kinds keep none: they look up sequences
    # already in the prompt, in memory, and touch no disk. `ngram-cache` is the exception --
    # `lookup_static` is read and never written, `lookup_dynamic` is updated as it
    # generates, so it carries what was learnt from one question into the next. That is
    # worth having when the same names keep coming back, which is what answering about one
    # community is.
    lookup_static: str | Path | None = None
    lookup_dynamic: str | Path | None = None
    # Carry a KV cache across prompts that merely *share a prefix*, by shifting rather than
    # reprocessing. Worth having wherever a system prompt and a set of tool schemas go out
    # ahead of every question -- a benchmark reprocesses that same preamble once per
    # question otherwise, twenty or thirty times a run.
    cache_reuse: int | None = None
    # The empty run at startup. Off saves a little of every load, which matters when a
    # comparison puts the same model up several times.
    warmup: bool = True
    # Context per slot, said directly. The alternative is a total divided by the slot count,
    # which is arithmetic done at the call site and got wrong once here -- a model served at
    # 8k per slot against everything else at 32k, visible only because the table prints it.
    context_per_slot: int | None = None
    # How the server holds several conversations at once. None on each leaves the build's
    # own default, so a spec that says nothing about them serves as it always has.
    #
    # One KV buffer shared across the slots, rather than one buffer each: a slot that is
    # idle costs nothing and a busy one may use what the idle ones are not. False emits the
    # `--no-` form, for a build whose default has gone the other way.
    kv_unified: bool | None = None
    # A prompt evicted from a slot is kept in RAM, up to this many MiB, and restored when a
    # conversation comes back -- the cost of holding more conversations than there are
    # slots is then a copy rather than a reprocessing of the whole history. 0 disables it.
    cache_ram_mb: int | None = None
    # Whether an idle slot's cache is moved to RAM to make room, or left where it is.
    cache_idle_slots: bool | None = None
    # How alike (0..1) a new prompt must be to a slot's held one before that slot is chosen
    # for it, so a returning conversation lands on the slot that still holds its prefix.
    slot_prompt_similarity: float | None = None
    # A directory a slot's cache can be saved to and restored from through the `/slots`
    # API -- a conversation put down and picked up across a restart.
    slot_save_path: str | Path | None = None
    # A chat template file to serve with, in place of the one the weights carry.
    chat_template_file: str | Path | None = None
    # Where individual tensors live, as `pattern=buffer` -- the way to keep part of a model
    # off the GPU without keeping all of it off.
    #
    # The case for it is Qwen3.8-Flash-Next's N-gram Embedding on a *discrete* GPU -- a
    # different thing entirely from the n-gram *speculation* above. That is a decoding
    # trick; this is architecture: a 51B-parameter table looked up by the current token
    # and the few before it, adding capacity at almost no compute per token. A gather
    # touches a few rows a token, so on a card whose VRAM the 27G table would not fit
    # beside the weights, naming its tensor here keeps it in host memory at no real cost.
    #
    # On unified memory -- a Mac -- do not: the CPU and GPU halves are one pool of RAM,
    # the build already leaves the table mapped on the host side of its own accord (the
    # `CPU_Mapped` line of the load log, which `fit` reads back), and an override changes nothing
    # but adds a flag to explain. ml-stack never sets one by itself; `--on-cpu` is the
    # person's call, for the machine in front of them.
    #
    # Find the pattern from the model rather than guessing: `gguf_dump` or llama-server's
    # own load log lists the tensor names.
    override_tensor: tuple[str, ...] = ()
    # MoE experts on the CPU: all of them, or the first N layers' worth. The same trade for
    # a different shape of model -- a 35B with 3B active fits a small machine this way.
    cpu_moe: bool = False
    n_cpu_moe: int | None = None
    # How the *main* model's KV cache is stored -- not the draft's, which
    # `spec_draft_type_k/v` already covers. "" leaves the server's own default, which is
    # f16. A preflight's fit estimate reads these back to size the KV cache it predicts.
    cache_type_k: str = ""
    cache_type_v: str = ""
    # mmap is the server's own default; False trades load-time paging for a slower first
    # load that pages in once rather than on every touch, which matters on a machine where
    # a model larger than RAM would otherwise thrash. mlock pins what is loaded so it
    # cannot be swapped back out; True costs the whole model's weight in wired memory.
    mmap: bool | None = None
    mlock: bool | None = None
    # How many tokens a thinking model may spend reasoning before it is made to answer.
    # `n_predict` is a ceiling over thinking, tool calls and answer together, and what a low
    # ceiling cuts is always the answer; this is the budget that stops the thinking instead.
    # None leaves the server's own default (-1: unlimited). Measured 2026-09-01: gemma-4-26B
    # spent 252 s and 505 s on two questions under a 16k ceiling, all of it in the thinking.
    reasoning_budget: int | None = None
    # RoPE/YaRN: how a context longer than the model's own training length is read. ""
    # leaves the server's own default (linear unless the model's header names one) --
    # LlamaServerBackend.resolved_context sets these to "yarn" plus a scale and the
    # trained length itself when a spec asks for more context than the model trained at,
    # and leaves them untouched otherwise. None on the ext/attn/beta factors leaves
    # llama.cpp's own YaRN defaults, which is what a person who has not measured a better
    # one should serve with.
    rope_scaling: str = ""
    rope_scale: float | None = None
    yarn_orig_ctx: int | None = None
    yarn_ext_factor: float | None = None
    yarn_attn_factor: float | None = None
    yarn_beta_fast: float | None = None
    yarn_beta_slow: float | None = None
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
        # the file keeps its subdirectory: a draft head lives under MTP/, and a reference
        # that lost it fetched nothing and served an empty draft path (measured 2026-09-01)
        return f"{parts[0]}/{parts[1]}", ("/".join(parts[2:]) if len(parts) > 2 else "")


@dataclass(frozen=True, slots=True)
class ServerInfo:
    base_url: str
    port: int
    pid: int | None
    backend: str
    adopted: bool = False
    """True when this server was already running and we did not start it."""
    log_path: Path | None = field(default=None, repr=False)
    # Wall time from the process starting to the health check answering -- a fact, not a
    # log line to be grepped for later, and where `status --json` reads it from.
    load_s: float | None = None
    # Wall time the post-health warm-up completion took, when one was sent. Compiling
    # shaders and allocating the KV cache happen on the first real request whether or not
    # anything measures them; this is what makes the *next* one the first that pays for it.
    warmup_s: float | None = None


@dataclass(frozen=True)
class Lease:
    """The manager's proof that a server about to start is already on the record.

    A backend launches nothing without one, so a server this code base spawns is recorded
    by construction -- before the process exists, with the pid filled in after -- and
    "a server nobody recorded" cannot come out of the library. Adam: "this wouldn't
    happen in the first place if we ensured that it wasn't possible to have an untracked
    server." Only `ServerManager` makes these; a test that wants a server goes through a
    manager on a temporary state file, the way everything else does.
    """

    port: int
    owner_pid: int
    state_file: str


class ServerBackend(ABC):
    """One way of putting a model behind an HTTP endpoint."""

    name: str

    @abstractmethod
    def start(self, spec: ServerSpec, *, lease: Lease, timeout: float = 300.0) -> ServerInfo:
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
        build: str | None = None,
        quiet: bool = True,
    ) -> None:
        self._explicit = binary
        self._vendor_dir = vendor_dir
        self._build = build
        self.quiet = quiet

    @property
    def binary(self) -> Path:
        return require_binary("llama-server", explicit=self._explicit,
                              vendor_dir=self._vendor_dir, build=self._build)

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

        argv += ["-np", str(max(1, spec.parallel))]
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
            elif not drafted[1]:
                argv += ["-hfd", drafted[0]]
            else:
                # llama-server's -hfd takes owner/repo[:quant], never a file, so a head named
                # by file is fetched into the cache first (`resolved_draft`) and passed as a
                # path. Reaching here means start() was bypassed.
                raise ServerFailed(
                    f"a draft named by file ({spec.draft}) must be fetched before serving; "
                    f"LlamaServerBackend.start does that, or `ml-stack-models fetch`")
        if spec.spec_type:
            argv += ["--spec-type", str(spec.spec_type)]
        for flag, value in (("--spec-draft-type-k", spec.spec_draft_type_k or None),
                            ("--spec-draft-type-v", spec.spec_draft_type_v or None),
                            ("--spec-draft-n-max", spec.spec_draft_max),
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
        if spec.cache_reuse is not None:
            argv += ["--cache-reuse", str(spec.cache_reuse)]
        if not spec.warmup:
            argv += ["--no-warmup"]
        if spec.context_per_slot is not None:
            argv += ["--kv-unified-per-slot", str(spec.context_per_slot)]
        for flag, choice in (("--kv-unified", spec.kv_unified),
                             ("--cache-idle-slots", spec.cache_idle_slots)):
            if choice is not None:
                argv += [flag if choice else flag.replace("--", "--no-", 1)]
        if spec.cache_ram_mb is not None:
            argv += ["--cache-ram", str(spec.cache_ram_mb)]
        if spec.slot_prompt_similarity is not None:
            argv += ["--slot-prompt-similarity", str(spec.slot_prompt_similarity)]
        if spec.slot_save_path:
            argv += ["--slot-save-path", str(spec.slot_save_path)]
        if spec.chat_template_file:
            argv += ["--chat-template-file", str(spec.chat_template_file)]
        if spec.lookup_static:
            argv += ["--lookup-cache-static", str(spec.lookup_static)]
        if spec.lookup_dynamic:
            argv += ["--lookup-cache-dynamic", str(spec.lookup_dynamic)]
        if spec.flash_attn:
            argv += ["-fa", "on"]
        if spec.jinja and not spec.embedding:
            argv += ["--jinja"]
        if spec.cache_type_k:
            argv += ["--cache-type-k", str(spec.cache_type_k)]
        if spec.cache_type_v:
            argv += ["--cache-type-v", str(spec.cache_type_v)]
        if spec.mmap is not None and not spec.mmap:
            argv += ["--no-mmap"]
        if spec.mlock:
            argv += ["--mlock"]
        if spec.reasoning_budget is not None:
            argv += ["--reasoning-budget", str(spec.reasoning_budget)]
        if spec.rope_scaling:
            argv += ["--rope-scaling", str(spec.rope_scaling)]
        if spec.rope_scale is not None:
            argv += ["--rope-scale", str(spec.rope_scale)]
        if spec.yarn_orig_ctx is not None:
            argv += ["--yarn-orig-ctx", str(spec.yarn_orig_ctx)]
        for flag, value in (("--yarn-ext-factor", spec.yarn_ext_factor),
                            ("--yarn-attn-factor", spec.yarn_attn_factor),
                            ("--yarn-beta-fast", spec.yarn_beta_fast),
                            ("--yarn-beta-slow", spec.yarn_beta_slow)):
            if value is not None:
                argv += [flag, str(value)]

        argv += list(spec.extra_args)
        return argv

    @staticmethod
    def resolved_draft(spec: ServerSpec) -> ServerSpec:
        """The spec with a draft named by `hf:` file turned into the cached file's path.

        `-hfd` downloads a repository's quant, not a file, and a head under `MTP/` is a file;
        so the file is fetched (or found in the cache) with `ml_stack.hub.fetch` and served
        by path. A quant-style reference (`hf:owner/repo`) is left for the server.
        """
        parts = spec.hf_parts(spec.draft) if spec.draft else None
        if not parts or not parts[1]:
            return spec
        from ml_stack.hub import fetch

        try:
            return replace(spec, draft=str(fetch(str(spec.draft))))
        except Exception as exc:  # noqa: BLE001 - whatever the Hub said, say it here
            raise ServerFailed(f"could not fetch the draft {spec.draft}: {exc}") from exc

    @staticmethod
    def resolved_context(spec: ServerSpec) -> tuple[ServerSpec, str]:
        """The spec with YaRN turned on when ``context`` asks for more than the model
        trained at, and the line to say about it -- "" when nothing changed.

        Untouched when the spec already names any rope or YaRN field: whoever set one has
        already made the choice. Untouched too when the trained length is not known -- the
        file is not on disk yet, or its header does not say -- since an unknown length
        gets no opinion rather than a guess. Otherwise ``rope_scaling`` is set to
        ``"yarn"``, ``yarn_orig_ctx`` to the trained length, and ``rope_scale`` to the
        ratio between the two, rounded up to two places so the served context always
        covers what was asked.
        """
        if (spec.rope_scaling or spec.rope_scale is not None
                or spec.yarn_orig_ctx is not None):
            return spec, ""
        trained = trained_context(spec.model)
        if trained <= 0 or spec.context <= trained:
            return spec, ""
        scale = math.ceil((spec.context / trained) * 100) / 100
        said = (
            f"{spec.context:,} tokens is beyond the {trained:,} this model trained at; "
            f"turning on YaRN (--rope-scaling yarn --rope-scale {scale} --yarn-orig-ctx "
            f"{trained}). llama.cpp applies this scaling to every position, so a "
            f"conversation shorter than {trained:,} tokens on this server pays for it too "
            "-- ask for that much or less to keep the model's own trained context unscaled."
        )
        return replace(spec, rope_scaling="yarn", rope_scale=scale,
                       yarn_orig_ctx=trained), said

    def start(self, spec: ServerSpec, *, lease: Lease, timeout: float = 300.0,
              check_flags: bool = True, preflight: bool = True,
              warmup_request: bool = True) -> ServerInfo:
        """Launch and wait until healthy. Raises ``ServerFailed`` with the log tail.

        A flag this build does not have raises ``UnknownFlag`` before anything is started;
        ``check_flags=False`` skips that, for a stand-in binary that prints no help.

        ``preflight=True`` (the default) runs every other check worth asking before a load
        -- shards present, architecture read by this build, an estimate against what this
        machine may use -- and raises ``PreflightFailed`` with the report's own lines when
        one comes back wrong. It runs after the port is confirmed free and before anything
        is spawned, so a refusal here still costs nothing: no process, no load, no GPU.

        ``warmup_request=True`` sends one short completion once the health check passes,
        so shader compilation and the first KV allocation are paid for here rather than by
        whatever the first real question turns out to be.
        """
        spec = self.resolved_draft(spec)
        spec, yarn_said = self.resolved_context(spec)
        if yarn_said:
            logger.warning(yarn_said)
        if not spec.is_hf_ref:
            model = Path(spec.model)
            if not model.is_file():
                raise ServerFailed(f"no model file at {model}")

        argv = self.command(spec)
        if check_flags:
            lacking = unknown_flags(argv, flags_of(self.binary))
            if lacking:
                raise UnknownFlag("\n".join(
                    f"this llama-server has no {flag}" + (f"; it has {near}" if near else "")
                    for flag, near in lacking))

        if lease.port != spec.port:
            raise ServerFailed(f"the lease is for port {lease.port}, not {spec.port}: a server "
                               "starts only on the port its record names")
        if not port_is_free(spec.port):
            reclaim_port(spec.port)
            if not port_is_free(spec.port):
                raise ServerFailed(
                    f"port {spec.port} is held by a process that is not one of ours; "
                    "refusing to kill it"
                )

        if preflight:
            from ml_stack.hub import room
            from ml_stack.serve.preflight import Preflight, PreflightFailed

            report = Preflight(spec, binary=self.binary, limit_bytes=room())
            if not report.ok:
                raise PreflightFailed(report.said())

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        if spec.slot_save_path:
            # llama-server refuses to start rather than create this itself: "not a
            # directory" is its whole complaint.
            Path(spec.slot_save_path).mkdir(parents=True, exist_ok=True)
        log_path = LOG_DIR / f"llama-server-{spec.port}.log"
        logger.info("starting: %s", " ".join(argv))

        extra_env = {}
        if spec.slot_save_path:
            # A slot's own prompt text is otherwise unreadable -- to_json() only fills
            # "prompt"/"generated" when slots_debug is set, and that is env-only, read
            # once at startup. Without it, escalate()'s summariser has a token count and
            # nothing to summarise.
            extra_env["LLAMA_SERVER_SLOTS_DEBUG"] = "1"

        started_at = time.monotonic()
        log_handle = log_path.open("wb")
        process = subprocess.Popen(
            argv,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=child_env(self.binary, extra_env or None),
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

        load_s = time.monotonic() - started_at

        warmup_s = None
        if warmup_request:
            warmup_s = self._warm_up(base_url, timeout=timeout)

        return ServerInfo(
            base_url=base_url,
            port=spec.port,
            pid=process.pid,
            backend=self.name,
            adopted=False,
            log_path=log_path,
            load_s=load_s,
            warmup_s=warmup_s,
        )

    def _warm_up(self, base_url: str, *, timeout: float) -> float | None:
        """One short completion through the real client, not curl -- shader compilation
        and the first KV allocation happen on some request; better this one than the first
        measured question. A warm-up that fails is logged and otherwise ignored: a server
        that answered health is a server, and what it does with a real prompt is somebody
        else's check to make."""
        from ml_stack.client import Client

        started = time.monotonic()
        try:
            Client(base_url, n_predict=8, timeout=min(timeout, 60.0)).complete(
                "hello", n_predict=8)
        except Exception as exc:  # noqa: BLE001
            logger.debug("warm-up request to %s did not complete: %s", base_url, exc)
            return None
        return time.monotonic() - started


def tail(path: Path, lines: int = 40) -> str:
    """The last ``lines`` of a log file, for an error message."""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return f"(no log at {path})"
    return "\n".join(content[-lines:])
