"""The runner proves itself on no GPU before it spends one.

A measuring command is minutes of loading before the first question is asked, and the
first question is where a new flag meets the client. On 2026-09-02 a new ``--also tight``
way reached ``Client.__init__`` as a keyword argument and took an 87G load down with it: the
smoke run had been left out of the plan, and the variant's test had faked a client that
accepts anything. Neither the plan nor the test is where that belongs. `selfcheck` drives
the exact subcommand and flags a person asked for -- every ``--also`` way, the store and
the shortlist, the KV cache, the draft lengths, the per-question cap -- through the whole
path with a scripted model that takes exactly what `Client` takes, a served model that
never starts, the real preflight over facts that read nothing, the invented community and
two of its questions, into a scratch store it reads back. Ten seconds, and `main` runs it
before it takes the measuring lock, so the check no longer depends on a person remembering
it.

The preflight is the real one on purpose. Twice on 2026-09-02 the self-check said ok and
the run then died inside the preflight -- `command()` refusing a draft head still named by
`hf:` file -- because the self-check had replaced `Preflight` whole, so no preflight code
ran. Now only its readers are replaced, through `Preflight`'s own seams (``shards_of``,
``read_header``, ``arches``, ``flags``, ``ref_bytes``): the shards are present, the header
is a dense model the build reads, the build accepts every flag `command` can emit. Every
check's code runs over the exact spec the run builds, the argv included; and `fake_serve`
walks what `start()` does before `Popen` -- the draft resolved, the argv built, its flags
checked -- so the lease's refusal is met here too.

`ScriptedModel` and `ScriptedReader` are the fakes the tests use too. One fake shared
between the tests and the runner is the point: a fake with ``**kwargs`` in its signature
is what let the flag through.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import inspect
import io
import json
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

# the real client's signature, read before anything is patched over it, is what the fakes
# are strict against -- so they stay strict as `Client` changes, with nothing to maintain;
# the real preflight, bound the same way, is what the self-check runs whatever a test has
# put over `preflight.Preflight` for the run it then makes
from ml_stack.client.chat import Client as _RealClient
from ml_stack.client.families import GENERIC
from ml_stack.serve.preflight import Preflight as _RealPreflight

__all__ = ["ScriptedModel", "ScriptedReader", "SelfCheckFailed", "selfcheck"]


class SelfCheckFailed(RuntimeError):
    """The dry run did not get through. Carries the traceback and what the run printed."""


def _bind(base_url: str, settings: Mapping[str, Any]) -> None:
    """Raise the `TypeError` the real `Client` would for a keyword it does not take."""
    inspect.signature(_RealClient.__init__).bind(None, base_url, **settings)


@dataclass
class _Reply:
    content: str
    raw: dict
    tool_calls: list | None = None
    thinking: str | None = None
    finish_reason: str | None = None


class ScriptedModel:
    """A model that takes exactly what `Client` takes, calls look_up once, then answers.

    Built the way `served` and `run` build the real one -- `Run.client`, the only place a
    run becomes a client -- and bound against the real ``__init__``'s signature, so a
    keyword the client does not take fails here, naming itself, rather than on the GPU.
    Keeps every message it was
    shown in ``seen``; ``told`` is what look_up answered, as the model saw it. Has the
    ``card`` a ``--also card`` way reads and the ``sampling`` the run writes down.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", **settings: Any) -> None:
        _bind(base_url, settings)
        self.base_url = base_url
        self.timeout = settings.get("timeout", 180.0)
        self.sampling: dict[str, Any] = dict(settings)
        self.card: dict[str, Any] = {"temperature": 1.0, "top_k": 64}
        self.text = "compilers"
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], tools: Any = None, **_: Any) -> Any:
        self.seen.append([dict(m) for m in messages])
        offered = {str((t.get("function") or {}).get("name")) for t in (tools or [])}
        # the first turn of a conversation looks something up; a later turn -- a
        # conversation carried into `concurrent` -- already has an answer to give
        asked_before = sum(1 for m in messages if m.get("role") == "tool")
        if "look_up" in offered and not asked_before:
            return _Reply(content="", raw={}, tool_calls=[{"id": "c1", "function": {
                "name": "look_up", "arguments": json.dumps({"texts": [self.text]})}}])
        return _Reply(content="a compiler person", raw={})

    def told(self) -> str:
        """What look_up answered, as the model saw it."""
        return " ".join(str(m["content"]) for turn in self.seen for m in turn
                        if m.get("role") == "tool")


class ScriptedReader:
    """A model that reads the sender off an extraction prompt and returns them, and nothing
    else -- through the real `Client.extract`, so the prompt, the schema and the JSON are
    exercised -- with the strict signature `ScriptedModel` has."""

    family = GENERIC
    card: dict[str, Any] = {}

    def __init__(self, base_url: str = "http://127.0.0.1:8080", **settings: Any) -> None:
        _bind(base_url, settings)
        self.base_url = base_url
        self.timeout = settings.get("timeout", 180.0)
        self.sampling: dict[str, Any] = dict(settings)
        self.seen: list[list[dict[str, Any]]] = []

    def chat(self, messages: Sequence[Mapping[str, Any]], **_: Any) -> Any:
        self.seen.append([dict(m) for m in messages])
        head = str(messages[-1]["content"]).split(":\n", 1)[0]
        sender = head.removeprefix("From ").split(" in ")[0]
        got = {"people": [{"name": sender, "role": "", "org": "", "place": ""}], "orgs": [],
               "topics": [], "places": [], "relations": []}
        return _Reply(content=json.dumps(got),
                      raw={"usage": {"prompt_tokens": 40, "completion_tokens": 12},
                           "timings": {"prompt_n": 30, "cache_n": 10}})


# -- the dry run ----------------------------------------------------------------------------------

def _scratch_store(where: Path) -> Path:
    """The invented community in a store of its own, word index built, as `prepare` builds
    the real one; what look_up searches and a shortlist reads when the run names one."""
    from ml_stack.graph.community import graph as invented
    from ml_stack.graph.store import replace

    replace(where, invented())
    return where


def _scratch_world(where: Path) -> Path:
    """A tiny invented company, written the way `ml-stack-world make --out` writes one;
    `load_world` then simulates its messages, as it does for any world without them."""
    from ml_stack.world.organisation import make

    world = make("company", "small", seed=1)
    where.mkdir(parents=True, exist_ok=True)
    (where / "graph.json").write_text(json.dumps(world.graph), encoding="utf-8")
    (where / "personas.json").write_text(json.dumps(world.personas), encoding="utf-8")
    (where / "world.json").write_text(json.dumps(
        {"kind": "company", "size": "small", "seed": 1, "people": world.people}),
        encoding="utf-8")
    return where


def _rewritten(args: argparse.Namespace, scratch: Path) -> argparse.Namespace:
    """The same command, pointed at scratch: the invented community and two of its
    questions, a store under the scratch home when the run has one, runs kept in a scratch
    store. Every flag about the asking or the serving is left exactly as given."""
    from ml_stack.graph import bench
    from ml_stack.graph.bench.extract import SMOKE_MESSAGES

    out = argparse.Namespace(**vars(args))
    home = scratch / "home"
    home.mkdir(parents=True, exist_ok=True)
    out.kept = str(scratch / "runs.ladybug")
    out.detach = False
    out.no_queue = False
    if hasattr(out, "graph"):
        out.graph = ""
    if hasattr(out, "questions"):
        out.questions = ""
    # a store when the run has one: named, or `prepare`'s default on this machine (which
    # `drafts` takes without being told) -- built under the scratch home, where `prepared`
    # finds it once HOME points there
    wants_store = bool(getattr(args, "store", "") or bench.prepared()
                       or (args.cmd == "run" and getattr(args, "shortlist", 0)))
    if wants_store:
        _scratch_store(home / "graph.ladybug")
    if hasattr(out, "store"):
        out.store = str(home / "graph.ladybug") if wants_store else ""
    if args.cmd == "extract":
        out.world = str(_scratch_world(scratch / "world"))
        out.sample = SMOKE_MESSAGES
    elif args.cmd == "speed":
        # the grid at its smallest that still has two cells and two streams
        out.prompts, out.streams, out.generate = "32,64", "1,2", 8
    elif not getattr(out, "smoke", False):
        # two questions, but not as `--smoke`: a run that was not asked for as a smoke
        # runs its own smoke first, and that path is checked here too
        if hasattr(out, "sample"):
            out.sample = bench.SMOKE
        if hasattr(out, "short"):
            out.short = False
        if args.cmd == "concurrent":
            out.conversations, out.turns = 2, 1
    return out


# What the faked facts say about any model: a dense header with the keys the KV estimate
# reads, so the estimate is a number and not a 0 that skips the arithmetic.
_HEADER: dict[str, object] = {"general.architecture": "llama", "llama.block_count": 32,
                              "llama.attention.head_count_kv": 8,
                              "llama.attention.key_length": 128}
_WEIGHTS_BYTES = 4 << 30
_COMPANION_BYTES = 1 << 30


def _stand_in_binary(home: Path) -> Path:
    """A file named llama-server that exists and does nothing: `command()` resolves its
    binary through `require_binary`, which needs a file, and a bare name would fall
    through to whatever this machine has -- or nothing, on a machine with none."""
    where = home / "bin" / "llama-server"
    where.parent.mkdir(parents=True, exist_ok=True)
    where.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    where.chmod(0o755)
    return where


def _mainline_flags(binary: str | Path) -> frozenset[str]:
    """Every flag `command` can emit, as the build's --help: the flags check then judges
    the argv it builds for real, rather than giving an unread build no opinion and
    never building one."""
    from ml_stack.serve.backend import LlamaServerBackend, emitted_flags

    return frozenset(emitted_flags(LlamaServerBackend(binary=binary)))


@contextlib.contextmanager
def _faked(args: argparse.Namespace, home: Path, built: list[Any]):
    """Everything a measuring command reaches for that is a server, a file or a machine,
    replaced for the block: the served model never starts, the preflight runs for real
    over facts that touch nothing, the client is scripted (and strict), nothing is asked
    of a port."""
    import ml_stack.client
    import ml_stack.hub
    import ml_stack.serve
    import ml_stack.serve.binary
    import ml_stack.serve.preflight as preflight
    from ml_stack.graph import bench
    from ml_stack.graph.bench import extract as bench_extract
    from ml_stack.serve import ServerInfo
    from ml_stack.serve.backend import (
        LlamaServerBackend,
        ServerSpec,
        UnknownFlag,
        unknown_flags,
    )
    from ml_stack.serve.preflight import Check, Report

    fake_client = ScriptedReader if args.cmd == "extract" else ScriptedModel
    stand_in = _stand_in_binary(home)

    class Built(fake_client):  # type: ignore[misc,valid-type]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            built.append(self)

    def find_stand_in(name: str = "llama-server", *, explicit: Any = None,
                      **_: Any) -> Path:
        # `command()` resolves its binary through here; a --binary that exists is kept,
        # anything else is the stand-in, so the argv is built whatever this machine has
        given = Path(str(explicit)).expanduser() if explicit else None
        return given.resolve() if given is not None and given.is_file() else stand_in

    def fetched(reference: str) -> Path:
        # a head named by hf: file is fetched into the cache and served by path; here
        # it lands under the scratch home, as a file that is never read
        head = home / "cache" / str(reference).removeprefix("hf:").rsplit("/", 1)[-1]
        head.parent.mkdir(parents=True, exist_ok=True)
        head.touch()
        return head

    @contextlib.contextmanager
    def fake_serve(model: Any, *, port: int | None = None, context: int = 4096,
                   timeout: float | None = None, manager: Any = None, **spec_kwargs: Any):
        # the spec is built for real: a keyword the server does not take fails here --
        # and then everything `start()` does before Popen: the draft resolved to a file,
        # the argv built, every flag in it checked. A spec the backend refuses is refused
        # here, naming what it refused (measured 2026-09-02: a head still named by hf:
        # file, refused in the lease after a self-check that built no argv)
        spec = ServerSpec(model=model, port=port or 1, context=context, **spec_kwargs)
        backend = getattr(manager, "backend", None) or LlamaServerBackend(binary=stand_in)
        argv = backend.command(backend.resolved_draft(spec))
        lacking = unknown_flags(argv, _mainline_flags(stand_in) | _raw_flags(spec))
        if lacking:
            raise UnknownFlag("\n".join(f"selfcheck: command() emits {flag}, which no build "
                                        f"accepts" + (f"; nearest {near}" if near else "")
                                        for flag, near in lacking))
        yield ServerInfo(base_url="http://127.0.0.1:1", port=1, pid=None, backend="selfcheck",
                         load_s=0.0, warmup_s=0.0)

    def present(spec: Any) -> tuple[int, Path | None, Check]:
        name = str(spec.model).rsplit("/", 1)[-1] or "model.gguf"
        return (_WEIGHTS_BYTES, home / "cache" / name,
                Check("shards", True, "selfcheck: taken as complete on this machine"))

    def checked_preflight(spec: Any, *, binary: str, limit_bytes: int = 0) -> Report:
        # the real thing, over the spec the run built, with only the readers replaced:
        # the shards are present at a plausible size, the header is a dense model this
        # build reads, the build accepts every flag `command` can emit, a companion
        # costs a gigabyte. Every check's own code runs, and the argv is built for real.
        try:
            report = _RealPreflight(spec, binary=binary, limit_bytes=limit_bytes,
                                    shards_of=present, read_header=lambda path: dict(_HEADER),
                                    arches=lambda build: {"llama"},
                                    flags=lambda build: _mainline_flags(build) | _raw_flags(spec),
                                    ref_bytes=lambda ref: _COMPANION_BYTES if ref else 0)
        except Exception as exc:  # noqa: BLE001 - the point is to say so before the GPU
            raise SelfCheckFailed(
                f"the preflight raised over the spec the run builds for {spec.model}"
                + (f" with draft {spec.draft}" if spec.draft else "")
                + f": {type(exc).__name__}: {exc}\n\n{traceback.format_exc()}") from exc
        if not report.ok:
            raise SelfCheckFailed(
                f"the preflight refused the spec the run builds for {spec.model}:\n"
                + report.said())
        return report

    def fake_footprint(url: str, client: Any = None) -> dict[str, Any]:
        return {"base_url": url, "context": int(getattr(args, "context", 0) or 32768),
                "slots": int(getattr(args, "parallel", 1) or 1), "model": "selfcheck.gguf",
                "resident_bytes": 0}

    real_ask_from = bench.ask_from
    wanted_client = str(getattr(args, "client", "") or "")

    def fake_ask_from(spec: str) -> Callable[..., Any]:
        return Built if wanted_client and spec == wanted_client else real_ask_from(spec)

    def no_embedder(*a: Any, **k: Any) -> Any:
        raise ConnectionError("selfcheck: nothing embeds here; the words vote")

    with contextlib.ExitStack() as patched:
        patch = patched.enter_context
        patch(mock.patch.object(bench, "HOME", home))
        patch(mock.patch.object(bench, "find_model", lambda named: named))
        patch(mock.patch.object(bench_extract, "find_model", lambda named: named))
        patch(mock.patch.object(bench, "busy", lambda url: 0))
        patch(mock.patch.object(bench, "slot_count", lambda url: 1))
        patch(mock.patch.object(bench, "footprint", fake_footprint))
        patch(mock.patch.object(bench_extract, "footprint", fake_footprint))
        patch(mock.patch.object(bench, "ask_from", fake_ask_from))
        patch(mock.patch.object(ml_stack.serve, "serve", fake_serve))
        patch(mock.patch.object(ml_stack.serve.binary, "find_binary", find_stand_in))
        patch(mock.patch.object(ml_stack.hub, "fetch", fetched))
        # `--serve-draft auto` asks the one resolver; here it answers with a head that
        # never loads, and says so the way the real one does
        patch(mock.patch.object(ml_stack.hub, "choose_head",
                                lambda model, **k: ml_stack.hub.Chosen(
                                    f"mtp-{Path(str(model)).name}", "",
                                    "selfcheck: a head that never loads", False)))
        patch(mock.patch.object(preflight, "Preflight", checked_preflight))
        patch(mock.patch.object(ml_stack.hub, "room", lambda: 1 << 40))
        patch(mock.patch.object(ml_stack.client, "Client", Built))
        # the module, not the function the package re-exports under the same name
        patch(mock.patch.object(importlib.import_module("ml_stack.client.embed"), "embed",
                                no_embedder))
        yield


def _raw_flags(spec: Any) -> frozenset[str]:
    """The flags a run named raw (`--serve-arg`): the person's explicit choice, which the
    stand-in build cannot know and the real preflight, against the real binary, still
    checks. Without this every `--serve-arg` sweep failed its self-check on the flag it
    was written to measure (2026-09-02, `-ub 2048`)."""
    return frozenset(a.split("=", 1)[0] for a in (getattr(spec, "extra_args", ()) or ())
                     if str(a).startswith("-"))


def selfcheck(argv: Sequence[str]) -> str:
    """Drive ``ml-stack-bench argv`` through the whole path with no server and no GPU.

    The exact subcommand and flags given, with the model, the server and the machine
    faked and everything else real: the ways are built, the spec is built, the real
    preflight runs over it (its readers faked, its checks and its argv not), what
    `start()` does before `Popen` is done to it, every client is constructed as the run
    would construct it, two questions are asked of
    the invented community and scored, the run is saved to a scratch store and read back
    the way `show` reads it. `extract` reads a tiny invented world the same way. A run
    that was not asked for as ``--smoke`` runs its own smoke first here too.

    Returns one line saying what got through. Raises `SelfCheckFailed`, carrying the
    traceback and everything the run printed, otherwise. ``--detach``, ``--no-queue`` and
    ``--no-selfcheck`` are ignored; a command line that does not parse exits as it would.
    """
    import tempfile

    from ml_stack.graph import bench
    from ml_stack.graph.bench import extract as bench_extract

    rest = [a for a in argv if a not in ("--detach", "--no-queue", "--no-selfcheck")]
    args = bench._parser().parse_args(rest)
    if args.cmd not in bench.MEASURING:
        raise ValueError(f"{args.cmd} measures nothing; there is nothing to check")
    began = time.monotonic()
    built: list[Any] = []
    said = io.StringIO()
    with tempfile.TemporaryDirectory(prefix="ml-stack-selfcheck-") as scratch:
        where = Path(scratch)
        try:
            with contextlib.redirect_stdout(said), contextlib.redirect_stderr(said):
                asked = _rewritten(args, where)
                with _faked(asked, where / "home", built):
                    code = bench._run(asked)
                    if code != 0:
                        raise SelfCheckFailed(f"{args.cmd} returned {code}")
                    kept = bench.runs(asked.kept)
                    if not kept:
                        raise SelfCheckFailed(f"{args.cmd} kept no run")
                    reading = bench_extract.read_back if args.cmd == "extract" else bench.read_back
                    reading(asked.kept, [r["key"] for r in kept])
                    shown = sum(1 for r in kept for _ in (r.get("rows") or ()))
                    if not shown:
                        raise SelfCheckFailed(f"{args.cmd} kept {len(kept)} run(s) with no rows")
                    turns = sum(len(c.seen) for c in built)
                    if not turns:
                        raise SelfCheckFailed("the scripted model was never asked anything")
        except SelfCheckFailed as why:
            raise SelfCheckFailed(f"{why}\n\n{_printed(said)}") from None
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - the point is to catch it
            raise SelfCheckFailed(f"{type(exc).__name__}: {exc}\n\n"
                                  f"{traceback.format_exc()}\n{_printed(said)}") from exc
    labels = sorted({str(r.get("label", "")) for r in kept})
    unit = "message" if args.cmd == "extract" else "question"
    return (f"{args.cmd}: {len(kept)} run(s) kept and read back -- {', '.join(labels)} -- "
            f"{shown} {unit}(s) through {len(built)} scripted client(s), "
            f"{time.monotonic() - began:.1f}s")


def _printed(said: io.StringIO) -> str:
    text = said.getvalue().strip()
    return "the run printed:\n" + "\n".join(f"  {ln}" for ln in text.splitlines()) if text \
        else "the run printed nothing"
