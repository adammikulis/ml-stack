"""The shape a model is served in, written down once, and one server per port to sit at.

llama.cpp serves **one shape per port**. Two parts of a program that lease the same model
with different context, different seats, or a draft head on one and not the other are not
two clients of one server: whichever leases second finds a mismatch, stops the first server
and loads the weights again. On a large model that is a minute of nothing working, and it
happens the moment a lease is spelled out in two places and one of them is edited.

So a :class:`Shape` is the whole shape in one object -- the model, its port, how many
conversations it holds and how much context each gets, the KV cache's precision, the draft
head and how far ahead it guesses, the vision projector, the thinking budget, and which
llama.cpp build serves it -- and :meth:`Shape.lease` is the only place those become the
keyword arguments :func:`ml_stack.serve.serve` takes. Everything that wants the model asks
:func:`seat` for a seat on it: the server is started once per port and held for the process,
and each caller gets a :class:`~ml_stack.client.Client` pinned to a slot of its own, so two
conversations at once do not reprocess each other's context.

    shape = Shape(model="hf:owner/repo/weights.gguf", port=8080, seats=4,
                  seat_context=32768, cache_type="q8_0", draft=head, draft_n_max=4)
    client = seat(shape, index=request_number, n_predict=16384)

A shape is one third of what a model needs. :class:`Run` is all three -- the :class:`Shape`
to serve it in, the :class:`Asking` to ask it with, and the :class:`Talking` the client is
built from -- so a bench row, a page answer and a seated client for one model are the same
lease and the same asking by construction rather than by three places agreeing::

    run = profile_for(model).run(port=8080, seats=4)
    serve(run.shape.model, **run.lease())                  # the server
    converse(question, graph, client, **run.converse())    # the asking
    client = seat(run, index=request_number)                # the client

:func:`draft_for` and :func:`projector_for` answer 'auto' the way `ml-stack-serve up` does,
because a lease built by hand has to resolve what the CLI resolves for itself.
"""

from __future__ import annotations

import contextlib
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

__all__ = ["Asking", "Run", "Shape", "Talking", "draft_for", "held", "projector_for",
           "release_all", "seat"]

# The sampler settings a `Talking` carries. They are the client's, never the server's.
SAMPLERS = ("temperature", "top_p", "top_k", "min_p")


@dataclass(frozen=True)
class Shape:
    """One model, served one way. :meth:`lease` is what :func:`ml_stack.serve.serve` takes."""

    model: str
    port: int = 8080
    # One conversation each, with its own KV cache. The server divides the context it was
    # given between them, so what is asked for is seats x seat_context and a seat is what
    # any one conversation actually gets.
    seats: int = 1
    seat_context: int = 4096
    # How the KV cache is stored. "" leaves the server's own f16; "q8_0" halves it, which is
    # twice the seats at a context -- measure whether the answers change before taking it.
    cache_type: str = ""
    # A small model or a head of the same family, guessing ahead for the large one to check
    # in one pass. A path, or hf:owner/repo[/file.gguf]; "" for none. Which `--spec-type` it
    # needs is read from what it is called, so a head is never served as the wrong method.
    draft: str = ""
    draft_n_max: int | None = None      # tokens guessed ahead; None leaves the default
    # Which method the head implements. "" reads it off the head's own name, which is right
    # whenever the name says so; a profile that measured one says it outright, and a head
    # that lives inside the weights -- `--spec-type draft-mtp` with no `-md` -- can only be
    # asked for this way.
    spec_type: str = ""
    mmproj: str = ""                    # the vision projector, so the model can see
    reasoning_budget: int | None = None  # tokens a turn may think for; 0 turns it off
    mlock: bool = False                 # hold the weights in memory rather than let them page
    flash_attn: bool | None = None      # None leaves the spec's own, which is on
    # A named build from `ml-stack-serve build --name`, or "" for the managed master: an
    # architecture or a head newer than any release loads only on the build that has it.
    build: str = ""
    # Anything else llama-server takes that no field here names -- `-ub 2048`,
    # `--spec-draft-p-min 0.5`. Measured flags, not remembered ones: they are here because a
    # profile carries what a measurement found, and a run that found `-ub 2048` worth 4.7x
    # has nowhere else to put it.
    extra_args: tuple[str, ...] = ()

    @property
    def context(self) -> int:
        """What the server is asked for: every seat's context, added up."""
        return self.seat_context * self.seats

    def lease(self) -> dict[str, Any]:
        """The keyword arguments :func:`ml_stack.serve.serve` takes, model aside.

        Only what was actually asked for appears, so a shape that says nothing about a
        draft, a projector or thinking serves exactly as the build's own defaults do.
        """
        out: dict[str, Any] = {"port": self.port, "context": self.context,
                               "parallel": self.seats}
        if self.cache_type:
            out["cache_type_k"] = out["cache_type_v"] = self.cache_type
        if self.draft:
            from ml_stack.hub import spec_for

            out["draft"] = self.draft
            out["spec_type"] = self.spec_type or spec_for(self.draft)
            if self.draft_n_max is not None:
                out["spec_draft_max"] = self.draft_n_max
        elif self.spec_type:
            # a head inside the weights: the method, no -md, and how far it guesses
            out["spec_type"] = self.spec_type
            if self.draft_n_max is not None:
                out["spec_draft_max"] = self.draft_n_max
        if self.mmproj:
            out["mmproj"] = self.mmproj
        if self.reasoning_budget is not None:
            out["reasoning_budget"] = self.reasoning_budget
        if self.mlock:
            out["mlock"] = True
        if self.flash_attn is not None:
            out["flash_attn"] = bool(self.flash_attn)
        if self.extra_args:
            out["extra_args"] = tuple(self.extra_args)
        return out

    def manager(self) -> Any | None:
        """The :class:`~ml_stack.serve.ServerManager` for a named build, else None.

        None is not "no manager": it is the default one, which finds the binary the usual
        way. A build is named only when the model needs it.
        """
        if not self.build:
            return None
        from ml_stack.serve.backend import LlamaServerBackend
        from ml_stack.serve.manager import ServerManager

        return ServerManager(LlamaServerBackend(build=self.build))


@dataclass(frozen=True)
class Asking:
    """One model, asked one way. :meth:`converse` is what
    :func:`ml_stack.graph.ask.converse` takes.

    Every field here changes what the model is asked, and none of them is anything a client
    or a server has ever heard of. ``tight`` reached ``Client.__init__`` once and took an
    87G load down with it; keeping the ways in their own record is what makes that
    impossible rather than caught.
    """

    tight: bool = True
    batch: bool = False
    single: bool = False                 # one entry to a read, more turns -- batch's opposite
    few: bool = False                    # three tools offered, not eight
    kinds: bool = False
    summary: bool = False                # `converse`'s summary_tool, named as the bench is
    rich: bool = False
    terse: bool = False                  # `tools_for`'s schemas, not `converse`'s
    reach: int | None = None             # tokens one tool result may carry
    rounds: int | None = None            # tool-calling turns one question may spend
    constrain_ids: bool = False          # id arguments held to the graph's ids by grammar

    def converse(self) -> dict[str, Any]:
        """The keyword arguments :func:`ml_stack.graph.ask.converse` takes.

        Only what was asked for, so a way that asked for nothing leaves `converse` exactly
        as it was. ``summary`` becomes ``summary_tool``: `converse`'s own ``summary`` is a
        thread's rolling summary and passing one as the other hands a bool to something
        that reads text. ``terse`` is not here and cannot be -- it chooses the tool
        *schemas*, which is :func:`~ml_stack.graph.ask.tools_for`'s argument.
        """
        out: dict[str, Any] = {"tight": bool(self.tight)}
        for way in ("batch", "kinds", "rich", "single", "few", "constrain_ids"):
            if getattr(self, way):
                out[way] = True
        if self.summary:
            out["summary_tool"] = True
        if self.reach is not None:
            out["reach"] = int(self.reach)
        if self.rounds is not None:
            out["rounds"] = int(self.rounds)
        return out

    def tools(self) -> dict[str, Any]:
        """The keyword arguments :func:`~ml_stack.graph.ask.tools_for` takes about the
        asking -- the terse set is chosen outside `converse` and handed in."""
        return {"tight": bool(self.tight), "reach": self.reach,
                "batch": bool(self.batch), "single": bool(self.single),
                "few": bool(self.few), "summary": bool(self.summary)}

    def said(self) -> dict[str, Any]:
        """The way itself, in the words the bench keeps beside its rows: ``summary`` under
        its own name, ``terse`` said outright, and nothing that was not asked for."""
        out: dict[str, Any] = {"tight": bool(self.tight), "terse": bool(self.terse)}
        for way in ("rich", "batch", "kinds", "summary", "single", "few", "constrain_ids"):
            if getattr(self, way):
                out[way] = True
        if self.reach:
            out["reach"] = int(self.reach)
        if self.rounds:
            out["rounds"] = int(self.rounds)
        return out


@dataclass(frozen=True)
class Talking:
    """One model, talked to one way: what a :class:`~ml_stack.client.Client` is built from.

    ``timeout`` is the cap on one call, which is the bench's per-question cap said once.
    ``think`` is deliberately absent from :meth:`client`: the client takes it per call --
    ``chat(..., think=)`` -- and handing it to ``Client.__init__`` raises.
    """

    n_predict: int = 16384               # a ceiling, not a budget
    timeout: float = 300.0
    sampling: Mapping[str, Any] = field(default_factory=dict)
    think: bool | None = None

    def client(self) -> dict[str, Any]:
        """The keyword arguments :class:`~ml_stack.client.Client` takes."""
        out: dict[str, Any] = {"n_predict": int(self.n_predict), "timeout": float(self.timeout)}
        out.update({k: v for k, v in dict(self.sampling).items() if v is not None})
        return out


@dataclass(frozen=True)
class Run:
    """One model, whole: served one way, asked one way, talked to one way.

    Three sections, because three different pieces of code read them, and one object,
    because a bench row, a page answer and a seated client that build their own drift.
    :meth:`lease` is the server's, :meth:`converse` the asking's, :meth:`client` the
    client's, and :meth:`over` is how a caller changes one knob without knowing which
    section owns it.
    """

    shape: Shape
    asking: Asking = field(default_factory=Asking)
    talking: Talking = field(default_factory=Talking)

    @property
    def model(self) -> str:
        """The model reference served."""
        return self.shape.model

    @property
    def port(self) -> int:
        return self.shape.port

    def lease(self) -> dict[str, Any]:
        """The keyword arguments :func:`ml_stack.serve.serve` takes, model aside."""
        return self.shape.lease()

    def converse(self) -> dict[str, Any]:
        """The keyword arguments :func:`ml_stack.graph.ask.converse` takes."""
        return self.asking.converse()

    def client(self, base_url: str, *, index: int | None = None, **over: Any) -> Any:
        """A :class:`~ml_stack.client.Client` on this run's server.

        ``index`` pins it to a slot -- whose seat it is, taken modulo the seats -- and
        None leaves the server to choose, which is what a run measuring one conversation
        at a time wants.
        """
        from ml_stack.client import Client

        asked = {**self.talking.client(), **over}
        if index is not None:
            asked["slot"] = index % max(1, self.shape.seats)
        return Client(base_url, **asked)

    def over(self, **fields: Any) -> Run:
        """This run with ``fields`` laid over it, each routed to the section that owns it.

        A sampler setting no field names -- ``temperature``, ``top_k`` -- goes into
        ``talking.sampling``. A name no section knows is a `TypeError` here rather than a
        keyword the client refuses at the far end of a load.
        """
        parts: dict[str, dict[str, Any]] = {"shape": {}, "asking": {}, "talking": {}}
        sampling = dict(self.talking.sampling)
        for name, value in fields.items():
            if name in Shape.__dataclass_fields__:
                parts["shape"][name] = value
            elif name in Asking.__dataclass_fields__:
                parts["asking"][name] = value
            elif name in Talking.__dataclass_fields__:
                parts["talking"][name] = value
            elif name in SAMPLERS:
                sampling[name] = value
            else:
                raise TypeError(f"no such run field: {name}")
        if sampling != dict(self.talking.sampling):
            parts["talking"]["sampling"] = sampling
        return replace(self, **{name: replace(getattr(self, name), **taken)
                                for name, taken in parts.items() if taken})


# One held server per port, for the life of the process. There is more than one model in a
# program that reads with a large one and answers with a small one, and a single slot here
# handed whichever was asked for first to both of them.
_LOCK = threading.Lock()
_STACKS: dict[int, contextlib.ExitStack] = {}
_URLS: dict[int, str] = {}


def seat(shape: Shape | Run, *, index: int, n_predict: int | None = None,
         timeout: float | None = None, **client_kwargs: Any) -> Any:
    """A client on one seat of ``shape``'s server, started on first ask and held after.

    ``shape`` is a :class:`Shape` or the whole :class:`Run`; given a run, the ceiling, the
    timeout and the sampling are its ``talking``'s and need not be said again.

    ``index`` is whose seat it is -- a request number, a worker id -- taken modulo the
    seats, so each conversation keeps its own KV cache and a busy port cycles through them
    rather than fighting over one. Sampling is the client's default, which is greedy: a
    task that calls tools with exact ids is one where sampling noise becomes a wrong
    argument rather than a livelier sentence.
    """
    run = shape if isinstance(shape, Run) else Run(shape=shape)
    if n_predict is not None:
        client_kwargs["n_predict"] = n_predict
    if timeout is not None:
        client_kwargs["timeout"] = timeout
    with _LOCK:
        if run.port not in _URLS:
            from ml_stack.serve import serve

            stack = contextlib.ExitStack()
            server = stack.enter_context(
                serve(run.model, manager=run.shape.manager(), **run.lease()))
            _STACKS[run.port], _URLS[run.port] = stack, server.base_url
        where = _URLS[run.port]
    return run.client(where, index=index, **client_kwargs)


def held() -> dict[int, str]:
    """port -> base url, for every server :func:`seat` is holding."""
    with _LOCK:
        return dict(_URLS)


def release_all() -> None:
    """Let go of every held server. What that does is the manager's business: a server this
    process started stops, one it adopted stays up for whoever else is using it."""
    with _LOCK:
        stacks = list(_STACKS.values())
        _STACKS.clear()
        _URLS.clear()
    for stack in stacks:
        stack.close()


def draft_for(model: str, asked: str, *, build: str = "",
              log: Callable[[str], None] | None = None) -> str:
    """The draft head to serve beside ``model``, resolving 'auto', or "" if there is none.

    'auto' reads the repository's own listing rather than guessing a filename, and only an
    ``hf:`` reference can be resolved that way -- a local path says nothing about where it
    came from. ``build`` is the named build the head has to load on, since a head withheld
    from mainline is not a head this server can use.

    A head that cannot be found is served without, out loud: ``log`` is told why. Said in
    silence once, and a model ran undrafted for an hour with nothing to show for it.
    """
    from ml_stack.serve.cli import drafted

    try:
        binary: str | Path | None = None
        if build:
            from ml_stack.serve.backend import LlamaServerBackend

            binary = LlamaServerBackend(build=build).binary
        return drafted(str(model), asked, binary=binary)
    except Exception as exc:  # noqa: BLE001 - a draft that cannot be found is served without
        if log:
            log(f"no draft head: {exc}")
        return ""


def projector_for(model: str, asked: str, *,
                  log: Callable[[str], None] | None = None) -> str:
    """The vision projector to serve beside ``model``, resolving 'auto', or "" for none.

    :func:`ml_stack.serve.serve` hands ``mmproj`` straight to the ``ServerSpec`` and
    resolves nothing, so 'auto' is answered here the way the CLI answers it: beside the
    weights, then the directory above, then every revision of the same repository, taking
    the *most precise* projector found -- quantising one costs sight out of all proportion
    to what it saves.
    """
    from ml_stack.serve.cli import alongside

    try:
        return alongside(str(model), asked, "mmproj-", best=True)
    except Exception as exc:  # noqa: BLE001 - a projector not found is served without
        if log:
            log(f"no projector: {exc}")
        return ""
