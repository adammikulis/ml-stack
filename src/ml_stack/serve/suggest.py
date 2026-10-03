"""The best settings for a model on a machine, and the best model for a machine.

``suggest`` picks context, slots, offload, cache types and batch for one model so that
`estimate.verdict` stays at or under a level; ``suggest_model`` ranks models for a goal.
Both are deterministic: the same header, sizes and machine give the same answer.
"""

from __future__ import annotations

import contextlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

from ml_stack import hub
from ml_stack.hub import naming, remote
from ml_stack.hub.discover import ModelInfo
from ml_stack.hub.probe import MachineMemory, machine_memory
from ml_stack.platform import is_apple_silicon
from ml_stack.serve import estimate as est
from ml_stack.units import human_bytes

__all__ = ["Candidate", "Choice", "Option", "Recommendation", "Suggestion", "Want",
           "recommend", "suggest", "suggest_meta", "suggest_model"]

GOALS = {
    "agent": (32768, 8192),
    "chat": (16384, 4096),
    "long-context": (1_048_576, 8192),
    "fast": (4096, 2048),
}
"""``goal -> (context to aim for, the smallest context worth offering)``."""

STEPS = tuple(2**n for n in range(20, 8, -1))
BATCHES = {"agent": (2048, 1024, 512), "long-context": (2048, 1024, 512)}
OFFLOADS = (1.0, 0.75, 0.5, 0.25, 0.0)

QUALITY = {"F32": 1.0, "F16": 1.0, "BF16": 1.0, "Q8": 1.0, "Q6": 0.99, "Q5": 0.97,
           "Q4": 0.93, "IQ4": 0.92, "MXFP4": 0.92, "Q3": 0.8, "IQ3": 0.78, "Q2": 0.6,
           "IQ2": 0.55, "IQ1": 0.4}
"""How much of its full-precision quality a quantisation keeps, by prefix. A rough ranking
aid, not a measurement."""

KV_PER_PARAMETER = 2.0e-5
"""Bytes of q8_0 KV cache per token per parameter, for a model whose header is not at hand
(fit to a 4B dense model: 78 kB a token)."""

NOT_CHAT = ("bert", "embed", "clip", "whisper", "ltxv", "t5", "rerank", "bge", "minilm")


@dataclass(frozen=True, slots=True)
class Option:
    """One way of serving, with what it costs and how it rates."""

    label: str
    context: int
    n_gpu_layers: int | str
    verdict: str
    total_bytes: int


@dataclass(frozen=True, slots=True)
class Suggestion:
    """Settings for one model on one machine."""

    context: int
    parallel: int
    n_gpu_layers: int | str
    kv_cache_type: str
    flash_attn: bool
    batch: int
    estimate: est.Estimate
    verdict: str
    reasons: tuple[str, ...]
    alternatives: tuple[Option, ...] = ()

    def setup(self) -> est.Setup:
        """The settings as an `estimate.Setup`."""
        return est.Setup(self.context, self.parallel, self.n_gpu_layers, self.kv_cache_type,
                         self.flash_attn, self.batch)

    def lease(self) -> dict[str, object]:
        """The keyword arguments of `ServerSpec` (and `serve`) for these settings, model
        aside: ``-c`` is every slot's context added up, ``-ctk``/``-ctv`` come from the
        cache type, and a batch other than 512 is ``-ub``."""
        k, v = est.split_kv(self.kv_cache_type)
        out: dict[str, object] = {
            "context": self.context * self.parallel, "parallel": self.parallel,
            "n_gpu_layers": self.n_gpu_layers, "cache_type_k": k, "cache_type_v": v,
            "flash_attn": self.flash_attn}
        if self.batch != 512:
            out["extra_args"] = ("-ub", str(self.batch))
        return out

    def as_dict(self) -> dict[str, object]:
        return {"context": self.context, "parallel": self.parallel,
                "n_gpu_layers": self.n_gpu_layers, "kv_cache_type": self.kv_cache_type,
                "flash_attn": self.flash_attn, "batch": self.batch,
                "estimate": self.estimate.as_dict(), "verdict": self.verdict,
                "reasons": list(self.reasons),
                "alternatives": [{"label": o.label, "context": o.context,
                                  "n_gpu_layers": o.n_gpu_layers, "verdict": o.verdict,
                                  "total_bytes": o.total_bytes} for o in self.alternatives]}


def tokens(count: int) -> str:
    """``16384`` as ``16k``."""
    return f"{count // 1024}k" if count >= 1024 and count % 1024 == 0 else str(count)


def _contexts(goal: str, trained: int) -> list[int]:
    aim = GOALS[goal][0]
    top = min(aim, trained) if trained else aim
    return sorted({top, *(s for s in STEPS if s < top)}, reverse=True)


@dataclass(frozen=True, slots=True)
class Want:
    """What a suggestion is for: ``goal``, the worst rating accepted, and the memory kept
    back for the system (`estimate.reserve_default` when ``None``)."""

    goal: str = "agent"
    max_verdict: str = "green"
    reserve_bytes: int | None = None
    mmproj_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _Ask:
    found: Mapping[str, object]
    weights: int
    machine: MachineMemory
    want: Want
    flash: bool

    @property
    def trained(self) -> int:
        arch = str(self.found.get("general.architecture") or "")
        return int(self.found.get(f"{arch}.context_length") or 0)  # type: ignore[call-overload]

    @property
    def layers(self) -> int:
        arch = str(self.found.get("general.architecture") or "")
        return int(self.found.get(f"{arch}.block_count") or 0)  # type: ignore[call-overload]

    @property
    def gpu(self) -> bool:
        return bool(self.machine.gpus) or self.machine.unified

    def kv(self, kind: str) -> str:
        return kind if self.flash else f"{kind}/f16"

    def rate(self, use: est.Setup) -> tuple[est.Estimate, str]:
        got = est.estimate_meta(self.found, self.weights, use)
        return got, est.verdict(got, self.machine, self.want.reserve_bytes)

    def within(self, rated: str) -> bool:
        return est.ORDER[rated] <= est.ORDER[self.want.max_verdict]

    def pick(self, start: est.Setup, steps: list[int]
             ) -> tuple[est.Estimate, est.Setup, str] | None:
        """The largest of ``steps`` whose rating is accepted."""
        for ctx in steps:
            use = replace(start, context=ctx)
            got, rated = self.rate(use)
            if self.within(rated):
                return got, use, rated
        return None

    def headroom(self, got: est.Estimate) -> str:
        pool, free, _ = est.headroom(got, self.machine, self.want.reserve_bytes)
        return f"{human_bytes(max(free, 0))} of {pool} left"

    def alternatives(self, use: est.Setup, steps: list[int]) -> tuple[Option, ...]:
        out: list[Option] = []
        limit = self.trained or STEPS[0]
        larger = next((s for s in reversed(STEPS) if use.context < s <= limit), 0)
        for label, ctx in (("smaller and faster", max(use.context // 2, 512)),
                           ("bigger and tighter", larger)):
            if ctx and ctx != use.context:
                got, rated = self.rate(replace(use, context=ctx))
                out.append(Option(f"{label}: {tokens(ctx)} context", ctx, use.n_gpu_layers,
                                  rated, got.total_bytes))
        return tuple(out)

    def reasons(self, use: est.Setup, got: est.Estimate, rated: str) -> tuple[str, ...]:
        said = [f"Context {tokens(use.context)}: the largest that stays {rated} "
                f"with {self.headroom(got)}"]
        if self.trained and use.context >= self.trained:
            said.append(f"Context is capped at the model's trained {tokens(self.trained)}")
        if use.kv_cache_type == "q8_0":
            said.append("KV cache q8_0: about half the memory of f16, near-identical output")
        elif use.kv_cache_type.startswith("q8_0/"):
            said.append("Flash attention is not available here, so the V cache stays f16 "
                        "(a quantised V cache needs it); K is q8_0")
        elif use.kv_cache_type.startswith("q4_0"):
            said.append("KV cache q4_0: nothing larger fit; it saves more and costs quality")
        if use.batch > 512:
            said.append(f"Batch {use.batch}: faster prompt processing, "
                        f"{human_bytes(got.compute_buffer_bytes)} of compute buffers")
        if use.n_gpu_layers not in ("auto", 0):
            said.append(f"{use.n_gpu_layers} layers on the GPU; the model does not fit whole")
        if self.want.goal == "agent":
            said.append("One slot: tool output is long, and slots divide the context")
        return tuple(said)

    def offloaded(self, base: est.Setup, steps: list[int]
                  ) -> tuple[est.Estimate, est.Setup, str] | None:
        if not self.gpu or self.machine.unified or not self.layers:
            return None
        for share in OFFLOADS[1:]:
            got = self.pick(replace(base, n_gpu_layers=int(self.layers * share)), steps)
            if got:
                return got
        return None

    def batched(self, use: est.Setup, got: est.Estimate, rated: str
                ) -> tuple[est.Estimate, est.Setup]:
        for batch in BATCHES.get(self.want.goal, (512,)):
            more = replace(use, batch=batch)
            tried, rating = self.rate(more)
            if est.ORDER[rating] <= est.ORDER[rated]:
                return tried, more
        return got, use

    def smallest(self, base: est.Setup, steps: list[int]) -> Suggestion:
        least = replace(base, kv_cache_type=self.kv("q4_0"), context=steps[-1],
                        n_gpu_layers=base.n_gpu_layers if self.machine.unified else 0)
        got, rated = self.rate(least)
        said = (f"Nothing fits at {self.want.max_verdict} or better: this is the smallest "
                f"option, rated {rated}", f"The weights alone are {human_bytes(self.weights)}")
        return Suggestion(least.context, 1, least.n_gpu_layers, least.kv_cache_type,
                          self.flash, 512, got, rated, said, self.alternatives(least, steps))


def suggest_meta(found: Mapping[str, object], weights_bytes: int,
                 machine: MachineMemory | None = None, want: Want | None = None
                 ) -> Suggestion:
    """`suggest` for a header already read."""
    want = want or Want()
    if want.goal not in GOALS:
        raise ValueError(f"goal is one of {', '.join(GOALS)}, not {want.goal!r}")
    ask = _Ask(found, weights_bytes, machine or machine_memory(), want,
               est.flash_attention_likely(found))
    steps = _contexts(want.goal, ask.trained)
    big = [s for s in steps if s >= GOALS[want.goal][1]] or steps
    base = est.Setup(kv_cache_type=ask.kv("q8_0"), flash_attn=ask.flash,
                     mmproj_bytes=want.mmproj_bytes, n_gpu_layers="auto" if ask.gpu else 0)
    tight = replace(base, kv_cache_type=ask.kv("q4_0"))
    chosen = (ask.pick(base, big) or ask.offloaded(base, big) or ask.pick(base, steps)
              or ask.pick(tight, big))
    if not chosen:
        return ask.smallest(base, steps)
    got, use, rated = chosen
    got, use = ask.batched(use, got, rated)
    return Suggestion(use.context, use.parallel, use.n_gpu_layers, use.kv_cache_type,
                      ask.flash, use.batch, got, rated, ask.reasons(use, got, rated),
                      ask.alternatives(use, steps))


def suggest(model: str | Path, machine: MachineMemory | None = None, *, goal: str = "agent",
            max_verdict: str = "green", reserve_bytes: int | None = None) -> Suggestion:
    """Settings for an installed model on ``machine`` (this one by default).

    ``goal`` is ``agent`` (tool calling: up to 32k, one slot), ``chat`` (16k),
    ``long-context`` (the largest that fits) or ``fast`` (4k). The answer is never rated
    worse than ``max_verdict`` unless nothing fits, and then it is the smallest option with
    the reason. Context never exceeds the model's trained context. Quantised KV needs
    flash attention; where the head size or the build lacks it V stays f16.
    """
    path = est._locate(model)
    shown = [m for m in hub.discover(formats=("gguf",)) if m.path == path]
    mm = shown[0].mmproj if shown and shown[0].mmproj else None
    return suggest_meta(est.read_meta(path), est._bytes_of(path), machine,
                        Want(goal, max_verdict, reserve_bytes,
                             est._bytes_of(mm) if mm else 0))


@dataclass(frozen=True, slots=True)
class Candidate:
    """A model that may be installed or only downloadable, as far as it is known."""

    name: str
    size_bytes: int
    parameters: int = 0
    quantization: str = ""
    architecture: str = ""
    context_length: int = 0
    ref: str = ""
    path: Path | None = None


@dataclass(frozen=True, slots=True)
class Choice:
    """A candidate with how it rates on this machine."""

    candidate: Candidate
    verdict: str
    score: float
    reason: str


def candidate_of(info: ModelInfo) -> Candidate:
    """A `Candidate` for an installed model."""
    return Candidate(info.name, info.size_bytes, info.parameters, info.quantization,
                     info.architecture, info.context_length, info.id, info.path)


def _quality(quant: str) -> float:
    text = quant.upper()
    for prefix in sorted(QUALITY, key=len, reverse=True):
        if text.startswith(prefix):
            return QUALITY[prefix]
    return 0.9


def _chat(one: Candidate) -> bool:
    text = f"{one.architecture} {one.name}".lower()
    return not any(word in text for word in NOT_CHAT)


def _found_for(one: Candidate) -> Mapping[str, object]:
    if one.path:
        return est.read_meta(one.path)
    return {"general.architecture": one.architecture}


def _effective(one: Candidate) -> float:
    params = one.parameters or one.size_bytes / 0.6
    moe = 0.5 if "moe" in one.architecture.lower() else 1.0
    return params * _quality(one.quantization) * moe


def is_iq(one: Candidate) -> bool:
    """Whether ``one`` is an IQ-family quantisation, by its header's file type or its name."""
    found = naming.QUANT.search(f"{one.name} {one.ref}")
    return one.quantization.upper().startswith("IQ") or bool(
        found and found.group(1).upper().startswith("IQ"))


def suggest_model(candidates: Iterable[Candidate | ModelInfo],
                  machine: MachineMemory | None = None, goal: str = "agent", *,
                  max_verdict: str = "green", reserve_bytes: int | None = None
                  ) -> list[Choice]:
    """Candidates ranked for ``goal`` on ``machine``, best first.

    Models whose settings (`suggest_meta`) rate better come first. Within a rating, the
    score is parameters times a quantisation quality factor (a mixture of experts counts
    half), so a larger model wins when it fits; for ``fast`` it is the smaller of the
    models of 1.5 billion parameters or more. Embedding, vision-encoder and speech models
    are left out. No benchmark scores are used: the ranking is a size heuristic. On Apple
    silicon an IQ quantisation ranks after a
    candidate it ties with on rating and score, and its note says it may be slower on Metal.
    """
    machine = machine or machine_memory()
    apple = is_apple_silicon()
    floor = GOALS[goal][1]
    rows: list[Choice] = []
    for item in candidates:
        one = candidate_of(item) if isinstance(item, ModelInfo) else item
        if not _chat(one) or one.size_bytes <= 0:
            continue
        probe = est.estimate_meta(_found_for(one), one.size_bytes, est.Setup(context=floor))
        if not probe.kv_cache_bytes and not probe.state_bytes:
            kv = int(KV_PER_PARAMETER * (one.parameters or one.size_bytes / 0.6) * floor)
            probe = replace(probe, total_bytes=probe.total_bytes + kv,
                            gpu_bytes=probe.gpu_bytes + kv)
        rated = est.verdict(probe, machine, reserve_bytes)
        score = _effective(one)
        if goal == "fast":
            score = -one.size_bytes if _effective(one) >= 1.5e9 else -1e18 + one.size_bytes
        note = f"{human_bytes(one.size_bytes)} {one.quantization or 'weights'}, rated {rated}"
        if apple and is_iq(one):
            note += "; IQ quantisation, may be slower on Metal"
        rows.append(Choice(one, rated, score, note))
    rows.sort(key=lambda r: (est.ORDER[r.verdict] > est.ORDER[max_verdict],
                             est.ORDER[r.verdict], -r.score, apple and is_iq(r.candidate),
                             r.candidate.name))
    return rows



@dataclass(frozen=True, slots=True)
class Recommendation:
    """A model worth using on this machine, installed or still to download."""

    choice: Choice
    installed: bool
    ref: str


_PARAMS = re.compile(r"(\d+(?:\.\d+)?)\s*[Bb](?![a-z])")
_ACTIVE = re.compile(r"-A\d+(?:\.\d+)?B", re.IGNORECASE)


def download_candidates(repos: Iterable[remote.Repo]) -> list[Candidate]:
    """One `Candidate` per build of each repository, sized from the listing."""
    out = []
    for repo in repos:
        found = _PARAMS.search(repo.id.split("/")[-1])
        params = int(float(found.group(1)) * 1e9) if found else 0
        moe = "moe" if _ACTIVE.search(repo.id) else ""
        for build, size, _shards, quant in repo.builds():
            first = next(f for f in repo.files if f.build == build)
            out.append(Candidate(f"{repo.id} {build}", size, params, quant, moe, 0,
                                 f"hf:{repo.id}/{first.path}"))
    return out


def recommend(machine: MachineMemory | None = None, goal: str = "agent", *, query: str = "",
              limit: int = 8) -> list[Recommendation]:
    """Installed models, and with ``query`` the GGUF builds the Hub offers for it, ranked
    for ``goal`` on ``machine`` and cut to ``limit``. Ranking is `suggest_model`'s; the
    search is skipped when the Hub cannot be reached."""
    machine = machine or machine_memory()
    installed = hub.discover(formats=("gguf",))
    pool: list[Candidate | ModelInfo] = list(installed)
    owned = {m.id for m in installed}
    if query:
        with contextlib.suppress(remote.RemoteError, OSError):
            pool += [c for c in download_candidates(remote.search(query, remote.Filters(limit=6)))
                     if c.ref not in owned]
    rows = suggest_model(pool, machine, goal)
    return [Recommendation(r, r.candidate.ref in owned, r.candidate.ref)
            for r in rows if r.verdict != "none"][:limit]
