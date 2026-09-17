"""``ml-stack-bench tree``: tree speculative decoding on MLX, checked lossless and timed by arm.

``lossless`` loads the target once and, for each drafter, decodes fixed prompts greedily
through the tree engine. A reply identical to plain greedy decoding passes. One that differs
passes only where every drafted token is within twice the model's own shape noise of plain
decoding's best token: the logit gap between the model decoding one token at a time and
reading the same tokens in one batched pass, measured over the same positions in the run.

``speed`` times plain MLX decoding and each tree drafter in one process, then each llama.cpp
arm leased through the broker, over chat, code and math prompts with thinking on and off,
greedy, at each token count, and keeps every sample with its settings in one JSON.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import metal_smi
import mlx.core as mx

from ml_stack import bench
from ml_stack.bench.quiet import look
from ml_stack.bench.speed import prompt_text
from ml_stack.client import Client
from ml_stack.client.health import serving_params
from ml_stack.files import write_json
from ml_stack.lock import only_one
from ml_stack.log import say, warn
from ml_stack.serve import broker_wire
from ml_stack.spec.cost import load_curve
from ml_stack.spec.decode import Asked, Session, decode
from ml_stack.spec.drafters import Budget
from ml_stack.spec.engine import EngineConfig, load_target, make_drafter, weights
from ml_stack.spec.layout import layout_for

__all__ = ["PROMPTS", "Target", "run", "witness"]

MODEL = "mlx-community/Qwen3.8-Flash-Next-4bit"
PROMPTS = {
    "chat": "Explain to someone planting their first vegetable garden how to plan one growing "
            "season: soil, layout, planting times, watering and pests.",
    "code": "Write a Python module with an LRU cache class offering get, put and resize, with "
            "type hints, docstrings and unittest tests.",
    "math": "Solve step by step: pipe A fills a tank in 6 hours, pipe B in 9 hours, and pipe C "
            "drains it in 12 hours. With all three open from empty, when is it full? Then "
            "solve it for fill times a and b and drain time c.",
}
#: tokens a witness prompt must exceed so its passes attend sparsely past the indexer budget
LONG_TOKENS = 2200
#: the largest self-disagreement, in logits, under which a witness can tell drafts from noise
MAX_NOISE = 1.0


def long_prompt(tokenizer: Any) -> str:
    """Numbered report lines to past `LONG_TOKENS` tokens, then a request to summarise them."""
    size = LONG_TOKENS
    while True:
        text = prompt_text(size) + "\nSummarise the report above in one paragraph."
        if len(tokenizer.encode(text)) > LONG_TOKENS:
            return text
        size += 200


class Target:
    """One loaded MLX model: its layout, its tokenizer, and plain greedy decoding."""

    def __init__(self, model: str) -> None:
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])
        self.path = weights(model)
        self.model, self.tokenizer = load_target(self.path)
        self.layout = layout_for(self.model)
        self.text = getattr(self.model, "language_model", self.model)
        self.eos = frozenset(self.tokenizer.eos_token_ids)

    def prompt(self, question: str, thinking: bool) -> list[int]:
        return list(self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], add_generation_prompt=True,
            enable_thinking=thinking))

    def logits(self, tokens: Sequence[int], cache: Any) -> Any:
        """Float32 logits [T, V] for ``tokens`` read after ``cache``."""
        return self.text(mx.array([list(tokens)]), cache=cache).logits[0].astype(mx.float32)

    def plain(self, prompt: Sequence[int], count: int) -> tuple[list[int], float]:
        """Greedy tokens one forward each, and the seconds they took after the prefill."""
        cache = self.layout.make_cache()
        row = self.logits(prompt, cache)[-1]
        out: list[int] = []
        began = time.perf_counter()
        while len(out) < count:
            out.append(int(row.argmax().item()))
            if out[-1] in self.eos or len(out) == count:
                break
            row = self.logits(out[-1:], cache)[0]
        return out, time.perf_counter() - began

    def session(self, kind: str, head: str, max_nodes: int) -> Any:
        budget = Budget(max_nodes=max_nodes, cost=load_curve(self.layout, self.path.name))
        config = EngineConfig(model=str(self.path), drafter=kind, drafter_model=head)
        return Session(self.layout, make_drafter(config, self.layout, budget), keep=0)


def witness(target: Target, prompt: Sequence[int], drafted: Sequence[int]) -> dict[str, Any]:
    """Whether every drafted token is plain greedy decoding's choice, up to measured noise."""
    reference, _ = target.plain(prompt, len(drafted))
    if list(drafted) == reference:
        return {"exact": True, "ok": True, "tokens": len(drafted)}
    cache = target.layout.make_cache()
    stepped = [target.logits(prompt, cache)[-1]]
    stepped += [target.logits([token], cache)[0] for token in drafted[:-1]]
    batched = target.logits(list(prompt) + list(drafted[:-1]),
                            target.layout.make_cache())[len(prompt) - 1:]
    noise, gaps = 0.0, []
    for i, token in enumerate(drafted):
        best = int(stepped[i].argmax().item())
        picked = mx.array([best, token])
        drift = mx.abs(stepped[i][picked] - batched[i][picked]).max()
        gap = stepped[i][best] - stepped[i][token]
        noise = max(noise, float(drift.item()))
        gaps.append(float(gap.item()))
    first = next((i for i, (a, b) in enumerate(zip(drafted, reference, strict=False))
                  if a != b), min(len(drafted), len(reference)))
    worst = max(gaps)
    return {"exact": False, "ok": noise <= MAX_NOISE and worst <= 2 * noise,
            "tokens": len(drafted), "first_difference": first, "worst_gap": worst,
            "noise": noise, "tolerance": 2 * noise}


def _drafters(named: Sequence[str]) -> list[tuple[str, str]]:
    """``ngram``, ``mtp=HEAD`` or ``dflash=HEAD`` as (kind, head)."""
    out = []
    for one in named:
        kind, _, head = one.partition("=")
        if kind not in ("ngram", "mtp", "dflash") or (kind != "ngram" and not head):
            raise SystemExit(f"error: --drafter {one!r}: name ngram, mtp=HEAD or dflash=HEAD")
        out.append((kind, head))
    return out


def lossless(args: argparse.Namespace) -> int:
    """Exit 0 when every drafter passes the witness on every prompt, else 1."""
    target = Target(args.model)
    prompts = [(name, question, False) for name, question in PROMPTS.items()]
    prompts += [("math", PROMPTS["math"], True), ("long", long_prompt(target.tokenizer), False)]

    results, failed = [], 0
    for kind, head in _drafters(args.drafter):
        session = target.session(kind, head, args.max_nodes)
        for name, question, thinking in prompts:
            session.reset()
            tokens = target.prompt(question, thinking)
            out = decode(session, tokens, Asked(args.tokens, eos=target.eos))
            verdict = {"drafter": kind, "head": head, "prompt": name, "thinking": thinking,
                       "prompt_tokens": len(tokens), "tokens_per_pass": out.tokens_per_pass,
                       **witness(target, tokens, out.tokens)}
            failed += not verdict["ok"]
            results.append(verdict)
            say(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in verdict.items() if k != "head"), flush=True)
    kept = _keep(args, "lossless", {"model": args.model, "results": results})
    say(f"{'FAIL' if failed else 'PASS'}: {len(results) - failed}/{len(results)} "
        f"lossless; kept {kept}")
    return 1 if failed or not results else 0


@dataclass(frozen=True)
class Sample:
    """One timed completion."""

    arm: str
    prompt: str
    thinking: bool
    max_tokens: int
    sample: int
    tokens: int
    seconds: float
    tokens_per_second: float
    passes: int | None
    tokens_per_pass: float | None
    accepted_per_pass: float | None
    pass_ms: float | None
    peak_resident_bytes: int | None
    peak_mlx_bytes: int | None = None


def _peak(pid: int) -> int | None:
    info = metal_smi.proc_info(pid)
    return int(info["peak_memory"]) if info else None


def _mlx_samples(target: Target, arm: str, run: Callable[[list[int], int], tuple],
                 args: argparse.Namespace) -> list[Sample]:
    out = []
    run(target.prompt(PROMPTS["chat"], False), 16)
    for count in args.tokens_list:
        for name, question in PROMPTS.items():
            for thinking in (False, True):
                tokens = target.prompt(question, thinking)
                for n in range(args.samples):
                    mx.reset_peak_memory()
                    written, seconds, passes, accepted = run(tokens, count)
                    out.append(Sample(
                        arm, name, thinking, count, n, written, seconds, written / seconds,
                        passes, written / passes if passes else None,
                        accepted / passes if passes else None,
                        1000 * seconds / passes if passes else None,
                        _peak(os.getpid()), int(mx.get_peak_memory())))
                    say(f"{arm} {name} thinking={thinking} {count}: {written} tokens "
                        f"{written / seconds:.1f} tok/s", flush=True)
    return out


def mlx_arms(args: argparse.Namespace) -> list[Sample]:
    """Plain MLX greedy decoding and every tree drafter, on one loaded target."""

    target = Target(args.model)

    def plain(tokens: list[int], count: int) -> tuple:
        written, seconds = target.plain(tokens, count)
        return len(written), seconds, len(written), 0

    samples = _mlx_samples(target, "mlx-plain", plain, args)
    for kind, head in _drafters(args.drafter):
        session = target.session(kind, head, args.max_nodes)

        def tree(tokens: list[int], count: int, session: Any = session) -> tuple:
            session.reset()
            out = decode(session, tokens, Asked(count, eos=target.eos))
            return (len(out.tokens), out.seconds, len(out.passes),
                    sum(p.accepted for p in out.passes))

        samples += _mlx_samples(target, f"mlx-tree-{kind}", tree, args)
    return samples


def llama_arms(args: argparse.Namespace) -> tuple[list[Sample], list[dict[str, Any]]]:
    """llama.cpp without a draft and with the shared MTP head, each leased through the broker."""
    arms = [("llama-plain", {"context": args.context}),
            ("llama-mtp", {"context": args.context, "draft": args.gguf_mtp,
                           "spec_type": "draft-mtp", "spec_draft_max": args.draft_max})]
    samples, served = [], []
    for arm, spec in arms:
        grant = broker_wire.lease("bench-tree", [args.gguf], spec=spec, timeout=1800.0)
        try:
            pid = next(int(s["pid"]) for s in broker_wire.status()["servers"]
                       if s["port"] == grant.port)
            params = serving_params(grant.base_url)
            served.append({"arm": arm, "spec": spec, "pid": pid, "base_url": grant.base_url,
                           "params": asdict(params) if params is not None else None})
            for count in args.tokens_list:
                client = Client(grant.base_url, temperature=0.0, n_predict=count)
                client.chat([{"role": "user", "content": "Say hello."}], think=False)
                for name, question in PROMPTS.items():
                    for thinking in (False, True):
                        for n in range(args.samples):
                            reply = client.chat([{"role": "user", "content": question}],
                                                think=thinking)
                            samples.append(_llama_sample(arm, (name, thinking, count, n),
                                                         reply.raw["timings"], _peak(pid)))
                            say(f"{arm} {name} thinking={thinking} {count}: "
                                f"{samples[-1].tokens_per_second:.1f} tok/s", flush=True)
        finally:
            broker_wire.release(grant.lease)
    return samples, served


def _llama_sample(arm: str, cell: tuple[str, bool, int, int], timings: dict[str, Any],
                  peak: int | None) -> Sample:
    written, seconds = int(timings["predicted_n"]), float(timings["predicted_ms"]) / 1000
    accepted = int(timings.get("draft_n_accepted") or 0)
    passes = int(timings.get("verify_n") or written - accepted)
    return Sample(arm, *cell, written, seconds, written / seconds, passes,
                  written / passes, accepted / passes, 1000 * seconds / passes, peak)


def _quiet(args: argparse.Namespace) -> list[str]:
    """Why a timing taken now would not be this model's alone; empty when quiet."""
    reasons = list(look().reasons)
    busy = metal_smi.system_gpu_stats()["device_utilization"]
    if busy > args.gpu_busy:
        reasons.append(f"the GPU is {busy}% utilized before anything was loaded")
    return reasons


def speed(args: argparse.Namespace) -> int:
    """Time every arm on a quiet machine, waiting in bounded checks; 3 when it stays busy."""
    reasons = _quiet(args)
    for _ in range(args.wait_checks):
        if not reasons:
            break
        warn("not quiet: " + "; ".join(reasons) + f"; checking again in {args.wait_s:g}s")
        time.sleep(args.wait_s)
        reasons = _quiet(args)
    if reasons and not args.anyway:
        warn("refused: " + "; ".join(reasons))
        return 3
    samples: list[Sample] = []
    served: list[dict[str, Any]] = []
    if "mlx" in args.engines:
        samples += mlx_arms(args)
        gc.collect()
        mx.clear_cache()
    if "llama" in args.engines:
        more, served = llama_arms(args)
        samples += more
    kept = _keep(args, "speed", {"model": args.model, "gguf": args.gguf,
                                 "gguf_mtp": args.gguf_mtp, "draft_max": args.draft_max,
                                 "drafters": args.drafter, "max_nodes": args.max_nodes,
                                 "busy": reasons, "served": served,
                                 "samples": [asdict(s) for s in samples]})
    say(f"kept {kept}")
    return 0


def _keep(args: argparse.Namespace, kind: str, body: dict[str, Any]) -> Path:
    rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                         cwd=Path(__file__).resolve().parent, check=False).stdout.strip()
    where = args.out or bench.home_dir() / "tree" / f"{kind}-{time.strftime('%Y%m%d-%H%M%S')}.json"
    where.parent.mkdir(parents=True, exist_ok=True)
    write_json(where, {"kind": kind, "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "code_rev": rev, "mlx": mx.__version__, "argv": vars(args) | {
                           "out": str(args.out or "")}, **body})
    return where


def run(args: argparse.Namespace) -> int:
    """``ml-stack-bench tree lossless|speed``, under the measuring lock."""
    if not args.drafter:
        raise SystemExit("error: name at least one --drafter")
    with only_one(bench.home_dir() / "measuring.lock", announce=warn):
        return lossless(args) if args.action == "lossless" else speed(args)
