"""``ml-stack-bench speed``: how fast a served model reads and writes, by prompt size and
by how many ask at once.

One cell per (prompt size, streams): ``streams`` identical-length requests sent together,
greedy, thinking off, each writing ``--generate`` tokens. Per cell: ``prefill_tps`` (what
the server read over the time it took, per stream and over all of them), ``decode_tps``
(what it wrote, per stream and summed for throughput), ``ttft_s`` (the time to the first
token, from the server's own prompt clock and marked so), the wall, and the same memory
and `served_by` record every run carries. Kept in the store as a run of kind ``speed``
(`KIND`), one row per cell; `speed_table` prints them, one table per label.

The prompt is built to a token count rather than guessed at: `prompt_text` writes numbered
lines, `calibrated` measures them -- through ``/tokenize`` where the server has one, else
by a one-token request read back for what the server counted -- and scales the text until
it lands within `TOLERANCE`. Each stream gets its own lines, so no two share a prefix and
the prompt cache reads nothing for the second.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ml_stack.graph import bench
from ml_stack.graph.bench.backends import client_for, describe, http_of, timings_of
from ml_stack.graph.bench.keep import read_back, save

KIND = "speed"
PROMPTS = (512, 4096, 16384)
STREAMS = (1, 2, 4)
GENERATE = 256
# How far a built prompt may miss the token count asked for, as a fraction, and how many
# times it is scaled and measured again before the miss is recorded as it is.
TOLERANCE = 0.02
TRIES = 4
# The first guess at how many characters a token is, before anything is measured.
CHARS_PER_TOKEN = 4.0

_WORDS = ("ledger", "crate", "copper", "harbour", "signal", "quarter", "meadow", "winter",
          "lantern", "furnace", "orchard", "compass", "granite", "willow", "ferry", "anvil")


def prompt_text(tokens: int, *, seed: int = 0, chars_per_token: float = CHARS_PER_TOKEN,
                chars: int | None = None) -> str:
    """A prompt of about ``tokens`` tokens: numbered lines of plain words, different for
    each ``seed`` so two streams never share a prefix. ``chars`` fixes the length outright."""
    wanted = int(chars if chars is not None else max(1, tokens) * chars_per_token)
    lines = [f"Report {seed}. Read every line and then write a summary of the report."]
    n = 0
    while sum(len(ln) + 1 for ln in lines) < wanted:
        a, b = _WORDS[(seed * 7 + n * 3) % len(_WORDS)], _WORDS[(seed * 11 + n * 5) % len(_WORDS)]
        lines.append(f"Line {n + 1}: the {a} yard counted {(seed * 13 + n * 17) % 97 + 3} "
                     f"units of {b} on day {n + 1}.")
        n += 1
    text = "\n".join(lines)
    return text[:wanted] if len(text) > wanted else text


def count_tokens(client: Any, text: str) -> tuple[int | None, str]:
    """``(tokens, how)``: what the server counts ``text`` as -- ``tokenize`` through its
    ``/tokenize``, else ``reply`` from a one-token request's prompt count; ``(None,
    "unknown")`` when it will say neither."""
    if hasattr(client, "tokenize"):
        try:
            got = client.tokenize(text)
            if got:
                return len(got), "tokenize"
        except Exception:  # noqa: BLE001 - no /tokenize on this program; ask by reply
            pass
    try:
        reply = client.chat([{"role": "user", "content": text}], think=False, n_predict=1)
    except Exception:  # noqa: BLE001 - a server that will not answer counts nothing
        return None, "unknown"
    timings = timings_of(reply)
    if timings.get("prompt_n") is not None:
        return int(timings["prompt_n"]) + int(timings.get("cache_n") or 0), "reply"
    usage = (getattr(reply, "raw", None) or {}).get("usage") or {}
    if usage.get("prompt_tokens"):
        return int(usage["prompt_tokens"]), "reply"
    return None, "unknown"


def calibrated(client: Any, tokens: int, *, seed: int = 0, tries: int = TRIES,
               tolerance: float = TOLERANCE) -> tuple[str, int | None, dict[str, Any]]:
    """A prompt measured to ``tokens``: ``(text, measured, built)``.

    Built at the first guess, counted by the server, scaled by the miss and counted again
    until within ``tolerance`` or ``tries`` are spent. ``built`` says how: the method the
    count came from and every ``(chars, tokens)`` step, so the record says what the prompt
    really was rather than what was asked for. ``measured`` is None when the server will
    not count at all, and the text is then the first guess.
    """
    chars = int(tokens * CHARS_PER_TOKEN)
    steps: list[list[int]] = []
    text, measured, how = "", None, "unknown"
    for _ in range(max(1, tries)):
        text = prompt_text(tokens, seed=seed, chars=chars)
        measured, how = count_tokens(client, text)
        steps.append([len(text), int(measured) if measured is not None else -1])
        if measured is None or not measured:
            break
        if abs(measured - tokens) <= tolerance * tokens:
            break
        chars = max(8, int(chars * tokens / measured))
    return text, measured, {"method": how, "steps": steps, "tolerance": tolerance,
                            "how": "numbered lines at a guessed chars-per-token, counted by "
                                   "the server, scaled by the miss and counted again"}


def _one(client: Any, text: str, *, generate: int) -> dict[str, Any]:
    """One request through the client, with its timings and wall clock."""
    began = time.time()
    error = ""
    reply = None
    try:
        reply = client.chat([{"role": "user", "content": text}], think=False)
    except Exception as exc:  # noqa: BLE001 - a failed request is a result of the cell
        error = f"{type(exc).__name__}: {exc}"[:200]
    wall = time.time() - began
    timings = timings_of(reply) if reply is not None else {}
    usage = ((getattr(reply, "raw", None) or {}).get("usage") or {}) if reply is not None else {}
    out: dict[str, Any] = {"wall_s": round(wall, 3), "error": error,
                           "generate": int(generate),
                           "completion_tokens": (int(usage["completion_tokens"])
                                                 if usage.get("completion_tokens") is not None
                                                 else None)}
    for key in ("prompt_ms", "predicted_ms", "prompt_n", "cache_n", "predicted_n",
                "draft_n", "draft_n_accepted"):
        out[key] = timings.get(key)
    return out


def _rate(count: Any, ms: Any) -> float | None:
    """Tokens per second from a count and a clock in milliseconds, None when either is
    not there."""
    if count is None or ms is None or not float(ms):
        return None
    return float(count) * 1000.0 / float(ms)


def _mean(values: Sequence[float | None]) -> float | None:
    said = [float(v) for v in values if v is not None]
    return sum(said) / len(said) if said else None


def _sum(values: Sequence[float | None]) -> float | None:
    said = [float(v) for v in values if v is not None]
    return sum(said) if said else None


def cell(client: Any, *, tokens: int, streams: int, generate: int, seed: int = 0,
         log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """One cell: ``streams`` prompts of ``tokens`` each, sent at once, each written to
    ``generate`` tokens. Every figure is None where the program did not report it."""
    prompts = []
    measured = []
    built: dict[str, Any] = {}
    for n in range(streams):
        text, got, built = calibrated(client, tokens, seed=seed * 64 + n)
        prompts.append(text)
        measured.append(got)
    began = time.time()
    with ThreadPoolExecutor(max_workers=streams) as pool:
        got = list(pool.map(lambda text: _one(client, text, generate=generate), prompts))
    wall = time.time() - began
    prefill_each = [_rate(r["prompt_n"], r["prompt_ms"]) for r in got]
    decode_each = [_rate(r["predicted_n"], r["predicted_ms"]) for r in got]
    ttft_each = [r["prompt_ms"] / 1000.0 if r["prompt_ms"] is not None else None for r in got]
    out: dict[str, Any] = {
        "prompt_tokens": int(tokens),
        "prompt_measured": _mean([m for m in measured]),
        "built": built,
        "streams": int(streams),
        "generate": int(generate),
        "wall_s": round(wall, 3),
        # what the server read and wrote, per stream and over the cell
        "prompt_read": _sum([r["prompt_n"] for r in got]),
        "cached_tokens": _sum([r["cache_n"] for r in got]),
        "predicted_tokens": _sum([r["predicted_n"] for r in got]),
        "completion_tokens": _sum([r["completion_tokens"] for r in got]),
        "prefill_tps": _rate(_sum([r["prompt_n"] for r in got]),
                             _mean([r["prompt_ms"] for r in got])),
        "prefill_tps_per_stream": _mean(prefill_each),
        "decode_tps": _sum(decode_each),
        "decode_tps_per_stream": _mean(decode_each),
        # the first token, from the server's own prompt clock: nothing here streams, so
        # what is known is how long the prompt took to read, and the record says so
        "ttft_s": _mean(ttft_each),
        "ttft_from": "prompt_ms" if any(t is not None for t in ttft_each) else None,
        "draft_tokens": _sum([r["draft_n"] for r in got]),
        "draft_taken": _sum([r["draft_n_accepted"] for r in got]),
        "errors": sum(1 for r in got if r["error"]),
        "requests": [dict(r) for r in got],
    }
    if out["decode_tps"] is None and out["completion_tokens"] is not None and wall > 0:
        # a program with no clocks of its own: what was written over the wall, marked
        out["decode_tps_wall"] = float(out["completion_tokens"]) / wall
    if log:
        log(f"  {tokens:>6} tok x{streams}: "
            + (f"prefill {out['prefill_tps']:.0f} tok/s, " if out["prefill_tps"] is not None
               else "prefill -, ")
            + (f"decode {out['decode_tps']:.1f} tok/s" if out["decode_tps"] is not None
               else "decode -")
            + (f" ({out['decode_tps_per_stream']:.1f}/stream)" if streams > 1
               and out["decode_tps_per_stream"] is not None else "")
            + (f", ttft {out['ttft_s']:.2f}s" if out["ttft_s"] is not None else "")
            + f", {wall:.1f}s wall"
            + (f", {out['errors']} failed" if out["errors"] else ""))
    return out


def grid(client: Any, *, prompts: Sequence[int], streams: Sequence[int], generate: int,
         log: Callable[[str], None] | None = None, smoke: bool = False,
         sample: int = 0) -> list[dict[str, Any]]:
    """Every (prompt, streams) cell in turn -- the smallest of each alone for ``smoke``,
    the first ``sample`` of them when asked."""
    sizes = [min(prompts)] if smoke else list(prompts)
    widths = [min(streams)] if smoke else list(streams)
    out = []
    for i, tokens in enumerate(sizes):
        for j, width in enumerate(widths):
            if sample and len(out) >= sample:
                return out
            out.append(cell(client, tokens=int(tokens), streams=int(width), generate=generate,
                            seed=i * len(widths) + j + 1, log=log))
    return out


def _ints(text: str, default: Sequence[int]) -> list[int]:
    words = [w.strip() for w in str(text or "").split(",") if w.strip()]
    return [int(w) for w in words] if words else list(default)


def add_arguments(sub: Any) -> argparse.ArgumentParser:
    """The ``speed`` subcommand's own flags; the common measuring flags are added by the
    parser beside every other measuring command's."""
    from ml_stack.graph.vectors import MARGIN  # noqa: F401 - the sweep's defaults are here

    one = sub.add_parser("speed", allow_abbrev=False,
                         help="how fast a served model reads and writes: prefill and decode "
                              "tokens per second and the time to the first token, by "
                              "prompt size and by how many ask at once")
    one.add_argument("model", nargs="?", default="",
                     help="a model to put up and measure, the same as --serve MODEL")
    one.add_argument("--on", action="append", metavar="NAME=URL", default=[],
                     help="a server somebody else started, e.g. flash=http://127.0.0.1:8080 "
                          "or flash-ollama=ollama://127.0.0.1:11434/model; repeatable")
    one.add_argument("--serve", action="append", default=[], metavar="MODEL",
                     help="a model to put up in its measured shape, measure and take down; "
                          "repeatable")
    one.add_argument("--serve-label", default="", metavar="NAME",
                     help="what the --serve'd model's runs are labelled, instead of the "
                          "first 14 characters of its file's name")
    one.add_argument("--serve-draft", action="append", default=[], metavar="PATH_OR_AUTO",
                     help="a draft head for the matching --serve, positionally")
    one.add_argument("--no-draft", action="store_true",
                     help="serve without the head the profile measured best with; the "
                          "label ends -nodraft")
    one.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                     help="serve each model in its measured shape from ml-stack's profiles")
    one.add_argument("--label-suffix", default="", metavar="TEXT",
                     help="appended to every label this measures")
    one.add_argument("--prompts", default=",".join(str(p) for p in PROMPTS), metavar="N,N",
                     help="prompt sizes in tokens (default: %(default)s)")
    one.add_argument("--streams", "--users", default=",".join(str(s) for s in STREAMS),
                     metavar="N,N",
                     help="how many ask at once, per cell (default: %(default)s)")
    one.add_argument("--generate", type=int, default=GENERATE, metavar="N",
                     help="tokens each request writes (default: %(default)s)")
    one.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                     help="where to keep the runs (default: %(default)s)")
    one.add_argument("--context", type=int, default=0, metavar="N",
                     help="total context for a --serve'd model (default: what the largest "
                          "cell needs, per slot)")
    one.add_argument("--parallel", type=int, default=0, metavar="N",
                     help="slots for a --serve'd model (default: the most streams asked)")
    one.add_argument("--serve-port", type=int, default=8099)
    one.add_argument("--binary", default="", metavar="PATH")
    one.add_argument("--serve-kv", default="", metavar="TYPE")
    one.add_argument("--n-max", type=int, default=None, metavar="N")
    one.add_argument("--reasoning-budget", type=int, default=None, metavar="N")
    one.add_argument("--sample", type=int, default=0, metavar="N",
                     help="only the first N cells of the grid, smallest prompt first "
                          "(default: every cell)")
    one.add_argument("--smoke", action="store_true",
                     help="one cell -- the smallest prompt, one stream -- through the whole "
                          "path, kept and read back")
    one.add_argument("--anyway", action="store_true",
                     help="measure even when the server is already busy")
    one.add_argument("--trace", action=argparse.BooleanOptionalAction, default=None,
                     help=argparse.SUPPRESS)
    return one


def _client_settings(args: Any, *, timeout: float) -> dict[str, Any]:
    return {"timeout": float(timeout), "n_predict": int(args.generate), "temperature": 0.0}


def measure_on(args: Any, named: Sequence[tuple[str, str]], *, smoke: bool,
               smoking_first: bool) -> list[str]:
    """Every ``--on`` server: the grid, kept as one run of kind ``speed`` per label."""
    from ml_stack.graph.bench.measure import _idle
    from ml_stack.graph.bench.run import sampling_from

    keys = []
    prompts, streams = _ints(args.prompts, PROMPTS), _ints(args.streams, STREAMS)
    context = int(getattr(args, "context", 0) or 0) or None
    for name, url in named:
        label = f"{name}{getattr(args, 'label_suffix', '') or ''}-speed"
        if not _idle(http_of(url), args):
            return keys
        settings = {**_client_settings(args, timeout=float(args.per_question)),
                    **{k: v for k, v in sampling_from(args).items() if k != "n_predict"}}
        client = client_for(url, context=context, **settings)
        print(f"\n{label} on {url}")
        if smoking_first:
            print("  smoke: one cell first")
            proved = grid(client, prompts=prompts, streams=streams, generate=args.generate,
                          log=print, smoke=True)
            key = save(args.kept, proved, held=_held(url, client), kind=KIND, label=label)
            _proved(read_back(args.kept, [key]), f"{label} smoke")
            keys.append(key)
            print("  smoke: ok")
        cells = grid(client, prompts=prompts, streams=streams, generate=args.generate,
                     log=print, smoke=smoke, sample=int(getattr(args, "sample", 0) or 0))
        keys.append(save(args.kept, cells, held=_held(url, client), kind=KIND, label=label))
    return keys


def _held(url: str, client: Any) -> dict[str, Any]:
    """The run's ``server`` record: what serves on ``url``, and what it holds."""
    from ml_stack.graph.bench.measure import said_by

    held = bench.footprint(url, client)
    held.setdefault("base_url", url)
    if "served_by" not in held:
        record = said_by(client)
        if record:
            held["served_by"] = record
    return held


def _proved(kept: Sequence[Mapping[str, Any]], what: str) -> None:
    from ml_stack.graph.bench.serve import SmokeFailed

    rows = [r for one in kept for r in (one.get("rows") or ())]
    if not rows:
        raise SmokeFailed(f"{what}: no cell was kept")
    if all(int(r.get("errors") or 0) >= int(r.get("streams") or 1) for r in rows):
        raise SmokeFailed(f"{what}: every request failed -- "
                          f"{(rows[0].get('requests') or [{}])[0].get('error') or '?'}")


def measure_served(args: Any, *, smoke: bool, smoking_first: bool) -> list[str]:
    """Every ``--serve`` model: put up in its measured shape (minus the head with
    ``--no-draft``), the grid through `up`, taken down."""
    from ml_stack.graph.bench.run import measured_shape, swept
    from ml_stack.graph.bench.serve import NotLoaded, up
    from ml_stack.serve.backend import ServerFailed
    from ml_stack.serve.preflight import PreflightFailed

    keys: list[str] = []
    prompts, streams = _ints(args.prompts, PROMPTS), _ints(args.streams, STREAMS)
    seats = int(getattr(args, "parallel", 0) or 0) or max(streams)
    # each seat holds the largest prompt and what is written after it, with room to spare
    per_seat = int(getattr(args, "context", 0) or 0) // seats if getattr(args, "context", 0) \
        else max(4096, ((max(prompts) + int(args.generate) + 512 + 1023) // 1024) * 1024)
    for n, wanted in enumerate(list(getattr(args, "serve", []) or [])):
        model = bench.find_model(wanted)
        heads = list(getattr(args, "serve_draft", []) or [])
        head = heads[n] if n < len(heads) else ""
        if head.lower() == "auto":
            from ml_stack import hub

            chosen = hub.choose_head(model, binary=args.binary or None)
            head = chosen.path
            print(f"    draft head: {head or 'none'} -- {chosen.why}")
        stem = (str(getattr(args, "serve_label", "") or "")
                or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14])
        label = (stem + ("-nodraft" if getattr(args, "no_draft", False) else "")
                 + str(getattr(args, "label_suffix", "") or "") + "-speed")
        print(f"\n{label}: {len(prompts)} prompt size(s) x {len(streams)} stream count(s)")
        args.parallel = seats
        run = swept(args, model, measured_shape(args, model, head, heads, n),
                    context=per_seat * seats, port=args.serve_port,
                    head=head if n < len(heads) else None)
        run = run.over(**{k: v for k, v in _client_settings(
            args, timeout=float(args.per_question)).items()})
        try:
            with up(run, binary=args.binary or "", name=label) as (server, held_up):
                held_up.pop("baseline", None)
                held_up.pop("loaded", None)
                client = run.client(server.base_url)
                if smoking_first:
                    print("  smoke: one cell first, on this load")
                    proved = grid(client, prompts=prompts, streams=streams,
                                  generate=args.generate, log=print, smoke=True)
                    key = save(args.kept, proved, held={**_held(server.base_url, client),
                                                        **held_up}, kind=KIND, label=label)
                    _proved(read_back(args.kept, [key]), f"{label} smoke")
                    keys.append(key)
                    print("  smoke: ok")
                cells = grid(client, prompts=prompts, streams=streams, generate=args.generate,
                             log=print, smoke=smoke,
                             sample=int(getattr(args, "sample", 0) or 0))
                keys.append(save(args.kept, cells,
                                 held={**_held(server.base_url, client), **held_up},
                                 kind=KIND, label=label))
        except (NotLoaded, PreflightFailed) as why:
            print(f"    preflight refused {label}; not loaded:\n"
                  + "\n".join(f"      {line}" for line in str(why).splitlines()))
        except ServerFailed as why:
            print(f"    {label} did not load; moving on:\n"
                  + "\n".join(f"      {line}" for line in str(why).splitlines()[:6]))
    return keys


def main(args: Any) -> int:
    """The ``speed`` subcommand after the parse."""
    from ml_stack.graph.bench.backends import parse_on
    from ml_stack.graph.bench.run import wants_smoke

    named = []
    for one in args.on:
        try:
            name, url, _ = parse_on(one)
        except ValueError as why:
            print(f"error: {why}", file=sys.stderr)
            return 2
        named.append((name, url))
    if getattr(args, "model", ""):
        args.serve = [args.model, *list(args.serve or [])]
    if not named and not args.serve:
        print("error: nothing to measure; pass --on NAME=URL for a server that is already "
              "up, or --serve MODEL to put one up", file=sys.stderr)
        return 2
    smoke = bool(getattr(args, "smoke", False))
    first = wants_smoke(args)
    keys = measure_on(args, named, smoke=smoke, smoking_first=first)
    keys += measure_served(args, smoke=smoke, smoking_first=first)
    print()
    kept = read_back(args.kept, keys) if keys else []
    speed_table(kept if keys else only(bench._kept(args.kept)))
    return 0 if keys or not (named or args.serve) else 1


def only(kept: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The speed runs among ``kept``."""
    return [dict(r) for r in kept if r.get("kind") == KIND]


def _f(value: Any, fmt: str = ".0f", unit: str = "") -> str:
    return "-" if value is None else f"{float(value):{fmt}}{unit}"


def speed_table(kept: Sequence[Mapping[str, Any]]) -> None:
    """One table per speed run: a line per cell."""
    runs = only(kept)
    if not runs:
        print("no speed run kept yet: ml-stack-bench speed --on NAME=URL, or --serve MODEL")
        return
    for one in runs:
        server = one.get("server") or {}
        served = describe(server.get("served_by"), build=str(server.get("build") or ""))
        peak = server.get("resident_peak") or server.get("resident_bytes")
        print(f"{one.get('label', '')}  ({one.get('at', '')}"
              + (f", {served}" if served else "")
              + (f", peak {int(peak) / 2**30:.1f}G" if peak else "")
              + (f", load {float(server['load_s']):.0f}s" if server.get("load_s") is not None
                 else "") + ")")
        head = (f"  {'prompt':>7} {'x':>2} {'prefill':>9} {'decode':>9} {'/stream':>8} "
                f"{'ttft':>7} {'wall':>7} {'draft':>6} {'errors':>6}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for c in one.get("rows") or []:
            drafted = c.get("draft_tokens")
            accept = (f"{100 * float(c.get('draft_taken') or 0) / float(drafted):.0f}%"
                      if drafted else ("none" if drafted == 0 else "-"))
            ttft = c.get("ttft_s")
            print(f"  {int(c.get('prompt_tokens') or 0):>7} {int(c.get('streams') or 1):>2} "
                  f"{_f(c.get('prefill_tps')):>9} {_f(c.get('decode_tps'), '.1f'):>9} "
                  f"{_f(c.get('decode_tps_per_stream'), '.1f'):>8} "
                  f"{(_f(ttft, '.2f', 's') + ('*' if c.get('ttft_from') == 'prompt_ms' else '')):>7} "
                  f"{_f(c.get('wall_s'), '.1f', 's'):>7} {accept:>6} "
                  f"{int(c.get('errors') or 0):>6}")
        print("  prefill and decode in tokens/s over the cell; /stream is one stream's "
              "decode; * ttft is the server's prompt clock, not a streamed first token")
        print()
