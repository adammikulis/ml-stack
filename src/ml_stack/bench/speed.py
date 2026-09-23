"""``ml-stack-bench speed``: how fast a served model reads and writes, by prompt size and
by how many ask at once.

One cell per (prompt size, streams): that many requests of that length sent together,
greedy, thinking off, each writing ``--generate`` tokens. A cell records ``prefill_tps``,
``decode_tps``, ``ttft_s`` and the wall, per stream and over all of them. Kept as a run of
kind `KIND`, one row per cell; `speed_table` prints them. `calibrated` builds each prompt
to a token count within `TOLERANCE`, with its own lines so no two streams share a prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ml_stack import bench, hub
from ml_stack.bench.askings import sampling_from
from ml_stack.bench.backends import client_for, describe, http_of, parse_on, timings_of
from ml_stack.bench.holding import _idle, said_by
from ml_stack.bench.keep import read_back, save
from ml_stack.bench.ops import measured_run, swept
from ml_stack.bench.serve import NotLoaded, SmokeFailed, refused, up
from ml_stack.bench.underway import wants_smoke
from ml_stack.client.settings import Request, Transport
from ml_stack.client.tokens import CHARS_PER_TOKEN
from ml_stack.http import ServerError
from ml_stack.log import say, warn
from ml_stack.serve.backend import ServerFailed
from ml_stack.serve.preflight import PreflightFailed

KIND = "speed"
PROMPTS = (512, 4096, 16384)
STREAMS = (1, 2, 4)
GENERATE = 256
# How far a built prompt may miss the token count asked for, as a fraction, and how many
# times it is scaled and measured again before the miss is recorded as it is.
TOLERANCE = 0.02
TRIES = 4

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
        with contextlib.suppress(ServerError, NotImplementedError):  # no /tokenize here
            got = client.tokenize(text)
            if got:
                return len(got), "tokenize"
    try:
        reply = client.chat([{"role": "user", "content": text}], think=False, n_predict=1)
    except (ServerError, NotImplementedError):  # a server that will not answer counts nothing
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
    """One request through the client, with its timings, wall clock and first token.

    Streamed on every api but Ollama's, so ``ttft_s`` is the clock to the first piece
    that arrived (``ttft_from`` "stream"); Ollama's, and a reply that streamed nothing,
    take the server's prompt clock instead ("prompt_ms")."""
    began = time.time()
    started = time.perf_counter()
    first: list[float] = []
    error = ""
    reply = None

    def arrived(_kind: str, _text: str) -> None:
        if not first:
            first.append(time.perf_counter() - started)

    streamed = getattr(client, "api", "") != "ollama"
    try:
        reply = client.chat([{"role": "user", "content": text}], think=False,
                            **({"on_delta": arrived} if streamed else {}))
    except (ServerError, NotImplementedError) as exc:  # a failed request is a result
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
                "draft_n", "draft_n_accepted", "draft_n_verified", "draft_ms", "verify_ms",
                "verify_n"):
        out[key] = timings.get(key)
    if first:
        out["ttft_s"], out["ttft_from"] = round(first[0], 4), "stream"
    elif out["prompt_ms"] is not None:
        out["ttft_s"], out["ttft_from"] = out["prompt_ms"] / 1000.0, "prompt_ms"
    else:
        out["ttft_s"], out["ttft_from"] = None, None
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


def cell(client: Any, *, tokens: int, streams: int, generate: int,
         seed: int = 0) -> dict[str, Any]:
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
    ttft_each = [r["ttft_s"] for r in got]
    clocks = {r["ttft_from"] for r in got if r["ttft_from"]}
    ttft_from = "stream" if "stream" in clocks else ("prompt_ms" if clocks else None)
    out: dict[str, Any] = {
        "prompt_tokens": int(tokens),
        "prompt_measured": _mean(list(measured)),
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
        # the first token: the clock to the first streamed piece, or the server's own
        # prompt clock where nothing streamed, and the record says which
        "ttft_s": _mean(ttft_each),
        "ttft_from": ttft_from,
        "draft_tokens": _sum([r["draft_n"] for r in got]),
        "draft_taken": _sum([r["draft_n_accepted"] for r in got]),
        "draft_ms": _sum([r["draft_ms"] for r in got]),
        "verify_ms": _sum([r["verify_ms"] for r in got]),
        "verify_n": _sum([r["verify_n"] for r in got]),
        "errors": sum(1 for r in got if r["error"]),
        "requests": [dict(r) for r in got],
    }
    if out["decode_tps"] is None and out["completion_tokens"] is not None and wall > 0:
        # a program with no clocks of its own: what was written over the wall, marked
        out["decode_tps_wall"] = float(out["completion_tokens"]) / wall
    return out


def said(one: Mapping[str, Any]) -> str:
    """One cell as the line a sweep prints for it."""
    streams = int(one["streams"])
    return (f"  {one['prompt_tokens']:>6} tok x{streams}: "
            + (f"prefill {one['prefill_tps']:.0f} tok/s, " if one["prefill_tps"] is not None
               else "prefill -, ")
            + (f"decode {one['decode_tps']:.1f} tok/s" if one["decode_tps"] is not None
               else "decode -")
            + (f" ({one['decode_tps_per_stream']:.1f}/stream)" if streams > 1
               and one["decode_tps_per_stream"] is not None else "")
            + (f", ttft {one['ttft_s']:.2f}s" if one["ttft_s"] is not None else "")
            + f", {one['wall_s']:.1f}s wall"
            + (f", {one['errors']} failed" if one["errors"] else ""))


def pairs(prompts: Sequence[int], streams: Sequence[int], *, smoke: bool = False,
          sample: int = 0) -> list[tuple[int, int]]:
    """Every (prompt, streams) cell in order -- the smallest of each alone for ``smoke``,
    the first ``sample`` of them when asked."""
    sizes = [min(prompts)] if smoke else list(prompts)
    widths = [min(streams)] if smoke else list(streams)
    every = [(int(tokens), int(width)) for tokens in sizes for width in widths]
    return every[:sample] if sample else every


def grid(client: Any, cells: Sequence[tuple[int, int]], *, generate: int,
         log: Callable[[str], None] | None = None) -> list[dict[str, Any]]:
    """Each of ``cells`` measured in turn, each said to ``log``."""
    out = []
    for n, (tokens, width) in enumerate(cells):
        out.append(cell(client, tokens=tokens, streams=width, generate=generate, seed=n + 1))
        if log:
            log(said(out[-1]))
    return out


def _ints(text: str, default: Sequence[int]) -> list[int]:
    words = [w.strip() for w in str(text or "").split(",") if w.strip()]
    return [int(w) for w in words] if words else list(default)


def add_arguments(sub: Any) -> argparse.ArgumentParser:
    """The ``speed`` subcommand's own flags; the common measuring flags are added by the
    parser beside every other measuring command's."""
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
                     help="a model to put up in the settings it scored best with, measure and take down; "
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
                     help="serve each model in the settings it scored best with from ml-stack's profiles")
    one.add_argument("--label-suffix", default="", metavar="TEXT",
                     help="appended to every label this measures")
    one.add_argument("--prompts", default=",".join(str(p) for p in PROMPTS), metavar="N,N",
                     help="prompt sizes in tokens (default: %(default)s)")
    one.add_argument("--streams", "--users", default=",".join(str(s) for s in STREAMS),
                     metavar="N,N",
                     help="how many ask at once, per cell (default: %(default)s)")
    one.add_argument("--generate", type=int, default=GENERATE, metavar="N",
                     help="tokens each request writes (default: %(default)s)")
    one.add_argument("--kept", default=str(bench.home_dir() / "runs.ladybug"),
                     help="where to keep the runs (default: %(default)s)")
    one.add_argument("--context", type=int, default=0, metavar="N",
                     help="total context for a --serve'd model (default: what the largest "
                          "cell needs, per slot)")
    one.add_argument("--parallel", type=int, default=0, metavar="N",
                     help="slots for a --serve'd model (default: the most streams asked)")
    one.add_argument("--serve-port", type=int, default=8099)
    one.add_argument("--binary", default="", metavar="PATH")
    one.add_argument("--serve-kv", default="", metavar="TYPE")
    one.add_argument("--serve-kv-unified", action=argparse.BooleanOptionalAction, default=None,
                     help="serve with one cache pool for every slot (or, --no-serve-kv-unified, "
                          "a cache per slot); unset leaves the build's default")
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




def measure_on(args: Any, named: Sequence[tuple[str, str]], *, smoke: bool,
               smoking_first: bool) -> list[str]:
    """Every ``--on`` server: the grid, kept as one run of kind ``speed`` per label."""
    keys = []
    prompts, streams = _ints(args.prompts, PROMPTS), _ints(args.streams, STREAMS)
    context = int(getattr(args, "context", 0) or 0) or None
    for name, url in named:
        label = f"{name}{getattr(args, 'label_suffix', '') or ''}-speed"
        if not _idle(http_of(url), args):
            return keys
        sampling = {"temperature": 0.0, **sampling_from(args), "n_predict": int(args.generate)}
        client = client_for(url, request=Request(context=context, **sampling),
                            transport=Transport(timeout=float(args.per_question)))
        say(f"\n{label} on {url}")
        if smoking_first:
            say("  smoke: one cell first")
            proved = grid(client, pairs(prompts, streams, smoke=True),
                          generate=args.generate, log=print)
            key = save(args.kept, proved, server=_server_record(url, client), kind=KIND, label=label)
            _proved(read_back(args.kept, [key]), f"{label} smoke")
            keys.append(key)
            say("  smoke: ok")
        cells = grid(client, pairs(prompts, streams, smoke=smoke,
                                   sample=int(getattr(args, "sample", 0) or 0)),
                     generate=args.generate, log=print)
        keys.append(save(args.kept, cells, server=_server_record(url, client), kind=KIND, label=label))
    return keys


def _server_record(url: str, client: Any) -> dict[str, Any]:
    """The run's ``server`` record: what serves on ``url``, and what it holds."""
    server = bench.footprint(url, client)
    server.setdefault("base_url", url)
    if "served_by" not in server:
        record = said_by(client)
        if record:
            server["served_by"] = record
    return server


def _proved(kept: Sequence[Mapping[str, Any]], what: str) -> None:
    rows = [r for one in kept for r in (one.get("rows") or ())]
    if not rows:
        raise SmokeFailed(f"{what}: no cell was kept")
    if all(int(r.get("errors") or 0) >= int(r.get("streams") or 1) for r in rows):
        raise SmokeFailed(f"{what}: every request failed -- "
                          f"{(rows[0].get('requests') or [{}])[0].get('error') or '?'}")


def measure_served(args: Any, *, smoke: bool, smoking_first: bool) -> list[str]:
    """Every ``--serve`` model: put up in the settings it scored best with (minus the head with
    ``--no-draft``), the grid through `up`, taken down."""
    keys: list[str] = []
    prompts, streams = _ints(args.prompts, PROMPTS), _ints(args.streams, STREAMS)
    slots = int(getattr(args, "parallel", 0) or 0) or max(streams)
    # each slot holds the largest prompt and what is written after it, with room to spare
    per_slot = int(getattr(args, "context", 0) or 0) // slots if getattr(args, "context", 0) \
        else max(4096, ((max(prompts) + int(args.generate) + 512 + 1023) // 1024) * 1024)
    for n, wanted in enumerate(list(getattr(args, "serve", []) or [])):
        model = str(hub.located(wanted, loose=True) or wanted)
        heads = list(getattr(args, "serve_draft", []) or [])
        head = heads[n] if n < len(heads) else ""
        if head.lower() == "auto":
            chosen = hub.choose_head(model, binary=args.binary or None)
            head = chosen.path
            say(f"    draft head: {head or 'none'} -- {chosen.why}")
        stem = (str(getattr(args, "serve_label", "") or "")
                or str(model).rsplit("/", 1)[-1].removesuffix(".gguf")[:14])
        label = (stem + ("-nodraft" if getattr(args, "no_draft", False) else "")
                 + str(getattr(args, "label_suffix", "") or "") + "-speed")
        say(f"\n{label}: {len(prompts)} prompt size(s) x {len(streams)} stream count(s)")
        args.parallel = slots
        run = swept(args, model, measured_run(args, model, head, heads, n),
                    context=per_slot * slots, head=head if n < len(heads) else None)
        run = run.over(port=int(args.serve_port), timeout=float(args.per_question),
                       n_predict=int(args.generate), temperature=0.0)
        try:
            with up(run, binary=args.binary or "") as (server, held_up):
                held_up.pop("baseline", None)
                held_up.pop("loaded", None)
                client = run.client(server.base_url)
                if smoking_first:
                    say("  smoke: one cell first, on this load")
                    proved = grid(client, pairs(prompts, streams, smoke=True),
                                  generate=args.generate, log=print)
                    key = save(args.kept, proved, server={**_server_record(server.base_url, client),
                                                        **held_up}, kind=KIND, label=label)
                    _proved(read_back(args.kept, [key]), f"{label} smoke")
                    keys.append(key)
                    say("  smoke: ok")
                cells = grid(client, pairs(prompts, streams, smoke=smoke,
                                           sample=int(getattr(args, "sample", 0) or 0)),
                             generate=args.generate, log=print)
                keys.append(save(args.kept, cells,
                                 server={**_server_record(server.base_url, client),
                                         **held_up},
                                 kind=KIND, label=label))
        except (NotLoaded, PreflightFailed) as why:
            say(refused(label, why))
        except ServerFailed as why:
            say(f"    {label} did not load; moving on:\n"
                + "\n".join(f"      {line}" for line in str(why).splitlines()[:6]))
    return keys


def run(args: Any) -> int:
    """The ``speed`` subcommand after the parse."""
    named = []
    for one in args.on:
        try:
            name, url, _ = parse_on(one)
        except ValueError as why:
            warn(f"error: {why}")
            return 2
        named.append((name, url))
    if getattr(args, "model", ""):
        args.serve = [args.model, *list(args.serve or [])]
    if not named and not args.serve:
        warn("error: nothing to measure; pass --on NAME=URL for a server that is already "
             "up, or --serve MODEL to put one up")
        return 2
    smoke = bool(getattr(args, "smoke", False))
    first = wants_smoke(args)
    keys = measure_on(args, named, smoke=smoke, smoking_first=first)
    keys += measure_served(args, smoke=smoke, smoking_first=first)
    say()
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
        say("no speed run kept yet: ml-stack-bench speed --on NAME=URL, or --serve MODEL")
        return
    for one in runs:
        server = one.get("server") or {}
        served = describe(server.get("served_by"), build=str(server.get("build") or ""))
        peak = server.get("resident_peak") or server.get("resident_bytes")
        say(f"{one.get('label', '')}  ({one.get('at', '')}"
            + (f", {served}" if served else "")
            + (f", peak {int(peak) / 2**30:.1f}G" if peak else "")
            + (f", load {float(server['load_s']):.0f}s" if server.get("load_s") is not None
                 else "") + ")")
        head = (f"  {'prompt':>7} {'x':>2} {'prefill':>9} {'decode':>9} {'/stream':>8} "
                f"{'ttft':>7} {'wall':>7} {'draft':>6} {'errors':>6}")
        say(head)
        say("  " + "-" * (len(head) - 2))
        for c in one.get("rows") or []:
            drafted = c.get("draft_tokens")
            accept = (f"{100 * float(c.get('draft_taken') or 0) / float(drafted):.0f}%"
                      if drafted else ("none" if drafted == 0 else "-"))
            ttft = c.get("ttft_s")
            say(f"  {int(c.get('prompt_tokens') or 0):>7} {int(c.get('streams') or 1):>2} "
                f"{_f(c.get('prefill_tps')):>9} {_f(c.get('decode_tps'), '.1f'):>9} "
                f"{_f(c.get('decode_tps_per_stream'), '.1f'):>8} "
                f"{(_f(ttft, '.2f', 's') + ('*' if c.get('ttft_from') == 'prompt_ms' else '')):>7} "
                f"{_f(c.get('wall_s'), '.1f', 's'):>7} {accept:>6} "
                f"{int(c.get('errors') or 0):>6}")
        say("  prefill and decode in tokens/s over the cell; /stream is one stream's "
            "decode; ttft is the clock to the first streamed token, * the server's "
            "prompt clock instead")
        say()
