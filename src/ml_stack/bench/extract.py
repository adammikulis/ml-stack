"""Whether a model reads a message right, scored against the truth that wrote it.

A world's messages are sampled, a model extracts each into the generic shape
`contracts/extraction.schema.json` holds, the extractions are folded into one graph by
name and that graph is scored against what the sampled messages assert. `extract_one` is
one message through the model, `measure` the whole sample folded and scored, `save` and
`read_back` how a run is kept beside the answering ones, and `table` how it prints:
coverage and precision as separate columns per kind, never only F1.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ml_stack import hub
from ml_stack.bench import (
    PER_QUESTION,
    Counting,
    RunNotKept,
    _idle,
    _shown,
    _which,
    footprint,
    home_dir,
    runs,
    sampling_from,
    smoke_first,
    smoked,
    stamped,
    wants_smoke,
)
from ml_stack.bench.folding import consistency, fold, score
from ml_stack.bench.truth import BUCKETS, gold, load_world, sample_messages, schema
from ml_stack.extraction import Checking, Prompting
from ml_stack.log import say, warn

__all__ = [
    "GUESS_SECONDS",
    "INSTRUCTIONS",
    "KIND",
    "SAMPLE",
    "SMOKE_MESSAGES",
    "MessageRow",
    "add_arguments",
    "detail",
    "estimate",
    "extract_one",
    "measure",
    "only",
    "read_back",
    "run",
    "save",
    "table",
]

# The record's `kind`, which is what tells an extraction run from an answering one in the
# one store both are kept in.
KIND = "extract"
# The workload these runs measure, which is the profile slot `report --profile` writes.
WORKLOAD = "ingest"
SAMPLE = 40
# Three, not `bench.SMOKE`'s two: two messages can both land in one stratum, and a smoke
# run exists to prove the path -- the stratified sample included.
SMOKE_MESSAGES = 3
# What a message is guessed to cost before any run of that model has said otherwise; the
# estimate is printed before the clock starts so the wall clock is known up front.
GUESS_SECONDS = 15.0
INSTRUCTIONS = (
    "Read this message from an organised group and list the people, organisations, topics, "
    "places and relations it states; invent nothing. The sender is named before the "
    "message: include them among the people. A topic is a subject area the message names "
    "-- a field, a technology, a craft, an activity -- written as the message writes it, "
    "at most three; the message's purpose, a project, an event, a request or a feeling is "
    "not a topic, and a message about nothing in particular has none. An organisation is "
    "a named company, team, institution or group, not a role or a department in passing. "
    "A field the message does not give is an empty string. A relation joins two names "
    "from this message with a short lower-case verb phrase, underscores for spaces, from "
    "these where one fits: works_at, works_with, works_on, reports_to, part_of, based_in, "
    "advises, attended, experienced_in; only otherwise a phrase of your own. "
    "Return only JSON matching the schema."
)


@dataclass
class MessageRow:
    """One message, extracted once, and everything it cost."""

    id: str
    sender: str
    channel: str = ""
    arc: str = ""
    kind: str = ""
    seconds: float = 0.0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    processed_tokens: int = 0
    completion_tokens: int = 0
    timed_out: bool = False
    error: str = ""
    # whether the message's gold is exact (template-written) or a lower bound (model-written)
    exact: bool = True
    extracted: dict[str, Any] = field(default_factory=dict)


class _Extracting(Counting):
    """`Counting`, with `Client.extract` run through its own counted, deadlined `chat`.

    `Counting` delegates what it does not define to the client, and the client's `extract`
    calls the client's own `chat` -- uncounted, and past no deadline. Binding the extractor
    here puts every call it makes through the counting one.
    """

    def extract(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        from ml_stack.client.chat import Client

        return Client.extract(self, *args, **kwargs)  # type: ignore[arg-type]

    def _chat_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._chat_extractor(self, *args, **kwargs)  # type: ignore[arg-type]

    def _raw_extractor(self, *args: Any, **kwargs: Any) -> Any:
        from ml_stack.client.chat import Client

        return Client._raw_extractor(self, *args, **kwargs)  # type: ignore[arg-type]


def prompt_for(message: Mapping[str, Any], sender: str) -> list[dict[str, str]]:
    """The two turns a message is extracted from: the instructions, and the message with
    its sender named ahead of it."""
    where = str(message.get("channel") or "")
    head = f"From {sender}" + (f" in {where}" if where else "") + ":\n"
    return [{"role": "system", "content": INSTRUCTIONS},
            {"role": "user", "content": head + str(message.get("text") or "")}]


def extract_one(client: Any, message: Mapping[str, Any], sender: str, shape: Mapping[str, Any],
                *, per_message: float = PER_QUESTION) -> MessageRow:
    """One message through ``client.extract``, and what it cost.

    Past ``per_message`` seconds the row is kept as timed out -- nothing extracted, the cap
    as its wall clock -- and the next message is read. ``think=False``: an extraction is a
    reading, not a reasoning, and the thinking channel is where a ceiling is spent.
    """
    attrs = message.get("attrs") or {}
    row = MessageRow(id=str(message.get("id") or ""), sender=sender,
                     channel=str(message.get("channel") or ""),
                     arc=str(attrs.get("arc") or ""), kind=str(attrs.get("kind") or ""),
                     exact=bool(attrs.get("asserts_exact", True)))
    began = time.time()
    counting = _Extracting(client, deadline=began + per_message if per_message else None)
    try:
        got = counting.extract(str(message.get("text") or ""), dict(shape),
                               prompting=Prompting(messages=prompt_for(message, sender)),
                               checking=Checking(tries=1))
        row.extracted = got if isinstance(got, dict) else {}
    except Exception as exc:  # noqa: BLE001 - a failure is a result, not the end of the run
        row.error = f"{type(exc).__name__}: {exc}"[:200]
    row.seconds = round(time.time() - began, 2)
    if counting.timed_out:
        row.timed_out = True
        row.error = f"timed out after {per_message:.0f}s"
        row.seconds = float(per_message)
        row.extracted = {}
    row.prompt_tokens = counting.prompt_tokens
    row.cached_tokens = int(counting.cached_tokens or 0)
    row.processed_tokens = counting.processed_tokens
    row.completion_tokens = counting.completion_tokens
    return row


def measure(client: Any, messages: Sequence[Mapping[str, Any]], graph: Mapping[str, Any], *,
            per_message: float = PER_QUESTION,
            log: Callable[[str], None] | None = None) -> tuple[list[MessageRow], dict[str, Any]]:
    """Extract every message, fold, and score against the gold those messages assert.

    Returns ``(rows, scores)`` where ``scores`` is `score` over the messages whose gold is
    exact, with the ``folded`` graph it scored, plus ``"lower_bound"``: the same over the
    model-written ones, when there are any, whose coverage is against a lower bound. The
    gold is read before anything is extracted, so a corpus without one costs no model time.
    """
    exact = [m for m in messages if (m.get("attrs") or {}).get("asserts_exact", True)]
    loose = [m for m in messages if m not in exact]
    golds = {"exact": gold(graph, exact), "lower_bound": gold(graph, loose) if loose else None}
    labels = {str(n.get("id")): str(n.get("label") or n.get("id"))
              for n in (graph.get("nodes") or ())}
    shape = schema()
    rows: list[MessageRow] = []
    for m in messages:
        sender = labels.get(str(m.get("sender") or ""), str(m.get("sender") or ""))
        row = extract_one(client, m, sender, shape, per_message=per_message)
        rows.append(row)
        if log:
            got = row.extracted
            counted = (f"{len(got.get('people') or ())}p {len(got.get('orgs') or ())}o "
                       f"{len(got.get('topics') or ())}t {len(got.get('places') or ())}pl "
                       f"{len(got.get('relations') or ())}r") if got else "-"
            log(f"  {row.seconds:5.1f}s {counted:>18}  {str(m.get('text') or '')[:56]}"
                + ("  TIMED OUT" if row.timed_out else f"  {row.error}" if row.error else ""))
    got = {r.id: r.extracted for r in rows}
    asserted = lambda ms: [(m.get("attrs") or {}).get("asserts") or {} for m in ms]  # noqa: E731
    folded = fold([got[str(m.get("id") or "")] for m in exact])
    scores = score(folded, golds["exact"], per_message=asserted(exact))
    scores["folded"] = folded
    if golds["lower_bound"] is not None:
        scores["lower_bound"] = score(fold([got[str(m.get("id") or "")] for m in loose]),
                                      golds["lower_bound"], per_message=asserted(loose))
    return rows, scores


# -- keeping and showing --------------------------------------------------------------------------

def save(store: str | Path, rows: Sequence[MessageRow], *, label: str, model: str,
         world: Mapping[str, Any], scores: Mapping[str, Any], sample: Mapping[str, Any],
         server: Mapping[str, Any] | None = None) -> str:
    """Keep an extraction run beside the answering ones, and read it back before returning.

    The same discipline as `bench.save`, for the same reason: the store once took twelve
    runs and gave back nothing, and the read-back is the only proof a run exists.
    """
    from ml_stack.bench import _plain
    from ml_stack.graph.store import GraphStore

    stem = f"bench:{label}:{time.strftime('%Y%m%dT%H%M%S')}"
    record = _plain({"at": time.strftime("%FT%T"), "label": label, "kind": KIND,
                     "workload": WORKLOAD, "model": model,
                     "world": dict(world), "sample": dict(sample), "server": stamped(server),
                     "scores": dict(scores), "rows": [asdict(r) for r in rows]})
    record = json.loads(json.dumps(record))
    with GraphStore(store) as writer:
        key, n = stem, 1
        while writer.get_doc(key) is not None:
            key, n = f"{stem}-{n}", n + 1
        writer.put_doc(key, record)
    back = next((r for r in runs(store) if r.get("key") == key), None)
    if back is None:
        raise RunNotKept(f"{key} was written to {store} and did not come back")
    back = {k: v for k, v in back.items() if k != "key"}
    if back != record:
        differs = sorted(k for k in set(back) | set(record) if back.get(k) != record.get(k))
        raise RunNotKept(f"{key} came back from {store} changed: {', '.join(differs)} differ")
    return key


def only(kept: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The extraction runs among ``kept``."""
    return [dict(r) for r in kept if r.get("kind") == KIND]


def read_back(store: str | Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    """The extraction runs under ``keys``, read the way `show` reads them."""
    kept = {r["key"]: r for r in only(runs(store))}
    lost = [k for k in keys if k not in kept]
    if lost:
        raise RunNotKept(f"{len(lost)} run(s) saved to {store} did not come back: "
                         + ", ".join(lost))
    return [kept[k] for k in keys]


def estimate(store: str | Path, model: str, n: int) -> tuple[float, str]:
    """``(seconds per message, where that came from)``: the mean over earlier runs of the
    same model that were not timeouts, else `GUESS_SECONDS`."""
    if Path(store).expanduser().exists():
        seen = [float(r.get("seconds") or 0) for one in only(runs(store))
                if one.get("model") == model
                for r in (one.get("rows") or ()) if not r.get("timed_out")]
        if seen:
            return sum(seen) / len(seen), f"{len(seen)} earlier messages of {model}"
    return GUESS_SECONDS, "a guess, no earlier run of this model"


def _pct(v: Any) -> str:
    return f"{100 * float(v):.0f}%" if v is not None else "-"


def _line(label: str, model: str, rows: Sequence[Mapping[str, Any]], scores: Mapping[str, Any],
          rss: Any) -> str:
    n = len(rows)
    by = scores.get("by_kind") or {}
    nodes, rels = scores.get("nodes") or {}, scores.get("relations") or {}
    made = scores.get("invented") or {}
    attrs = scores.get("attrs") or {}
    seconds = sum(float(r.get("seconds") or 0) for r in rows)
    tokens = sum(int(r.get("prompt_tokens") or 0) + int(r.get("completion_tokens") or 0)
                 for r in rows)
    timed = sum(1 for r in rows if r.get("timed_out"))

    def pair(bucket: str) -> str:
        row = by.get(bucket) or {}
        return f"{_pct(row.get('coverage')):>7} {_pct(row.get('precision')):>8}"

    def ratio(key: str) -> str:
        row = attrs.get(key) or {}
        return f"{row.get('right', 0)}/{row.get('stated', 0)}" if row else "-"

    invented = (f"{made.get('count', 0)} ({_pct(made.get('rate'))})"
                if made.get("of") else f"{made.get('count', 0)}")
    return (f"{_shown(label):28} {_shown(model, 22):22} {n:>4} "
            f"{(seconds / n if n else 0):>6.1f} {(tokens / n if n else 0):>7.0f} "
            f"{(str(timed) if timed else ''):>4} "
            f"{pair('people')} {pair('orgs')} {pair('topics')} {pair('places')} "
            f"{_pct(rels.get('coverage')):>7} {_pct(rels.get('precision')):>8} "
            f"{_pct(nodes.get('f1')):>5} {_pct(rels.get('f1')):>5} {invented:>10} "
            f"{ratio('org'):>5} {ratio('place'):>5} "
            f"{(f'{rss / 2**30:.2f}G' if rss else '-'):>9}")


def table(kept: Sequence[Mapping[str, Any]]) -> None:
    """Every extraction run, one per line, and its topology under it.

    Coverage and precision are separate columns per kind on purpose -- an F1 alone cannot
    say whether a model missed things or made them up, and those are fixed by opposite
    changes to the asking. `invented` is the count and rate of extracted people and
    organisations matching nothing in the gold. A run with model-written messages gets a
    second line, ``~ lower bound``, scored against the gold those messages are known to
    hold at least; its coverage reads high for that reason and its precision does not.
    """
    if not kept:
        say("no extraction runs kept yet")
        return
    head = (f"{'run':28} {'model':22} {'msgs':>4} {'s/msg':>6} {'tok/msg':>7} {'t/o':>4} "
            f"{'ppl-cov':>7} {'ppl-prec':>8} {'org-cov':>7} {'org-prec':>8} "
            f"{'top-cov':>7} {'top-prec':>8} {'plc-cov':>7} {'plc-prec':>8} "
            f"{'rel-cov':>7} {'rel-prec':>8} {'n-F1':>5} {'r-F1':>5} {'invented':>10} "
            f"{'org':>5} {'place':>5} {'resident':>9}")
    say(head)
    say("-" * len(head))
    for one in kept:
        rows = one.get("rows") or []
        scores = one.get("scores") or {}
        rss = (one.get("server") or {}).get("resident_bytes")
        exact = [r for r in rows if r.get("exact", True)]
        loose = [r for r in rows if not r.get("exact", True)]
        say(_line(str(one.get("label", "")), str(one.get("model", "")), exact, scores, rss))
        if scores.get("lower_bound") is not None:
            say(_line("  ~ lower bound", "", loose, scores["lower_bound"], None))
        for line in detail(scores):
            say(f"  {line}")


def detail(scores: Mapping[str, Any]) -> list[str]:
    """The lines under a run's row: topology, conformance, survival, resolution, and the
    consistency of a second reading when there was one."""
    out: list[str] = []
    shape = scores.get("topology") or {}
    if shape:
        got, want = shape.get("extracted") or {}, shape.get("gold") or {}
        out.append(f"topology: extracted {got.get('nodes', 0)} nodes, {got.get('edges', 0)} "
                   f"edges, {got.get('components', 0)} components, largest "
                   f"{_pct(got.get('largest_share', 0))} -- gold {want.get('nodes', 0)} nodes, "
                   f"{want.get('edges', 0)} edges, {want.get('components', 0)} components, "
                   f"largest {_pct(want.get('largest_share', 0))}")
    conf = scores.get("conformance") or {}
    if conf:
        r, e = conf.get("relations") or {}, conf.get("entities") or {}
        out.append(f"conformance: {r.get('in_vocabulary', 0)}/{r.get('of', 0)} relations in "
                   f"the world's vocabulary, {e.get('in_schema', 0)}/{e.get('of', 0)} entries "
                   f"in the schema; off-schema {conf.get('off_schema', 0)}")
    lived = scores.get("survival") or {}
    if lived.get("messages"):
        out.append(f"survival: {_pct(lived.get('mean'))} of each message's assertions in the "
                   f"folded graph, over {lived['messages']} messages")
    res = scores.get("resolution") or {}
    if res.get("splits") is not None:
        out.append(f"resolution: splits {res['splits']:.2f}, merges {res['merges']:.2f} "
                   f"(1.00 is perfect)")
    alike = scores.get("consistency") or {}
    if alike:
        j = lambda v: f"{v:.2f}" if v is not None else "-"  # noqa: E731
        out.append(f"consistency: nodes J={j(alike.get('nodes'))}, relations "
                   f"J={j(alike.get('relations'))} (second reading with {alike.get('with', '?')})")
    return out


# -- the subcommand -------------------------------------------------------------------------------

def add_arguments(sub: Any) -> Any:
    """Register ``extract`` on the bench's subparsers."""
    ap = sub.add_parser("extract", allow_abbrev=False,
                        help="read a world's messages with a model and score the extraction "
                             "against the truth that wrote them")
    ap.add_argument("label", help="what this run is, e.g. flash-next-extract")
    ap.add_argument("--world", required=True, metavar="DIR",
                    help="what ml-stack-world make --out wrote, or what simulate wrote "
                         "(with messages.jsonl). Without messages, a few working days are "
                         "simulated with the template writer into a temporary directory")
    ap.add_argument("--serve", action="append", default=[], metavar="MODEL",
                    help="a model to put up, read with, and take down: a name, a path or "
                         "an hf: reference. One at a time")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080",
                    help="the model reading, when nothing is served (default: %(default)s)")
    ap.add_argument("--sample", type=int, default=SAMPLE, metavar="N",
                    help="how many messages to read, arcs and chatter both kept "
                         "(default: %(default)s)")
    ap.add_argument("--seed", type=int, default=0, help="which N (default: %(default)s)")
    ap.add_argument("--context", type=int, default=65536, metavar="N",
                    help="total context for a --serve'd model (default: %(default)s)")
    ap.add_argument("--parallel", type=int, default=1, metavar="N",
                    help="slots for a --serve'd model (default: %(default)s -- extraction "
                         "reads one message at a time and never splits the GPU; a second "
                         "slot only halves the context each has)")
    ap.add_argument("--serve-port", type=int, default=8099)
    ap.add_argument("--n-max", type=int, default=None, metavar="N",
                    help="tokens the draft head guesses ahead each step, over the profile's "
                         "measured length. Extraction repeats what it has just read -- "
                         "names, ids, the schema's own keys -- so a longer draft may pay "
                         "here where it lost on answering; this is how that is measured")
    ap.add_argument("--per-message", type=float, default=PER_QUESTION, metavar="SECONDS",
                    help="the most one message may take before it is recorded as timed out "
                         "-- nothing extracted, the cap as its wall clock -- and the next is "
                         "read (default: %(default)s)")
    ap.add_argument("--kept", default=str(home_dir() / "runs.ladybug"),
                    help="where to keep the run (default: %(default)s)")
    ap.add_argument("--smoke", action="store_true",
                    help=f"read only {SMOKE_MESSAGES} messages, to prove the whole path -- "
                         f"serve, read, fold, score, save and read the run back -- before "
                         f"spending the GPU on it")
    ap.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True,
                    help="serve the model in the settings it scored best with from ml-stack's profiles "
                         "(build, head, cache type, thinking budget, raw flags); "
                         "--no-profile serves it bare")
    ap.add_argument("--twice", action="store_true",
                    help="read the sample a second time with the model's own card (or the "
                         "same settings again when the card names none) and report the "
                         "Jaccard of the two graphs: a model that gives a different graph "
                         "each run is a finding")
    ap.add_argument("--anyway", action="store_true",
                    help="measure even when the server is already busy")
    ap.add_argument("--temperature", type=float, default=None,
                    help="override the sampling temperature; the default is the model's card")
    ap.add_argument("--top-p", type=float, default=None, help="override top_p")
    ap.add_argument("--top-k", type=int, default=None, help="override top_k")
    ap.add_argument("--min-p", type=float, default=None, help="override min_p")
    ap.add_argument("--no-queue", action="store_true",
                    help="fail at once if another measurement holds the GPU, rather than "
                         "queue behind it")
    ap.add_argument("--detach", action="store_true",
                    help="run this in the background, owned by nobody's terminal, with its "
                         f"output in a log under {home_dir() / 'logs'}; status, tail -f and stop "
                         "as for run")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="do not download an hf: model before the measuring lock is taken")
    return ap


def twice(client: Any, picked: Sequence[Mapping[str, Any]], graph: Mapping[str, Any],
          scores: Mapping[str, Any], *, per_message: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """The sample read again -- with the card's sampling when the model names any, else at
    the same settings -- and how alike the two folds are. Returns ``(consistency, what the
    second reading was)``; the second reading's rows are kept under ``server.twice``."""
    card = dict(getattr(client, "card", None) or {})
    if card:
        again = type(client)(client.base_url, timeout=per_message, **card)
        how = "the card: " + " ".join(f"{k[:4]}{v}" for k, v in sorted(card.items()))
    else:
        again = client
        how = "the same settings again"
    say(f"\n  reading the sample again with {how}")
    rows, second = measure(again, picked, graph, per_message=per_message, log=print)
    alike = {**consistency(scores.get("folded") or {}, second.get("folded") or {}), "with": how}
    return alike, {"with": how, "sampling": dict(getattr(again, "sampling", {}) or {}),
                   "rows": [asdict(r) for r in rows], "scores": {k: v for k, v in second.items()
                                                                 if k != "folded"}}


def run(args: Any) -> int:
    """``ml-stack-bench extract``: sample, read, fold, score, keep, and print the table."""
    if len(args.serve) > 1:
        warn("error: --serve takes one model; a comparison is one run per model")
        return 2
    graph, messages, note = load_world(args.world)
    if note:
        say(note)
    if not messages:
        warn(f"error: no messages in {args.world}")
        return 2
    n = SMOKE_MESSAGES if args.smoke else args.sample
    picked = sample_messages(messages, n, seed=args.seed)
    arcs = sum(1 for m in picked if (m.get("attrs") or {}).get("arc"))
    loose = sum(1 for m in picked if not (m.get("attrs") or {}).get("asserts_exact", True))
    try:
        truth = gold(graph, picked)
    except ValueError as why:
        warn(f"error: {why}")
        return 2
    meta = (graph.get("meta") or {}).get("world") or {}
    world = {"kind": meta.get("kind", ""), "size": meta.get("size", ""),
             "seed": meta.get("seed", 0), "organisation": meta.get("organisation", ""),
             "digest": _which(graph), "messages": len(messages), "where": str(args.world)}
    sample = {"n": len(picked), "seed": args.seed, "arcs": arcs, "chatter": len(picked) - arcs,
              "model_written": loose,
              "gold": {b: len(truth["nodes"][b]) for b in BUCKETS} | {
                  "others": len(truth["others"]), "relations": len(truth["relations"])}}

    model = (str(hub.located(args.serve[0], loose=True) or args.serve[0])
             .rsplit("/", 1)[-1].removesuffix(".gguf")
             if args.serve else "")
    if not args.serve:
        model = str(footprint(args.base_url).get("model") or "").removesuffix(".gguf") or args.label
    per, source = estimate(args.kept, model, len(picked))
    say(f"{args.label}: {len(picked)} of {len(messages)} messages ({arcs} from arcs"
        + (f", {loose} model-written, scored against a lower bound" if loose else "")
        + f") over a {world['kind'] or 'small'} world; the gold holds "
        + ", ".join(f"{v} {k}" for k, v in sample["gold"].items())
        + f"; about {per:.0f} s/msg ({source}), so about {len(picked) * per / 60:.0f} min")

    from ml_stack.client import Client, Request, Transport

    sampling = sampling_from(args)

    def read_and_keep(client: Any, reading: Sequence[Mapping[str, Any]], *, server: Mapping[str, Any],
                      twice_over: bool, n: int) -> str:
        """``reading`` through ``client``, folded, scored, kept and read back: the key."""
        rows, scores = measure(client, reading, graph, per_message=args.per_message, log=print)
        server = dict(server)
        if twice_over:
            scores["consistency"], server["twice"] = twice(client, reading, graph, scores,
                                                         per_message=args.per_message)
        key = save(args.kept, rows, label=args.label, model=model, world=world, scores=scores,
                   sample={**sample, "n": n}, server=server)
        say(f"kept as {key}")
        table(read_back(args.kept, [key]))
        return key

    if args.serve:
        from ml_stack.serve import serve

        found = str(hub.located(args.serve[0], loose=True) or args.serve[0])
        began = time.time()
        # the settings the model scored best with -- its build, head, cache type, thinking budget, raw
        # flags -- unless told to serve it bare: an extraction measured on mainline without
        # the head (2026-09-02) measured a different program from the one that answers
        lease: dict[str, Any] = {"port": args.serve_port, "context": args.context,
                                 "parallel": args.parallel, "timeout": 900.0,
                                 "cache_reuse": 256, "warmup": False}
        manager = None
        if getattr(args, "profile", True):
            from ml_stack.serve.profile import profile_for

            measured = profile_for(str(found), workload="ingest")
            if measured is not None:
                serving = measured.serving(port=args.serve_port, slots=args.parallel)
                lease = {**lease, **{k: v for k, v in serving.lease().items()
                                     if k not in ("port", "context", "parallel")}}
                manager = serving.manager()
                say(f"    serving in the settings it scored best with: {measured.said()}"
                    if hasattr(measured, "said") else "    serving in the settings it scored best with")
        if getattr(args, "n_max", None) is not None:
            if not lease.get("draft"):
                warn("    --n-max: no draft head is being served, so there is no draft "
                     "to lengthen")
                return 2
            lease["spec_draft_max"] = int(args.n_max)
            say(f"    draft length {args.n_max} over the profile's")
        with serve(found, manager=manager, **lease) as server:
            say(f"    up in {time.time() - began:.0f}s")
            client = Client(server.base_url, request=Request(**sampling),
                            transport=Transport(timeout=args.per_message))
            server = {**footprint(server.base_url), "sampling": dict(client.sampling),
                    "load_s": getattr(server, "load_s", None)}
            if lease.get("spec_draft_max") is not None:
                server["spec_draft_max"] = int(lease["spec_draft_max"])
            if wants_smoke(args):
                # first, on this load: a few messages through the whole path, kept and
                # read back, before the sample that costs the GPU
                few = sample_messages(messages, SMOKE_MESSAGES, seed=args.seed)
                say(f"\n  smoke: {len(few)} message(s) through the whole path first")
                key = read_and_keep(client, few, server=server, twice_over=False, n=len(few))
                smoked(read_back(args.kept, [key]), f"{args.label} smoke")
                say("  smoke: ok\n")
            read_and_keep(client, picked, server=server, twice_over=args.twice, n=len(picked))
    else:
        if wants_smoke(args):
            smoke_first(args)
        if not _idle(args.base_url, args):
            return 3
        client = Client(args.base_url, request=Request(**sampling),
                        transport=Transport(timeout=args.per_message))
        read_and_keep(client, picked, server={**footprint(args.base_url),
                                            "sampling": dict(client.sampling)},
                      twice_over=args.twice, n=len(picked))
    return 0
