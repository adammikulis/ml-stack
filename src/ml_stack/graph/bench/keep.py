"""Where runs are kept, and the two sizes every module agrees on.

A run is worth keeping because the point of one is to compare it with another, later; so
the store, its home, what a saved run must be made of (`_plain`), and the read-back that
proves a run exists all live here, under everything else. `SHORT` and `SMOKE` are here
too, because the scorer, the parser and the runner all need the same two numbers.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The package is the namespace the tests and `selfcheck` patch -- `bench.HOME`,
# `bench.runs` -- so anything patchable is looked up there at call time, never bound here
# at import.
from ml_stack.graph import bench

if TYPE_CHECKING:
    from ml_stack.graph.bench.score import Row

# Runs are worth keeping: the point of one is to compare it with another, later, and a
# benchmark written to a temporary directory answers no question a week from now.
# ``MLSTACK_BENCH_HOME`` moves it: a fleet daemon rooted somewhere else sets it on the
# benches it launches, so they record under the home whose lock the daemon watches.
HOME = Path(os.environ.get("MLSTACK_BENCH_HOME") or "~/.ml-stack/bench").expanduser()


def _plain(value: Any) -> Any:
    """``value`` as nothing but dicts, lists, strings, numbers, booleans and None.

    What `save` writes has to come back, and the store keeps JSON. Keys that are not
    strings become strings, dataclasses become their fields, sets and tuples become lists,
    and anything else becomes its ``str`` -- nothing is dropped, because a run that lost
    a field is not the run that was measured. What comes out is fed to ``json.dumps``
    with no ``default``, so anything this missed raises before the store sees it.
    """
    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, Mapping):
        return {(k if isinstance(k, str) else str(k)): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(v) for v in (sorted(value, key=str) if isinstance(value, (set, frozenset))
                                    else value)]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


_commit_here: str | None = None


def _commit(root: Path | None = None) -> str:
    """The short sha this package runs from, ``(dirty)`` appended when the tree has changes;
    "" when there is no repository or no git. Best effort, never a reason not to run.

    On every run record (`stamped`), at the top of every detached log and in
    `measuring_file`, because a day of logs with no commit on them was a day nobody could
    tell which code had measured what -- and a fleet must not compose one commit's
    accuracy with another's cost. ``root`` is the working tree to ask; unset, the one this
    file sits in, read once per process -- a pinned worktree keeps a ``.git`` file, and
    `repo_root` reads that too.
    """
    global _commit_here
    if root is None and _commit_here is not None:
        return _commit_here
    from ml_stack.paths import repo_root

    where = root if root is not None else repo_root(Path(__file__).parent)
    out = ""
    if where is not None:
        try:
            def git(*words: str) -> str:
                return subprocess.run(["git", "-C", str(where), *words], capture_output=True,
                                      text=True, timeout=15, check=True).stdout.strip()

            sha = git("rev-parse", "--short", "HEAD")
            out = f"{sha} (dirty)" if sha and git("status", "--porcelain") else sha
        except Exception:  # noqa: BLE001 - a commit line is never worth a run not starting
            out = ""
    if root is None:
        _commit_here = out
    return out


def stamped(held: Mapping[str, Any] | None) -> dict[str, Any]:
    """A run's ``server`` record with ``host`` and ``commit`` on it, unless it names them.

    Every run says which machine measured it and which code: a fleet gathers runs from
    several hosts into one store, and the table, the ranking and the frontier must tell
    them apart. A record that arrives with a host -- gathered from a peer, or measured
    under ``--host`` -- keeps it; an empty one is filled in.
    """
    out = dict(held or {})
    out.setdefault("host", socket.gethostname())
    out.setdefault("commit", _commit())
    return out


def prepared() -> str:
    """The store `prepare` builds by default, when it has been built, else "".

    `run` and `sweep` take it as their `--store` default so that a machine that has run
    `prepare` measures the look_up that ships -- characters, the word index and vectors fused
    -- without being told to. Without one, the bench measures character matching alone, and
    every run recorded that way ranked something nobody runs.
    """
    where = bench.HOME / "graph.ladybug"
    return str(where) if where.exists() else ""


# How many questions a short run asks. Chosen by measuring what survives, not by feel: at
# twenty, every kind of answer is still asked about and the mean number of answers expected
# is 2.2 -- the same as the whole set, so the difficulty is preserved and not only the
# variety. Below about fourteen the rarer kinds start to go, and a shorter benchmark that
# has stopped asking about places is not a shorter benchmark but a different one.
#
# The cost is granularity: eighteen scored questions make each one worth 5.6 points of F1
# against 2.0 on the full set of fifty, so a small difference is noise on a short run and
# signal on a full one. The `n` column on every line is what keeps the two from being read
# together.
SHORT = 20
# Enough to walk the whole path -- serve, ask, score, measure the server, save, summarise --
# and short enough that finding out it is broken costs a minute. A sweep once answered every
# question and then raised while writing a summary line, losing all of it.
SMOKE = 2


class RunNotKept(RuntimeError):
    """A run was written and did not come back the way `runs` reads it."""


def _told(row: dict[str, Any]) -> dict[str, Any]:
    """One row as it is kept: its transcript when it has one, and no key when it has not.

    A traced row carries what was said call by call (`bench.measure.Counting.trace`), which
    is the field a fine-tune is built from and the only one measured in kilobytes. An
    untraced row carries an empty list, and an empty list in every row of every run is a
    key that says nothing -- so it is dropped, and a run kept before tracing existed and a
    run kept without it read back identically.
    """
    if not row.get("trace"):
        row.pop("trace", None)
    return row


def asked_with(asking: Mapping[str, Any] | None,
               held: Mapping[str, Any]) -> dict[str, Any]:
    """A run's ``asking`` record: the way it asked, with the sampling it asked at.

    `measure.asking` knows the keywords it handed `converse` and nothing about the client,
    and the sampler settings are the other half of how a question was put -- so the two are
    joined here, once, rather than at every call site. An explicit ``sampling`` in
    ``asking`` wins; no asking at all stays no record.
    """
    if asking is None:
        return {}
    out = dict(asking)
    if "sampling" not in out and held.get("sampling"):
        out["sampling"] = dict(held["sampling"])
    return out


def save(store: str | Path, rows: Sequence[Row], *, held: dict[str, Any] | None = None,
         asking: Mapping[str, Any] | None = None) -> str:
    """Keep a run where it can be compared with another one, later, by anybody.

    Then read it back the way `runs` will, on a fresh handle, and refuse to return until
    what came back is what went in. Twelve runs -- half an hour of GPU -- were once kept
    as nothing: the store took them, a scan of it returned an empty string for each, and
    the sweep printed its summary from memory, so nobody knew until the next morning. The
    read-back is the only proof that a run exists, and it is cheap next to the run.

    ``held`` is the server record -- what was serving, and how. ``asking`` is the other
    half, the keywords `converse` was asked with (`measure.asking`), which used to live
    only in the end of a label -- ``...-plain-batch-kv-q8_0-rb0`` -- where nothing could
    group on them or even read them. Written only when there is one, so a run kept before
    it existed and a run kept without it read back identically: an empty record in every
    run is a key that says nothing.
    """
    from ml_stack.graph.bench.score import prefix_hits
    from ml_stack.graph.store import GraphStore

    server = stamped(held)
    # the run's prompt-cache figure beside the per-question ones, so the table can carry
    # it without adding up every row. Only when there is one: a run with no turn to judge
    # carries no key, like a run kept before it was counted -- not counted is not zero
    if "prefix_hits" not in server:
        hits = prefix_hits([asdict(r) for r in rows])
        if hits is not None:
            server["prefix_hits"] = hits
    stem = f"bench:{rows[0].label}:{time.strftime('%Y%m%dT%H%M%S')}" if rows else "bench:empty"
    kept_rows = [_told(asdict(r)) for r in rows]
    asked = asked_with(asking, server)
    record = _plain({"at": time.strftime("%FT%T"), "label": rows[0].label if rows else "",
                     "server": server, **({"asking": asked} if asked else {}),
                     "rows": kept_rows,
                     # how many of the rows carry their transcript, so `show` and
                     # `train-tools from-bench` can say "this run kept none" rather than
                     # "this run found none"
                     "traced": sum(1 for r in kept_rows if r.get("trace")),
                     "unread_named": sum(r.unread_named for r in rows)})
    record = json.loads(json.dumps(record))      # no default=: anything left raises here
    with GraphStore(store) as writer:
        # Two runs of one label inside a second used to land on the same key and the later
        # one silently replaced the earlier. A run took minutes when that was written; with
        # answers cached it can take no time at all, so the collision is real now.
        key, n = stem, 1
        while writer.get_doc(key) is not None:
            key, n = f"{stem}-{n}", n + 1
        writer.put_doc(key, record)
    back = next((r for r in bench.runs(store) if r.get("key") == key), None)
    tries = 1
    while back is None and tries < 4:
        # CI (Linux, ladybug 0.18.x, 2026-09-02): a run written to a fresh store came back on
        # a later read and not the first. A run that needs a second read is still a run that
        # came back; one that never does is the error below. HANDOFF has the investigation.
        time.sleep(0.2 * tries)
        tries += 1
        back = next((r for r in bench.runs(store) if r.get("key") == key), None)
    if back is None:
        raise RunNotKept(f"{key} was written to {store} and did not come back")
    back = {k: v for k, v in back.items() if k != "key"}
    if back != record:
        differs = sorted(k for k in set(back) | set(record) if back.get(k) != record.get(k))
        raise RunNotKept(f"{key} came back from {store} changed: {', '.join(differs)} differ")
    return key


def runs(store: str | Path, label: str = "") -> list[dict[str, Any]]:
    """Every run kept in ``store``, newest last, optionally only one label's.

    Each carries its ``key``. A doc that reads back empty is left out -- it is not a run,
    and a row of dashes in the table said nothing about why -- and `empties` names them.
    """
    from ml_stack.graph.store import GraphStore

    with GraphStore(store, read_only=True) as held:
        kept = held.docs()
    found = [{**kept[k], "key": k} for k in sorted(kept)
             if k.startswith("bench:") and isinstance(kept[k], dict) and kept[k]]
    return [r for r in found if not label or r.get("label") == label]


def empties(store: str | Path) -> list[str]:
    """The keys of runs that read back as nothing, which `forget --empty` removes."""
    from ml_stack.graph.store import GraphStore

    if not Path(store).expanduser().exists():
        return []
    with GraphStore(store, read_only=True) as held:
        kept = held.docs()
    return sorted(k for k, v in kept.items()
                  if k.startswith("bench:") and not (isinstance(v, dict) and v))


def forget(store: str | Path, *, label: str = "", empty: bool = False) -> list[str]:
    """Delete runs: every empty one, or every run of one label. Returns what went."""
    from ml_stack.graph.store import GraphStore

    going = (empties(store) if empty
             else [r["key"] for r in bench.runs(store, label)] if label else [])
    if not going:
        return []
    with GraphStore(store) as held:
        for key in going:
            held.delete_doc(key)
    return going


def _kept(store: str | Path) -> list[dict[str, Any]]:
    """Every run in ``store``, or none when there is no store yet -- a sweep whose every
    model was refused at preflight has kept nothing and must still print its table."""
    return bench.runs(store) if Path(store).expanduser().exists() else []


def read_back(store: str | Path, keys: Sequence[str]) -> list[dict[str, Any]]:
    """The runs under ``keys``, read from the store the way `show` reads them.

    What a smoke run exists to prove: the whole path, and the last step of the path is
    the store giving the run back. Summarising from memory proved everything but that,
    and a sweep passed its smoke and then kept twelve runs as nothing.
    """
    kept = {r["key"]: r for r in bench.runs(store)}
    lost = [k for k in keys if k not in kept]
    if lost:
        raise RunNotKept(f"{len(lost)} run(s) saved to {store} did not come back: "
                         + ", ".join(lost))
    return [kept[k] for k in keys]


def resumable(store: str | Path, *, questions: int, context: int, parallel: int,
              since: str = "") -> Callable[[str], Mapping[str, Any] | None]:
    """``already(label)``: the run kept under ``label`` that makes measuring it again a
    waste, or None.

    A kept run counts when it asked this many questions at this context per slot and this
    many slots -- a run at another context is another measurement, as the `ctx` column
    says -- and is no older than ``since``, which defaults to the start of today. A run
    from last week is what the sweep was started to replace.
    """
    floor = since or time.strftime("%FT00:00:00")
    per_slot = int(context) // max(1, int(parallel))
    kept = bench.runs(store) if Path(store).expanduser().exists() else []

    def already(label: str) -> Mapping[str, Any] | None:
        for one in reversed(kept):
            server = one.get("server") or {}
            if (one.get("label") == label
                    and str(one.get("at", "")) >= floor
                    and len(one.get("rows") or ()) == questions
                    and int(server.get("context") or 0) == per_slot
                    and int(server.get("slots") or 0) == parallel):
                return one
        return None

    return already
