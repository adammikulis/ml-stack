"""``ml-stack-bench compare``: several configurations of one model as one document.

`assemble` takes the store and a list of labels and writes, per label, what served it
(`served_by`), the graph bench's newest run under that label, the newest speed run
(`speed.KIND`) beside it, the memory either measured, the standard sets a
``ml-stack-bench standard`` JSON scored under the same label, and the draft head's
acceptance. Nothing absent is 0: a configuration measured on the graph and not for
speed carries ``"speed": null``.

``--export`` writes it as JSON, refusing a path inside a repository the way ``show
--export`` does, and carries graph runs only over the community that ships with this
package.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.graph import bench
from ml_stack.graph.bench.backends import describe
from ml_stack.graph.bench.score import _head_of, derived, invented_digest
from ml_stack.graph.bench.speed import KIND as SPEED
from ml_stack.paths import repo_root

# The way suffixes a graph run's label ends in, which a speed run's does not.
WAYS = ("plain", "shortlist")


def stem_of(label: str) -> str:
    """``flash-plain-batch`` -> ``flash``: the label with its way and everything after
    it taken off, which is what a speed run of the same server is labelled by."""
    words = str(label or "").split("-")
    for n, word in enumerate(words):
        if word in WAYS and n > 0:
            return "-".join(words[:n])
    return str(label or "")


def _graph_runs(kept: Sequence[Mapping[str, Any]], label: str, *, anyway: bool = False
                ) -> list[Mapping[str, Any]]:
    """The graph runs labelled ``label``, over the invented community unless ``anyway``."""
    mine = "" if anyway else invented_digest()
    out = []
    for one in kept:
        if str(one.get("kind") or "") or str(one.get("label") or "") != label:
            continue
        if not any(r.get("expected") for r in (one.get("rows") or [])):
            continue
        if mine and str((one.get("server") or {}).get("graph") or "") != mine:
            continue
        out.append(one)
    return out


def _speed_runs(kept: Sequence[Mapping[str, Any]], label: str) -> list[Mapping[str, Any]]:
    """The speed runs that go with ``label``: labelled the same, or by its stem with
    ``-speed`` on the end."""
    wanted = {label, f"{label}-speed", f"{stem_of(label)}-speed", stem_of(label)}
    return [one for one in kept
            if str(one.get("kind") or "") == SPEED and str(one.get("label") or "") in wanted]


def _newest(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The newest by ``at``, the later in the store on a tie -- two runs kept inside one
    second are ordered by their keys."""
    if not runs:
        return None
    return max(enumerate(runs), key=lambda pair: (str(pair[1].get("at") or ""), pair[0]))[1]


def _standard_for(standards: Sequence[Mapping[str, Any]], label: str) -> dict[str, Any] | None:
    """The standard-set record whose ``label`` is this one or its stem."""
    wanted = {label, stem_of(label), f"{stem_of(label)}-speed"}
    for one in standards:
        if str(one.get("label") or "") in wanted:
            sets = one.get("sets") or {}
            return {name: {"score": got.get("score"), "n": got.get("n"),
                           "metric": got.get("metric"), "seconds": got.get("seconds")}
                    for name, got in sets.items() if isinstance(got, Mapping)} or None
    return None


def _drafted(server: Mapping[str, Any]) -> bool | None:
    """Whether a head was served: True with one on the record, False for a llama-server
    served without one, None for a run that cannot say."""
    if _head_of({"server": server}):
        return True
    said = server.get("served_by") or {}
    if isinstance(said, Mapping) and said.get("draft") is not None:
        return bool(said["draft"])
    if server.get("binary"):
        return False
    return None


def _acceptance(one: Mapping[str, Any] | None) -> float | None:
    if one is None:
        return None
    rows = one.get("rows") or []
    guessed = [float(r["draft_tokens"]) for r in rows if r.get("draft_tokens") is not None]
    if not guessed or not sum(guessed):
        return None
    taken = sum(float(r.get("draft_taken") or 0) for r in rows if r.get("draft_tokens") is not None)
    return round(taken / sum(guessed), 4)


def _memory(*records: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """``peak_gb``, ``load_s`` and ``disk_gb`` from whichever run measured each, None
    for what none did."""
    peak = load = disk = None
    for one in records:
        server = (one or {}).get("server") or {}
        got = server.get("resident_peak") or server.get("resident_bytes")
        if got:
            peak = max(float(got), peak or 0.0)
        if load is None and server.get("load_s") is not None:
            load = float(server["load_s"])
        if disk is None and server.get("weights_bytes"):
            disk = float(server["weights_bytes"])
    if peak is None and load is None and disk is None:
        return None
    return {"peak_gb": round(peak / 2**30, 2) if peak is not None else None,
            "load_s": round(load, 1) if load is not None else None,
            "disk_gb": round(disk / 2**30, 2) if disk is not None else None}


def _graph(one: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if one is None:
        return None
    got = derived(one)
    if not got:
        return None
    return {"f1": round(got["right"], 4), "recall": round(got["recall"], 4),
            "precision": round(got["precision"], 4),
            "seconds_per_question": round(got["seconds_per_question"], 2),
            "calls_per_question": round(got.get("calls_per_question", 0.0), 2),
            "questions": int(got["questions"]), "label": one.get("label"),
            "at": one.get("at")}


def _speed(one: Mapping[str, Any] | None) -> list[dict[str, Any]] | None:
    if one is None:
        return None
    return [{"prompt_tokens": c.get("prompt_tokens"), "streams": c.get("streams"),
             "prefill_tps": c.get("prefill_tps"), "decode_tps": c.get("decode_tps"),
             "decode_tps_per_stream": c.get("decode_tps_per_stream"),
             "ttft_s": c.get("ttft_s"), "ttft_from": c.get("ttft_from")}
            for c in (one.get("rows") or [])] or None


def machine() -> str:
    """``Mac (128 GB)``: the kind of machine and its memory, as far as this one says."""
    kind = {"Darwin": "Mac", "Linux": "Linux", "Windows": "Windows"}.get(
        platform.system(), platform.system() or "machine")
    try:
        import psutil

        total = int(psutil.virtual_memory().total)
        return f"{kind} ({round(total / 2**30)} GB)"
    except Exception:  # noqa: BLE001 - a machine that will not say its memory
        return kind


def assemble(kept: Sequence[Mapping[str, Any]], labels: Sequence[str], *,
             standards: Sequence[Mapping[str, Any]] = (), title: str = "",
             machine_name: str = "", anyway: bool = False) -> dict[str, Any]:
    """The document: one entry per label, absent things None."""
    configs = []
    for label in labels:
        graph_run = _newest(_graph_runs(kept, label, anyway=anyway))
        speed_run = _newest(_speed_runs(kept, label))
        server = dict((graph_run or speed_run or {}).get("server") or {})
        said = server.get("served_by") if isinstance(server.get("served_by"), Mapping) else None
        build = str(server.get("build") or "")
        configs.append({
            "label": describe(said, build=build) or label,
            "run": label,
            "program": (said or {}).get("program"),
            "version": (said or {}).get("version"),
            "format": (said or {}).get("format"),
            "runtime": (said or {}).get("runtime"),
            "quant": (said or {}).get("quant"),
            "model": server.get("model") or (said or {}).get("model"),
            "draft": _drafted(server),
            "graph": _graph(graph_run),
            "speed": _speed(speed_run),
            "memory": _memory(graph_run, speed_run),
            "standard": _standard_for(standards, label),
            "acceptance": _acceptance(graph_run),
        })
    return {"made_at": time.strftime("%FT%T"), "machine": machine_name or machine(),
            "title": title or "", "configs": configs}


def read_standards(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Every standard-set JSON named: ``{"label", "sets": {...}}`` each."""
    out = []
    for path in paths:
        got = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
        if isinstance(got, list):
            out += [dict(one) for one in got if isinstance(one, Mapping)]
        elif isinstance(got, Mapping):
            out.append(dict(got))
    return out


def write(document: Mapping[str, Any], where: str | Path, *, anyway: bool = False) -> str:
    """Write the document as JSON, refusing a path inside a repository."""
    target = Path(where).expanduser()
    repo = repo_root(target.parent)
    if repo and not anyway:
        raise ValueError(
            f"{target} is inside the git repository at {repo}. These numbers describe one "
            f"machine and one build and go stale with the next model release -- write it "
            f"somewhere outside a repository.")
    target.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


def add_arguments(sub: Any) -> Any:
    one = sub.add_parser("compare", allow_abbrev=False,
                         help="several configurations as one document: the graph bench, "
                              "the speed grid, the memory and the standard sets per label")
    one.add_argument("labels_given", nargs="*", metavar="LABEL",
                     help="the graph runs' labels, newest of each; a speed run matches by "
                          "the same label, or its stem with -speed on the end")
    one.add_argument("--labels", default="", metavar="A,B,C",
                     help="the same labels, comma-separated")
    one.add_argument("--last", nargs="?", const=3, type=int, default=0, metavar="N",
                     help="the N newest labels the store holds graph runs under "
                          "(default 3 when given bare), newest first")
    one.add_argument("--standard", action="append", default=[], metavar="FILE.json",
                     help="a standard-set result written by ml-stack-bench standard, "
                          "matched to a label by its own; repeatable")
    one.add_argument("--export", required=True, metavar="FILE.json",
                     help="where to write the document; not inside a repository")
    one.add_argument("--title", default="", help="the document's title")
    one.add_argument("--machine", default="", help="the machine, e.g. 'Mac (128 GB)' "
                                                   "(default: this one)")
    one.add_argument("--kept", default=str(bench.HOME / "runs.ladybug"),
                     help="the store the runs are in (default: %(default)s)")
    one.add_argument("--anyway-export", action="store_true", dest="export_anyway",
                     help="include graph runs over some other community, and allow a "
                          "path inside a repository")
    return one


def newest_labels(kept: Sequence[Mapping[str, Any]], n: int) -> list[str]:
    """The ``n`` labels most recently kept graph runs under, newest first."""
    seen: dict[str, str] = {}
    for one in kept:
        if str(one.get("kind") or "") or not any(r.get("expected")
                                                 for r in (one.get("rows") or [])):
            continue
        label = str(one.get("label") or "")
        seen[label] = max(seen.get(label, ""), str(one.get("at") or ""))
    return sorted(seen, key=lambda label: seen[label], reverse=True)[:max(0, n)]


def main(args: Any) -> int:
    kept = bench._kept(args.kept)
    labels = [*(getattr(args, "labels_given", None) or []),
              *[w.strip() for w in str(args.labels or "").split(",") if w.strip()]]
    if getattr(args, "last", 0):
        labels += [label for label in newest_labels(kept, int(args.last)) if label not in labels]
    if not labels:
        print("error: name at least one label, or pass --last", file=sys.stderr)
        return 2
    try:
        standards = read_standards(args.standard or [])
    except (OSError, ValueError) as why:
        print(f"error: could not read a --standard file: {why}", file=sys.stderr)
        return 2
    document = assemble(kept, labels, standards=standards, title=args.title,
                        machine_name=args.machine, anyway=bool(args.export_anyway))
    try:
        where = write(document, args.export, anyway=bool(args.export_anyway))
    except ValueError as why:
        print(f"error: {why}", file=sys.stderr)
        return 2
    for one in document["configs"]:
        has = [name for name in ("graph", "speed", "memory", "standard")
               if one.get(name) is not None]
        print(f"{one['run']}: {one['label']}"
              + (f" -- {', '.join(has)}" if has else " -- nothing kept under this label"))
    print(where)
    return 0
