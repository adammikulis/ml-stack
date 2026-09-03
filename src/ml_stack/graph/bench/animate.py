"""A comparison of model builds as an animated graphic: the tables each scene draws from,
the plan of scenes with their seconds, and the command that renders them with manim.

The document is JSON: ``title``, ``machine``, ``made_at`` and a list of ``configs``, each
with a ``label``, ``program``/``format``/``quant``/``draft``, and the measurements
``graph``, ``speed``, ``memory``, ``standard`` and ``acceptance``, any of which may be
absent or null. Everything null is shown as ``NOT_MEASURED``, never as a zero.

Nothing here imports manim until ``render`` is called; the tables are plain dicts.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

NOT_MEASURED = "not measured"

# Okabe & Ito's palette: distinguishable under the common forms of colour blindness.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7",
           "#999999"]

SCENES: list[tuple[str, str, float]] = [
    ("title", "Title card", 0.08),
    ("decode", "Decode tokens/s", 0.14),
    ("prefill", "Prefill tokens/s", 0.10),
    ("ttft", "Time to first token", 0.06),
    ("concurrency", "Decode at 1, 2 and 4 streams", 0.08),
    ("memory", "Peak memory and load time", 0.12),
    ("graph", "The graph bench: F1 against seconds per question", 0.14),
    ("calls", "Tool calls per question", 0.06),
    ("standard", "Standard sets", 0.12),
    ("closing", "Closing card", 0.10),
]


def _get(config: Mapping[str, Any], *path: str) -> Any:
    at: Any = config
    for key in path:
        if not isinstance(at, Mapping):
            return None
        at = at.get(key)
    return at


def _num(value: float | None, places: int | None = None) -> str:
    if value is None:
        return NOT_MEASURED
    if places is None:
        places = 0 if abs(value) >= 100 else 1 if abs(value) >= 10 else 2
    return f"{value:.{places}f}"


def _pct(value: float | None) -> str:
    return NOT_MEASURED if value is None else f"{100 * value:.0f}%"


def scale(values: Iterable[float | None]) -> float:
    """The smallest round ceiling (1, 1.5, 2, 3, 4, 5 x 10^k) at or above the largest value."""
    want = max((v for v in values if v is not None), default=0)
    if want <= 0:
        return 1
    power = 10 ** math.floor(math.log10(want))
    for step in (1, 1.5, 2, 3, 4, 5, 10):
        if step * power >= want:
            return step * power
    return 10 * power


def config_line(config: Mapping[str, Any]) -> str:
    """``program · format · quant``, and ``· draft`` when the config serves a draft head."""
    parts = [str(config.get(k)) for k in ("program", "format", "quant") if config.get(k)]
    if config.get("draft"):
        parts.append("draft")
    return " · ".join(parts)


def colours(doc: Mapping[str, Any]) -> dict[str, str]:
    """One palette colour per config label, in document order."""
    return {c["label"]: PALETTE[i % len(PALETTE)] for i, c in enumerate(doc["configs"])}


def _date(doc: Mapping[str, Any]) -> str:
    return str(doc.get("made_at") or "")[:10]


def title_card(doc: Mapping[str, Any]) -> dict[str, Any]:
    """The title, machine, date and one ``config_line`` per config."""
    return {"title": doc.get("title") or "", "machine": doc.get("machine") or "",
            "date": _date(doc), "configs": [config_line(c) for c in doc["configs"]]}


def _bar(doc: Mapping[str, Any], config: Mapping[str, Any], value: float | None,
         shown: str) -> dict[str, Any]:
    return {"config": config["label"], "colour": colours(doc)[config["label"]],
            "value": value, "shown": shown}


def _panel(doc: Mapping[str, Any], key: str, title: str, unit: str,
           groups: Sequence[tuple[str, list[float | None]]], *, places: int | None = None,
           line: Mapping[str, Any] | None = None, note: str = "") -> dict[str, Any]:
    built = [{"label": label,
              "bars": [_bar(doc, c, v, _num(v, places)) for c, v in zip(doc["configs"], vs)]}
             for label, vs in groups]
    every = [v for _, vs in groups for v in vs]
    if line:
        every.append(line["value"])
    return {"key": key, "title": title, "unit": unit, "groups": built, "scale": scale(every),
            "line": dict(line) if line else None, "note": note}


def _speed_row(config: Mapping[str, Any], prompt: int, streams: int) -> Mapping[str, Any]:
    for row in config.get("speed") or []:
        if row.get("prompt_tokens") == prompt and row.get("streams") == streams:
            return row
    return {}


def _prompt_sizes(doc: Mapping[str, Any]) -> list[int]:
    return sorted({row["prompt_tokens"] for c in doc["configs"] for row in c.get("speed") or []
                   if row.get("prompt_tokens") is not None})


def speed_panels(doc: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Decode, prefill and time to first token per prompt size at one stream, then decode
    at every stream count measured at the smallest prompt."""
    sizes = _prompt_sizes(doc)
    per_size = lambda field: [(f"{n} tokens", [_speed_row(c, n, 1).get(field)  # noqa: E731
                                                for c in doc["configs"]]) for n in sizes]
    panels = [_panel(doc, "decode", "Decode", "tokens/s", per_size("decode_tps")),
              _panel(doc, "prefill", "Prefill", "tokens/s", per_size("prefill_tps")),
              _panel(doc, "ttft", "Time to first token", "s", per_size("ttft_s"))]
    smallest = sizes[0] if sizes else None
    streams = sorted({row["streams"] for c in doc["configs"] for row in c.get("speed") or []
                      if row.get("prompt_tokens") == smallest and row.get("streams")})
    groups = [(f"{s} stream" + ("s" if s != 1 else ""),
               [_speed_row(c, smallest, s).get("decode_tps") for c in doc["configs"]])
              for s in streams]
    panels.append(_panel(doc, "concurrency", "Decode as streams are added", "tokens/s", groups,
                         note=f"{smallest}-token prompt" if smallest else ""))
    return panels


def machine_gb(machine: str | None) -> float | None:
    """The memory in a machine's name (``Mac (128 GB)`` -> 128), or None."""
    found = re.search(r"(\d+(?:\.\d+)?)\s*GB", machine or "", re.IGNORECASE)
    return float(found.group(1)) if found else None


def memory_panels(doc: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Peak GB per config against a line at the machine's memory, and load seconds."""
    room = machine_gb(doc.get("machine"))
    line = {"value": room, "label": f"{room:g} GB on this machine"} if room else None
    peak = _panel(doc, "peak_gb", "Peak memory", "GB",
                  [("peak", [_get(c, "memory", "peak_gb") for c in doc["configs"]])],
                  places=0, line=line)
    load = _panel(doc, "load_s", "Load time", "s",
                  [("load", [_get(c, "memory", "load_s") for c in doc["configs"]])])
    return peak, load


def frontier(points: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The points no other point beats on both axes (lower x, higher y), by x."""
    kept = []
    for p in points:
        beaten = any((q["x"] <= p["x"] and q["y"] >= p["y"]
                      and (q["x"] < p["x"] or q["y"] > p["y"])) for q in points)
        if not beaten:
            kept.append(dict(p))
    return sorted(kept, key=lambda p: (p["x"], -p["y"]))


def graph_scatter(doc: Mapping[str, Any]) -> dict[str, Any]:
    """F1 against seconds per question per config, with the frontier and the axis scales."""
    points, missing = [], []
    palette = colours(doc)
    for c in doc["configs"]:
        f1, secs = _get(c, "graph", "f1"), _get(c, "graph", "seconds_per_question")
        if f1 is None or secs is None:
            missing.append(c["label"])
            continue
        points.append({"config": c["label"], "colour": palette[c["label"]], "x": secs, "y": f1,
                       "shown": f"F1 {f1:.2f} · {secs:.1f} s",
                       "questions": _get(c, "graph", "questions")})
    return {"points": points, "frontier": frontier(points),
            "x_scale": scale([p["x"] for p in points]), "y_scale": 1,
            "x_label": "seconds per question", "y_label": "F1", "not_measured": missing}


def calls_panel(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Tool calls per question as one group of bars."""
    return _panel(doc, "calls", "Tool calls per question", "calls",
                  [("calls", [_get(c, "graph", "calls_per_question") for c in doc["configs"]])],
                  places=1)


def standard_grid(doc: Mapping[str, Any]) -> dict[str, Any]:
    """Every standard set any config ran, as a row of cells per config."""
    sets: list[str] = []
    for c in doc["configs"]:
        for name in (c.get("standard") or {}):
            if name not in sets:
                sets.append(name)
    palette = colours(doc)
    rows = []
    for c in doc["configs"]:
        cells = []
        for name in sets:
            score = _get(c, "standard", name, "score")
            cells.append({"set": name, "score": score, "n": _get(c, "standard", name, "n"),
                          "shown": _pct(score)})
        rows.append({"config": c["label"], "colour": palette[c["label"]], "cells": cells})
    return {"sets": sets, "rows": rows}


def _headline(config: Mapping[str, Any]) -> str:
    sizes = sorted({row["prompt_tokens"] for row in config.get("speed") or []
                    if row.get("prompt_tokens") is not None})
    parts = []
    decode = _speed_row(config, sizes[0], 1).get("decode_tps") if sizes else None
    if decode is not None:
        parts.append(f"{_num(decode)} tok/s")
    f1, secs = _get(config, "graph", "f1"), _get(config, "graph", "seconds_per_question")
    if f1 is not None:
        parts.append(f"F1 {f1:.2f}")
    if secs is not None:
        parts.append(f"{secs:.1f} s/question")
    if f1 is None and secs is None:
        parts.append("graph not measured")
    peak = _get(config, "memory", "peak_gb")
    if peak is not None:
        parts.append(f"{peak:.0f} GB peak")
    if config.get("acceptance") is not None:
        parts.append(f"{100 * config['acceptance']:.0f}% accepted")
    return " · ".join(parts)


def closing(doc: Mapping[str, Any]) -> dict[str, Any]:
    """One line of headline numbers per config, and where and when it was measured."""
    palette = colours(doc)
    return {"title": doc.get("title") or "",
            "lines": [{"config": c["label"], "colour": palette[c["label"]],
                       "numbers": _headline(c)} for c in doc["configs"]],
            "footer": f"measured on {doc.get('machine') or 'this machine'}, {_date(doc)}"}


def _has(doc: Mapping[str, Any], key: str) -> bool:
    if key in ("title", "closing"):
        return True
    if key in ("decode", "prefill", "ttft", "concurrency"):
        panel = next(p for p in speed_panels(doc) if p["key"] == key)
        return any(b["value"] is not None for g in panel["groups"] for b in g["bars"])
    if key == "memory":
        return any(b["value"] is not None for p in memory_panels(doc)
                   for g in p["groups"] for b in g["bars"])
    if key == "graph":
        return bool(graph_scatter(doc)["points"])
    if key == "calls":
        return any(b["value"] is not None for b in calls_panel(doc)["groups"][0]["bars"])
    if key == "standard":
        return bool(standard_grid(doc)["sets"])
    return False


def plan(doc: Mapping[str, Any], seconds: float = 50, *,
         only: Sequence[str] | None = None) -> list[dict[str, Any]]:
    """The scenes with something measured, in order, with seconds that add up to the cut."""
    chosen = [(k, t, w) for k, t, w in SCENES if _has(doc, k) and (not only or k in only)]
    total = sum(w for _, _, w in chosen) or 1
    return [{"key": k, "title": t, "seconds": seconds * w / total} for k, t, w in chosen]


def load(where: str | Path) -> dict[str, Any]:
    """The comparison document at ``where``; raises ValueError when it has no configs."""
    doc = json.loads(Path(where).read_text(encoding="utf-8"))
    if not doc.get("configs"):
        raise ValueError("no configs in the comparison document")
    for c in doc["configs"]:
        if not c.get("label"):
            raise ValueError("a config has no label")
    return doc


QUALITY = {"l": "low_quality", "m": "medium_quality", "h": "high_quality"}


def render(doc: Mapping[str, Any], *, out: str | Path, png: str | Path | None = None,
           quality: str = "h", seconds: float = 50, only: Sequence[str] | None = None,
           work: str | Path | None = None) -> Path:
    """Render the plan for ``doc`` to ``out`` (and its last frame to ``png``); returns ``out``."""
    from . import animate_scene
    return animate_scene.render(doc, plan(doc, seconds, only=only), out=Path(out),
                                png=Path(png) if png else None, quality=QUALITY[quality],
                                work=Path(work) if work else None)


def describe(doc: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]]) -> str:
    """The plan as text: the title, the configs, and one line per scene with its seconds."""
    lines = [f"{doc.get('title') or 'comparison'} -- {len(doc['configs'])} configs, "
             f"{sum(s['seconds'] for s in scenes):.1f} s"]
    for c in doc["configs"]:
        lines.append(f"  {c['label']}  ({config_line(c)})")
    for s in scenes:
        lines.append(f"  {s['key']:<12}{s['seconds']:>6.1f} s  {s['title']}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """``ml-stack-bench animate COMPARISON.json --out FILE.mp4``."""
    parser = argparse.ArgumentParser(
        prog="ml-stack-bench animate",
        description="Render a comparison document as an animated graphic with manim.")
    parser.add_argument("comparison", help="the comparison document (JSON)")
    parser.add_argument("--out", required=True, help="the .mp4 to write")
    parser.add_argument("--png", help="also write the last frame as a still")
    parser.add_argument("--quality", choices=sorted(QUALITY), default="h",
                        help="l 480p15, m 720p30, h 1080p60 (default h)")
    parser.add_argument("--seconds", type=float, default=50,
                        help="the length of the whole cut (default 50)")
    parser.add_argument("--only", help="comma-separated scene keys to render alone")
    parser.add_argument("--work", help="where manim keeps its partial renders")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the scene plan and write nothing")
    args = parser.parse_args(argv)
    try:
        doc = load(args.comparison)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"{args.comparison}: {e}", file=sys.stderr)
        return 2
    only = [k.strip() for k in args.only.split(",")] if args.only else None
    scenes = plan(doc, args.seconds, only=only)
    print(describe(doc, scenes))
    if args.dry_run:
        return 0
    try:
        import manim  # noqa: F401
    except ImportError:
        print("manim is not installed: pip install 'ml-stack[viz]'", file=sys.stderr)
        return 2
    where = render(doc, out=args.out, png=args.png, quality=args.quality,
                   seconds=args.seconds, only=only, work=args.work)
    print(f"wrote {where}" + (f" and {args.png}" if args.png else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
