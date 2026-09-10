"""What a run cost to be right: the Pareto frontier, the rates table and the plot.

`pareto` is the runs nothing beats on both accuracy and cost, `rates` tables every run
against it, and `plot` writes the same as a self-contained HTML scatter. `AXES` names each
cost a frontier can be drawn against.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml_stack.bench.score import COSTS, NOISE, composed, derived, host_of, hosts_of
from ml_stack.bench.show import _shown
from ml_stack.log import say

__all__ = ["AXES", "UNITS", "pareto", "plot", "rates"]


def pareto(kept: Sequence[Mapping[str, Any]], *,
           cost: str = "seconds") -> list[Mapping[str, Any]]:
    """The runs nothing else beats on both accuracy and ``cost``.

    A run is dominated when another is at least as accurate *and* costs no more; those are
    the ones there is never a reason to choose. What is left is the frontier — every point
    on it is the best available at some budget, and choosing among them is choosing a budget
    rather than choosing a better run.

    ``cost`` is a name in `COSTS` -- seconds, paid_tokens, kv_bytes -- compared on the key
    it maps to: per question for time and tokens, a total for memory.
    """
    key = COSTS.get(cost, cost)
    scored = [(one, derived(one)) for one in kept]
    scored = [(one, d) for one, d in scored if d and d.get(key) is not None]
    front = []
    for one, mine in scored:
        beaten = any(
            other is not one
            and theirs["right"] >= mine["right"] and theirs[key] <= mine[key]
            and (theirs["right"] > mine["right"] or theirs[key] < mine[key])
            for other, theirs in scored)
        if not beaten:
            front.append(one)
    return sorted(front, key=lambda one: derived(one)[key])


def rates(kept: Sequence[Mapping[str, Any]], *, cost: str = "seconds",
          noise: float = NOISE) -> None:
    """Every run by what it cost to be right, frontier marked -- and each model once more
    as `composed` composes it, marked ``=``, so the frontier holds what a model can do at
    the speed of the configuration that held its accuracy."""
    if not kept:
        say("nothing kept yet")
        return
    points = list(kept) + composed(kept, noise=noise)
    on_front = {id(one) for one in pareto(points, cost=cost)}
    several = len(hosts_of(points)) > 1
    head = (f"{'run':28} {'n':>3} {'F1':>5} {'rec':>5} {'prec':>5} {'lit/q':>6} "
            f"{'F1/min':>8} {'F1/1k tok':>10} {'F1/GB':>7} {'s per':>7} {'tok per':>8}")
    say(head)
    say("-" * len(head))
    for one in sorted(points, key=lambda o: -(derived(o).get("right") or 0)):
        d = derived(one)
        if not d:
            continue
        def num(key: str, fmt: str) -> str:
            return format(d[key], fmt) if key in d else "-"
        mark = ("*" if id(one) in on_front else " ") + ("=" if one.get("composed") else " ")
        # by host when several measured: the frontier is one clock, and a point from
        # another machine is on it only by name
        named = (_shown(f"{one.get('label', '')}@{host_of(one) or '?'}", 18) if several
                 else str(one.get("label", ""))[:18])
        say(f"{named:18}{mark} {d['questions']:>3.0f} "
            f"{100 * d['right']:>4.0f}% {100 * d['recall']:>4.0f}% "
            f"{100 * d['precision']:>4.0f}% {d['shown_per_question']:>6.1f} "
            f"{num('right_per_minute', '8.2f')} {num('right_per_1k', '10.4f')} "
            f"{num('right_per_gb', '7.3f')} {num('seconds_per_right', '7.1f')} "
            f"{num('tokens_per_right', '8.0f')}")
    say(f"\n* on the frontier for accuracy against {AXES.get(cost, cost)}: nothing is "
        f"both more accurate and cheaper.")
    say(f"= a model composed: accuracy from its largest run, cost from its fastest run "
        f"within {noise * 100:g} points of it, scaled to the same number of questions.")


AXES = {"seconds": "wall clock per question (s)",
        "paid_tokens": "tokens paid for per question (read + written)",
        "kv_bytes": "KV cache and runtime (GB)"}
UNITS = {"seconds": "s", "paid_tokens": "tokens", "kv_bytes": "bytes"}


def plot(kept: Sequence[Mapping[str, Any]], where: str | Path, *,
         cost: str = "seconds", noise: float = NOISE) -> str:
    """Write accuracy against cost as a self-contained HTML scatter, frontier joined.

    Plain SVG built here rather than a plotting library: this has to open on a machine with
    no network and no packages, and a chart of a dozen points is a dozen circles. The
    frontier is drawn as a line through the runs nothing beats on both axes, so the shape of
    the trade is visible rather than inferred from a column of numbers. Each model's
    `composed` point is drawn as a ring beside its runs.
    """
    key = COSTS.get(cost, cost)
    points = [(one, derived(one)) for one in list(kept) + composed(kept, noise=noise)]
    points = [(one, d) for one, d in points if d and d.get(key) and d[key] > 0]
    if not points:
        raise ValueError("nothing to plot")
    front = {id(one) for one in pareto([one for one, _ in points], cost=cost)}
    several = len(hosts_of([one for one, _ in points])) > 1
    each = f" {UNITS.get(cost, cost)}" + (" per question" if key != cost else "")

    wide, tall, pad = 900, 520, 70
    costs = [d[key] for _, d in points]
    lo, hi = 0.0, max(costs) * 1.08
    best = max(d["right"] for _, d in points)
    top = min(1.0, best * 1.15)

    def x(v: float) -> float:
        return pad + (v - lo) / (hi - lo or 1) * (wide - 2 * pad)

    def y(v: float) -> float:
        return tall - pad - (v / (top or 1)) * (tall - 2 * pad)

    marks, dots = [], []
    for one, d in sorted(points, key=lambda kv: kv[1][key]):
        cx, cy = x(d[key]), y(d["right"])
        on = id(one) in front
        label = _shown(f"{one.get('label', '')}@{host_of(one) or '?'}" if several
                       else one.get("label", ""), 28)
        kind = "composed" if one.get("composed") else ("front" if on else "dot")
        note = (f'\ncomposed: cost from {one.get("from") or "its own run"}'
                if one.get("composed") else "")
        if several:
            note += f'\nhost: {host_of(one) or "?"}'
        dots.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="{6 if on else 4.5}" '
            f'class="{kind}{" front" if on and kind == "composed" else ""}"><title>{label}\n'
            f'{100 * d["right"]:.0f}% right, {d[key]:.1f}{each}\n'
            f'{d["questions"]:.0f} questions{note}</title></circle>')
        if on:
            marks.append((cx, cy))
            dots.append(f'<text x="{cx + 9:.1f}" y="{cy - 8:.1f}" class="tag">{label}</text>')
    line = ("<polyline class=\"edge\" points=\""
            + " ".join(f"{a:.1f},{b:.1f}" for a, b in sorted(marks)) + "\"/>") if marks else ""

    ticks = []
    for n in range(5):
        v = lo + (hi - lo) * n / 4
        ticks.append(f'<line class="grid" x1="{x(v):.1f}" y1="{y(0):.1f}" '
                     f'x2="{x(v):.1f}" y2="{y(top):.1f}"/>'
                     f'<text class="ax" x="{x(v):.1f}" y="{y(0) + 20:.1f}" '
                     f'text-anchor="middle">{v:.0f}</text>')
        r = top * n / 4
        ticks.append(f'<line class="grid" x1="{x(lo):.1f}" y1="{y(r):.1f}" '
                     f'x2="{x(hi):.1f}" y2="{y(r):.1f}"/>'
                     f'<text class="ax" x="{x(lo) - 10:.1f}" y="{y(r) + 4:.1f}" '
                     f'text-anchor="end">{100 * r:.0f}%</text>')

    out = Path(where)
    out.write_text(f"""<title>Answering the graph: accuracy against {AXES.get(cost, cost)}</title>
<style>
  :root {{ --ink:#1b1b1f; --thin:#d8d8de; --front:#1a6b4a; --dot:#8a8a95; --paper:#fbfbfd; }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
    --ink:#e9e9ef; --thin:#33333c; --front:#5fd3a0; --dot:#6f6f7c; --paper:#141418; }} }}
  :root[data-theme="dark"] {{ --ink:#e9e9ef; --thin:#33333c; --front:#5fd3a0;
    --dot:#6f6f7c; --paper:#141418; }}
  body {{ background: var(--paper); color: var(--ink); margin: 0; padding: 24px;
          font: 14px/1.5 ui-sans-serif, system-ui, -apple-system, sans-serif; }}
  h1 {{ font-size: 17px; margin: 0 0 4px; }}
  p  {{ margin: 0 0 18px; color: var(--dot); max-width: 62ch; }}
  .wrap {{ overflow-x: auto; }}
  .grid {{ stroke: var(--thin); stroke-width: 1; }}
  .ax   {{ fill: var(--dot); font-size: 11px; }}
  .dot  {{ fill: var(--dot); }}
  .front{{ fill: var(--front); }}
  .composed {{ fill: none; stroke: var(--ink); stroke-width: 2; }}
  .composed.front {{ stroke: var(--front); }}
  .edge {{ fill: none; stroke: var(--front); stroke-width: 2; stroke-dasharray: 5 4; }}
  .tag  {{ fill: var(--ink); font-size: 11px; }}
</style>
<h1>Answering the graph: accuracy against {AXES.get(cost, cost)}</h1>
<p>Each point is one benchmark run; a ring is one model composed &mdash; accuracy from its
largest run, cost from its fastest run that held that accuracy. Green points are the Pareto
frontier &mdash; nothing is both more accurate and cheaper &mdash; so choosing among them is
choosing a budget, not choosing a better run. Hover a point for its numbers.</p>
<div class="wrap"><svg width="{wide}" height="{tall}" viewBox="0 0 {wide} {tall}"
  role="img" aria-label="accuracy against {cost}">
  {"".join(ticks)}
  {line}
  {"".join(dots)}
  <text class="ax" x="{wide / 2:.0f}" y="{tall - 14}" text-anchor="middle">
    {AXES.get(cost, cost)} &mdash; lower is better</text>
</svg></div>
""", encoding="utf-8")
    return str(out)
