"""Drawing what fits: two panels over the measured records, who fits and what it costs."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from ml_stack.hub import pretty_name
from ml_stack.serve.fit import Fit
from ml_stack.units import human_bytes

__all__ = ["COMMON_VRAM_GB", "PLOT_CONTEXTS", "READ_CONTEXTS", "SLIDER_CONTEXTS",
           "label_of", "plot"]

# Where the first panel's x axis runs: 2k is a question with its tools, 128k is a whole
# conversation kept open, and everything interesting about a cache happens between them.
PLOT_CONTEXTS: tuple[int, ...] = (2048, 4096, 8192, 16384, 32768, 65536, 131072)

# Where the fit view's context slider stands, 1k to 256k by doubling.
SLIDER_CONTEXTS: tuple[int, ...] = tuple(1024 * 2 ** k for k in range(9))

# The contexts a slot count is worked out at, eight to the octave over the same range, so a
# reader dragging across the chart reads a measured answer rather than one worked out again
# in the browser.
READ_CONTEXTS: tuple[int, ...] = tuple(round(1024 * 2 ** (k / 8)) for k in range(65))

# One line style per room asked about, so a model keeps its colour across all of them and
# the rooms are told apart by the line rather than by a second set of colours.
_ROOM_STYLES = ("-", "--", ":", "-.")

# The cards and machines people actually have, in GB, drawn faintly behind the second panel.
# A chart that only knows about *this* machine answers "will it fit here"; these are what
# turn it into "and what would I need". Nothing is measured about them -- they are gridlines
# with names on.
COMMON_VRAM_GB: tuple[int, ...] = (6, 8, 12, 16, 24, 32, 48, 64, 96, 128)


def _pyplot():
    """matplotlib, or a refusal that says how to get it.

    Optional the way every other heavy dependency here is optional -- `torch_ops`,
    `vision.payloads` both raise with the install line rather than failing on
    an ImportError somebody has to interpret. Note that `ml-stack-bench show --plot` is
    *not* the precedent it looks like: that one writes hand-built SVG with no library at
    all, on purpose, because it has to open on a machine with no packages. A PNG cannot be
    written that way, which is why this one has a dependency and that one does not.
    """

    try:
        import matplotlib

        matplotlib.use("Agg")           # a file, never a window: nothing here is interactive
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - the install line is the whole message
        raise RuntimeError(
            "drawing the fit chart needs matplotlib: pip install 'ml-stack[plot]'") from exc
    return plt


def label_of(fit: Fit) -> str:
    """What a line is called in the legend: the model, what its cache is stored as, and
    whether it was measured with a draft head -- the three things that make two records of
    the same weights different measurements."""

    parts = [pretty_name(fit.model)]
    if fit.cache_type and fit.cache_type != "f16":
        parts.append(fit.cache_type)
    if fit.spec:
        parts.append("+draft")
    return " ".join(parts)


def _k(tokens: int) -> str:
    """32768 -> `32k`, for a legend that has to say a context without spending a line on it."""
    return f"{tokens // 1024}k" if tokens >= 1024 and tokens % 1024 == 0 else f"{tokens:,}"


def _rooms_of(fits: Sequence[Fit], rooms: Sequence[int]) -> list[int]:
    """The rooms to draw, in the order asked for. None asked means whatever the records
    were recorded against, which is this machine."""
    asked = [int(r) for r in rooms if int(r) > 0]
    if asked:
        return asked
    return [max((f.room for f in fits), default=0)] if fits else [0]


def plot(fits: Iterable[Fit], where: str | Path, *, rooms: Sequence[int] = (),
         at: int = 32768, contexts: Sequence[int] = PLOT_CONTEXTS,
         machine: str = "") -> str:
    """Two panels over the measured records: who fits, and what it costs.

    The panels answer the same question from the two ends, and the second is the one worth
    having. A model is usually chosen by its weights, and the weights are the part that does
    *not* grow: a 30B whose cache costs 4 KiB a token overtakes an 8B whose cache costs 32
    KiB somewhere around the fourth or fifth user, and no table of per-model numbers makes
    that crossing visible. Drawn as memory against users at one context, it is where the
    lines cross.

    1. **Users that fit against the per-user context** (log2 x, log y): one line per record,
       one line style per room. A record whose weights do not leave room for a single cache
       draws nothing and says "does not fit" in the legend, which is a fact about the
       machine rather than an empty space to be interpreted.
    2. **Memory against users** at ``at`` tokens each: a straight line per record, starting
       at zero users -- where the height is the model with an empty cache -- and climbing
       by `cost(at)` per user. The rooms in force are drawn across it, and the familiar card
       sizes (`COMMON_VRAM_GB`) faintly behind, so the chart answers "what would I need"
       as well as "does it fit here". Where a line leaves a room is the last user that
       machine holds. `Fit.line(context)` is the pair of numbers each line is drawn from.

    ``where`` names the format: `.png`, `.svg`, `.pdf`. Returns the path written.
    """
    plt = _pyplot()

    rows = [f for f in fits]
    if not rows:
        raise ValueError("nothing to plot: no model has been measured")
    out = Path(where).expanduser()
    if out.suffix.lower() not in {".png", ".svg", ".pdf"}:
        raise ValueError(f"cannot draw a {out.suffix or 'nameless'} file; "
                         "name it .png, .svg or .pdf")
    spans = [int(c) for c in contexts if int(c) > 0] or list(PLOT_CONTEXTS)
    spans.sort()
    drawn_rooms = _rooms_of(rows, rooms)
    at = max(1, int(at))

    figure, (left, right) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    colours = plt.rcParams["axes.prop_cycle"].by_key().get("color") or ["C0"]

    # -- panel one: who fits, at every context, in every room -------------------------
    for index, fit in enumerate(rows):
        colour = colours[index % len(colours)]
        drew = False
        for style, room in zip(_ROOM_STYLES, drawn_rooms):
            here = fit.at_room(room)
            points = [(c, here.users(c)) for c in spans]
            points = [(c, n) for c, n in points if n > 0]
            if not points:
                continue
            drew = True
            left.plot([c for c, _ in points], [n for _, n in points], style, color=colour,
                      marker="o", markersize=3.5,
                      label=(f"{label_of(fit)} ({human_bytes(fit.loaded())} loaded)"
                             if style == _ROOM_STYLES[0] else None))
        if not drew:
            left.plot([], [], "-", color=colour,
                      label=f"{label_of(fit)} ({human_bytes(fit.loaded())}) -- does not fit")

    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xticks(spans)
    left.set_xticklabels([f"{c // 1024}k" for c in spans])
    left.set_xlabel("context each user gets (tokens)")
    left.set_ylabel("users that fit")
    left.set_title("How many fit, and at what context")
    left.grid(True, which="both", alpha=0.25)
    people = left.legend(fontsize=8, loc="upper right")
    if len(drawn_rooms) > 1:
        left.add_artist(people)
        left.legend(handles=[plt.Line2D([], [], color="0.35", linestyle=style,
                                        label=f"{human_bytes(room)} of room")
                             for style, room in zip(_ROOM_STYLES, drawn_rooms)],
                    fontsize=8, loc="lower left")

    # -- panel two: what it costs as the users arrive ---------------------------------
    #
    # Every line here is `loaded() + users * cost(at)` -- a straight line from the model
    # sitting there with an empty cache, climbing by one user's worth of cache each step.
    # Starting at zero users is the point: the intercept is the part a person already knows
    # (the weights) and the slope is the part nobody does, and the crossing between a heavy
    # model with a cheap cache and a light one with an expensive cache is only visible when
    # both are drawn from the axis rather than from the first user.
    most = max((f.at_room(max(drawn_rooms)).users(at) for f in rows), default=0)
    most = min(max(most, 8), 256)
    people_axis = list(range(0, most + 1))
    gb = float(1024 ** 3)
    reach = 0.0
    for index, fit in enumerate(rows):
        colour = colours[index % len(colours)]
        intercept, each = fit.line(at)
        heights = [(intercept + n * each) / gb for n in people_axis]
        reach = max(reach, heights[-1])
        right.plot(people_axis, heights, "-", color=colour, linewidth=1.8,
                   label=f"{label_of(fit)}: {human_bytes(intercept)} + "
                         f"{each / gb:.2f}G/user at {_k(at)}")

    # A little above whichever is higher: the rooms actually being asked about, or the
    # largest familiar card any of these lines climbs past. Below that and a crossing is
    # cut off the top; far above it and every line is squashed into the bottom inch.
    crossed = [size for size in COMMON_VRAM_GB if size <= reach]
    top = max([room / gb for room in drawn_rooms] + [max(crossed, default=0)]) * 1.08
    right.set_xlim(0, most)
    right.set_ylim(0, top or 1.0)

    for size in COMMON_VRAM_GB:
        if size > top:
            continue
        right.axhline(size, color="0.75", linewidth=0.8, zorder=0)
        right.annotate(f"{size}G", xy=(most, size), xytext=(-3, 2),
                       textcoords="offset points", fontsize=7, color="0.55",
                       va="bottom", ha="right", zorder=0)
    for style, room in zip(_ROOM_STYLES, drawn_rooms):
        right.axhline(room / gb, color="#b0413e", linestyle=style, linewidth=1.4)
        right.annotate(f"{human_bytes(room)} of room", xy=(0, room / gb), xytext=(4, 3),
                       textcoords="offset points", fontsize=8, color="#b0413e",
                       va="bottom", ha="left")

    right.set_xlabel(f"users, each with {at:,} tokens")
    right.set_ylabel("memory needed (GB)")
    right.set_title("What it costs as they arrive")
    right.grid(True, axis="x", alpha=0.25)
    right.legend(fontsize=8, loc="upper left")

    builds = sorted({f.build for f in rows if f.build})
    figure.suptitle(
        (machine or "this machine") + f" -- {human_bytes(drawn_rooms[0])} of room"
        + (f", measured on llama.cpp {', '.join(builds)}" if builds else ""),
        fontsize=11)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=140)
    plt.close(figure)
    return str(out)
