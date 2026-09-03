"""The manim scene behind ``ml-stack-bench animate``: draws the tables ``animate`` builds.

Only ``Text`` is used for lettering, never ``Tex``, so no LaTeX is needed to render.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import manimpango
from manim import (DOWN, LEFT, RIGHT, UP, AnimationGroup, Create, DashedVMobject, Dot, FadeIn,
                   FadeOut, GrowFromEdge, LaggedStart, Line, Rectangle, Scene, Square, Text,
                   ValueTracker, VGroup, VMobject, always_redraw, tempconfig)

from . import animate as tables

BG = "#0F1117"
INK = "#F2F2F5"
MUTED = "#8A8F9C"
FAINT = "#2A2E3A"

# the frame is 14.22 x 8 units, centred on the origin
LEFT_X, RIGHT_X = -6.4, 6.4
TOP_Y, BOTTOM_Y = 3.6, -3.6

FONTS = ["Helvetica Neue", "Inter", "Avenir Next", "Helvetica", "Arial", "DejaVu Sans",
         "Liberation Sans"]


def font() -> str:
    """The first of ``FONTS`` this machine has, or '' for Pango's default."""
    have = set(manimpango.list_fonts())
    return next((f for f in FONTS if f in have), "")


FONT = font()


def _text(s: str, size: float = 24, colour: str = INK, **kw: Any) -> Text:
    return Text(s, font_size=size, color=colour, font=FONT, **kw)


def _fit(m: VMobject, width: float) -> VMobject:
    if m.width > width:
        m.scale_to_fit_width(width)
    return m


def _swatch(colour: str, side: float = 0.22) -> Square:
    return Square(side_length=side, fill_color=colour, fill_opacity=1, stroke_width=0)


def _clear(m: VMobject, others: Sequence[VMobject], pad: float = 0.1) -> bool:
    for o in others:
        if (m.get_left()[0] < o.get_right()[0] + pad and m.get_right()[0] > o.get_left()[0] - pad
                and m.get_bottom()[1] < o.get_top()[1] + pad
                and m.get_top()[1] > o.get_bottom()[1] - pad):
            return False
    return True


def _placed(tag: VMobject, dot: VMobject, others: Sequence[VMobject], left: float,
            right: float, bottom: float, top: float) -> VMobject:
    """``tag`` beside ``dot`` in the first of eight positions clear of ``others`` and inside
    the box; the last position when none is."""
    for side in (RIGHT + UP * 0.6, RIGHT + DOWN * 0.6, LEFT + UP * 0.6, LEFT + DOWN * 0.6,
                 UP, DOWN, RIGHT + UP * 1.6, LEFT + UP * 1.6):
        tag.next_to(dot, side, buff=0.15)
        inside = (tag.get_left()[0] >= left - 0.9 and tag.get_right()[0] <= right + 0.3
                  and tag.get_bottom()[1] >= bottom - 0.3 and tag.get_top()[1] <= top + 0.4)
        if inside and _clear(tag, others):
            break
    return tag


class Comparison(Scene):
    """Every scene of the plan, one after another, in ``seconds`` seconds."""

    def __init__(self, doc: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]],
                 png: Path | None = None, **kw: Any) -> None:
        super().__init__(**kw)
        self.doc = doc
        self.scenes = list(scenes)
        self.png = png
        self.colours = tables.colours(doc)

    # -- the run ----------------------------------------------------------------------------

    def construct(self) -> None:
        self.camera.background_color = BG
        draw = {"title": self.title, "decode": self.speed, "prefill": self.speed,
                "ttft": self.speed, "concurrency": self.speed, "memory": self.memory,
                "graph": self.graph, "calls": self.calls, "standard": self.standard,
                "closing": self.closing}
        for scene in self.scenes:
            self.budget = float(scene["seconds"])
            draw[scene["key"]](scene["key"])
            if scene is not self.scenes[-1]:
                self.play(FadeOut(*self.mobjects), run_time=min(0.4, self.budget * 0.1))
                self.clear()
        self.wait(0.1)
        if self.png:
            self.renderer.update_frame(self)
            self.camera.get_image().save(str(self.png))

    def _hold(self, share: float) -> None:
        self.wait(max(0.05, self.budget * share))

    # -- shared pieces ----------------------------------------------------------------------

    def _header(self, title: str, sub: str = "") -> VGroup:
        head = _fit(_text(title, 36), 12.0).move_to([LEFT_X, TOP_Y, 0], aligned_edge=LEFT + UP)
        group = VGroup(head)
        if sub:
            group.add(_fit(_text(sub, 20, MUTED), 12.0)
                      .next_to(head, DOWN, buff=0.12, aligned_edge=LEFT))
        return group

    def _legend(self, labels: Sequence[str] | None = None) -> VGroup:
        labels = list(labels or self.colours)
        rows = VGroup()
        for label in labels:
            row = VGroup(_swatch(self.colours[label]), _fit(_text(label, 18), 5.6))
            row.arrange(RIGHT, buff=0.15)
            rows.add(row)
        per_row = 2 if len(labels) > 1 else 1
        lines = VGroup(*[VGroup(*rows[i:i + per_row]).arrange(RIGHT, buff=0.7,
                                                                aligned_edge=DOWN)
                         for i in range(0, len(rows), per_row)])
        lines.arrange(DOWN, buff=0.15, aligned_edge=LEFT)
        for line in lines:
            for row in line:
                row.align_to(line, DOWN)
        return lines.move_to([LEFT_X, BOTTOM_Y, 0], aligned_edge=LEFT + DOWN)

    def _bars(self, panel: Mapping[str, Any], left: float, right: float, bottom: float,
              top: float, *, grow: float, label_size: float = 20) -> None:
        """Grouped bars for ``panel`` inside the box, with counting labels."""
        scale = float(panel["scale"])
        height = top - bottom
        unit = panel["unit"]

        static = VGroup()
        for n in (0, 0.5, 1):
            y = bottom + n * height
            static.add(Line([left, y, 0], [right, y, 0], stroke_color=FAINT if n else MUTED,
                            stroke_width=1 if n else 2))
            static.add(_text(f"{n * scale:g}", 14, MUTED).next_to([left, y, 0], LEFT, buff=0.12))
        static.add(_text(unit, 14, MUTED).next_to([left, top, 0], UP, buff=0.15)
                   .align_to([left - 0.9, 0, 0], LEFT))
        line = panel.get("line")
        if line and line.get("value") is not None:
            y = bottom + float(line["value"]) / scale * height
            static.add(Line([left, y, 0], [right, y, 0], stroke_color=MUTED, stroke_width=1.5)
                       .set_opacity(0.7))
            static.add(_text(line["label"], 14, MUTED)
                       .next_to([right, y, 0], UP, buff=0.06).align_to([right, 0, 0], RIGHT))

        groups = panel["groups"]
        slot = (right - left) / max(1, len(groups))
        per = max(1, len(groups[0]["bars"])) if groups else 1
        gap = min(0.12, slot * 0.05)
        width = min(1.2, (slot * 0.78 - gap * (per - 1)) / per)
        span = per * width + (per - 1) * gap

        grown, labels, blanks = [], [], []
        for g, group in enumerate(groups):
            cx = left + slot * (g + 0.5)
            x0 = cx - span / 2
            if len(groups) > 1 or group["label"] not in ("peak", "load", "calls"):
                static.add(_fit(_text(group["label"], 16, MUTED), slot * 0.95)
                           .next_to([cx, bottom, 0], DOWN, buff=0.18))
            for i, bar in enumerate(group["bars"]):
                bx = x0 + i * (width + gap) + width / 2
                if bar["value"] is None:
                    ghost = Rectangle(width=width, height=0.4, stroke_color=MUTED,
                                      stroke_width=1.2, fill_opacity=0)
                    ghost = DashedVMobject(ghost, num_dashes=18)
                    ghost.move_to([bx, bottom, 0], aligned_edge=DOWN)
                    note = _fit(_text("not\nmeasured", 14, MUTED, line_spacing=0.7),
                                width * 1.3).next_to(ghost, UP, buff=0.08)
                    blanks.append(VGroup(ghost, note))
                    continue
                value = float(bar["value"])
                h = max(0.02, value / scale * height)
                rect = Rectangle(width=width, height=h, fill_color=bar["colour"],
                                 fill_opacity=0.92, stroke_width=0)
                rect.move_to([bx, bottom, 0], aligned_edge=DOWN)
                places = len(bar["shown"].split(".")[1]) if "." in bar["shown"] else 0
                tracker = ValueTracker(0.0)
                grown.append(AnimationGroup(GrowFromEdge(rect, DOWN),
                                            tracker.animate.set_value(value)))
                label = always_redraw(
                    lambda t=tracker, p=places, r=rect, w=width:
                    _fit(_text(f"{t.get_value():.{p}f}", label_size), w * 1.3)
                    .next_to(r, UP, buff=0.08).set_opacity(1 if t.get_value() > 0 else 0))
                labels.append(label)

        self.add(static, *blanks)
        self.add(*labels)
        if grown:
            self.play(LaggedStart(*grown, lag_ratio=0.5 / len(grown)), run_time=grow)
            for label in labels:
                label.clear_updaters()

    # -- scenes -----------------------------------------------------------------------------

    def title(self, key: str) -> None:
        card = tables.title_card(self.doc)
        head = _fit(_text(card["title"], 60), 12.5).move_to([0, 2.2, 0])
        where = _fit(_text(" · ".join(p for p in (card["machine"], card["date"]) if p), 26,
                           MUTED), 12.0).next_to(head, DOWN, buff=0.35)
        rows = VGroup()
        for cfg, line in zip(self.doc["configs"], card["configs"]):
            row = VGroup(_swatch(self.colours[cfg["label"]], 0.3), _fit(_text(line, 28), 10.5))
            rows.add(row.arrange(RIGHT, buff=0.25))
        rows.arrange(DOWN, buff=0.35, aligned_edge=LEFT).next_to(where, DOWN, buff=0.9)
        if rows.get_bottom()[1] < BOTTOM_Y:
            rows.scale_to_fit_height(where.get_bottom()[1] - 0.9 - BOTTOM_Y)
        self.play(FadeIn(head, shift=UP * 0.3), run_time=self.budget * 0.25)
        self.play(FadeIn(where), run_time=self.budget * 0.12)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.5),
                  run_time=self.budget * 0.33)
        self._hold(0.3)

    def speed(self, key: str) -> None:
        panel = next(p for p in tables.speed_panels(self.doc) if p["key"] == key)
        sub = {"decode": "tokens generated per second, one stream",
               "prefill": "prompt tokens read per second, one stream",
               "ttft": "seconds until the first token, one stream",
               "concurrency": "decode tokens per second as streams are added"}[key]
        if panel.get("note"):
            sub += f" · {panel['note']}"
        self.add(self._header(panel["title"], sub), self._legend())
        self._bars(panel, LEFT_X + 0.9, RIGHT_X, -2.0, 1.9, grow=self.budget * 0.5)
        self._hold(0.4)

    def memory(self, key: str) -> None:
        peak, load = tables.memory_panels(self.doc)
        self.add(self._header("Memory", "peak while answering, and seconds to load"),
                 self._legend())
        mid = LEFT_X + (RIGHT_X - LEFT_X) * 0.58
        self.add(_text("Peak memory (GB)", 20).move_to([(LEFT_X + 0.9 + mid) / 2, 2.25, 0]))
        self.add(_text("Load time (s)", 20).move_to([(mid + 1.4 + RIGHT_X) / 2, 2.25, 0]))
        self._bars(peak, LEFT_X + 0.9, mid - 0.4, -2.0, 1.9, grow=self.budget * 0.35)
        self._bars(load, mid + 1.4, RIGHT_X, -2.0, 1.9, grow=self.budget * 0.25)
        self._hold(0.3)

    def graph(self, key: str) -> None:
        sc = tables.graph_scatter(self.doc)
        self.add(self._header("Answering the graph",
                              "F1 against wall clock per question — up and left is better"),
                 self._legend())
        left, right, bottom, top = LEFT_X + 1.1, RIGHT_X - 0.6, -2.0, 1.9
        xs, ys = float(sc["x_scale"]), float(sc["y_scale"])

        def at(x: float, y: float) -> list[float]:
            return [left + x / xs * (right - left), bottom + y / ys * (top - bottom), 0]

        axes = VGroup(Line(at(0, 0), at(xs, 0), stroke_color=MUTED, stroke_width=2),
                      Line(at(0, 0), at(0, ys), stroke_color=MUTED, stroke_width=2))
        for n in range(5):
            x, y = xs * n / 4, ys * n / 4
            if n:
                axes.add(Line(at(x, 0), at(x, ys), stroke_color=FAINT, stroke_width=1),
                         Line(at(0, y), at(xs, y), stroke_color=FAINT, stroke_width=1))
            axes.add(_text(f"{x:g}", 14, MUTED).next_to(at(x, 0), DOWN, buff=0.12),
                     _text(f"{y:.2f}", 14, MUTED).next_to(at(0, y), LEFT, buff=0.12))
        axes.add(_text(sc["x_label"], 16, MUTED).next_to(at(xs, 0), DOWN, buff=0.45)
                 .align_to([right, 0, 0], RIGHT),
                 _text(sc["y_label"], 16, MUTED).next_to(at(0, ys), UP, buff=0.15))
        self.add(axes)

        dots, tags = [], []
        for p in sc["points"]:
            dot = Dot(at(p["x"], p["y"]), radius=0.13, color=p["colour"])
            tag = VGroup(_fit(_text(p["config"], 16), 4.2),
                         _text(p["shown"], 14, MUTED)).arrange(DOWN, buff=0.05,
                                                                aligned_edge=LEFT)
            dots.append(dot)
            tags.append(_placed(tag, dot, dots[:-1] + tags, left, right, bottom, top))
        self.play(LaggedStart(*[FadeIn(d, scale=0.3) for d in dots], lag_ratio=0.4),
                  run_time=self.budget * 0.2)
        self.play(LaggedStart(*[FadeIn(t) for t in tags], lag_ratio=0.3),
                  run_time=self.budget * 0.15)
        front = sc["frontier"]
        if len(front) > 1:
            path = VMobject(stroke_color=INK, stroke_width=2.5)
            path.set_points_as_corners([at(p["x"], p["y"]) for p in front])
            self.play(Create(DashedVMobject(path, num_dashes=40)), run_time=self.budget * 0.2)
        if front:
            ring = VGroup(*[Dot(at(p["x"], p["y"]), radius=0.22, color=INK, fill_opacity=0,
                                stroke_width=2, stroke_color=INK) for p in front])
            note = _text("ringed: on the Pareto frontier — nothing is both more accurate "
                         "and faster", 14, MUTED)
            note.next_to(self._legend(), UP, buff=0.25).align_to([LEFT_X, 0, 0], LEFT)
            self.play(FadeIn(ring), FadeIn(note), run_time=self.budget * 0.1)
        if sc["not_measured"]:
            missing = _fit(_text("not on the graph bench: " + ", ".join(sc["not_measured"]),
                                 14, MUTED), 12.0)
            missing.move_to([RIGHT_X, TOP_Y - 0.95, 0], aligned_edge=RIGHT + UP)
            self.play(FadeIn(missing), run_time=self.budget * 0.05)
        self._hold(0.3)

    def calls(self, key: str) -> None:
        panel = tables.calls_panel(self.doc)
        self.add(self._header("Tool calls per question", "fewer calls is less to wait for"),
                 self._legend())
        self._bars(panel, LEFT_X + 0.9, RIGHT_X, -2.0, 1.9, grow=self.budget * 0.5)
        self._hold(0.4)

    def standard(self, key: str) -> None:
        grid = tables.standard_grid(self.doc)
        self.add(self._header("Standard sets", "share of the set answered right"))
        sets, rows = grid["sets"], grid["rows"]
        name_w = 4.6
        cell_w = (RIGHT_X - LEFT_X - name_w) / max(1, len(sets))
        row_h = min(1.1, (2.3 - -3.4) / (len(rows) + 1))
        y0 = 2.0
        header = VGroup()
        for j, name in enumerate(sets):
            cx = LEFT_X + name_w + cell_w * (j + 0.5)
            header.add(_fit(_text(name, 20, MUTED), cell_w * 0.9).move_to([cx, y0, 0]))
        self.add(header)
        names, cells = [], []
        for i, row in enumerate(rows):
            y = y0 - row_h * (i + 1)
            name = VGroup(_swatch(row["colour"]), _fit(_text(row["config"], 18), name_w - 0.6))
            name.arrange(RIGHT, buff=0.15).move_to([LEFT_X, y, 0], aligned_edge=LEFT)
            names.append(name)
            self.add(Line([LEFT_X, y + row_h / 2, 0], [RIGHT_X, y + row_h / 2, 0],
                          stroke_color=FAINT, stroke_width=1))
            for j, cell in enumerate(row["cells"]):
                cx = LEFT_X + name_w + cell_w * (j + 0.5)
                if cell["score"] is None:
                    m = _text(tables.NOT_MEASURED, 13, MUTED).move_to([cx, y, 0])
                else:
                    m = VGroup(_text(cell["shown"], 30, row["colour"]))
                    if cell.get("n"):
                        m.add(_text(f"n = {cell['n']}", 12, MUTED))
                    m.arrange(DOWN, buff=0.05).move_to([cx, y, 0])
                cells.append(m)
        self.play(LaggedStart(*[FadeIn(n, shift=RIGHT * 0.2) for n in names], lag_ratio=0.3),
                  run_time=self.budget * 0.2)
        self.play(LaggedStart(*[FadeIn(c, scale=0.6) for c in cells],
                              lag_ratio=0.8 / max(1, len(cells))),
                  run_time=self.budget * 0.45)
        self._hold(0.3)

    def closing(self, key: str) -> None:
        card = tables.closing(self.doc)
        head = _fit(_text(card["title"], 44), 12.5).move_to([0, 2.9, 0])
        rows = VGroup()
        for line in card["lines"]:
            name = VGroup(_swatch(line["colour"], 0.26), _fit(_text(line["config"], 26), 11.5))
            name.arrange(RIGHT, buff=0.22)
            numbers = _fit(_text(line["numbers"], 22, MUTED), 11.5)
            rows.add(VGroup(name, numbers).arrange(DOWN, buff=0.1, aligned_edge=LEFT))
        rows.arrange(DOWN, buff=0.45, aligned_edge=LEFT).next_to(head, DOWN, buff=0.7)
        rows.align_to([LEFT_X + 0.4, 0, 0], LEFT)
        footer = _fit(_text(card["footer"], 22, MUTED), 12.0).move_to([0, BOTTOM_Y + 0.2, 0])
        if rows.get_bottom()[1] < footer.get_top()[1] + 0.3:
            rows.scale_to_fit_height(head.get_bottom()[1] - 0.7 - footer.get_top()[1] - 0.3)
            rows.next_to(head, DOWN, buff=0.7).align_to([LEFT_X + 0.4, 0, 0], LEFT)
        self.play(FadeIn(head), run_time=self.budget * 0.15)
        self.play(LaggedStart(*[FadeIn(r, shift=RIGHT * 0.3) for r in rows], lag_ratio=0.5),
                  run_time=self.budget * 0.4)
        self.play(FadeIn(footer), run_time=self.budget * 0.15)
        self._hold(0.3)


def render(doc: Mapping[str, Any], scenes: Sequence[Mapping[str, Any]], *, out: Path,
           png: Path | None, quality: str, work: Path | None = None) -> Path:
    """Render ``scenes`` of ``doc`` to ``out`` (mp4) and the last frame to ``png``."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if png:
        Path(png).parent.mkdir(parents=True, exist_ok=True)
    keep = tempfile.TemporaryDirectory() if work is None else None
    media = Path(work) if work else Path(keep.name)  # type: ignore[union-attr]
    try:
        with tempconfig({"quality": quality, "output_file": out.stem, "media_dir": str(media),
                         "disable_caching": True, "progress_bar": "none",
                         "verbosity": "ERROR", "background_color": BG, "preview": False,
                         "write_to_movie": True, "flush_cache": True}):
            scene = Comparison(doc, scenes, png=Path(png) if png else None)
            scene.render()
            made = Path(scene.renderer.file_writer.movie_file_path)
            shutil.copyfile(made, out)
    finally:
        if keep:
            keep.cleanup()
    return out

