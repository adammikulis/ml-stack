/* <ml-sparkline>: a small line of recent values. */
import { MlElement, define, s } from "./base.js";
import { fmt } from "./format.js";
import { normalize } from "./verdict.js";

const STYLES = `
:host { display: inline-block; vertical-align: middle; }
svg { display: block; overflow: visible; }
.line { fill: none; stroke: var(--v, var(--ml-accent)); stroke-width: 1.6; stroke-linejoin: round;
  stroke-linecap: round; vector-effect: non-scaling-stroke; }
.area { fill: color-mix(in srgb, var(--v, var(--ml-accent)) 16%, transparent); stroke: none; }
.dot { fill: var(--v, var(--ml-accent)); }
.none { stroke: var(--ml-line-strong); stroke-width: 1; stroke-dasharray: 3 3; }
`;

class MlSparkline extends MlElement {
  static props = { values: "object", width: "number", height: "number", min: "number",
                   max: "number", verdict: "string", label: "string", format: "string",
                   unit: "string", area: "bool" };
  static styles = STYLES;

  build() {
    this.svg = s("svg", { role: "img" });
    this.root.append(this.svg);
  }

  update() {
    const w = this.width || 120, hgt = this.height || 28, pad = 3;
    const all = (Array.isArray(this.values) ? this.values : []).map(Number);
    const pts = all.filter((n) => Number.isFinite(n));
    this.svg.setAttribute("viewBox", `0 0 ${w} ${hgt}`);
    this.svg.setAttribute("width", String(w));
    this.svg.setAttribute("height", String(hgt));
    const v = this.verdict ? normalize(this.verdict) : "";
    this.style.setProperty("--v", v ? `var(--ml-${v === "none" ? "line-strong" : v})` : "");
    if (!pts.length) {
      this.svg.setAttribute("aria-label", `${this.label || "trend"}: no data`.trim());
      this.svg.replaceChildren(s("line", { class: "none", x1: pad, x2: w - pad,
                                           y1: hgt / 2, y2: hgt / 2 }));
      return;
    }
    const lo = this.min ?? Math.min(...pts), top = this.max ?? Math.max(...pts);
    const span = top - lo || 1;
    const x = (i) => (pts.length === 1 ? w / 2 : pad + (i / (pts.length - 1)) * (w - 2 * pad));
    const y = (n) => hgt - pad - ((Math.min(top, Math.max(lo, n)) - lo) / span) * (hgt - 2 * pad);
    const d = pts.map((n, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(n).toFixed(1)}`).join("");
    const last = pts[pts.length - 1];
    const kids = [];
    if (this.area && pts.length > 1) {
      kids.push(s("path", { class: "area",
        d: `${d}L${x(pts.length - 1).toFixed(1)},${hgt - pad}L${x(0).toFixed(1)},${hgt - pad}Z` }));
    }
    if (pts.length > 1) kids.push(s("path", { class: "line", d }));
    kids.push(s("circle", { class: "dot", cx: x(pts.length - 1).toFixed(1), cy: y(last).toFixed(1), r: 2.4 }));
    this.svg.replaceChildren(...kids);
    const f = (n) => fmt(n, this.format || "number", this.unit);
    this.svg.setAttribute("aria-label",
      `${this.label || "trend"}: latest ${f(last)}, low ${f(Math.min(...pts))}, high ${f(Math.max(...pts))}, ${pts.length} points`);
  }
}
define("ml-sparkline", MlSparkline);
