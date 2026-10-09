/* <ml-meter>: a resource bar of labelled segments against a capacity, with a verdict. */
import "./chip.js";
import { MlElement, define, h, uid } from "./base.js";
import { clamp, fmt } from "./format.js";
import { LABELS, THRESHOLDS, normalize, verdictOf } from "./verdict.js";

const STYLES = `
.head { display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
  font-size: 12px; color: var(--ml-muted); margin-bottom: 5px; }
.label { overflow-wrap: anywhere; flex: 1 1 40%; min-width: 0; }
.caption { display: inline-flex; align-items: center; justify-content: flex-end; flex: 0 1 auto;
  flex-wrap: wrap; max-width: 62%; overflow-wrap: anywhere; gap: 4px 8px; text-align: right;
  color: var(--ml-text); font-family: var(--ml-mono); font-weight: 600; }
.track { position: relative; height: 9px; background: var(--ml-sunken); border-radius: 999px;
  border: 1px solid var(--ml-line-strong); overflow: hidden; display: flex; }
.seg { height: 100%; flex: none; min-width: 0; background: var(--ml-series-1);
  transition: width 1.2s cubic-bezier(.4, 0, .2, 1), background-color .3s; }
.seg + .seg { box-shadow: -2px 0 0 var(--ml-sunken); }
.seg.t1 { background: var(--ml-series-1); } .seg.t2 { background: var(--ml-series-2); }
.seg.t3 { background: var(--ml-series-3); } .seg.t4 { background: var(--ml-series-4); }
.seg.t5 { background: var(--ml-series-5); } .seg.t6 { background: var(--ml-series-6); }
.seg.solo { background: var(--v); }
.cap-mark { position: absolute; top: -1px; bottom: -1px; width: 2px; background: var(--ml-text); }
.over .track { border-color: var(--v); }
.legend { list-style: none; display: flex; flex-wrap: wrap; gap: 4px 14px; margin: 7px 0 0;
  padding: 0; font-size: 12px; color: var(--ml-muted); }
.legend[hidden] { display: none; }
.legend li { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
.legend i { width: 9px; height: 9px; border-radius: 2px; flex: none; background: var(--ml-series-1); }
.legend i.t2 { background: var(--ml-series-2); } .legend i.t3 { background: var(--ml-series-3); }
.legend i.t4 { background: var(--ml-series-4); } .legend i.t5 { background: var(--ml-series-5); }
.legend i.t6 { background: var(--ml-series-6); }
.legend b { color: var(--ml-text); font-family: var(--ml-mono); font-weight: 600; }
.empty .track { border-style: dashed; }
`;

class MlMeter extends MlElement {
  static props = {
    label: "string", capacity: "number", value: "number", segments: "object",
    format: "string", unit: "string", digits: "number", verdict: "string",
    yellowAt: "number", redAt: "number", caption: "string", verdictText: "string",
    compact: "bool", legend: "string",
  };
  static styles = STYLES;

  build() {
    this.uid = uid("meter");
    this.head = h("div", { class: "head" });
    this.track = h("div", { class: "track", role: "meter" });
    this.list = h("ul", { class: "legend" });
    this.wrap = h("div", { part: "meter" }, this.head, this.track, this.list);
    this.root.append(this.wrap);
  }

  parts() {
    const given = this.segments;
    if (Array.isArray(given)) {
      return given.map((g, i) => ({ label: String(g?.label ?? ""),
        value: Math.max(0, Number(g?.value) || 0), tone: Number(g?.tone) || (i % 6) + 1 }));
    }
    const v = this.value;
    return v === undefined ? [] : [{ label: "", value: Math.max(0, v), tone: 1 }];
  }

  update() {
    const parts = this.parts();
    const total = parts.reduce((a, p) => a + p.value, 0);
    const cap = this.capacity > 0 ? this.capacity : undefined;
    const t = { yellowAt: this.yellowAt ?? THRESHOLDS.yellowAt, redAt: this.redAt ?? THRESHOLDS.redAt };
    const named = this.verdict;
    const v = named ? normalize(named) : parts.length ? verdictOf(total, cap, t) : "none";
    const f = (n) => fmt(n, this.format || "number", this.unit, this.digits ?? 1);
    const scale = Math.max(cap ?? total, total) || 1;
    const word = this.verdictText || LABELS[v];
    const said = !parts.length ? "no data"
      : this.caption || (cap === undefined ? f(total) : `${f(total)} of ${f(cap)}`);
    const pct = cap ? ` (${Math.round((total / cap) * 100)}%)` : "";

    this.wrap.className = `v-${v}${total > (cap ?? Infinity) ? " over" : ""}${parts.length ? "" : " empty"}`;
    this.head.replaceChildren(
      h("span", { class: "label", id: `${this.uid}-l` }, this.label),
      h("span", { class: "caption" }, said,
        v === "none" && !parts.length ? null
          : h("ml-chip", { verdict: v, "icon-only": this.compact && v === "green" ? "" : null },
              word)));

    this.track.setAttribute("aria-labelledby", `${this.uid}-l`);
    this.track.setAttribute("aria-valuemin", "0");
    this.track.setAttribute("aria-valuemax", String(cap ?? scale));
    this.track.setAttribute("aria-valuenow", String(clamp(total, 0, cap ?? scale)));
    this.track.setAttribute("aria-valuetext", parts.length ? `${said}${pct}, ${word}` : "no data");
    const solo = parts.length === 1 && !this.segments;
    const nodes = parts.map((p) => h("i", {
      class: `seg t${p.tone}${solo ? " solo" : ""}`,
      style: { width: `${(p.value / scale) * 100}%` } }));
    if (cap !== undefined && total > cap) {
      nodes.push(h("b", { class: "cap-mark", style: { left: `${(cap / scale) * 100}%` } }));
    }
    this.track.replaceChildren(...nodes);

    const show = this.legend === "show" || (this.legend !== "hide" && parts.length > 1);
    this.list.hidden = !show;
    this.list.replaceChildren(...(show ? parts.map((p) => h("li", {},
      h("i", { class: `t${p.tone}` }), p.label && `${p.label} `, h("b", {}, f(p.value)))) : []));
  }
}
define("ml-meter", MlMeter);
