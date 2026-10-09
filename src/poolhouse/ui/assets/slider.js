/* <ml-slider>: a range with tick labels. `stops` makes it step through a list of values. */
import { define, h } from "./base.js";
import { FIELD_STYLES, FieldElement } from "./field.js";
import { fmt } from "./format.js";

const STYLES = `
${FIELD_STYLES}
.top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; }
.top label { margin-bottom: 0; }
output { font-family: var(--ml-mono); font-size: 13px; font-weight: 600; color: var(--ml-text); }
input[type=range] { width: 100%; margin: 10px 0 0; accent-color: var(--ml-accent); background: transparent; height: 22px; }
.ticks { position: relative; height: 30px; margin-top: 2px; }
.tick { position: absolute; top: 0; transform: translateX(-50%); display: flex; flex-direction: column;
  align-items: center; font-size: 11px; color: var(--ml-muted); white-space: nowrap; }
.tick::before { content: ""; width: 1px; height: 6px; background: var(--ml-line-strong); margin-bottom: 3px; }
.tick.on { color: var(--ml-text); font-weight: 600; }
.tick.on::before { background: var(--ml-accent); width: 2px; }
.ticks[hidden] { display: none; }
`;

const THUMB = 16;
const compact = (n) => (Math.abs(n) >= 1000 && n % 1024 === 0 ? `${n / 1024}k`
  : Math.abs(n) >= 1000 ? `${+(n / 1000).toFixed(1)}k` : String(n));

class MlSlider extends FieldElement {
  static props = { ...FieldElement.common, value: "number", min: "number", max: "number", step: "number",
                   marks: "object", stops: "object", format: "string", unit: "string", valueText: "string" };
  static styles = STYLES;

  build() {
    this.control = h("input", { type: "range", id: "c", oninput: () => this.moved() });
    this.out = h("output", { for: "c" });
    this.ticks = h("div", { class: "ticks", "aria-hidden": "true" });
    this.chrome(h("div", {}, this.control, this.ticks));
    this.top = h("div", { class: "top" }, this.lab, this.out);
    this.field.insertBefore(this.top, this.field.firstChild);
  }

  list() { return Array.isArray(this.stops) && this.stops.length ? this.stops.map(Number) : null; }

  moved() {
    const stops = this.list();
    const value = stops ? stops[Number(this.control.value)] : Number(this.control.value);
    this.value = value;
    this.announce({ value });
  }

  formValue() { return String(this.value ?? ""); }

  text(v) { return fmt(v, this.format || "number", this.unit, this.step && this.step % 1 ? 1 : 0); }

  update() {
    const stops = this.list();
    const lo = stops ? 0 : this.min ?? 0, hi = stops ? stops.length - 1 : this.max ?? 100;
    this.control.min = String(lo);
    this.control.max = String(hi);
    this.control.step = String(stops ? 1 : this.step ?? 1);
    const at = stops ? Math.max(0, stops.indexOf(this.value ?? stops[0])) : this.value ?? lo;
    this.control.value = String(at);
    const shown = stops ? stops[at] : Number(this.control.value);
    this.out.textContent = this.valueText || this.text(shown);
    this.control.setAttribute("aria-valuetext", this.valueText || this.text(shown));
    const marks = Array.isArray(this.marks) ? this.marks
      : stops ? stops.map((v) => ({ value: v })) : [];
    const frac = (v) => (hi === lo ? 0 : ((stops ? stops.indexOf(v) : v) - lo) / (hi - lo));
    this.ticks.hidden = !marks.length;
    this.ticks.replaceChildren(...marks.filter((m) => !stops || stops.includes(m.value)).map((m) => h("span", {
      class: `tick${m.value === shown ? " on" : ""}`,
      style: { left: `calc(${frac(m.value)} * (100% - ${THUMB}px) + ${THUMB / 2}px)` } },
      String(m.label ?? compact(m.value)))));
    this.paintChrome(this.control);
    this.control.setAttribute("aria-labelledby", this.ids.l);
  }
}
define("ml-slider", MlSlider);
