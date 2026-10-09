/* <ml-progress>: determinate or indeterminate progress with bytes, rate, ETA and cancel. */
import { MlElement, define, h, uid } from "./base.js";
import { clamp, duration, fmt } from "./format.js";

const STYLES = `
.head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline;
  font-size: 13px; margin-bottom: 5px; }
.label { font-weight: 600; overflow-wrap: anywhere; }
.stats { color: var(--ml-muted); font-family: var(--ml-mono); font-size: 12px; text-align: right; }
.row { display: flex; align-items: center; gap: 10px; }
.track { flex: 1; position: relative; height: 8px; border-radius: 999px; overflow: hidden;
  background: var(--ml-sunken); border: 1px solid var(--ml-line-strong); }
.fill { height: 100%; width: 0; background: var(--ml-accent);
  transition: width .4s cubic-bezier(.4, 0, .2, 1); }
.running .fill { background: var(--ml-accent); }
.done .fill { background: var(--ml-green); }
.failed .fill { background: var(--ml-red); }
.paused .fill { background: var(--ml-yellow);
  background-image: repeating-linear-gradient(45deg, transparent 0 4px, rgba(0,0,0,.28) 4px 8px); }
.indeterminate .fill { width: 38%; position: absolute; animation: slide 1.4s ease-in-out infinite; }
@keyframes slide { from { left: -40%; } to { left: 102%; } }
button { background: transparent; border: 1px solid var(--ml-line-strong); border-radius: 7px;
  padding: 3px 10px; cursor: pointer; font-size: 12px; color: var(--ml-text); }
button:hover { border-color: var(--ml-text); }
.msg { margin-top: 5px; font-size: 12px; color: var(--ml-muted); }
.failed .msg { color: var(--ml-red-ink); }
@media (prefers-reduced-motion: reduce) {
  .indeterminate .fill { animation: none; left: 0; width: 100%; opacity: .45; }
}
`;

class MlProgress extends MlElement {
  static props = {
    label: "string", done: "number", total: "number", value: "number", format: "string",
    unit: "string", rate: "number", eta: "number", state: "string", indeterminate: "bool",
    cancelable: "bool", message: "string", cancelLabel: "string",
  };
  static styles = STYLES;

  build() {
    this.uid = uid("progress");
    this.head = h("div", { class: "head" });
    this.fill = h("div", { class: "fill" });
    this.track = h("div", { class: "track", role: "progressbar" }, this.fill);
    this.cancelBtn = h("button", { type: "button", onclick: () => this.emit("ml-cancel") });
    this.msg = h("div", { class: "msg" });
    this.wrap = h("div", {}, this.head, h("div", { class: "row" }, this.track, this.cancelBtn), this.msg);
    this.root.append(this.wrap);
  }

  fraction() {
    if (this.value !== undefined) return clamp(this.value, 0, 1);
    if (this.total > 0 && this.done !== undefined) return clamp(this.done / this.total, 0, 1);
    return undefined;
  }

  update() {
    const state = ["running", "done", "failed", "paused"].includes(this.state) ? this.state : "running";
    const frac = this.fraction();
    const busy = state === "running" || state === "paused";
    const indeterminate = busy && (this.indeterminate || frac === undefined);
    const fm = this.format || "number";
    const bits = [];
    if (state === "done") bits.push("done");
    else if (state === "failed") bits.push("failed");
    else if (state === "paused") bits.push("paused");
    if (frac !== undefined && state !== "failed") {
      bits.push(this.total > 0 && this.done !== undefined
        ? `${fmt(this.done, fm, this.unit)} of ${fmt(this.total, fm, this.unit)} · ${Math.round(frac * 100)}%`
        : `${Math.round(frac * 100)}%`);
    }
    if (busy && this.rate > 0) bits.push(`${fmt(this.rate, fm, this.unit)}/s`);
    const eta = this.eta ?? (busy && this.rate > 0 && this.total > 0 && this.done !== undefined
      ? (this.total - this.done) / this.rate : undefined);
    if (busy && eta !== undefined) bits.push(`${duration(eta)} left`);

    this.wrap.className = `${state}${indeterminate ? " indeterminate" : ""}`;
    this.head.replaceChildren(h("span", { class: "label", id: `${this.uid}-l` }, this.label),
                              h("span", { class: "stats" }, bits.join(" · ")));
    this.track.setAttribute("aria-labelledby", `${this.uid}-l`);
    this.track.setAttribute("aria-busy", String(busy));
    if (indeterminate || frac === undefined) {
      this.track.removeAttribute("aria-valuenow");
      this.track.setAttribute("aria-valuetext", bits.join(", ") || "working");
      this.fill.style.removeProperty("width");
    } else {
      this.track.setAttribute("aria-valuemin", "0");
      this.track.setAttribute("aria-valuemax", "100");
      this.track.setAttribute("aria-valuenow", String(Math.round(frac * 100)));
      this.track.setAttribute("aria-valuetext", bits.join(", "));
      this.fill.style.setProperty("width", `${frac * 100}%`);
    }
    this.cancelBtn.hidden = !(this.cancelable && busy);
    this.cancelBtn.textContent = this.cancelLabel || "Cancel";
    this.cancelBtn.setAttribute("aria-label", `${this.cancelLabel || "Cancel"} ${this.label}`.trim());
    this.msg.hidden = !this.message;
    this.msg.textContent = this.message;
    if (this.message) this.msg.setAttribute("role", state === "failed" ? "alert" : "status");
  }
}
define("ml-progress", MlProgress);
