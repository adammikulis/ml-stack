/* <ml-stepper>: a wizard shell. Steps are named in `steps`; each step's content is a child
   with slot="step-<id>". */
import { MlElement, define, h, uid } from "./base.js";

const STYLES = `
.bar { display: flex; gap: 6px; list-style: none; margin: 0 0 20px; padding: 0; }
.bar li { flex: 1; min-width: 0; }
.bar button { display: block; width: 100%; text-align: left; background: transparent; border: 0;
  padding: 0; cursor: default; color: var(--ml-muted); font-size: 12px; }
.bar button:not([disabled]) { cursor: pointer; }
.bar i { display: block; height: 4px; border-radius: 2px; background: var(--ml-line-strong);
  margin-bottom: 6px; transition: background .3s; }
.bar .done i, .bar .current i { background: var(--ml-accent); }
.bar .current button { color: var(--ml-text); font-weight: 600; }
.bar .name { display: block; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
:host([compact]) .name { position: absolute; width: 1px; height: 1px; overflow: hidden;
  clip: rect(0 0 0 0); }
.err { margin: 14px 0 0; padding: 10px 12px; border-radius: 8px; font-size: 13px;
  color: var(--ml-red-ink); border: 1px solid color-mix(in srgb, var(--ml-red) 55%, transparent);
  background: color-mix(in srgb, var(--ml-red) 12%, var(--ml-surface)); }
.err[hidden] { display: none; }
.nav { display: flex; justify-content: space-between; gap: 10px; margin-top: 22px; }
.nav[hidden] { display: none; }
.nav button { padding: 9px 18px; border-radius: 8px; font-weight: 600; cursor: pointer;
  border: 1px solid var(--ml-line-strong); background: transparent; }
.nav button.primary { background: var(--ml-accent); border-color: var(--ml-accent);
  color: var(--ml-on-accent); }
.nav button[disabled] { opacity: .5; cursor: default; }
.nav .spacer { flex: 1; }
`;

class MlStepper extends MlElement {
  static props = { steps: "object", current: "number", controls: "string", compact: "bool",
                   backLabel: "string", nextLabel: "string", finishLabel: "string",
                   busy: "bool", validate: "object" };
  static styles = STYLES;

  completed = new Set();

  build() {
    this.bar = h("ol", { class: "bar", "aria-label": "Steps" });
    this.stage = h("div", { part: "stage", "aria-live": "polite" });
    this.err = h("div", { class: "err", role: "alert", hidden: true });
    this.back = h("button", { type: "button", onclick: () => this.step(-1) });
    this.next = h("button", { type: "button", class: "primary", onclick: () => this.step(1) });
    this.nav = h("div", { class: "nav" }, this.back, h("span", { class: "spacer" }), this.next);
    this.root.append(this.bar, this.stage, this.err, this.nav);
    this.root.addEventListener("keydown", (e) => this.key(e));
  }

  list() { return Array.isArray(this.steps) ? this.steps : []; }
  index() { return Math.max(0, Math.min(this.list().length - 1, Math.trunc(this.current ?? 0))); }

  /** The position and completed steps, to store and hand back through `state =`. */
  get state() { return { current: this.index(), completed: [...this.completed] }; }
  set state(v) {
    this.completed = new Set(Array.isArray(v?.completed) ? v.completed : []);
    this.current = Number(v?.current) || 0;
  }

  /** Go to step `to` without validating; fires `ml-step`. */
  goto(to) {
    const n = Math.max(0, Math.min(this.list().length - 1, to));
    if (n === this.index()) return;
    const from = this.index();
    this.current = n;
    this.err.hidden = true;
    this.emit("ml-step", { from, to: n, id: this.list()[n]?.id, state: this.state });
    this.updateNow();
  }

  updateNow() { this.update(); }

  async step(delta) {
    if (this.busy) return;
    const from = this.index(), to = from + delta, last = this.list().length - 1;
    if (delta > 0) {
      this.busy = true;
      let verdict = true;
      try {
        verdict = typeof this.validate === "function" ? await this.validate(from) : true;
      } catch (e) { verdict = String(e?.message || "This step could not be checked."); }
      this.busy = false;
      if (verdict !== true && verdict !== undefined) {
        this.showError(typeof verdict === "string" ? verdict : "This step is not complete.");
        return;
      }
      if (!this.emit("ml-before-next", { from, to, id: this.list()[from]?.id }, true)) return;
      this.completed.add(from);
      if (from >= last) { this.emit("ml-finish", { state: this.state }); this.updateNow(); return; }
    }
    this.goto(to);
  }

  showError(text) {
    this.err.textContent = text;
    this.err.hidden = false;
    this.err.scrollIntoView?.({ block: "nearest" });
  }

  key(e) {
    const btn = e.composedPath().find((n) => n.dataset?.i !== undefined);
    if (!btn || !["ArrowLeft", "ArrowRight", "Home", "End"].includes(e.key)) return;
    const items = [...this.bar.querySelectorAll("button:not([disabled])")];
    const at = items.indexOf(btn);
    const to = e.key === "ArrowLeft" ? at - 1 : e.key === "ArrowRight" ? at + 1
      : e.key === "Home" ? 0 : items.length - 1;
    items[Math.max(0, Math.min(items.length - 1, to))]?.focus();
    e.preventDefault();
  }

  update() {
    const steps = this.list(), at = this.index(), last = steps.length - 1;
    const reach = Math.max(at, ...this.completed, 0);
    this.bar.replaceChildren(...steps.map((st, i) => {
      const cls = i === at ? "current" : this.completed.has(i) || i < at ? "done" : "";
      const go = i <= reach && i !== at && !this.busy;
      const b = h("button", { type: "button", "data-i": i, disabled: !go && i !== at,
        "aria-current": i === at ? "step" : null,
        "aria-label": `Step ${i + 1} of ${steps.length}: ${st.title ?? st.id}`,
        onclick: () => go && this.goto(i) }, h("i"), h("span", { class: "name" }, st.title ?? st.id));
      if (i === at) b.removeAttribute("disabled");
      if (i === at) b.tabIndex = -1;
      return h("li", { class: cls }, b);
    }));
    const id = steps[at]?.id;
    this.stage.replaceChildren(id === undefined ? null : h("slot", { name: `step-${id}` }));
    this.nav.hidden = this.controls === "none";
    this.back.hidden = at === 0;
    this.back.textContent = this.backLabel || "Back";
    this.back.disabled = !!this.busy;
    this.next.textContent = at >= last ? this.finishLabel || "Finish" : this.nextLabel || "Continue";
    this.next.disabled = !!this.busy;
    this.next.setAttribute("aria-busy", String(!!this.busy));
  }
}
define("ml-stepper", MlStepper);
