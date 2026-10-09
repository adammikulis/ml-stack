/* <ml-toaster>: a live region that shows short messages and removes them. */
import { MlElement, define, h } from "./base.js";
import { icon, normalize } from "./verdict.js";

const STYLES = `
:host { position: fixed; inset: auto 16px 16px auto; z-index: 1000; display: block; width: min(380px, calc(100vw - 32px));
  pointer-events: none; }
:host([position="top"]) { inset: 16px 16px auto auto; }
.stack { display: flex; flex-direction: column; gap: 8px; }
.toast { pointer-events: auto; display: flex; gap: 10px; align-items: flex-start; padding: 11px 12px;
  border-radius: 10px; background: var(--ml-surface); color: var(--ml-text);
  border: 1px solid var(--v); box-shadow: var(--ml-shadow); animation: in .18s ease-out; }
.mark { color: var(--v-ink); padding-top: 2px; }
.text { flex: 1; min-width: 0; overflow-wrap: anywhere; font-size: 13.5px; }
.title { display: block; font-weight: 600; }
button { background: transparent; border: 1px solid var(--ml-line-strong); color: var(--ml-text);
  border-radius: 7px; padding: 2px 9px; cursor: pointer; font-size: 12px; }
button.x { border-color: transparent; color: var(--ml-muted); font-size: 16px; line-height: 1; padding: 2px 6px; }
@keyframes in { from { transform: translateY(6px); opacity: 0; } }
`;

class MlToaster extends MlElement {
  static props = { position: "string" };
  static styles = STYLES;
  timers = new Map();
  count = 0;

  build() {
    this.polite = h("div", { class: "stack", role: "status", "aria-live": "polite" });
    this.urgent = h("div", { class: "stack", role: "alert", "aria-live": "assertive" });
    this.root.append(this.polite, this.urgent);
  }

  /** Show a toast: `{message, title?, verdict?, timeout?, action?: {label}}`. Returns its id.
      Red toasts stay until dismissed unless `timeout` says otherwise. */
  show({ message = "", title = "", verdict = "none", timeout, action } = {}) {
    const v = normalize(verdict), id = `t${++this.count}`;
    const ms = timeout ?? (v === "red" ? 0 : 6000);
    const node = h("div", { class: `toast v-${v}`, "data-id": id },
      h("span", { class: "mark" }, icon(v)),
      h("span", { class: "text" }, title ? h("span", { class: "title" }, title) : null, message),
      action ? h("button", { type: "button", onclick: () => {
        this.emit("ml-action", { id }); this.dismiss(id); } }, String(action.label ?? "Undo")) : null,
      h("button", { type: "button", class: "x", "aria-label": "Dismiss",
        onclick: () => this.dismiss(id) }, "×"));
    (v === "red" ? this.urgent : this.polite).append(node);
    if (ms > 0) {
      const arm = () => this.timers.set(id, setTimeout(() => this.dismiss(id), ms));
      node.addEventListener("pointerenter", () => clearTimeout(this.timers.get(id)));
      node.addEventListener("pointerleave", arm);
      node.addEventListener("focusin", () => clearTimeout(this.timers.get(id)));
      arm();
    }
    return id;
  }

  /** Remove one toast; fires `ml-dismiss`. */
  dismiss(id) {
    clearTimeout(this.timers.get(id));
    this.timers.delete(id);
    for (const n of this.root.querySelectorAll(`[data-id="${CSS.escape(id)}"]`)) n.remove();
    this.emit("ml-dismiss", { id });
  }
}
define("ml-toaster", MlToaster);
