/* The shared shell of the form controls: label, hint, error, disabled. */
import { MlElement, h, uid } from "./base.js";

export const FIELD_STYLES = `
.field { display: block; }
label, .lbl { display: block; font-size: 12.5px; font-weight: 600; color: var(--ml-muted);
  margin-bottom: 6px; }
.hint { font-size: 12px; color: var(--ml-muted); margin-top: 6px; }
.err { font-size: 12.5px; margin-top: 6px; color: var(--ml-red-ink); display: flex; gap: 6px; align-items: center; }
.err[hidden], .hint[hidden], label[hidden] { display: none; }
input[type=text], select { width: 100%; padding: 9px 11px; border-radius: 8px; font: inherit; font-size: 14px;
  background: var(--ml-sunken); border: 1px solid var(--ml-line-strong); color: var(--ml-text); }
input:disabled, select:disabled, button:disabled { opacity: .55; cursor: not-allowed; }
input:focus-visible, select:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: 1px; }
.invalid input, .invalid select { border-color: var(--ml-red); }
.mono input { font-family: var(--ml-mono); font-size: 13px; }
`;

/** Base of select, slider, toggle and path: `label`, `hint`, `error`, `name`, `disabled`. */
export class FieldElement extends MlElement {
  static formAssociated = true;
  static common = { label: "string", hint: "string", error: "string", name: "string", disabled: "bool" };

  constructor() {
    super();
    this.internals = this.attachInternals?.();
  }

  /** The label, hint and error nodes, wired to `control` by id. */
  chrome(control) {
    this.ids = { l: uid("lbl"), h: uid("hint"), e: uid("err") };
    this.lab = h("label", { id: this.ids.l });
    this.hintEl = h("div", { class: "hint", id: this.ids.h });
    this.errEl = h("div", { class: "err", id: this.ids.e, role: "alert" });
    this.field = h("div", { class: "field" }, this.lab, control, this.hintEl, this.errEl);
    this.root.append(this.field);
  }

  /** Paint the shell from the common attributes; `control` gets the aria wiring. */
  paintChrome(control, labelFor = true) {
    this.lab.textContent = this.label;
    this.lab.hidden = !this.label;
    if (labelFor && control.id) this.lab.setAttribute("for", control.id);
    this.hintEl.textContent = this.hint;
    this.hintEl.hidden = !this.hint;
    this.errEl.textContent = this.error;
    this.errEl.hidden = !this.error;
    this.field.classList.toggle("invalid", !!this.error);
    const described = [this.hint && this.ids.h, this.error && this.ids.e].filter(Boolean).join(" ");
    if (described) control.setAttribute("aria-describedby", described);
    else control.removeAttribute("aria-describedby");
    control.setAttribute("aria-invalid", String(!!this.error));
    if (!labelFor) control.setAttribute("aria-labelledby", this.ids.l);
    control.toggleAttribute("disabled", this.disabled);
    this.internals?.setFormValue?.(this.formValue());
  }

  formValue() { return String(this.value ?? ""); }

  /** Fire `input` and `change` from the host with the new value in `detail`. */
  announce(detail) {
    this.internals?.setFormValue?.(this.formValue());
    for (const type of ["input", "change"]) {
      this.dispatchEvent(new CustomEvent(type, { detail, bubbles: true, composed: true }));
    }
  }

  formDisabledCallback(off) { if (off) this.setAttribute("disabled", ""); }
  focus(options) { this.control?.focus(options); }
}
