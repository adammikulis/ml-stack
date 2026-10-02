/* <ml-select>, <ml-toggle>, <ml-path>. */
import "./chip.js";
import { define, h } from "./base.js";
import { FIELD_STYLES, FieldElement } from "./field.js";

class MlSelect extends FieldElement {
  static props = { ...FieldElement.common, value: "string", options: "object", placeholder: "string" };
  static styles = FIELD_STYLES;

  build() {
    this.control = h("select", { id: "c", onchange: () => {
      this.value = this.control.value;
      this.announce({ value: this.control.value });
    } });
    this.chrome(this.control);
  }

  update() {
    const opts = Array.isArray(this.options) ? this.options : [];
    const kids = opts.map((o) => h("option", { value: String(o.value), disabled: !!o.disabled },
      String(o.label ?? o.value)));
    if (this.placeholder) kids.unshift(h("option", { value: "", disabled: true }, this.placeholder));
    this.control.replaceChildren(...kids);
    this.control.value = this.value;
    if (this.control.value !== this.value && this.placeholder) this.control.value = "";
    this.paintChrome(this.control);
  }
}
define("ml-select", MlSelect);

const TOGGLE = `
${FIELD_STYLES}
.row { display: flex; align-items: center; gap: 12px; }
.row .text { flex: 1; min-width: 0; }
.row label { margin: 0; color: var(--ml-text); cursor: pointer; }
.sw { flex: none; position: relative; width: 42px; height: 24px; border-radius: 999px; padding: 0;
  background: var(--ml-sunken); border: 1.5px solid var(--ml-line-strong); cursor: pointer;
  transition: background .15s, border-color .15s; }
.sw::after { content: ""; position: absolute; top: 3px; left: 3px; width: 15px; height: 15px; border-radius: 50%;
  background: var(--ml-muted); transition: transform .15s, background .15s; }
.sw[aria-checked=true] { background: var(--ml-accent); border-color: var(--ml-accent); }
.sw[aria-checked=true]::after { transform: translateX(18px); background: var(--ml-on-accent); }
.state { font-size: 12px; color: var(--ml-muted); min-width: 2.2em; text-align: right; }
`;

class MlToggle extends FieldElement {
  static props = { ...FieldElement.common, checked: "bool", onLabel: "string", offLabel: "string" };
  static styles = TOGGLE;

  build() {
    this.control = h("button", { type: "button", role: "switch", id: "c", class: "sw",
      onclick: () => {
        this.checked = !this.checked;
        this.announce({ checked: this.checked });
      } });
    this.state = h("span", { class: "state", "aria-hidden": "true" });
    this.lab = h("label", { for: "c", id: "l" });
    this.hintEl = h("div", { class: "hint", id: "h" });
    this.errEl = h("div", { class: "err", id: "e", role: "alert" });
    this.ids = { l: "l", h: "h", e: "e" };
    this.field = h("div", { class: "field" },
      h("div", { class: "row" }, h("div", { class: "text" }, this.lab, this.hintEl), this.state, this.control),
      this.errEl);
    this.root.append(this.field);
  }

  formValue() { return this.checked ? "on" : ""; }

  update() {
    this.control.setAttribute("aria-checked", String(this.checked));
    this.state.textContent = this.checked ? this.onLabel || "On" : this.offLabel || "Off";
    this.paintChrome(this.control);
    this.lab.removeAttribute("for");
    this.control.setAttribute("aria-labelledby", this.ids.l);
  }
}
define("ml-toggle", MlToggle);

const PATH = `
${FIELD_STYLES}
.row { display: flex; gap: 8px; align-items: center; }
.row input { flex: 1; min-width: 0; font-family: var(--ml-mono); font-size: 13px; }
.row button { flex: none; padding: 8px 14px; border-radius: 8px; cursor: pointer; font-weight: 600;
  background: transparent; border: 1px solid var(--ml-line-strong); }
.row button[hidden] { display: none; }
.status { margin-top: 6px; }
.status[hidden] { display: none; }
`;

class MlPath extends FieldElement {
  static props = { ...FieldElement.common, value: "string", placeholder: "string", browse: "string",
                   verdict: "string", verdictText: "string" };
  static styles = PATH;

  build() {
    this.control = h("input", { type: "text", id: "c", spellcheck: "false", autocomplete: "off",
      oninput: () => { this.value = this.control.value; this.announce({ value: this.control.value }); } });
    this.browseBtn = h("button", { type: "button",
      onclick: () => this.emit("ml-browse", { value: this.value }) });
    this.status = h("div", { class: "status" });
    this.chrome(h("div", { class: "row" }, this.control, this.browseBtn));
    this.field.insertBefore(this.status, this.hintEl);
  }

  update() {
    if (this.control.value !== this.value) this.control.value = this.value;
    this.control.placeholder = this.placeholder;
    this.browseBtn.hidden = !this.browse;
    this.browseBtn.textContent = this.browse || "";
    this.browseBtn.toggleAttribute("disabled", this.disabled);
    this.status.hidden = !this.verdict;
    this.status.replaceChildren(this.verdict
      ? h("ml-chip", { verdict: this.verdict }, this.verdictText || "") : null);
    this.paintChrome(this.control);
  }
}
define("ml-path", MlPath);
