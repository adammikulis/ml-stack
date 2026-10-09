/* <ml-chip>: a verdict or status label with an icon and a word. */
import { MlElement, define, h } from "./base.js";
import { LABELS, icon, normalize } from "./verdict.js";

const STYLES = `
:host { display: inline-block; vertical-align: middle; }
.chip { display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px 2px 7px;
  border-radius: 999px; font-size: 12px; font-weight: 600; line-height: 1.5; white-space: nowrap;
  max-width: 100%; color: var(--v-ink);
  background: color-mix(in srgb, var(--v) 16%, var(--ml-surface));
  border: 1px solid color-mix(in srgb, var(--v) 55%, transparent); }
.chip.plain { color: var(--ml-text); --v: var(--ml-line-strong);
  background: var(--ml-sunken); border-color: var(--ml-line-strong); }
.text { overflow: hidden; text-overflow: ellipsis; }
:host([icon-only]) .chip { padding: 2px 5px; }
`;

class MlChip extends MlElement {
  static props = { verdict: "string", label: "string", iconOnly: "bool" };
  static styles = STYLES;

  build() {
    this.chip = h("span", { class: "chip", part: "chip" });
    this.root.append(this.chip);
    this.observer = new MutationObserver(() => this.refresh());
    this.observer.observe(this, { childList: true, characterData: true, subtree: true });
  }

  disconnectedCallback() { this.observer?.disconnect(); }

  update() {
    const named = this.verdict;
    const v = normalize(named);
    const plain = !named || v === "none" && named !== "none";
    const text = this.label || this.textContent.trim() || (plain ? "" : LABELS[v]);
    this.chip.className = `chip v-${v}${plain ? " plain" : ""}`;
    const kids = [];
    if (!plain) kids.push(icon(v));
    kids.push(h("span", { class: this.iconOnly ? "sr" : "text" }, text));
    this.chip.replaceChildren(...kids);
    if (this.iconOnly) this.chip.setAttribute("title", text); else this.chip.removeAttribute("title");
  }
}
define("ml-chip", MlChip);
