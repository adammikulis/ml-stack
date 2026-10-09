/* <ml-sheet>: a modal dialog. Native <dialog> supplies the focus trap, Escape and aria-modal. */
import { MlElement, define, h, uid } from "./base.js";

const STYLES = `
dialog { border: 1px solid var(--ml-line-strong); border-radius: 14px; padding: 0;
  background: var(--ml-surface); color: var(--ml-text); width: min(var(--ml-sheet-width, 460px), 92vw);
  max-height: 90vh; box-shadow: 0 24px 60px -20px rgba(0,0,0,.7); overflow: auto;
  animation: rise .18s cubic-bezier(.2, .8, .2, 1); }
dialog::backdrop { background: color-mix(in srgb, var(--ml-sunken) 82%, transparent);
  backdrop-filter: blur(6px); animation: fade .14s ease-out; }
@keyframes rise { from { transform: translateY(8px); opacity: 0; } }
@keyframes fade { from { opacity: 0; } }
.body { padding: 24px 26px; position: relative; }
h2 { margin: 0 0 6px; font-size: 18px; letter-spacing: -.01em; padding-right: 28px; }
h2[hidden] { display: none; }
.close { position: absolute; top: 12px; right: 12px; width: 30px; height: 30px; border-radius: 8px;
  border: 1px solid transparent; background: transparent; cursor: pointer; font-size: 18px;
  line-height: 1; color: var(--ml-muted); }
.close:hover { border-color: var(--ml-line-strong); color: var(--ml-text); }
.close[hidden] { display: none; }
.foot { display: flex; justify-content: flex-end; gap: 8px; margin-top: 18px; }
.foot[hidden] { display: none; }
`;

class MlSheet extends MlElement {
  static props = { open: "bool", heading: "string", dismissible: "string", closeLabel: "string" };
  static styles = STYLES;

  build() {
    const id = uid("sheet");
    this.titleEl = h("h2", { id });
    this.closer = h("button", { class: "close", type: "button",
      onclick: () => this.close("close-button") }, "×");
    this.foot = h("div", { class: "foot" }, h("slot", { name: "footer" }));
    this.dialog = h("dialog", { "aria-labelledby": id },
      h("div", { class: "body" }, this.closer, this.titleEl, h("slot"), this.foot));
    this.dialog.addEventListener("cancel", (e) => {
      e.preventDefault();
      if (this.dismissible !== "false") this.close("escape");
    });
    this.dialog.addEventListener("click", (e) => {
      if (e.target === this.dialog && this.dismissible !== "false") this.close("backdrop");
    });
    this.root.append(this.dialog);
    new MutationObserver(() => this.refresh()).observe(this, { childList: true });
  }

  /** Ask to close; `ml-close` can be cancelled with preventDefault. */
  close(reason = "api") {
    if (this.emit("ml-close", { reason }, true)) this.removeAttribute("open");
  }

  update() {
    this.titleEl.textContent = this.heading;
    this.titleEl.hidden = !this.heading;
    this.closer.hidden = this.dismissible === "false";
    this.closer.setAttribute("aria-label", this.closeLabel || "Close");
    this.foot.hidden = !this.querySelector(":scope > [slot=footer]");
    if (this.open && !this.dialog.open) {
      this.returnTo = document.activeElement;
      this.dialog.showModal();
    } else if (!this.open && this.dialog.open) {
      this.dialog.close();
      if (this.returnTo?.isConnected) this.returnTo.focus();
    }
  }
}
define("ml-sheet", MlSheet);
