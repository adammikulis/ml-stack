const STYLE = `
ml-composer { display:flex; flex-wrap:wrap; gap:8px; flex:none; min-width:0; padding:12px;
  border:1px solid var(--ml-line-strong,#555); border-radius:12px; background:var(--ml-surface,#1b1f3a); }
ml-composer[hidden] { display:none; }
ml-composer:focus-within { border-color:var(--poolside-pink,#ff5fa2); }
ml-composer .composer-recipient { flex:0 0 100%; font-size:.85rem; color:var(--ml-muted,#aaa); }
ml-composer textarea { flex:1 1 100%; width:100%; box-sizing:border-box; min-height:48px;
  max-height:150px; resize:vertical; border:0; padding:6px; background:transparent; color:inherit; font:inherit; }
ml-composer input { flex:1 1 100%; min-width:0; background:var(--ml-bg); color:inherit; }
ml-composer button { margin-left:auto; padding:8px 18px; }
ml-composer .composer-hint { flex:0 0 100%; font-size:.75rem; color:var(--ml-muted,#aaa); }
ml-composer [role=alert] { flex:0 0 100%; }
`;
const styled = new WeakSet();

class Composer extends HTMLElement {
  static observedAttributes = ["recipient"];

  connectedCallback() {
    if (this.ready) return;
    this.ready = true;
    const root = this.getRootNode();
    if (!styled.has(root)) {
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(STYLE);
      root.adoptedStyleSheets = [...root.adoptedStyleSheets, sheet];
      styled.add(root);
    }
    this.label = document.createElement("div");
    this.label.className = "composer-recipient";
    this.prepend(this.label);
    this.updateRecipient();
    const hint = document.createElement("div");
    hint.className = "composer-hint";
    hint.textContent = "Enter to send · Shift + Enter for a new line";
    this.append(hint);
    this.input = this.querySelector("textarea");
    this.send = this.querySelector("[data-composer-send]");
    this.cancel = this.querySelector("[data-composer-cancel]");
    this.send?.addEventListener("click", () => this.submit());
    this.cancel?.addEventListener("click", () => this.dispatchEvent(new CustomEvent("composer-cancel", { bubbles:true, composed:true })));
    this.input?.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" || event.shiftKey || event.isComposing || event.keyCode === 229) return;
      event.preventDefault();
      this.submit();
    });
  }

  attributeChangedCallback() { this.updateRecipient(); }

  updateRecipient() {
    if (this.label) this.label.textContent = this.getAttribute("recipient") || "Message";
  }

  submit() {
    if (!this.input?.value.trim() || !this.send || this.send.disabled || this.send.hidden) return;
    this.dispatchEvent(new CustomEvent("composer-send", { bubbles:true, composed:true }));
  }
}

customElements.define("ml-composer", Composer);
