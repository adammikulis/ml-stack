/* The base class and the DOM builders every ml-ui element is made from. Text only ever
   reaches the page through textContent or setAttribute. */

const HTML = "http://www.w3.org/1999/xhtml";
const SVG = "http://www.w3.org/2000/svg";

function build(ns, tag, attrs, kids) {
  const n = document.createElementNS(ns, tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k.startsWith("on") && typeof v === "function") n.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") {
      for (const [p, val] of Object.entries(v)) n.style.setProperty(p, String(val));
    } else n.setAttribute(k, v === true ? "" : String(v));
  }
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
}

/** An HTML element with attributes, a `style` object and children; strings become text. */
export const h = (tag, attrs, ...kids) => build(HTML, tag, attrs, kids);
/** An SVG element, built the same way. */
export const s = (tag, attrs, ...kids) => build(SVG, tag, attrs, kids);

const SHARED = `
:host { display: block; box-sizing: border-box; font-family: var(--ml-font);
  color: var(--ml-text); font-size: 14px; line-height: 1.45; }
:host([hidden]) { display: none !important; }
*, *::before, *::after { box-sizing: inherit; }
.sr { position: absolute; width: 1px; height: 1px; margin: -1px; overflow: hidden;
  clip: rect(0 0 0 0); white-space: nowrap; border: 0; padding: 0; }
:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: 2px; }
.v-green { --v: var(--ml-green); --v-ink: var(--ml-green-ink); }
.v-yellow { --v: var(--ml-yellow); --v-ink: var(--ml-yellow-ink); }
.v-red { --v: var(--ml-red); --v-ink: var(--ml-red-ink); }
.v-none { --v: var(--ml-line-strong); --v-ink: var(--ml-muted); }
.vicon { flex: none; }
button { font: inherit; color: inherit; }
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { transition: none !important; animation: none !important; }
}
`;

const sheets = new Map();
function sheetFor(css) {
  if (!sheets.has(css)) {
    const sheet = new CSSStyleSheet();
    sheet.replaceSync(css);
    sheets.set(css, sheet);
  }
  return sheets.get(css);
}

const OBJECT = "object";

/** Base of every element: typed attribute-backed properties, a shadow root, batched renders. */
export class MlElement extends HTMLElement {
  static props = {};
  static styles = "";

  static get observedAttributes() {
    return Object.entries(this.props).filter(([, t]) => t !== OBJECT).map(([k]) => attr(k));
  }

  #pending = false;
  #built = false;
  data = {};

  connectedCallback() {
    if (!this.#built) {
      this.#built = true;
      this.root = this.attachShadow({ mode: "open", delegatesFocus: this.constructor.focusable });
      this.root.adoptedStyleSheets = [sheetFor(SHARED), sheetFor(this.constructor.styles)];
      for (const key of Object.keys(this.constructor.props)) {
        if (Object.hasOwn(this, key)) {
          const early = this[key];
          delete this[key];
          this[key] = early;
        }
      }
      this.build();
    }
    this.update();
  }

  attributeChangedCallback() { this.refresh(); }

  /** Re-run `update` once, after the current task's property writes. */
  refresh() {
    if (!this.#built || this.#pending) return;
    this.#pending = true;
    queueMicrotask(() => { this.#pending = false; this.update(); });
  }

  build() {}
  update() {}

  emit(name, detail = {}, cancelable = false) {
    const ev = new CustomEvent(name, { detail, bubbles: true, composed: true, cancelable });
    this.dispatchEvent(ev);
    return !ev.defaultPrevented;
  }
}

const attr = (key) => key.replace(/[A-Z]/g, (c) => `-${c.toLowerCase()}`);

/** Register `cls` as `tag` with an accessor for every entry in `cls.props`. */
export function define(tag, cls) {
  for (const [key, type] of Object.entries(cls.props)) {
    const name = attr(key);
    Object.defineProperty(cls.prototype, key, {
      configurable: true,
      get() {
        if (type === OBJECT) return this.data[key];
        const raw = this.getAttribute(name);
        if (type === "bool") return raw !== null && raw !== "false";
        if (type === "number") return raw === null || raw === "" ? undefined : Number(raw);
        return raw ?? "";
      },
      set(value) {
        if (type === OBJECT) { this.data[key] = value; this.refresh(); return; }
        if (type === "bool") {
          if (value) this.setAttribute(name, ""); else this.removeAttribute(name);
        } else if (value === null || value === undefined) this.removeAttribute(name);
        else this.setAttribute(name, String(value));
      },
    });
  }
  if (!customElements.get(tag)) customElements.define(tag, cls);
}

let counter = 0;
/** A document-unique id for aria wiring inside one shadow root. */
export const uid = (prefix) => `${prefix}-${++counter}`;
