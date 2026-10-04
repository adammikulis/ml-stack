/* <ml-board endpoint="/board">: the workspace's boards, threads and direct conversations, read-only.
   Every string from the route is shown as text after control and bidirectional characters are
   removed; nothing is parsed as markup and no link is made. The page holds no token. */
import { MlElement, define, h } from "./base.js";

const HIDDEN = /[\u0000-\u001f\u007f-\u009f\u061c\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/g;
const BODY_MAX = 4000;
const LINE_MAX = 200;
const POLL_MIN = 3000;
const POLL_MAX = 60000;

/** `value` on one line, hidden characters turned to spaces, at most `max` characters. */
export function line(value, max = LINE_MAX) {
  return String(value ?? "").replace(HIDDEN, " ").replace(/\s+/g, " ").trim().slice(0, max);
}

/** `value` with hidden characters removed (newlines kept), at most `max` characters. */
export function body(value, max = BODY_MAX) {
  return String(value ?? "").replace(/\r/g, "").replace(HIDDEN, (c) => (c === "\n" ? "\n" : " ")).slice(0, max);
}

/** The next poll delay: doubles from `POLL_MIN` up to `POLL_MAX` after a failure, resets on success. */
export function nextDelay(current, failed) {
  return failed ? Math.min(POLL_MAX, Math.max(POLL_MIN, current) * 2) : POLL_MIN;
}

const when = (ts) => {
  const d = new Date(Number(ts) * 1000);
  return Number.isFinite(d.getTime()) ? d.toLocaleString() : "";
};

const STYLES = `
:host { min-height: 320px; }
.shell { display: grid; grid-template-columns: minmax(150px, 220px) 1fr; height: 100%;
  min-height: inherit; border: 1px solid var(--ml-line); border-radius: var(--ml-radius);
  background: var(--ml-surface); overflow: hidden; }
@media (max-width: 560px) { .shell { grid-template-columns: 1fr; } }
nav { border-right: 1px solid var(--ml-line); background: var(--ml-sunken); overflow-y: auto;
  padding: 8px 0; }
nav h3 { margin: 10px 14px 4px; font-size: 11px; text-transform: uppercase; letter-spacing: .06em;
  color: var(--ml-muted); font-weight: 600; }
nav button { all: unset; box-sizing: border-box; display: flex; justify-content: space-between;
  gap: 8px; width: 100%; padding: 6px 14px; cursor: pointer; overflow-wrap: anywhere; }
nav button:hover, nav button[aria-current=true] {
  background: color-mix(in srgb, var(--ml-accent) 12%, transparent); }
nav button:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: -2px; }
.count { font-family: var(--ml-mono); font-size: 12px; color: var(--ml-accent-ink); font-weight: 600; }
main { overflow-y: auto; padding: 12px 16px; min-width: 0; }
main header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 8px; }
main h2 { margin: 0; font-size: 16px; overflow-wrap: anywhere; }
.badge { font-size: 11px; color: var(--ml-muted); border: 1px solid var(--ml-line-strong);
  border-radius: 999px; padding: 0 8px; }
.row { all: unset; box-sizing: border-box; display: block; width: 100%; padding: 8px 10px;
  border-bottom: 1px solid var(--ml-line); cursor: pointer; }
.row:hover { background: color-mix(in srgb, var(--ml-accent) 8%, transparent); }
.row:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: -2px; }
.row .subject { font-weight: 600; overflow-wrap: anywhere; }
.meta { color: var(--ml-muted); font-size: 12px; }
.msg { padding: 8px 0; border-bottom: 1px solid var(--ml-line); }
.msg.sent { margin-left: 24px; }
.msg .who { font-weight: 600; font-size: 13px; }
.msg pre { margin: 4px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; font: inherit; }
.cut { color: var(--ml-yellow-ink); font-size: 12px; }
.back { all: unset; cursor: pointer; color: var(--ml-accent-ink); font-size: 13px; }
.back:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: 2px; }
.state { padding: 24px 8px; color: var(--ml-muted); }
.state.error { color: var(--ml-red-ink); }
`;

class MlBoard extends MlElement {
  static props = { endpoint: "string", interval: "number" };
  static styles = STYLES;

  build() {
    this.view = { kind: "none" };
    this.boards = [];
    this.dms = [];
    this.head = "";
    this.delay = POLL_MIN;
    this.nav = h("nav", { "aria-label": "Boards and conversations" });
    this.main = h("main", { "aria-live": "polite" });
    this.root.append(h("div", { class: "shell" }, this.nav, this.main));
  }

  connectedCallback() {
    super.connectedCallback();
    this.stopped = false;
    this.load();
    this.schedule();
  }

  disconnectedCallback() {
    this.stopped = true;
    clearTimeout(this.timer);
  }

  base() {
    const e = this.endpoint || "/board";
    return e.endsWith("/") ? e.slice(0, -1) : e;
  }

  async get(route, params = {}) {
    const q = new URLSearchParams(params).toString();
    const r = await fetch(`${this.base()}/${route}${q ? `?${q}` : ""}`, {
      method: "GET", credentials: "same-origin", headers: { Accept: "application/json" } });
    if (!r.ok) throw new Error(`${r.status}`);
    return r.json();
  }

  schedule() {
    clearTimeout(this.timer);
    if (this.stopped) return;
    this.timer = setTimeout(() => this.poll(), this.interval || this.delay);
  }

  async poll() {
    try {
      const { head } = await this.get("head");
      this.delay = nextDelay(this.delay, false);
      if (head !== this.head) { this.head = head; await this.load(); }
    } catch {
      this.delay = nextDelay(this.delay, true);
    }
    this.schedule();
  }

  async load() {
    try {
      const [b, d] = await Promise.all([this.get("boards"), this.get("dms")]);
      this.boards = b.boards ?? [];
      this.dms = d.conversations ?? [];
      this.error = "";
      await this.open(this.view, false);
    } catch (e) {
      this.error = `The workspace did not answer (${line(e.message, 40)}).`;
    }
    this.update();
  }

  async open(view, render = true) {
    this.view = view;
    try {
      if (view.kind === "board") this.items = (await this.get("threads", { board: view.name })).threads;
      else if (view.kind === "thread") this.items = (await this.get("thread", { root: view.root })).messages;
      else if (view.kind === "dm") this.items = (await this.get("dm", { a: view.a, b: view.b })).messages;
      else this.items = [];
      this.error = "";
    } catch (e) {
      this.items = [];
      this.error = `That did not load (${line(e.message, 40)}).`;
    }
    if (render) this.update();
    this.emit("ml-board-view", { kind: view.kind });
  }

  update() {
    if (!this.nav) return;
    const item = (label, count, current, onclick) => h("button", {
      type: "button", "aria-current": current ? "true" : null, onclick },
    h("span", {}, line(label, 60)), count ? h("span", { class: "count" }, String(count)) : null);
    this.nav.replaceChildren(
      h("h3", {}, "Boards"),
      ...this.boards.map((b) => item(b.name, b.unread, this.view.kind !== "none" && this.view.name === b.name,
        () => this.open({ kind: "board", name: b.name }))),
      h("h3", {}, "Direct messages"),
      ...this.dms.map((c) => item(`${c.a} and ${c.b}`, c.unread,
        this.view.kind === "dm" && this.view.a === c.a && this.view.b === c.b,
        () => this.open({ kind: "dm", a: c.a, b: c.b }))));
    this.main.replaceChildren(...this.pane());
  }

  pane() {
    const v = this.view;
    if (this.error) return [h("p", { class: "state error", role: "alert" }, this.error)];
    if (v.kind === "none") return [h("p", { class: "state" }, "Choose a board or a conversation.")];
    if (v.kind === "board") return [this.title(line(v.name, 60), "read only"), ...this.threads()];
    const back = v.kind === "thread" && v.board
      ? h("button", { class: "back", type: "button", onclick: () => this.open({ kind: "board", name: v.board }) },
        `Back to ${line(v.board, 60)}`) : null;
    const title = v.kind === "dm" ? `${line(v.a, 48)} and ${line(v.b, 48)}` : `Thread ${Number(v.root) || ""}`;
    return [back, this.title(title, "read only"), ...this.messages(v.kind === "dm" ? v.a : "")];
  }

  title(text, badge) {
    return h("header", {}, h("h2", {}, text), h("span", { class: "badge" }, badge));
  }

  threads() {
    if (!this.items.length) return [h("p", { class: "state" }, "No threads yet.")];
    return this.items.map((t) => h("button", { class: "row", type: "button",
      onclick: () => this.open({ kind: "thread", root: t.root, board: this.view.name }) },
    h("span", { class: "subject" }, line(t.subject) || "(no subject)"),
    h("span", { class: "meta" }, ` ${line(t.from, 48)}, ${Number(t.replies) || 0} replies`
      + `${t.unread ? `, ${Number(t.unread)} unread` : ""}, ${when(t.last)}`)));
  }

  messages(first) {
    if (!this.items.length) return [h("p", { class: "state" }, "No messages.")];
    return this.items.map((m) => h("article", { class: `msg${first && m.from === first ? " sent" : ""}` },
      h("div", { class: "who" }, `${line(m.from, 48)}`,
        h("span", { class: "meta" }, `  ${line(m.type, 16)}, ${when(m.ts)}${m.held ? ", held in quarantine" : ""}`)),
      m.subject ? h("div", { class: "meta" }, line(m.subject)) : null,
      h("pre", {}, body(m.body)),
      m.truncated ? h("div", { class: "cut" }, "Shortened for display.") : null));
  }
}
define("ml-board", MlBoard);
