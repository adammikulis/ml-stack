/* <ml-board endpoint="/board">: the workspace's boards, threads and direct conversations, with a live feed and the person's composer (`readonly` removes it).
   Every string from the route is shown as text after control and bidirectional characters are
   removed; nothing is parsed as markup and the only link is a file's download (an attachment, never shown inline). The page holds no token. */
import { MlElement, define, h } from "./base.js";
import { loadPage, messageRoute, pageControls, renderFeed } from "./board-pages.js";
import { INTEGRATED_STYLES } from "./board-styles.js";
import "./composer.js";

const HIDDEN = /[\u0000-\u001f\u007f-\u009f\u061c\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/g;
const BODY_MAX = 4000;
const NAV_MAX = 100; // every node costs a frame: a list is cut before it is drawn, never after
const LINE_MAX = 200;
const POLL_MIN = 3000;
const POLL_MAX = 60000;
const WAIT_S = 20;
const POST_MAX = 16000;

/** `value` on one line, hidden characters turned to spaces, at most `max` characters. */
export function line(value, max = LINE_MAX) {
  return String(value ?? "").replace(HIDDEN, " ").replace(/\s+/g, " ").trim().slice(0, max);
}

/** `value` with hidden characters removed (newlines kept), at most `max` characters. */
export function body(value, max = BODY_MAX) {
  return String(value ?? "").replace(/\r/g, "").replace(HIDDEN, (c) => (c === "\n" ? "\n" : " ")).slice(0, max);
}

/** The pause before asking again: doubles from `POLL_MIN` up to `POLL_MAX` after a failure; a
    successful long poll asks again at once. */
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
@media (max-width: 560px) { .shell { grid-template-columns: 1fr; grid-template-rows:auto minmax(0,1fr); } nav { max-height:180px; } }
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
.channel-tabs { display:flex; gap:8px; margin:0 0 12px; }
.channel-tabs button { background:var(--ml-sunken); border:1px solid var(--ml-line); border-radius:6px; padding:5px 12px; cursor:pointer; }
.channel-tabs button[aria-pressed=true] { color:var(--ml-accent-ink); border-color:var(--ml-accent); }
.back { all: unset; cursor: pointer; color: var(--ml-accent-ink); font-size: 13px; }
.back:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: 2px; }
.composer { display: grid; gap: 6px; margin-top: 12px; padding-top: 10px;
  border-top: 1px solid var(--ml-line); }
.composer textarea, .composer input { font: inherit; color: var(--ml-text); background: var(--ml-sunken);
  border: 1px solid var(--ml-line-strong); border-radius: 6px; padding: 6px 8px; width: 100%; }
.composer textarea { min-height: 64px; resize: vertical; }
.composer button { justify-self: start; padding: 5px 14px; border-radius: 6px; cursor: pointer;
  background: var(--ml-accent); color: var(--ml-on-accent); border: 0; font-weight: 600; }
.composer button[disabled] { opacity: .5; cursor: default; }
.composer .err { color: var(--ml-red-ink); font-size: 12px; }
nav .new { display: flex; gap: 4px; padding: 6px 14px; }
nav .new input { flex: 1; min-width: 0; font: inherit; color: var(--ml-text);
  background: var(--ml-surface); border: 1px solid var(--ml-line-strong); border-radius: 6px; padding: 2px 6px; }
.state { padding: 24px 8px; color: var(--ml-muted); }
.state.error { color: var(--ml-red-ink); }
`;

class MlBoard extends MlElement {
  static props = { endpoint: "string", interval: "number", readonly: "bool", channelFeed:"bool", integrated:"bool" };
  static styles = STYLES + INTEGRATED_STYLES;

  build() {
    this.storageKey = `ml-stack-board:${this.base()}`;
    this.view = this.restoredView();
    this.requests = new Set();
    this.drafts = this.restoredDrafts();
    this.viewRevision = 0;
    this.agents = [];
    this.channelTab = this.channelFeed ? "messages" : "threads";
    this.boards = [];
    this.items = [];
    this.dms = [];
    this.head = "";
    this.delay = POLL_MIN;
    this.seq = 0;
    this.me = "";
    this.draft = { body: "", subject: "", error: "" };
    this.nav = h("nav", { "aria-label": "Boards and conversations" });
    this.feed = h("div", { "aria-live": "polite" });
    this.compose = h("div", {});
    this.history = h("div", {class:"history"});
    this.main = h("main", {}, this.feed, this.history, this.compose);
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
    this.loadRevision = (this.loadRevision || 0) + 1;
    clearTimeout(this.timer);
    for (const controller of this.requests) controller.abort();
  }

  setActive(active) {
    if (this.stopped === !active) return;
    this.stopped = !active;
    clearTimeout(this.timer);
    if (!active) {
      this.loadRevision = (this.loadRevision || 0) + 1; this.viewRevision++;
      for (const controller of this.requests) controller.abort();
    }
    else { this.load(); this.schedule(); }
  }

  base() {
    const e = this.endpoint || "/board";
    return e.endsWith("/") ? e.slice(0, -1) : e;
  }

  restoredView() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (raw?.length > 2048) return {kind:"none"};
      const view = JSON.parse(raw);
      if (!view || !["board", "thread", "dm"].includes(view.kind)) return {kind:"none"};
      if (view.kind === "thread" && (!Number.isSafeInteger(view.root) || view.root < 1)) return {kind:"none"};
      if ([view.name, view.board].some(value => value !== undefined && (typeof value !== "string" || value.length > 64))) return {kind:"none"};
      if ([view.a, view.b].some(value => value !== undefined && (typeof value !== "string" || value.length > 97))) return {kind:"none"};
      return view;
    } catch { return {kind:"none"}; }
  }

  restoredDrafts() {
    try {
      const raw = sessionStorage.getItem(`${this.storageKey}:drafts`);
      if (!raw || raw.length > 1000000) return new Map();
      const entries = JSON.parse(raw);
      if (!Array.isArray(entries) || entries.length > 50) return new Map();
      return new Map(entries.filter(entry => Array.isArray(entry) && typeof entry[0] === "string" && entry[0].length < 512
        && typeof entry[1]?.body === "string" && entry[1].body.length <= POST_MAX
        && typeof entry[1]?.subject === "string" && entry[1].subject.length <= 200));
    } catch { return new Map(); }
  }

  rememberDraft() {
    this.drafts.set(JSON.stringify(this.target()), {...this.draft});
    try { sessionStorage.setItem(`${this.storageKey}:drafts`, JSON.stringify([...this.drafts].slice(-50))); } catch {}
  }

  remember() {
    try { localStorage.setItem(this.storageKey, JSON.stringify(this.view)); } catch {}
  }

  async request(route, params = {}, document = null) {
    const controller = new AbortController();
    if (["messages", "threads", "thread", "dm"].includes(route)) {
      this.viewRequest?.abort(); this.viewRequest = controller;
    }
    this.requests.add(controller);
    const query = new URLSearchParams(params).toString();
    try {
      const response = await fetch(`${this.base()}/${route}${query ? `?${query}` : ""}`, {
        method: document ? "POST" : "GET", credentials:"same-origin", signal:controller.signal,
        headers:{Accept:"application/json", ...(document ? {"Content-Type":"application/json"} : {}),
                 ...(this.base().startsWith("/ui/") ? globalThis.fleetModel?.headers || {"X-ML-Stack-UI":"1"} : {})},
        ...(document ? {body:JSON.stringify(document)} : {})});
      const answer = await response.json().catch(() => ({}));
      if (!response.ok) {
        const error = Error(line(answer.error || response.status, 160));
        error.personSetupRequired = answer.person_setup_required === true;
        throw error;
      }
      return answer;
    } finally { this.requests.delete(controller); }
  }

  send(document) { return this.request("post", {}, document); }
  get(route, params = {}) { return this.request(route, params); }

  schedule(pause = 0) {
    clearTimeout(this.timer);
    if (this.stopped) return;
    this.timer = setTimeout(() => this.poll(), pause || this.interval || 0);
  }

  async poll() {
    let pause = 0;
    try {
      const { seq } = await this.get("wait", { after: this.seq, timeout: WAIT_S });
      this.delay = nextDelay(this.delay, false);
      if (seq > this.seq) { this.seq = seq; await this.load(); }
    } catch {
      this.delay = nextDelay(this.delay, true);
      pause = this.delay;
    }
    this.schedule(pause);
  }

  async load() {
    const revision = this.loadRevision = (this.loadRevision || 0) + 1;
    try {
      const [b, d, h] = await Promise.all([this.get("boards"), this.get("dms"), this.get("head")]);
      if (revision !== this.loadRevision || this.stopped) return;
      this.seq = Math.max(this.seq, Number(h.seq) || 0);
      this.me = line(b.me ?? "", 48);
      this.boards = b.boards ?? [];
      this.dms = d.conversations ?? [];
      try { this.agents = (await this.get("agents")).agents || []; } catch { this.agents = []; }
      if (revision !== this.loadRevision || this.stopped) return;
      this.error = "";
      this.personSetupRequired = false;
      const view = this.view.kind === "none" && this.boards.length
        ? {kind:"board", name:(this.boards.find(board => board.name === "#general") || this.boards[0]).name} : this.view;
      await this.open(view, false);
    } catch (e) {
      if (revision !== this.loadRevision || this.stopped) return;
      this.personSetupRequired = e.personSetupRequired === true;
      this.error = `The workspace did not answer (${line(e.message, 40)}).`;
    }
    this.update();
  }

  async open(view, render = true) {
    const previous = JSON.stringify(this.target());
    if (this.draftKey !== undefined) this.drafts.set(previous, {...this.draft});
    if (this.drafts.size > 50) this.drafts.delete(this.drafts.keys().next().value);
    this.view = view;
    this.draftKey = JSON.stringify(this.target());
    this.draft = this.drafts.get(this.draftKey) || {body:"", subject:"", error:""};
    this.remember();
    const revision = ++this.viewRevision;
    if (render) { this.loading = true; this.items = []; this.update(); }
    try {
      let items = [];
      if (messageRoute(this)) {
        if (render || !this.page?.has_newer) {
          await loadPage(this, !render && this.items.length ? "newer" : "latest");
        }
        items = this.items;
      } else if (view.kind === "board") {
        items = (await this.get("threads", {board:view.name})).threads;
      }
      if (revision !== this.viewRevision) return;
      this.items = items || [];
      this.error = "";
    } catch (error) {
      if (revision !== this.viewRevision || this.stopped) return;
      this.items = []; this.error = `That did not load (${line(error.message, 160)}).`;
    }
    this.loading = false;
    if (render) this.update();
    this.emit("ml-board-view", {kind:view.kind});
  }

  async markRead() {
    const messages = this.view.kind === "board" && this.channelTab === "threads"
      ? this.items.map(thread => thread.root) : this.items.map(message => message.seq);
    const through = Math.max(0, ...messages.map(Number).filter(Number.isSafeInteger));
    const target = this.view.kind === "dm" ? this.target()?.to : this.view.board || this.view.name;
    if (!through || !target) return;
    try {
      await this.request("ack", {}, {through, ...(target.startsWith("#") ? {board:target} : {to:target})});
      await this.load();
    } catch (error) { this.error = line(error.message, 160); this.update(); }
  }

  update() {
    if (!this.nav) return;
    const navigation = {boards:this.boards, dms:this.dms, agents:this.agents,
      me:this.me, view:this.view, personSetupRequired:this.personSetupRequired === true, error:this.error || ""};
    const navigationKey = JSON.stringify(navigation);
    if (navigationKey !== this.navigationKey) {
      this.navigationKey = navigationKey; this.emit("board-navigation", navigation);
    }
    const item = (label, count, current, onclick, title = "") => h("button", {title,
      type: "button", "aria-current": current ? "true" : null, onclick },
    h("span", {}, line(label, 160)), count ? h("span", { class: "count" }, String(count)) : null);
    this.nav.replaceChildren(
      h("h3", {}, "Channels"),
      ...this.boards.slice(0, NAV_MAX).map((b) => item(b.name, b.unread, this.view.name === b.name || this.view.board === b.name,
        () => this.open({ kind: "board", name: b.name }))),
      h("h3", {}, "Direct messages"),
      ...(this.readonly || !this.me ? [] : [this.newDm()]),
      ...this.dms.slice(0, NAV_MAX).map((c) => item(`${c.a} and ${c.b}`, c.unread,
        this.view.kind === "dm" && this.view.a === c.a && this.view.b === c.b,
        () => this.open({ kind: "dm", a: c.a, b: c.b }))),
      ...(this.agents.length ? [h("h3", {}, "Agents"), ...this.agents.slice(0, NAV_MAX).map(agent =>
        item(agent.display_name || agent.id, 0, this.view.kind === "dm" && [this.view.a, this.view.b].includes(agent.id),
          () => this.open({kind:"dm", a:this.me, b:agent.id}), `${agent.id} · ${agent.device?.verification || "unknown"}`))] : []));
    renderFeed(this, this.pane());
    const controls = pageControls(this);
    this.history.replaceChildren(...(controls ? [controls] : []));
    // the draft a conversation was opened with decides the composer too: one built before it was read is stale
    const key = JSON.stringify([this.target(), this.readonly, this.draft.error, this.draftKey]);
    if (key !== this.composerKey) {
      this.composerKey = key;
      const node = this.composer();
      this.compose.replaceChildren(...(node ? [node] : []));
    }
  }

  newDm() {
    const box = this.agents.length
      ? h("select", {"aria-label":"Message an agent"}, h("option", {value:""}, "Choose an agent…"),
          ...this.agents.map(agent => h("option", {value:agent.id}, line(agent.display_name || agent.id, 160))))
      : h("input", {type:"text", maxlength:"97", "aria-label":"Message an agent", placeholder:"agent id"});
    const open = () => {
      const name = line(box.value, 97);
      if (name) {
        this.emit("ml-board-open", {kind:"dm"});
        this.open({ kind: "dm", a: this.me, b: name });
      }
    };
    box.addEventListener("keydown", (e) => { if (e.key === "Enter") open(); });
    return h("div", { class: "new" }, box, h("button", { type: "button", onclick: open }, "Open"));
  }

  target() {
    const v = this.view;
    if (v.kind === "board") return { to: v.name };
    if (v.kind === "thread" && v.board) return { to: v.board, reply_to: Number(v.root) || 0 };
    if (v.kind === "dm" && (v.a === this.me || v.b === this.me)) return { to: v.a === this.me ? v.b : v.a };
    return null;
  }

  composer() {
    const to = this.target();
    if (this.readonly || !to) return null;
    const text = h("textarea", { "aria-label": "Message", maxlength: String(POST_MAX) });
    text.value = this.draft.body;
    text.addEventListener("input", () => { this.draft.body = text.value; this.rememberDraft(); });
    const subject = this.view.kind === "board" ? h("input", { type: "text", maxlength: "200", "aria-label": "Subject",
      placeholder: "subject (optional)", value: this.draft.subject }) : null;
    subject?.addEventListener("input", () => { this.draft.subject = subject.value; this.rememberDraft(); });
    const go = h("button", { type: "button", "data-composer-send":"" }, "Send");
    const post = async () => {
      if (go.disabled || !text.value.trim()) return;
      const targetKey = JSON.stringify(to);
      const submitted = {body:text.value, subject:subject ? subject.value : ""};
      go.disabled = true; go.textContent = "Sending…";
      try {
        await this.send({...to, ...submitted});
        const saved = this.drafts.get(targetKey) || submitted;
        const cleared = saved.body === submitted.body && saved.subject === submitted.subject
          ? {body:"", subject:"", error:""} : {...saved, error:""};
        this.drafts.set(targetKey, cleared);
        if (JSON.stringify(this.target()) === targetKey) this.draft = cleared;
        this.rememberDraft();
        text.value = cleared.body;
        if (subject) subject.value = cleared.subject;
        go.disabled = false; go.textContent = "Send";
        await this.load();
      } catch (e) {
        const failed = {...(this.drafts.get(targetKey) || {}), error:line(e.message, 160)};
        this.drafts.set(targetKey, failed);
        if (JSON.stringify(this.target()) === targetKey) this.draft = failed;
        go.disabled = false; go.textContent = "Send";
        this.update();
      }
    };
    const node = h("ml-composer", {class:"composer", recipient:this.view.kind === "thread"
      ? `Reply in ${to.to}` : `Message ${to.to}`}, subject, text, go,
      this.draft.error ? h("div", { class: "err", role: "alert" }, this.draft.error) : null);
    node.addEventListener("composer-send", post);
    return node;
  }

  pane() {
    const v = this.view;
    if (this.error) return [h("p", {class:"state error", role:"alert"}, this.error),
      h("button", {class:"back", type:"button", onclick:async () => {
        if (this.personSetupRequired) {
          try { await this.request("connect", {}, {}); }
          catch (error) { this.error = line(error.message, 160); this.update(); return; }
        }
        await this.load();
      }}, this.personSetupRequired ? "Join workspace as person" : "Retry")];
    if (this.loading) return [h("p", {class:"state", role:"status"}, "Loading conversation…")];
    if (v.kind === "none") return [h("p", { class: "state" }, "Choose a board or a conversation.")];
    if (v.kind === "board") return [this.title(line(v.name, 60), this.readonly ? "read only" : "live"),
      h("div", {class:"channel-tabs"}, ...["messages", "threads"].map(mode => h("button", {
        type:"button", "aria-pressed":String(this.channelTab === mode), onclick:() => {
          this.channelTab = mode; this.open(this.view);
        }}, mode === "messages" ? "Messages" : "Threads"))),
      ...(this.channelTab === "messages" ? this.messages("") : this.threads())];
    const back = v.kind === "thread" && v.board
      ? h("button", { class: "back", type: "button", onclick: () => this.open({ kind: "board", name: v.board }) },
        `Back to ${line(v.board, 60)}`) : null;
    const display = id => this.agents.find(agent => agent.id === id)?.display_name || id;
    const title = v.kind === "dm" ? `${line(display(v.a), 160)} and ${line(display(v.b), 160)}` : `Thread ${Number(v.root) || ""}`;
    return [...(back ? [back] : []), this.title(title, this.readonly ? "read only" : "live", v.kind === "dm" ? `${v.a} and ${v.b}` : ""),
      ...this.messages(v.kind === "dm" ? v.a : "")];
  }

  title(text, badge, identity = "") {
    return h("header", {}, h("h2", {title:identity}, text), h("span", {class:"badge"}, badge),
      this.base().startsWith("/ui/") ? h("button", {class:"back", type:"button", onclick:() => this.markRead()}, "Mark read") : null);
  }

  threads() {
    if (!this.items.length) return [h("p", { class: "state" }, "No threads yet.")];
    return this.items.map((t) => h("button", { class: "row", type: "button",
      onclick: () => this.open({ kind: "thread", root: t.root, board: this.view.name }) },
    h("span", { class: "subject" }, line(t.subject) || "(no subject)"),
    h("span", { class: "meta" }, ` ${line(t.display_name || t.from, 160)}, ${Number(t.replies) || 0} replies`
      + `${t.unread ? `, ${Number(t.unread)} unread` : ""}, ${when(t.last)}`)));
  }

  messages(first) {
    if (!this.items.length) return [h("p", { class: "state" }, "No messages.")];
    return this.items.map((m) => h("article", { class: `msg${first && m.from === first ? " sent" : ""}`, "data-seq":String(m.seq) },
      h("div", { class: "who", title:m.from }, `${line(m.display_name || m.from, 160)}`,
        m.role === "human" ? null : h("span", { class: "meta" }, ` (${m.model ? `${line(m.model, 80)}, ${line(m.model_state, 12)}` : "model unknown"})`),
        h("span", { class: "meta" }, `  ${line(m.type, 16)}, ${when(m.ts)}${m.held ? ", held in quarantine" : ""}`)),
      m.subject ? h("div", { class: "meta" }, line(m.subject)) : null,
      h("pre", {}, body(m.body)),
      m.file && /^[0-9a-f]{12}$/.test(String(m.file.id))
        ? h("a", { class: "file", href: `${this.base()}/file?id=${m.file.id}`, download: "",
          rel: "noopener" }, m.file.text ? `Download ${line(m.file.name, 80)} (text)` : `Details of ${line(m.file.name, 80)}`)
        : null,
      m.truncated ? h("div", { class: "cut" }, "Shortened for display.") : null,
      this.view.kind === "board" ? h("button", {class:"back", type:"button", onclick:() =>
        this.open({kind:"thread", root:Number(m.thread || m.seq), board:this.view.name})}, "Reply in thread") : null));
  }
}
define("ml-board", MlBoard);
