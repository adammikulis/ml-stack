/* <ml-requests>: every request that waits for a person, grouped by agent and project, with the
   buttons that answer it. Everything the server sends is put on the page as text. */
import { MlElement, define, h } from "./base.js";

const STYLES = `
:host { display: block; max-width: 960px; }
.bar { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
.bar select, .bar label { font-size: 13px; }
.count { font-weight: 600; }
.group { margin: 0 0 18px; }
.group > h3 { margin: 0 0 6px; font-size: 14px; color: var(--ml-muted); overflow-wrap: anywhere; }
.card { border: 1px solid var(--ml-line-strong); border-radius: 10px; background: var(--ml-surface);
  padding: 10px 12px; margin: 0 0 8px; }
.card.done { opacity: .75; }
.kind { font-size: 12px; font-weight: 600; color: var(--ml-muted); }
.subject { font-family: var(--ml-mono, monospace); font-size: 13px; overflow-wrap: anywhere; white-space: pre-wrap; margin: 4px 0; }
.reason { font-size: 13px; overflow-wrap: anywhere; margin: 4px 0 8px; }
.choices { display: flex; flex-wrap: wrap; gap: 6px; }
.choice { display: flex; flex-direction: column; align-items: flex-start; }
.choice small { color: var(--ml-muted); font-size: 11.5px; max-width: 26ch; }
button { background: var(--ml-sunken); border: 1px solid var(--ml-line-strong); border-radius: 7px; padding: 4px 11px; cursor: pointer; }
button.approving { border-color: var(--ml-green); }
button[disabled] { opacity: .5; cursor: default; }
.note { font-size: 12.5px; color: var(--ml-muted); margin-top: 6px; overflow-wrap: anywhere; }
.bulk { border: 1px dashed var(--ml-line-strong); border-radius: 8px; padding: 8px 10px; margin: 0 0 8px; }
.bulk ul { margin: 4px 0 8px 18px; padding: 0; font-size: 12.5px; overflow-wrap: anywhere; }
.empty { color: var(--ml-muted); }
`;

const HIDDEN = [[0x00, 0x1f], [0x7f, 0x9f], [0xad, 0xad], [0x61c, 0x61c], [0x180e, 0x180e], [0x200b, 0x200f],
  [0x2028, 0x202e], [0x2060, 0x206f], [0xfeff, 0xfeff], [0xfff9, 0xfffb]];
const hidden = (n) => HIDDEN.some(([lo, hi]) => n >= lo && n <= hi);

/** Text as one safe line: control, bidi and invisible characters written out, cut at `most`. */
export function plain(value, most = 300) {
  const text = Array.from(String(value ?? ""), (c) => {
    const n = c.codePointAt(0);
    if (!hidden(n)) return c;
    return n < 0x100 ? "\\x" + n.toString(16).padStart(2, "0") : "\\u" + n.toString(16).padStart(4, "0");
  }).join("");
  return text.length > most ? `${text.slice(0, most - 1)}…` : text;
}

const BULK = ["allow-once", "approve", "deny"];

class MlRequests extends MlElement {
  static props = { api: "string", csrf: "string", agent: "string", project: "string", kind: "string", history: "bool" };
  static styles = STYLES;
  rows = [];
  rev = "";
  notes = new Map();
  bulk = null;

  build() {
    this.bar = h("div", { class: "bar" });
    this.list = h("div", { class: "list", "aria-live": "polite" });
    this.root.append(this.bar, this.list);
    this.timer = setInterval(() => this.tick(), 2000);
    this.tick();
  }

  disconnectedCallback() { clearInterval(this.timer); }

  get base() { return this.api || "/requests/api"; }

  get token() {
    return this.csrf || document.querySelector('meta[name="ml-requests-csrf"]')?.content || "";
  }

  async get(path) {
    const r = await fetch(`${this.base}${path}`, { credentials: "same-origin", cache: "no-store" });
    if (!r.ok) throw new Error(String(r.status));
    return r.json();
  }

  async post(path, body) {
    const r = await fetch(`${this.base}${path}`, {
      method: "POST", credentials: "same-origin", cache: "no-store",
      headers: { "Content-Type": "application/json", "X-Requests-CSRF": this.token },
      body: JSON.stringify(body),
    });
    return { status: r.status, data: await r.json().catch(() => ({})) };
  }

  async tick() {
    try {
      const feed = await this.get("/feed");
      if (feed.rev !== this.rev || this.rev === "") { this.rev = feed.rev; await this.load(); }
      this.emit("ml-requests-count", { pending: feed.pending });
    } catch (err) {
      this.list.replaceChildren(h("p", { class: "empty" }, "The Requests page is not connected."));
    }
  }

  async load() {
    const q = new URLSearchParams();
    for (const key of ["agent", "project", "kind"]) if (this[key]) q.set(key, this[key]);
    this.rows = (await this.get(`/list?${q}`)).requests;
    this.update();
  }

  update() {
    if (!this.list) return;
    const pending = this.rows.filter((r) => r.state === "pending");
    const done = this.rows.filter((r) => r.state !== "pending");
    const shown = this.history ? [...pending, ...done] : pending;
    this.bar.replaceChildren(
      h("span", { class: "count" }, `${pending.length} waiting`),
      this.select("agent", [...new Set(this.rows.map((r) => r.agent))]),
      this.select("project", [...new Set(this.rows.map((r) => r.project))]),
      this.select("kind", [...new Set(this.rows.map((r) => r.kind))]),
      h("label", {}, h("input", { type: "checkbox", checked: this.history, onchange: (e) => { this.history = e.target.checked; } }), " history"),
    );
    if (!shown.length) { this.list.replaceChildren(h("p", { class: "empty" }, "Nothing is waiting for you.")); return; }
    const groups = new Map();
    for (const r of shown) {
      const key = `${r.agent}\u0000${r.project}`;
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(r);
    }
    this.list.replaceChildren(...[...groups].map(([key, rows]) => this.group(key, rows)));
  }

  select(name, values) {
    const sel = h("select", { "aria-label": name, onchange: (e) => { this[name] = e.target.value; this.load(); } },
      h("option", { value: "" }, `any ${name}`),
      ...values.filter(Boolean).sort().map((v) => h("option", { value: v, selected: this[name] === v }, plain(v, 40))));
    return sel;
  }

  group(key, rows) {
    const [agent, project] = key.split("\u0000");
    const live = rows.filter((r) => r.state === "pending");
    const nodes = [h("h3", {}, `${plain(rows[0].who || agent, 160) || "unknown agent"} / ${plain(project, 64) || "no project"}`)];
    const kinds = new Set(live.map((r) => r.kind));
    const bulkable = live.length > 1 && kinds.size === 1 && live.every((r) => !r.destructive && !r.human_only);
    if (bulkable) nodes.push(this.bulkBox(key, live));
    nodes.push(...rows.map((r) => this.card(r)));
    return h("section", { class: "group" }, ...nodes);
  }

  bulkBox(key, live) {
    const common = BULK.filter((id) => live.every((r) => r.choices.some((c) => c.id === id)));
    if (!common.length) return h("span");
    if (this.bulk !== key) {
      return h("div", { class: "bulk" }, h("button", { onclick: () => { this.bulk = key; this.update(); } },
        `answer all ${live.length} requests of kind ${plain(live[0].kind, 40)}…`));
    }
    return h("div", { class: "bulk" },
      h("div", {}, `This answers exactly these ${live.length} requests:`),
      h("ul", {}, ...live.map((r) => h("li", {}, plain(r.subject, 160)))),
      ...common.map((id) => h("button", { class: id === "deny" ? "" : "approving", onclick: () => this.answerAll(live, id) },
        `${id === "deny" ? "deny" : id === "approve" ? "approve" : "allow once"} all ${live.length}`)),
      h("button", { onclick: () => { this.bulk = null; this.update(); } }, "cancel"));
  }

  card(r) {
    const open = r.state === "pending";
    const note = this.notes.get(r.id) || (open ? "" : `${plain(r.state, 20)}${r.answer ? `: ${plain(r.answer, 30)}` : ""}${r.answered_by ? ` (${plain(r.answered_by, 12)})` : ""}`);
    return h("article", { class: `card${open ? "" : " done"}` },
      h("div", { class: "kind" }, `${plain(r.kind, 40)}${r.human_only ? " - the action asks again before it runs" : ""}`),
      h("div", { class: "subject" }, plain(r.subject, 300)),
      h("div", { class: "reason" }, plain(r.reason, 400)),
      h("div", { class: "choices" }, ...r.choices.map((c) => h("div", { class: "choice" },
        h("button", { class: c.approving ? "approving" : "", disabled: !open, title: plain(c.effect, 200),
          onclick: () => this.answer(r, c.id) }, plain(c.label, 40)),
        h("small", {}, plain(c.effect, 120))))),
      note ? h("div", { class: "note" }, note) : null);
  }

  async answer(r, choice) {
    const { status, data } = await this.post("/answer", { id: r.id, choice, fingerprint: r.fingerprint });
    this.notes.set(r.id, status === 200 ? `${plain(data.state, 20)}: ${plain(data.answer, 30)}` : plain(data.error || `refused (${status})`, 160));
    await this.load();
  }

  async answerAll(live, choice) {
    const { status, data } = await this.post("/answer-kind", { choice, items: live.map((r) => ({ id: r.id, fingerprint: r.fingerprint })) });
    for (const r of live) this.notes.set(r.id, status === 200 ? (data.answered.includes(r.id) ? "answered" : "skipped") : plain(data.error || `refused (${status})`, 160));
    this.bulk = null;
    await this.load();
  }
}
define("ml-requests", MlRequests);
