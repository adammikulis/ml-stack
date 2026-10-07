/* <ml-agents endpoint="/agents">: start a local model as a workspace agent, see the agents that run,
   stop one. Every string from the route is shown as text after control and bidirectional
   characters are removed; nothing is parsed as markup and no link is made. The page holds no
   token: start and stop travel on the person's browser session cookie. */
import "./chip.js";
import "./controls.js";
import { MlElement, define, h } from "./base.js";
import { fmt } from "./format.js";

const HIDDEN = /[\u0000-\u001f\u007f-\u009f\u061c\u200b-\u200f\u2028-\u202e\u2060-\u206f\ufeff]/g;
const POLL_MIN = 3000;
const POLL_MAX = 60000;

/** `value` on one line, hidden characters turned to spaces, at most `max` characters. */
export function line(value, max = 200) {
  return String(value ?? "").replace(HIDDEN, " ").replace(/\s+/g, " ").trim().slice(0, max);
}

/** The next poll delay: doubles up to `POLL_MAX` after a failure, resets on success. */
export function nextDelay(current, failed) {
  return failed ? Math.min(POLL_MAX, Math.max(POLL_MIN, current) * 2) : POLL_MIN;
}

const VERDICT = { idle: "green", working: "yellow", loading: "yellow", starting: "yellow",
  stopping: "yellow", failed: "red", stopped: "none" };

const STYLES = `
:host { max-width: 920px; }
section { border: 1px solid var(--ml-line); border-radius: var(--ml-radius); background: var(--ml-surface);
  padding: 14px 16px; margin-bottom: 16px; }
h2 { margin: 0 0 10px; font-size: 16px; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; align-items: end; }
.roles { margin: 12px 0 0; padding: 0; list-style: none; font-size: 13px; color: var(--ml-muted); }
.roles b { color: var(--ml-text); }
.row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 14px; }
button.go, button.stop { padding: 9px 16px; border-radius: 8px; cursor: pointer; font-weight: 600;
  border: 1px solid var(--ml-line-strong); background: transparent; }
button.go { background: var(--ml-accent); color: var(--ml-on-accent); border-color: var(--ml-accent); }
button:disabled { opacity: .55; cursor: not-allowed; }
.note { font-size: 13px; color: var(--ml-muted); overflow-wrap: anywhere; }
.note.error { color: var(--ml-red-ink); }
label { display: block; font-size: 12.5px; font-weight: 600; color: var(--ml-muted); margin-bottom: 6px; }
input { width: 100%; padding: 9px 11px; border-radius: 8px; font: inherit; font-size: 14px;
  background: var(--ml-sunken); border: 1px solid var(--ml-line-strong); color: var(--ml-text); }
.agent { border-top: 1px solid var(--ml-line); padding: 10px 0; display: grid; gap: 4px; }
.agent:first-of-type { border-top: 0; }
.head { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
.head b { overflow-wrap: anywhere; }
.meta { color: var(--ml-muted); font-size: 12.5px; overflow-wrap: anywhere; }
.spacer { flex: 1; }
`;

class MlAgents extends MlElement {
  static props = { endpoint: "string" };
  static styles = STYLES;

  build() {
    this.agents = [];
    this.roleList = [];
    this.model = null;
    this.message = "";
    this.failed = false;
    this.busy = false;
    this.delay = POLL_MIN;
    this.nameField = h("input", { type: "text", id: "name", spellcheck: "false", autocomplete: "off",
      placeholder: "local-" });
    this.modelField = h("input", { type: "text", id: "model", spellcheck: "false", autocomplete: "off",
      value: "auto" });
    this.modelField.addEventListener("input", () => this.update());
    this.projectField = h("input", { type: "text", id: "project", spellcheck: "false", autocomplete: "off",
      placeholder: "optional" });
    this.roleControl = h("ml-select", { label: "What it may do", value: "" });
    this.effort = h("ml-select", { label: "Reasoning effort", hint: "Off thinks least and answers fastest", value: "off" });
    this.profile = h("ml-select", { label: "Kind of work", hint: "Coding serves a 256K context with Qwen3.8-27B",
      value: "chat" });
    this.profile.options = [{value:"chat",label:"Chat"},{value:"coding",label:"Coding"}];
    this.contextField = h("input", { type:"text", id:"context", value:"32K" });
    this.outputField = h("input", { type:"number", id:"output-tokens", min:"1", step:"1", placeholder:"None", value:"" });
    this.capFields = Object.fromEntries(["rounds", "calls", "steps", "seconds"].map(key => [key,
      h("input", {type:"number", id:`task-${key}`, min:"1", step:key === "seconds" ? "any" : "1", placeholder:"None"})]));
    this.limitNote = h("p", {class:"note"});
    for (const field of [this.outputField, ...Object.values(this.capFields)]) field.addEventListener("input", () => this.update());
    this.harness = h("ml-select", { label:"Harness", value:"ml-stack-agent" });
    this.profile.addEventListener("change", event => { if (event.detail?.value) { this.contextField.value = event.detail.value === "coding" ? "256K" : "32K"; this.update(); } });
    this.ceiling = h("ml-select", { label: "Maximum reasoning effort", hint: "It can raise its own effort up to this",
      value: "medium" });
    this.go = h("button", { class: "go", type: "button", onclick: () => this.start() }, "Start a local agent");
    this.note = h("span", { class: "note", role: "status", "aria-live": "polite" });
    this.preview = h("p", { class: "note" });
    this.rolesEl = h("ul", { class: "roles" });
    this.list = h("div", {});
    this.root.append(
      h("section", {},
        h("h2", {}, "Start a local agent"),
        h("div", { class: "grid" },
          h("div", {}, h("label", { for: "name" }, "Name"), this.nameField),
          h("div", {}, h("label", { for: "project" }, "Project folder"), this.projectField),
          this.roleControl, this.profile),
        h("details", {}, h("summary", {}, "Advanced options"),
          h("div", {class:"grid"}, h("div", {}, h("label", {for:"model"}, "Exact model path or reference (auto selects for the kind of work)"), this.modelField), this.effort, this.ceiling, this.harness,
            h("div", {}, h("label", {for:"context"}, "Context length"), this.contextField),
            h("div", {}, h("label", {for:"output-tokens"}, "Maximum output tokens"), this.outputField),
            ...Object.entries({rounds:"Task turns",calls:"Tool calls per task",steps:"Model calls per task",seconds:"Task wall time (seconds)"}).map(([key,label]) =>
              h("div", {}, h("label", {for:`task-${key}`}, label), this.capFields[key])))),
        this.limitNote,
        this.rolesEl, this.preview,
        h("div", { class: "row" }, this.go, this.note)),
      h("section", {}, h("h2", {}, "Running agents"), this.list));
  }

  connectedCallback() {
    super.connectedCallback();
    this.stopped = false;
    this.load();
    this.loadModel();
  }

  disconnectedCallback() {
    this.stopped = true;
    clearTimeout(this.timer);
  }

  base() {
    const e = this.endpoint || "/agents";
    return e.endsWith("/") ? e.slice(0, -1) : e;
  }

  async call(route, body) {
    const init = body === undefined
      ? { method: "GET", credentials: "same-origin", headers: { Accept: "application/json" } }
      : { method: "POST", credentials: "same-origin",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body) };
    if (this.base().startsWith("/ui/")) Object.assign(init.headers, window.fleetModel?.headers || {"X-ML-Stack-UI":"1"});
    const r = await fetch(`${this.base()}/${route}`, init);
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw Object.assign(new Error(data.error || `${r.status}`), { hint: data.hint });
    return data;
  }

  async load() {
    try {
      const data = await this.call("list");
      this.agents = data.agents ?? [];
      this.saved = data.saved ?? [];
      this.harness.options = (data.harnesses ?? []).map(name => ({value:name,label:name}));
      if (!this.roleList.length) {
        this.roleList = data.roles ?? [];
        this.roleControl.options = this.roleList.map((r) => ({ value: r.name, label: r.name }));
        this.roleControl.value = line(data.default_role, 40);
        const levels = (data.efforts ?? []).map((e) => ({ value: e, label: e }));
        this.effort.options = levels;
        this.ceiling.options = levels.filter((l) => l.value !== "auto");
        this.effort.value = line(data.default_effort, 12);
        this.ceiling.value = line(data.default_max_effort, 12);
        this.outputField.value = data.default_max_output_tokens ?? "";
      }
      if (!this.settingsLoaded && this.saved.length) { this.applySettings(this.saved[0]); this.settingsLoaded = true; }
      this.failed = false;
      this.delay = nextDelay(this.delay, false);
    } catch {
      this.failed = true;
      this.delay = nextDelay(this.delay, true);
    }
    this.update();
    clearTimeout(this.timer);
    if (!this.stopped) this.timer = setTimeout(() => this.load(), this.delay);
  }

  async loadModel() {
    try { this.model = await this.call("model"); } catch { this.model = null; }
    this.update();
  }

  async start() {
    if (this.busy) return;
    if ([this.outputField, ...Object.values(this.capFields)].some(field => !field.checkValidity())) { this.say("Limits must be positive numbers, or blank for None.", true); return; }
    this.busy = true;
    this.say("Starting. Loading the model can take a minute.", false);
    try {
      const got = await this.call("start", {
        model: this.modelField.value.trim() || "auto", name: this.nameField.value.trim(),
        role: this.roleControl.value, profile: this.profile.value || "chat", effort: this.effort.value || "off", max_effort: this.ceiling.value || "medium", project: this.projectField.value.trim(),
        harness:this.harness.value, ctx:this.contextField.value.trim(), max_output_tokens:this.outputField.value === "" ? null : Number(this.outputField.value),
        task_caps:Object.fromEntries(Object.entries(this.capFields).map(([key, field]) => [key, field.value === "" ? null : Number(field.value)])) });
      this.say(got.already ? `${line(got.name, 48)} is already running.` : `${line(got.name, 48)} started.`, false);
    } catch (e) {
      this.say(`${line(e.message, 300)}${e.hint ? ` Run: ${line(e.hint, 200)}` : ""}`, true);
    }
    this.busy = false;
    this.load();
  }

  applySettings(saved) {
    if (!saved) return;
    this.nameField.value = saved.name; this.modelField.value = saved.model;
    this.projectField.value = saved.project; this.roleControl.value = saved.role;
    this.profile.value = saved.profile; this.effort.value = saved.effort;
    this.ceiling.value = saved.max_effort; this.harness.value = saved.harness;
    this.contextField.value = String(saved.ctx);
    this.outputField.value = saved.max_output_tokens ?? "";
    for (const [key, field] of Object.entries(this.capFields)) field.value = saved.task_caps?.[key] ?? "";
    this.update();
  }

  async stop(name) {
    this.say(`Stopping ${line(name, 48)}.`, false);
    try {
      await this.call("stop", { name });
      this.say(`${line(name, 48)} stopped.`, false);
    } catch (e) {
      this.say(line(e.message, 300), true);
    }
    this.load();
  }

  say(text, error) {
    this.message = text;
    this.note.textContent = text;
    this.note.className = error ? "note error" : "note";
  }

  update() {
    if (!this.list) return;
    const effective = [`output tokens: ${this.outputField.value || "None"}`,
      ...Object.entries(this.capFields).map(([key,field]) => `${({rounds:"turns",calls:"tool calls",steps:"model calls",seconds:"wall seconds"})[key]}: ${field.value || "None"}`)];
    this.limitNote.textContent = `Effective limits — ${effective.join(", ")}.`
      + (this.capFields.seconds.value === "" ? " Task wall time is None. Tasks could run indefinitely until you cancel them." : "");
    this.go.toggleAttribute("disabled", this.busy);
    this.rolesEl.replaceChildren(...this.roleList.map((r) =>
      h("li", {}, h("b", {}, line(r.name, 40)), ` ${line(r.summary, 200)}`)));
    const selected = this.modelField.value.trim() || "auto";
    const m = this.model?.profiles?.[this.profile.value || "chat"] || this.model;
    this.preview.textContent = selected !== "auto"
      ? `Selected model: ${line(selected.split(/[\\/]/).pop(), 120)}`
      : !m ? "Checking the automatic model choice…" : m.ok
        ? `Auto for ${this.profile.value || "chat"}: ${line(m.name, 80)}${m.size_bytes ? ` (${fmt(m.size_bytes, "bytes-iec")})` : ""}.`
        : `${line(m.problem, 300)}${m.hint ? ` Run: ${line(m.hint, 200)}` : ""}`;
    this.preview.title = selected === "auto" ? "" : selected;
    this.list.replaceChildren(...(this.failed
      ? [h("p", { class: "note error", role: "alert" }, "The workspace did not answer.")]
      : this.agents.length ? this.agents.map((a) => this.card(a))
        : [h("p", { class: "note" }, "No local agent is running.")]));
  }

  card(a) {
    const last = a.last_message && a.last_message.from
      ? `Last message from ${line(a.last_message.from, 48)} (${line(a.last_message.type, 16)}${a.last_message.obeyed ? "" : ", not acted on"})`
        + `${a.last_message.summary ? `: ${line(a.last_message.summary, 160)}` : ""}` : "No messages yet.";
    return h("div", { class: "agent" },
      h("div", { class: "head" },
        h("b", {}, line(a.name, 48)),
        h("ml-chip", { verdict: VERDICT[a.state] || "none", label: line(a.state, 20) }),
        h("span", { class: "spacer" }),
        h("button", {type:"button", onclick:() => this.applySettings(this.saved.find(row => row.name === a.name))}, "Use settings"),
        h("button", { class: "stop", type: "button", onclick: () => this.stop(a.name) }, "Stop")),
      h("div", { class: "meta" },
        `${line(a.model, 80)} on ${line(a.harness, 24)}, ${line(a.role, 40)}`
        + `, ${Math.round((Number(a.ctx) || 0) / 1024)}K context, effort ${line(a.effort, 12)} (ceiling ${line(a.max_effort, 12)})`
        + `, ${a.max_output_tokens ?? "None"} maximum output tokens`
        + `, ${fmt(a.memory_bytes, "bytes-iec")} held, ${Number(a.tasks) || 0} tasks, ${Number(a.steps) || 0} steps`),
      h("div", { class: "meta" }, a.device?.label ? `${line(a.device.label, 128)} · ${line(a.device.verification, 24)}` : "Device not recorded"),
      a.detail ? h("div", { class: "meta" }, line(a.detail, 200)) : null,
      h("div", { class: "meta" }, last));
  }
}
define("ml-agents", MlAgents);
