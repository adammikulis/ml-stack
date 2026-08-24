// The fleet UI. Vanilla ES modules on purpose: this package is device tier and ships
// with no build step, so the browser gets what is on disk. Everything worth testing
// lives behind a route on the daemon, not here.

const root = document.getElementById("root");
const H = { "X-ML-Stack-UI": "1", "Content-Type": "application/json" };

async function api(path, opts = {}) {
  const r = await fetch(path, { ...opts, headers: { ...H, ...(opts.headers || {}) } });
  let body = {};
  try { body = await r.json(); } catch { /* empty body is fine */ }
  return { ok: r.ok, status: r.status, ...body };
}

const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    n.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return n;
};

const show = (...nodes) => { root.replaceChildren(...nodes); };

// ---------------------------------------------------------------- vendors
// The badge has to be right: torch's ROCm build answers True to every CUDA question,
// so the daemon reports vendor separately and this only renders what it is told.
function vendorOf(d = {}) {
  if (d.rocm || d.vendor === "amd") return { key: "amd", text: "AMD · ROCm" };
  if (d.cuda || d.vendor === "nvidia") return { key: "nvidia", text: "NVIDIA · CUDA" };
  if (d.vendor === "apple") return { key: "apple", text: "Apple · Metal" };
  return { key: "cpu", text: `${d.cpus || "?"} CPU` };
}

const gb = (n) => (n === undefined || n === null ? null : `${Number(n).toFixed(1)} GB`);

// ---------------------------------------------------------------- peer card
function peerCard(p) {
  const d = p.device || {};
  const v = vendorOf(d);
  const slots = p.slots || 1;
  const free = p.free === undefined ? (p.busy ? 0 : slots) : p.free;
  const running = slots - free;
  const queued = p.queued || 0;

  const pips = el("div", { class: "slots" },
    Array.from({ length: Math.min(slots, 16) }, (_, i) =>
      el("i", { class: `slot${i < running ? " on" : ""}` })));

  const meters = [
    el("div", { class: "meter" },
      el("div", { class: "cap" },
        el("span", {}, "capacity"),
        // Spelled out: "0/4" beside the word "idle" reads as "none of four available",
        // which is the opposite of what it means.
        el("b", {}, running === 0 ? `all ${slots} free`
          : `${running} of ${slots} busy${queued ? ` · ${queued} waiting` : ""}`)),
      pips),
  ];

  if (d.vram_total_gb) {
    const used = d.vram_total_gb - (d.vram_free_gb ?? d.vram_total_gb);
    const pct = Math.max(0, Math.min(100, (used / d.vram_total_gb) * 100));
    meters.push(el("div", { class: "meter" },
      el("div", { class: "cap" },
        el("span", {}, d.vendor === "apple" ? "unified memory" : "video memory"),
        el("b", {}, `${gb(d.vram_free_gb)} free of ${gb(d.vram_total_gb)}`)),
      el("div", { class: "track" },
        el("div", { class: "fill vram", style: `width:${pct}%` }))));
  }

  // Temperature, clock and power, when a vendor tool or darwin-perf reported them.
  // Throttling is the one that changes a decision: a card clocking down looks exactly
  // like a bad hyperparameter from a loss curve, and the measured rates would record
  // the box as slow and send it less work — the wrong correction to a heat problem.
  const chips = [];
  if (d.temp_c !== undefined) {
    const hot = d.temp_c >= 85 ? " hot" : d.temp_c >= 75 ? " warm" : "";
    chips.push(el("span", { class: `label temp${hot}` }, `${Math.round(d.temp_c)}°C`));
  }
  if (d.clock_mhz) chips.push(el("span", { class: "label" }, `${Math.round(d.clock_mhz)} MHz`));
  if (d.power_w !== undefined) chips.push(el("span", { class: "label" }, `${d.power_w.toFixed(1)} W`));
  if (d.throttled) chips.push(el("span", { class: "label bad" }, "▼ throttled"));
  if (chips.length) meters.push(el("div", { class: "chips" }, chips));

  if (d.gpu_util_pct !== undefined) {
    meters.push(el("div", { class: "meter" },
      el("div", { class: "cap" },
        el("span", {}, "gpu"),
        el("b", {}, `${Math.round(d.gpu_util_pct)}% busy`)),
      el("div", { class: "track" },
        el("div", { class: "fill util", style: `width:${Math.min(100, d.gpu_util_pct)}%` }))));
  }

  const state = free === 0
    ? { cls: "busy", text: queued ? `working · ${queued} waiting` : "working" }
    : { cls: "", text: running ? `${running} running · ${free} free` : "idle" };

  return el("div", { class: `peer${p.is_self ? " self" : ""}` },
    el("div", { class: "row" },
      el("div", {},
        el("div", { class: "name" }, p.name),
        el("div", { class: "url" }, p.base_url || p.host || ""))),
    el("div", { class: "row", style: "margin-top:10px;gap:6px;flex-wrap:wrap" },
      el("span", { class: `badge ${v.key}` }, el("span", { class: "chip" }), v.text),
      d.gpu ? el("span", { class: "label" }, d.gpu) : null,
      d.ram_gb ? el("span", { class: "label" }, `${Math.round(d.ram_gb)} GB RAM`) : null),
    (d.labels || []).length
      ? el("div", { class: "labels" }, d.labels.map((l) => el("span", { class: "label" }, l)))
      : null,
    ...meters,
    el("div", { class: "state" },
      el("span", { class: `pulse ${state.cls}` }), state.text));
}

// ---------------------------------------------------------------- fleet view
let timer = null;
async function fleet() {
  const r = await api("/ui/peers");
  if (r.status === 401) return signIn();

  const peers = r.peers || [];
  const totalSlots = peers.reduce((a, p) => a + (p.slots || 1), 0);
  const busy = peers.reduce((a, p) => a + ((p.slots || 1) - (p.free ?? 0)), 0);
  const cards = peers.length
    ? el("div", { class: "grid" }, peers.map(peerCard))
    : el("div", { class: "empty" },
        el("div", { class: "big" }, "Just this machine so far"),
        el("div", {}, "Install ml-stack on another machine and open it."),
        el("div", {}, "Type the same passphrase, and it appears here on its own."));

  show(el("div", { class: "app" },
    el("header", { class: "top" },
      el("div", { class: "brand" },
        el("span", { class: "dot" }), "ml-stack",
        el("span", { class: "group" }, r.group ? `· ${r.group}` : "")),
      el("nav", { class: "tabs" },
        el("a", { class: "on", href: "#" }, "Cluster"),
        el("a", { href: "#", onclick: signOut }, "Sign out"))),
    el("main", {},
      el("h1", {}, "Cluster"),
      el("p", { class: "sub" },
        peers.length === 1 ? "One machine." : `${peers.length} machines on this network.`),
      el("div", { class: "stat" },
        el("div", {}, el("div", { class: "n" }, peers.length), el("div", { class: "k" }, "machines")),
        el("div", {}, el("div", { class: "n" }, `${busy}/${totalSlots}`), el("div", { class: "k" }, "slots in use")),
        el("div", {}, el("div", { class: "n" },
          peers.filter((p) => vendorOf(p.device || {}).key !== "cpu").length),
          el("div", { class: "k" }, "with a GPU"))),
      cards)));

  clearTimeout(timer);
  timer = setTimeout(fleet, 4000);
}

// ---------------------------------------------------------------- sign in
function signIn(msg) {
  const pass = el("input", { type: "password", id: "p", autofocus: "1",
                             placeholder: "the words you chose" });
  const go = el("button", {}, "Sign in");
  const box = el("div", { class: "card" },
    el("h1", {}, "Sign in"),
    el("p", { class: "sub" }, "Type the passphrase this cluster was set up with."),
    el("label", { for: "p" }, "Passphrase"), pass,
    go, msg ? el("div", { class: "err" }, msg) : null);

  const submit = async () => {
    go.disabled = true; go.textContent = "Checking…";
    const r = await api("/ui/session", { method: "POST",
      body: JSON.stringify({ passphrase: pass.value }) });
    if (r.ok) return fleet();
    signIn(r.error || "That did not match.");
  };
  go.onclick = submit;
  pass.onkeydown = (e) => { if (e.key === "Enter") submit(); };
  show(el("div", { class: "centre" }, box));
}

async function signOut(e) {
  e.preventDefault();
  clearTimeout(timer);
  await api("/ui/session", { method: "DELETE" });
  signIn();
}

// ---------------------------------------------------------------- wizard
const state = { step: 0, mode: "", name: "", group: "ml-stack", prefs: null };

function steps(n) {
  return el("div", { class: "steps" },
    [0, 1, 2, 3, 4].map((i) => el("i", { class: i <= n ? "on" : "" })));
}

// Step 3: what this machine should do. Every box is pre-ticked from what the machine
// actually is, and every one shows why, so a wrong guess is visible rather than silent.
async function prefsStep(next) {
  const r = await api("/ui/setup/suggest");
  const s = r.suggest || {};
  const m = r.machine || {};
  const pick = {
    labels: (s.labels?.value || []).join(","),
    slots: s.slots?.value ?? 1,
    autostart: s.autostart?.value || "manual",
    work_hours: !!s.work_hours?.value,
    on_paused: s.on_paused?.value || "stop",
  };

  const spec = (label, why) => el("span", {},
    el("b", {}, label), why ? el("span", { class: "why" }, why) : null);

  const radio = (group, value, label, why) => {
    const id = `${group}-${value}`;
    const input = el("input", { type: "radio", name: group, id,
                                ...(pick[group] === value ? { checked: "1" } : {}) });
    input.onchange = () => { pick[group] = value; };
    return el("label", { class: "opt", for: id }, input, spec(label, why));
  };

  const check = (key, label, why) => {
    const input = el("input", { type: "checkbox", id: key,
                                ...(pick[key] ? { checked: "1" } : {}) });
    input.onchange = () => { pick[key] = input.checked; };
    return el("label", { class: "opt", for: key }, input, spec(label, why));
  };

  const slots = el("input", { type: "number", min: "1", max: "64", value: String(pick.slots),
                              class: "num" });
  slots.oninput = () => { pick.slots = Math.max(1, parseInt(slots.value || "1", 10)); };

  const go = el("button", {}, "Continue");
  const outcome = el("div");
  go.onclick = async () => {
    go.disabled = true; go.innerHTML = '<span class="spin"></span> Applying…';
    const res = await api("/ui/setup/prefs", { method: "POST", body: JSON.stringify({
      ...pick, labels: pick.labels ? pick.labels.split(",") : [] }) });
    if (res.manual) {
      go.disabled = false; go.textContent = "Continue";
      outcome.replaceChildren(
        el("div", { class: "err" }, res.manual_why || "This part needs administrator rights."),
        el("pre", { class: "cmd" }, res.manual),
        el("button", { class: "ghost", onclick: () => next() }, "Skip this — continue"));
      return;
    }
    next();
  };

  show(el("div", { class: "centre" }, el("div", { class: "card wide" },
    steps(3),
    el("h1", {}, "What should this machine do?"),
    el("p", { class: "sub" },
      m.gpu ? `Found ${m.gpu}. These are set from that — change any of them.`
            : "These are set from what this machine is — change any of them."),

    el("div", { class: "group" },
      el("h2", {}, "Its job"),
      radio("labels", "train", "Train models", s.labels?.why),
      radio("labels", "prep", "Prepare data", "tokenizing and packing, while others train"),
      radio("labels", "train,prep", "Both", "")),

    el("div", { class: "group" },
      el("h2", {}, "At once"),
      el("label", { class: "opt inline" }, slots,
        spec("jobs at a time", s.slots?.why))),

    el("div", { class: "group" },
      el("h2", {}, "Starting up"),
      radio("autostart", "login", "When I log in",
        s.autostart?.value === "login" ? s.autostart.why : ""),
      radio("autostart", "boot", "When the computer starts",
        "even before anyone logs in — your Mac will ask for your password"),
      radio("autostart", "manual", "Only when I open it", "")),

    el("div", { class: "group" },
      el("h2", {}, "When I need the machine"),
      check("work_hours", "Don't take work on weekdays, 9 to 5",
        s.work_hours?.why || "for a machine somebody works on"),
      el("label", { class: "opt" },
        el("input", { type: "checkbox", checked: "1", disabled: "1" }),
        spec("Pausing stops what is running", s.on_paused?.why))),

    go, outcome)));
}

function wizard(setup, err) {
  const next = (s) => { state.step = s; wizard(setup); };

  if (state.step === 0) {
    state.name = state.name || setup.name || "";
    const name = el("input", { type: "text", value: state.name, id: "n" });
    return show(el("div", { class: "centre" }, el("div", { class: "card" },
      steps(0),
      el("h1", {}, "Set up this machine"),
      el("p", { class: "sub" },
        "Machines that share a passphrase find each other and can train together. "
        + "Nothing else needs configuring."),
      el("label", { for: "n" }, "What should this machine be called?"),
      name,
      el("div", { class: "hint" }, "Other machines will show it under this name."),
      el("button", { onclick: () => { state.name = name.value.trim() || setup.name; next(1); } },
        "Continue"))));
  }

  if (state.step === 1) {
    const pick = (mode) => { state.mode = mode; next(2); };
    return show(el("div", { class: "centre" }, el("div", { class: "card" },
      steps(1),
      el("h1", {}, "Is this your first machine?"),
      el("p", { class: "sub" }, "Both roads lead to the same passphrase — this just changes what happens next."),
      el("div", { class: "choices" },
        el("button", { class: "choice", onclick: () => pick("first") },
          el("span", { class: "t" }, "This is my first machine"),
          el("span", { class: "d" }, "You will choose a passphrase, then type the same one on the others.")),
        el("button", { class: "choice", onclick: () => pick("join") },
          el("span", { class: "t" }, "Another machine is already set up"),
          el("span", { class: "d" }, "Type the passphrase you used there and this one joins it."))))));
  }

  if (state.step === 2) {
    const p1 = el("input", { type: "password", id: "p1", autofocus: "1" });
    const p2 = el("input", { type: "password", id: "p2" });
    const grp = el("input", { type: "text", value: state.group, id: "g" });
    const go = el("button", {}, state.mode === "first" ? "Create the cluster" : "Join");
    const note = el("div", { class: "hint" });

    const check = () => {
      const a = p1.value, b = p2.value;
      if (a.trim().length && a.trim().length < 8) {
        note.textContent = "At least 8 characters. A few words you will remember beats a short complicated one.";
      } else if (a && b && a !== b) {
        note.textContent = "These do not match yet.";
      } else { note.textContent = "Everyone who knows this can run commands on every machine in the group."; }
      go.disabled = !(a.trim().length >= 8 && a === b);
    };
    p1.oninput = p2.oninput = check;
    check();

    go.onclick = async () => {
      go.disabled = true;
      go.innerHTML = '<span class="spin"></span> Setting up — this takes a moment';
      const r = await api("/ui/setup/join", { method: "POST",
        body: JSON.stringify({ passphrase: p1.value, group: grp.value.trim() || "ml-stack" }) });
      if (!r.ok) return wizard(setup, r.error || "Could not set up.");
      state.group = grp.value.trim() || "ml-stack";
      next(3);
    };

    return show(el("div", { class: "centre" }, el("div", { class: "card" },
      steps(2),
      el("h1", {}, state.mode === "first" ? "Choose a passphrase" : "Type the passphrase"),
      el("p", { class: "sub" },
        state.mode === "first"
          ? "You will type this same passphrase on every other machine."
          : "The same words you used on the machine that is already set up."),
      el("label", { for: "p1" }, "Passphrase"), p1,
      el("label", { for: "p2" }, "Again"), p2,
      note,
      el("label", { for: "g" }, "Group name"),
      grp,
      el("div", { class: "hint" },
        "Only matters if two separate clusters share this network. Leave it alone otherwise."),
      go,
      err ? el("div", { class: "err" }, err) : null)));
  }

  if (state.step === 3) return prefsStep(() => next(4));

  // step 4: joined — what next. Deliberately not a command: telling someone to open a
  // terminal is the same wall as telling them to paste a 32-byte key, one layer up.
  const looking = el("div", { class: "watching" },
    el("span", { class: "spin" }), " Watching for other machines…");
  const found = el("div");
  const skip = el("button", { class: "ghost", onclick: () => location.reload() },
    "Open the cluster");

  const poll = async () => {
    const r = await api("/ui/setup/peers");
    const others = (r.peers || []).filter((p) => !p.is_self);
    if (others.length) {
      looking.remove();
      // Replaces the skip button rather than sitting above it: two buttons that do the
      // same thing is a choice the reader has to stop and make, about nothing.
      skip.remove();
      found.replaceChildren(
        el("div", { class: "ok" },
          `Found ${others.map((p) => p.name).join(", ")}.`),
        el("button", { onclick: () => location.reload() }, "Open the cluster"));
      return;
    }
    setTimeout(poll, 3000);
  };
  poll();

  const step = (n, title, body) =>
    el("li", {}, el("span", { class: "num" }, n),
      el("div", {}, el("b", {}, title), body ? el("div", { class: "d" }, body) : null));

  show(el("div", { class: "centre" }, el("div", { class: "card" },
    steps(3),
    el("h1", {}, `Joined “${state.group}”`),
    el("p", { class: "sub" }, "Now add your other machines. On each one:"),
    el("ol", { class: "howto" },
      step(1, "Install ml-stack", "The same installer you used here."),
      step(2, "Open it", "It opens in your browser and asks the same questions."),
      step(3, "Type the same passphrase",
        `The words you just chose${state.group !== "ml-stack"
          ? `, and the group name “${state.group}”` : ""}.`)),
    el("p", { class: "hint" },
      "They find each other on their own. Nothing else to configure."),
    looking, found,
    el("details", { class: "aside" },
      el("summary", {}, "Already installed it, and happy with a terminal?"),
      el("pre", { class: "cmd" }, "ml-stack-peers setup")),
    skip)));
}

// ---------------------------------------------------------------- boot
(async function start() {
  const setup = await api("/ui/setup");
  if (setup.needs_setup) return wizard(setup);
  const s = await api("/ui/session");
  return s.signed_in ? fleet() : signIn();
})();


// ---------------------------------------------------------------- closing
// Asked once, by the native window, when someone clicks the close button.
window.mlStackAskOnClose = () => {
  if (document.querySelector(".sheet")) return;

  const remember = el("input", { type: "checkbox", id: "remember", checked: "1" });
  const choose = (mode) => async () => {
    sheet.remove();
    if (window.pywebview) await window.pywebview.api.close_choice(mode, remember.checked);
  };

  const sheet = el("div", { class: "sheet" },
    el("div", { class: "sheet-card" },
      el("h1", {}, "Close ml-stack?"),
      el("p", { class: "sub" },
        "This machine is part of your cluster. Keeping it running lets the others "
        + "send it work while the window is shut."),
      el("div", { class: "choices" },
        el("button", { class: "choice", onclick: choose("background") },
          el("span", { class: "t" }, "Keep running in the background"),
          el("span", { class: "d" }, "Stays in the cluster. Open it again any time.")),
        el("button", { class: "choice", onclick: choose("quit") },
          el("span", { class: "t" }, "Quit"),
          el("span", { class: "d" },
            "Leaves the cluster until you open it again."))),
      el("label", { class: "opt", for: "remember", style: "margin-top:16px" },
        remember,
        el("span", {},
          el("b", {}, "Save my setting and don't ask again"),
          el("span", { class: "why" },
            "You can change it later under this machine's settings."))),
      el("button", { class: "ghost", onclick: () => sheet.remove() }, "Cancel")));

  document.body.append(sheet);
};
