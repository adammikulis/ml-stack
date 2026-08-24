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
        el("b", {}, `${running}/${slots}${queued ? ` +${queued} waiting` : ""}`)),
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
        el("div", { class: "big" }, "No machines are answering yet"),
        el("div", {}, "Run ml-stack-peers setup on another machine, with the same passphrase,"),
        el("div", {}, "then start ml-stack-traind there. It will appear here on its own."));

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
const state = { step: 0, mode: "", name: "", group: "ml-stack", autostart: "login" };

function steps(n) {
  return el("div", { class: "steps" },
    [0, 1, 2, 3].map((i) => el("i", { class: i <= n ? "on" : "" })));
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
      go.innerHTML = '<span class="spin"></span> Setting up — this takes a moment on purpose';
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

  // step 3: joined — what next
  const cmd = el("pre", { class: "cmd" }, "ml-stack-peers setup");
  const looking = el("div", { class: "hint" },
    el("span", { class: "spin" }), " Watching for other machines…");
  const found = el("div");

  const poll = async () => {
    const r = await api("/ui/setup/peers");
    const others = (r.peers || []).filter((p) => !p.is_self);
    if (others.length) {
      looking.remove();
      found.replaceChildren(el("div", { class: "ok" },
        `Found ${others.map((p) => p.name).join(", ")}.`),
        el("button", { onclick: () => location.reload() }, "Open the cluster"));
      return;
    }
    setTimeout(poll, 3000);
  };
  poll();

  show(el("div", { class: "centre" }, el("div", { class: "card" },
    steps(3),
    el("h1", {}, `This machine has joined “${state.group}”`),
    el("p", { class: "sub" }, "Now do the same on every other machine you want to train with:"),
    cmd,
    el("div", { class: "hint" }, "Same passphrase, same group name. Then start the daemon there."),
    looking, found,
    el("button", { class: "ghost", onclick: () => location.reload() }, "Skip — open the cluster"))));
}

// ---------------------------------------------------------------- boot
(async function start() {
  const setup = await api("/ui/setup");
  if (setup.needs_setup) return wizard(setup);
  const s = await api("/ui/session");
  return s.signed_in ? fleet() : signIn();
})();
