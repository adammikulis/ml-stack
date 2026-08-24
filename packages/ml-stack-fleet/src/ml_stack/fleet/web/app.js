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

const row = (label, why) => el("span", {},
  el("b", {}, label), why ? el("span", { class: "why" }, why) : null);

const TABS = [["Chat", () => chat()], ["Cluster", () => fleet()],
              ["Models", () => models()], ["Settings", () => settings()]];

const top = (current, group = "") =>
  el("header", { class: "top" },
    el("div", { class: "brand" },
      el("span", { class: "dot" }), "ml-stack",
      el("span", { class: "group" }, group ? `· ${group}` : "")),
    el("nav", { class: "tabs" },
      ...TABS.map(([label, go]) => (label === current
        ? el("a", { class: "on", href: "#" }, label)
        : el("a", { href: "#", onclick: (e) => { e.preventDefault(); go(); } }, label))),
      el("a", { href: "#", onclick: signOut }, "Sign out")));

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
        el("span", {}, "jobs"),
        // Spelled out: "0/4" beside the word "idle" reads as "none of four
        // available", which is the opposite of what it means.
        el("b", {}, running === 0
          ? `none running · ${slots} slot${slots === 1 ? "" : "s"} free`
          : `${running} running of ${slots}${queued ? ` · ${queued} waiting` : ""}`)),
      pips),
  ];

  if (d.ram_gb) {
    const used = d.ram_used_gb;
    const pct = used === undefined ? null
      : Math.max(0, Math.min(100, (used / d.ram_gb) * 100));
    meters.push(el("div", { class: "meter" },
      el("div", { class: "cap" },
        el("span", {}, "memory"),
        el("b", {}, used === undefined ? `${gb(d.ram_gb)} total`
          : `${gb(used)} of ${gb(d.ram_gb)} in use`)),
      pct === null ? null
        : el("div", { class: "track" },
            el("div", { class: "fill", style: `width:${pct}%` }))));
  }

  if (d.cpu_pct !== undefined) {
    meters.push(el("div", { class: "meter" },
      el("div", { class: "cap" },
        el("span", {}, "processors"),
        el("b", {}, `${Math.round(d.cpu_pct)}% busy`)),
      el("div", { class: "track" },
        el("div", { class: "fill util",
                    style: `width:${Math.min(100, d.cpu_pct)}%` }))));
  }

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
    (p.clusters || []).length > 1
      ? el("div", { class: "labels" },
          p.clusters.map((c) => el("span", { class: "label" }, c)))
      : null,
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

  // Which clusters this machine is in, and the way in and out of one.
  const joined = el("div", { class: "group" },
    el("h2", {}, "Clusters"),
    el("div", { class: "hint" }, "Loading…"));
  const drawJoined = (rows) => {
    const words = el("input", { type: "password", id: "words",
      placeholder: "passphrase" });
    const named = el("input", { type: "text", id: "cname",
      placeholder: "name (optional)" });
    const note = el("div");
    const add = el("button", { class: "ghost small" }, "Join");
    add.onclick = async () => {
      const said = words.value.trim();
      if (said.length < 8) {
        note.replaceChildren(el("div", { class: "err" },
          "A passphrase needs at least 8 characters."));
        return;
      }
      add.disabled = true;
      const got = await api("/ui/clusters", { method: "POST",
        body: JSON.stringify({ passphrase: said, group: named.value.trim() }) });
      add.disabled = false;
      if (got.error) {
        note.replaceChildren(el("div", { class: "err" }, got.error));
        return;
      }
      words.value = ""; named.value = "";
      note.replaceChildren(el("div", { class: "ok" }, `Joined ${got.joined}.`));
      drawJoined(got.clusters || []);
      fleet();
    };

    joined.replaceChildren(
      el("h2", {}, "Clusters"),
      el("div", { class: "hint" },
        "This machine answers to every cluster listed. Machines find each other by "
        + "sharing a passphrase."),
      ...(rows.length
        ? rows.map((c, i) =>
            el("div", { class: "row" },
              el("span", {}, el("b", {}, c.group),
                el("span", { class: "why" },
                  i === 0 ? "this machine answers as this one" : "also in this one")),
              el("button", { class: "ghost small",
                onclick: async () => {
                  const left = await api("/ui/clusters", { method: "DELETE",
                    body: JSON.stringify({ group: c.group }) });
                  drawJoined(left.clusters || []);
                  fleet();
                } }, "Leave")))
        : [el("div", { class: "hint" }, "In no cluster. Join one below.")]),
      el("div", { class: "searchrow" }, words, named, add),
      note);
  };
  api("/ui/clusters").then((c) => drawJoined(c.clusters || []));

  show(el("div", { class: "app" },
    top("Cluster", r.group),
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
      cards,
      joined)));

  clearTimeout(timer);
  timer = setTimeout(fleet, 4000);
}

// ---------------------------------------------------------------- chat
let openChat = null;

function say(list, role, text) {
  const node = el("div", { class: `msg ${role}` }, text);
  list.append(node);
  list.scrollTop = list.scrollHeight;
  return node;
}

async function chat(cid) {
  clearTimeout(timer);
  const where = await api("/ui/chat");
  if (where.status === 401) return signIn();
  const saved = await api("/ui/conversations");
  const available = where.models || [];
  openChat = cid || openChat;

  const picker = el("select", { id: "model" },
    ...available.map((m) => el("option", { value: m.model },
      m.local ? m.model : `${m.model} — on ${m.peer}`)));

  const list = el("div", { class: "messages" });
  const box = el("textarea", { id: "ask", rows: "3",
    placeholder: available.length ? "Ask it something" : "" });
  const sendIt = el("button", {}, "Send");
  const note = el("div");

  const load = async (id) => {
    openChat = id;
    const one = await api(`/ui/conversations/${id}`);
    list.replaceChildren();
    for (const m of one.messages || []) say(list, m.role, m.content);
    if (one.model) picker.value = one.model;
  };

  const ask = async () => {
    const text = box.value.trim();
    if (!text) return;
    box.value = "";
    note.replaceChildren();
    if (!openChat) {
      const made = await api("/ui/conversations", { method: "POST",
        body: JSON.stringify({ model: picker.value }) });
      openChat = made.id;
    }
    say(list, "user", text);
    const reply = say(list, "assistant", "");
    sendIt.disabled = true;
    try {
      const r = await fetch("/ui/chat", { method: "POST", headers: H,
        body: JSON.stringify({ conversation: openChat, model: picker.value,
          messages: [{ role: "user", content: text }] }) });
      if (!r.ok) {
        const why = await r.json().catch(() => ({}));
        reply.remove();
        note.replaceChildren(el("div", { class: "err" },
          why.error || "That did not go through."));
        return;
      }
      const reader = r.body.getReader();
      const decode = new TextDecoder();
      let pending = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        pending += decode.decode(value, { stream: true });
        const lines = pending.split("\n");
        pending = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const frame = line.slice(5).trim();
          if (!frame || frame === "[DONE]") continue;
          try {
            const parsed = JSON.parse(frame);
            for (const c of parsed.choices || []) {
              const piece = c.delta?.content || c.message?.content;
              if (piece) {
                reply.append(document.createTextNode(piece));
                list.scrollTop = list.scrollHeight;
              }
            }
          } catch { /* a half-arrived frame catches up next read */ }
        }
      }
    } finally {
      sendIt.disabled = false;
      chatList();
    }
  };
  sendIt.onclick = ask;
  box.onkeydown = (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) { e.preventDefault(); ask(); }
  };

  const sidebar = el("div", { class: "chats" });
  const paint = (rows) => {
    sidebar.replaceChildren(
      el("button", { class: "ghost small",
        onclick: () => { openChat = null; list.replaceChildren(); paint(rows); } },
        "New chat"),
      ...rows.map((c) =>
        el("div", { class: `chatrow${c.id === openChat ? " on" : ""}` },
          el("a", { href: "#",
            onclick: (e) => { e.preventDefault(); load(c.id).then(chatList); } },
            c.title || "New chat"),
          el("button", { class: "ghost small",
            onclick: async () => {
              await api(`/ui/conversations/${c.id}`, { method: "DELETE" });
              if (openChat === c.id) { openChat = null; list.replaceChildren(); }
              chatList();
            } }, "Delete"))));
  };
  const chatList = async () =>
    paint((await api("/ui/conversations")).conversations || []);
  paint(saved.conversations || []);

  show(el("div", { class: "app" },
    top("Chat"),
    el("main", { class: "chatmain" },
      sidebar,
      el("div", { class: "talk" },
        available.length
          ? el("div", { class: "pickrow" },
              el("label", { for: "model" }, "Model"), picker)
          : el("div", { class: "hint" },
              "No model is running yet. Start one on the Models screen, or leave "
              + "another machine on your network serving one."),
        list,
        note,
        available.length ? el("div", { class: "askrow" }, box, sendIt) : null))));

  if (openChat) load(openChat);
}

// ---------------------------------------------------------------- models
const fileSize = (n) => {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)} GB`;
  if (n >= 1e6) return `${Math.round(n / 1e6)} MB`;
  if (n >= 1e3) return `${Math.round(n / 1e3)} KB`;
  return `${n || 0} bytes`;
};

async function models(message) {
  clearTimeout(timer);
  const m = await api("/ui/models");
  if (m.status === 401) return signIn();
  const run = await api("/ui/serving");
  const running = new Set((run.running || []).flatMap((x) => x.models || []));
  const portOf = (name) =>
    (run.running || []).find((x) => (x.models || []).includes(name))?.port;

  const note = el("div");

  const serve = async (name) => {
    note.replaceChildren(el("div", { class: "hint" },
      el("span", { class: "spin" }), ` Starting ${name}…`));
    const r = await api("/ui/serving", { method: "POST",
      body: JSON.stringify({ name }) });
    note.replaceChildren(r.error
      ? el("div", { class: "err" }, r.error)
      : el("div", { class: "ok" }, `${name} is running. Open Chat to use it.`));
    if (!r.error) setTimeout(() => models(), 600);
  };

  const halt = async (name) => {
    await api("/ui/serving", { method: "DELETE",
      body: JSON.stringify({ port: portOf(name) }) });
    models();
  };
  const get = async (name, source, draft) => {
    note.replaceChildren(el("div", { class: "hint" },
      el("span", { class: "spin" }), ` Getting ${name}…`));
    const r = await api("/ui/models", { method: "POST",
      body: JSON.stringify({ name, source, draft: draft || "" }) });
    if (r.error) {
      note.replaceChildren(el("div", { class: "err" }, r.error));
      return;
    }
    note.replaceChildren();
    models();
  };

  const bar = (g) => {
    const pct = g.total ? Math.min(100, Math.round((g.done / g.total) * 100)) : 0;
    return el("div", { class: "row" },
      el("span", { class: "grow" },
        el("b", {}, g.name),
        el("span", { class: "why" },
          g.state === "failed" ? g.error
            : g.total ? `${fileSize(g.done)} of ${fileSize(g.total)} — ${pct}%`
            : (g.note || "Starting…")),
        g.state === "getting"
          ? el("div", { class: "track" },
              el("div", { class: "fill vram",
                          style: `width:${g.total ? pct : 0}%` }))
          : null),
      g.state === "failed"
        ? el("span", { class: "label bad" }, "failed")
        : el("span", { class: "spin" }));
  };

  // Its own request: the hub is asked about every repository, and a slow answer
  // must not hold up the rest of the screen.
  const popularBox = el("div", { class: "group" },
    el("h2", {}, "Popular models"),
    el("div", { class: "hint" }, el("span", { class: "spin" }),
      " Asking Hugging Face what people are running…"));
  let rude = false;
  let page = 0;
  let on = null;
  let typed = "";
  const hunt = el("input", { type: "search", id: "hunt",
    placeholder: "Search models, or paste hf:owner/repo" });
  let waiting = null;
  hunt.oninput = () => {
    clearTimeout(waiting);
    waiting = setTimeout(() => {
      typed = hunt.value.trim(); page = 0; on = null; load();
    }, 350);
  };

  const load = async () => {
    const pop = await api(`/ui/models/popular?page=${page}&rude=${rude ? 1 : 0}`
      + `&q=${encodeURIComponent(typed)}`);
    const heading = () => [
      el("h2", {}, typed ? `Models matching “${typed}”` : "Popular models"),
      el("div", { class: "searchrow" }, hunt,
        typed ? el("button", { class: "ghost small",
          onclick: () => { hunt.value = ""; typed = ""; page = 0; on = null; load(); } },
          "Clear") : null),
    ];
    if (pop.error || !(pop.models || []).length) {
      const ref = typed.startsWith("hf:") || typed.startsWith("http");
      popularBox.replaceChildren(...heading(),
        el("div", { class: "hint" },
          pop.error || (typed ? "Nothing on the hub matches that."
                              : "Nothing that fits on this machine.")),
        ref ? el("button", { class: "ghost small",
          onclick: () => get(typed.split("/").pop(), typed) },
          `Get ${typed.split("/").pop()}`) : null);
      if (hunt.value) hunt.focus();
      return;
    }
    // Families come from the whole list, so ticking one off survives paging.
    if (on === null) on = new Set(pop.families || []);
    const paint = () => {
      const rows = pop.models.filter((x) => on.has(x.family));
      popularBox.replaceChildren(
        ...heading(),
        el("div", { class: "hint" },
          (typed ? "From Hugging Face" : "Most downloaded and trending now")
          + ", in a Q4 build that fits this machine. "
          + "💬 text · 🖼 images · 🔊 audio · 🎬 video."),
        el("div", { class: "chips" }, (pop.families || []).map((f) => {
          const box = el("input", { type: "checkbox", id: `fam_${f}`,
                                    ...(on.has(f) ? { checked: "1" } : {}) });
          box.onchange = () => { box.checked ? on.add(f) : on.delete(f); paint(); };
          return el("label", { class: "opt inline", for: `fam_${f}` }, box, f);
        })),
        (() => {
          const box = el("input", { type: "checkbox", id: "rude",
                                    ...(rude ? { checked: "1" } : {}) });
          box.onchange = () => { rude = box.checked; page = 0; load(); };
          return el("label", { class: "opt", for: "rude" }, box,
            row("Show uncensored builds",
              "abliterated, heretic and uncensored variants, which are "
              + "published with their refusals removed"));
        })(),
        ...(rows.length
          ? rows.map((x) => {
              const bits = [`${x.gb} GB`];
              if (x.params_b) bits.push(`${x.params_b}B`);
              if (x.moe) bits.push(`${x.active_b}B active`);
              if (x.draft_ref) bits.push(`+${x.draft_gb} GB draft`);
              bits.push(`${x.takes.join("")} → ${x.gives.join("")}`);
              bits.push(x.what);
              return el("div", { class: "row" },
                el("span", {}, el("b", {}, x.name),
                  el("span", { class: "why" }, bits.join(" · "))),
                el("button", { class: "ghost small",
                  onclick: () => get(x.file, x.ref, x.draft_ref) }, "Get"));
            })
          : [el("div", { class: "hint" }, "No family is ticked.")]),
        el("div", { class: "pager" },
          (() => {
            const back = el("button", { class: "ghost small" }, "← Newer");
            back.disabled = pop.page <= 0;
            back.onclick = () => { page = pop.page - 1; load(); };
            return back;
          })(),
          el("span", { class: "why" },
            `Page ${pop.page + 1} of ${pop.pages} · ${pop.total} models`),
          (() => {
            const on2 = el("button", { class: "ghost small" }, "More →");
            on2.disabled = pop.page + 1 >= pop.pages;
            on2.onclick = () => { page = pop.page + 1; load(); };
            return on2;
          })()));
    };
    paint();
  };
  load();

  const getting = (m.getting || []).filter((g) => g.state !== "done").map(bar);
  if ((m.getting || []).some((g) => g.state === "getting")) {
    timer = setTimeout(() => models(), 1000);
  }

  const here = (m.here || []).map((x) =>
    el("div", { class: "row" },
      el("span", {}, el("b", {}, x.name),
        el("span", { class: "why" },
          running.has(x.name) ? `${fileSize(x.size)} · running` : fileSize(x.size))),
      run.can_serve
        ? (running.has(x.name)
            ? el("button", { class: "ghost small", onclick: () => halt(x.name) }, "Stop")
            : el("button", { class: "ghost small", onclick: () => serve(x.name) }, "Run"))
        : el("span", { class: "label" }, "here")));

  const elsewhere = (m.elsewhere || []).map((x) => {
    const b = el("button", { class: "ghost small", onclick: () => get(x.name, "") },
      "Copy here");
    return el("div", { class: "row" },
      el("span", {}, el("b", {}, x.name),
        el("span", { class: "why" }, `on ${x.peers.join(", ")}`)), b);
  });

  const unfinished = (m.unfinished || []).map((x) =>
    el("div", { class: "row" },
      el("span", {}, el("b", {}, x.name.replace(/\.part$/, "")),
        el("span", { class: "why" }, `${fileSize(x.size)} copied, then stopped`)),
      el("button", { class: "ghost small",
        onclick: async () => {
          await api("/ui/models", { method: "DELETE",
            body: JSON.stringify({ name: x.name }) });
          models();
        } }, "Discard")));

  show(el("div", { class: "app" },
    top("Models"),
    el("main", {},
      el("h1", {}, "Models"),
      el("p", { class: "sub" }, `${m.free_gb} GB free on this machine.`),
      message ? el("div", { class: "ok" }, message) : null,

      el("div", { class: "group" },
        el("h2", {}, "On this machine"),
        here.length ? here : el("div", { class: "hint" }, "None yet.")),

      elsewhere.length
        ? el("div", { class: "group" },
            el("h2", {}, "On your other machines"),
            el("div", { class: "hint" },
              "Copied over your network rather than downloaded again."),
            ...elsewhere)
        : null,

      getting.length
        ? el("div", { class: "group" },
            el("h2", {}, "Coming in"),
            ...getting)
        : null,

      unfinished.length
        ? el("div", { class: "group" },
            el("h2", {}, "Started, not finished"),
            el("div", { class: "hint" },
              "Asking for the same model again picks up where it stopped."),
            ...unfinished)
        : null,

      popularBox,
      note)));
}

// ---------------------------------------------------------------- settings
async function settings(message) {
  clearTimeout(timer);
  const s = await api("/ui/settings");
  if (s.status === 401) return signIn();
  const cur = s.settings || {};
  const pick = {
    labels: (cur.labels || []).join(","),
    slots: cur.slots ?? 1,
    autostart: (s.autostart || {}).mode || cur.autostart || "manual",
    on_paused: cur.on_paused || "stop",
    on_close: cur.on_close || "",
    auto_update: cur.auto_update !== false,
    autodownload_models: cur.autodownload_models !== false,
  };

  const radio = (group, value, label, why) => {
    const id = `${group}-${value}`;
    const input = el("input", { type: "radio", name: group, id,
                                ...(pick[group] === value ? { checked: "1" } : {}) });
    input.onchange = () => { pick[group] = value; };
    return el("label", { class: "opt", for: id }, input, row(label, why));
  };
  const check = (key, label, why) => {
    const input = el("input", { type: "checkbox", id: key,
                                ...(pick[key] ? { checked: "1" } : {}) });
    input.onchange = () => { pick[key] = input.checked; };
    return el("label", { class: "opt", for: key }, input, row(label, why));
  };

  const slots = el("input", { type: "number", min: "1", max: "64",
                              value: String(pick.slots), class: "num" });
  slots.oninput = () => { pick.slots = Math.max(1, parseInt(slots.value || "1", 10)); };

  const save = el("button", {}, "Save");
  const note = el("div");
  save.onclick = async () => {
    save.disabled = true; save.innerHTML = '<span class="spin"></span> Saving…';
    const r = await api("/ui/settings", { method: "POST", body: JSON.stringify({
      ...pick, labels: pick.labels ? pick.labels.split(",") : [] }) });
    save.disabled = false; save.textContent = "Save";
    note.replaceChildren(r.manual
      ? el("div", {}, el("div", { class: "err" }, r.manual_why || "Needs permission."),
          el("pre", { class: "cmd" }, r.manual))
      : el("div", { class: "ok" }, "Saved."));
  };

  // libraries
  const libs = el("div", { class: "group" },
    el("h2", {}, "What this machine can train with"),
    el("div", { class: "hint" }, el("span", { class: "spin" }), " Looking…"));

  (async () => {
    const L = await api("/ui/libraries");
    if (L.status === 501) { libs.remove(); return; }
    const want = new Set((L.libraries || []).filter((x) => x.installed).map((x) => x.name));
    const boxes = (L.libraries || []).map((lib) => {
      const input = el("input", { type: "checkbox", id: `lib-${lib.name}`,
                                  ...(lib.installed || (!L.ready && lib.default)
                                      ? { checked: "1" } : {}) });
      if (!lib.installed && !L.ready && lib.default) want.add(lib.name);
      input.onchange = () => { input.checked ? want.add(lib.name) : want.delete(lib.name); };
      return el("label", { class: "opt", for: `lib-${lib.name}` }, input,
        el("span", {},
          el("b", {}, lib.title),
          el("span", { class: "why" },
            `${lib.blurb} · ${lib.size_mb >= 1000
              ? (lib.size_mb / 1000).toFixed(1) + " GB" : lib.size_mb + " MB"}`
            + (lib.version ? ` · installed ${lib.version}` : ""))));
    });

    const apply = el("button", { class: "ghost" }, "Apply");
    const out = el("div");
    apply.onclick = async () => {
      const before = new Set((L.libraries || []).filter((x) => x.installed).map((x) => x.name));
      const install = [...want].filter((n) => !before.has(n));
      const remove = [...before].filter((n) => !want.has(n));
      if (!install.length && !remove.length) {
        out.replaceChildren(el("div", { class: "hint" }, "Nothing to change."));
        return;
      }
      apply.disabled = true;
      apply.innerHTML = '<span class="spin"></span> This can take a while…';
      const r = await api("/ui/libraries", { method: "POST",
        body: JSON.stringify({ install, remove }) });
      apply.disabled = false; apply.textContent = "Apply";
      const failed = Object.entries(r.changed || {}).filter(([, v]) => !v.ok);
      out.replaceChildren(failed.length
        ? el("div", { class: "err" },
            failed.map(([k, v]) => `${k}: ${v.error}`).join("; "))
        : el("div", { class: "ok" }, "Done."));
    };

    libs.replaceChildren(
      el("h2", {}, "What this machine can train with"),
      el("div", { class: "hint" },
        L.ready ? "Installed alongside ml-stack, not in your system Python."
                : "Nothing installed yet. Tick what this machine should be able to do."),
      ...boxes, apply, out);
  })();

  // updates
  const updates = el("div", { class: "group" },
    el("h2", {}, "Updates"),
    el("div", { class: "hint" }, `You have version ${s.version || "?"}.`),
    check("autodownload_models", "Get models automatically",
      "from another machine on your network if one has it, otherwise the internet"),
    check("auto_update", "Download and install updates automatically",
      "checked once a day, put on when no job is running, and it restarts itself"));
  const status = el("div", { class: "hint", style: "margin-top:10px" },
    el("span", { class: "spin" }), " Checking for updates…");
  updates.append(status);

  const again = el("div", { class: "group" },
    el("h2", {}, "Set this machine up again"),
    el("div", { class: "hint" },
      "Walks through naming it and choosing a passphrase, as it did the first time. "
      + "The clusters it is already in are left alone."),
    el("button", { class: "ghost small",
      onclick: async () => {
        const state = await api("/ui/setup");
        wizard(state);
      } }, "Run setup again"));

  // removing it
  const removal = el("div", { class: "group" },
    el("h2", {}, "Remove ml-stack"),
    el("div", { class: "hint" }, "Loading what is on this machine…"));
  api("/ui/uninstall").then((u) => {
    if (u.error) {
      removal.replaceChildren(el("h2", {}, "Remove ml-stack"),
        el("div", { class: "err" }, u.error));
      return;
    }
    const want = {};
    for (const it of u.items || []) want[it.key] = it.default;
    const boxes = (u.items || []).map((it) => {
      const box = el("input", { type: "checkbox", id: `rm_${it.key}`,
                                ...(it.default ? { checked: "1" } : {}) });
      box.onchange = () => { want[it.key] = box.checked; };
      return el("label", { class: "opt", for: `rm_${it.key}` }, box,
        row(`${it.name} — ${fileSize(it.bytes)}`, it.why));
    });
    const out = el("div");
    const go = el("button", { class: "danger" }, "Remove");
    let armed = false;
    go.onclick = async () => {
      const chosen = Object.keys(want).filter((k) => want[k]);
      if (!armed) {
        armed = true;
        go.textContent = `Remove ${chosen.length} of these — click again`;
        out.replaceChildren(el("div", { class: "hint" },
          "This cannot be undone."));
        return;
      }
      go.disabled = true;
      const r = await api("/ui/uninstall", { method: "POST",
        body: JSON.stringify({ remove: chosen }) });
      out.replaceChildren(
        el("div", { class: "ok" },
          `Removed ${(r.removed || []).length} of ${chosen.length}, freeing `
          + `${fileSize(r.freed || 0)}.`),
        r.app ? el("div", { class: "hint" },
          `Drag ${r.app} to the bin to finish.`) : null,
        Object.keys(r.failed || {}).length
          ? el("div", { class: "err" }, Object.values(r.failed).join("; "))
          : null);
    };
    removal.replaceChildren(
      el("h2", {}, "Remove ml-stack"),
      el("div", { class: "hint" },
        "Ticked items go. Your models and your own files are left unless you say "
        + "otherwise."),
      ...boxes, go, out);
  });

  (async () => {
    const u = await api("/ui/updates");
    if (!u.checked) {
      status.replaceChildren(document.createTextNode(
        "Could not check for updates right now."));
      return;
    }
    if (!u.known) {
      status.replaceChildren(document.createTextNode(
        "This copy does not say which version it is, so it will not update itself. "
        + `The newest is ${u.latest}.`));
      return;
    }
    if (!u.newer) {
      status.replaceChildren(document.createTextNode("This is the newest version."));
      return;
    }
    const go = el("button", {}, `Update to ${u.latest}`);
    go.onclick = async () => {
      go.disabled = true; go.innerHTML = '<span class="spin"></span> Downloading…';
      const r = await api("/ui/updates/install", { method: "POST" });
      status.replaceChildren(r.ok
        ? el("div", { class: "ok" },
            !r.installed ? "Already up to date."
              : r.restarting ? `Updated to ${r.version}. Starting it now…`
              : `Updated to ${r.version}. Open it again to use it.`)
        : el("div", { class: "err" }, r.error || "Could not install the update."));
    };
    status.replaceChildren(
      el("div", {}, `Version ${u.latest} is available`
        + (u.size ? ` (${(u.size / 1e6).toFixed(0)} MB)` : "") + "."),
      go);
  })();

  show(el("div", { class: "app" },
    top("Settings", s.group),
    el("main", {},
      el("h1", {}, "Settings"),
      el("p", { class: "sub" },
        `${s.name}${s.machine?.gpu ? " — " + s.machine.gpu : ""}`),
      message ? el("div", { class: "ok" }, message) : null,

      el("div", { class: "two" },
        el("div", {},
          el("div", { class: "group" },
            el("h2", {}, "What this machine does"),
            radio("labels", "train", "Train models", ""),
            radio("labels", "prep", "Prepare data", ""),
            radio("labels", "train,prep", "Both", "")),

          el("div", { class: "group" },
            el("h2", {}, "At once"),
            el("label", { class: "opt inline" }, slots,
              row("jobs at a time", "one on a card; more on a machine preparing data"))),

          el("div", { class: "group" },
            el("h2", {}, "Starting up"),
            radio("autostart", "login", "When I log in", ""),
            radio("autostart", "boot", "When the computer starts",
              "your computer will ask for your password"),
            radio("autostart", "manual", "Only when I open it", ""))),

        el("div", {},
          libs,
          updates,

          el("div", { class: "group" },
            el("h2", {}, "When you need the machine"),
            radio("on_paused", "stop", "Pausing stops what is running",
              "the run picks up from its last checkpoint"),
            radio("on_paused", "finish", "Pausing lets the current job finish", "")),

          el("div", { class: "group" },
            el("h2", {}, "Closing the window"),
            radio("on_close", "", "Ask me each time", ""),
            radio("on_close", "background", "Keep running in the background",
              "stays in the cluster"),
            radio("on_close", "quit", "Quit", "leaves the cluster")),

          (s.schedule?.windows || []).length
            ? el("div", { class: "group" },
                el("h2", {}, "Not available"),
                ...s.schedule.windows.map((w) =>
                  el("div", { class: "label", style: "display:block;margin-bottom:4px" }, w)))
            : null)),
      save,
      again,
      removal, note)));
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
