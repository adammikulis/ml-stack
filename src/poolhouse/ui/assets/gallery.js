/* Every primitive in every state, built from data. */
import { h } from "./base.js";

const GB = 1e9;
const HOSTILE = `<img src=x onerror="window.__pwned=1"> "quoted" 'single' </script><b>bold</b> &amp; ‮`;
const LONG = "an-extremely-long-model-name-with-no-break-opportunities-Q4_K_XL-00001-of-00004-and-more-and-more.gguf";

const theme = new URLSearchParams(location.search).get("theme");
if (theme === "light" || theme === "dark") document.documentElement.dataset.theme = theme;

function el(tag, props = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "text") n.textContent = v;
    else if (typeof v === "object" && v !== null || k === "validate") n[k] = v;
    else if (v === true) n.setAttribute(k, "");
    else if (v !== false && v !== null && v !== undefined) n.setAttribute(k, String(v));
  }
  n.append(...kids.flat());
  return n;
}
const cell = (title, ...kids) => h("div", { class: "cell" }, h("h3", {}, title), ...kids);
const wide = (title, ...kids) => { const c = cell(title, ...kids); c.classList.add("wide"); return c; };
const section = (name, ...cells) => [h("h2", { id: name.toLowerCase().replace(/\W+/g, "-") }, name), h("div", { class: "grid" }, ...cells)];

const meter = (title, props) => cell(title, el("ml-meter", props));
const meters = section("ml-meter",
  meter("green: 8 of 24 GB", { label: "Video memory", format: "bytes", capacity: 24 * GB,
    segments: [{ label: "weights", value: 5 * GB }, { label: "cache", value: 2 * GB }, { label: "compute", value: 1 * GB }] }),
  meter("yellow: 20 of 24 GB", { label: "Video memory", format: "bytes", capacity: 24 * GB,
    segments: [{ label: "weights", value: 14 * GB }, { label: "cache", value: 5 * GB }, { label: "compute", value: 1 * GB }] }),
  meter("red: over capacity", { label: "Video memory", format: "bytes", capacity: 24 * GB,
    segments: [{ label: "weights", value: 20 * GB }, { label: "cache", value: 9 * GB }, { label: "compute", value: 2 * GB }] }),
  meter("single value, green", { label: "Processors", value: 41, capacity: 100, unit: "%", caption: "41% busy" }),
  meter("single value, red, compact", { label: "GPU", value: 98, capacity: 100, unit: "%", caption: "98% busy", compact: true }),
  meter("compact green (icon only)", { label: "Memory", value: 12, capacity: 64, unit: "GB", compact: true }),
  meter("empty (no data)", { label: "Memory" }),
  meter("explicit verdict from an estimator", { label: "Fits at 32k", value: 3, capacity: 100, unit: "GB", verdict: "red", "verdict-text": "Does not fit" }),
  meter("unknown verdict", { label: "Cache", value: 3, verdict: "none" }),
  meter("long label and caption", { label: LONG, format: "bytes", capacity: 96 * GB, value: 60 * GB }),
  meter("hostile label", { label: HOSTILE, value: 10, capacity: 100, caption: HOSTILE }),
  meter("zero capacity", { label: "Zero", value: 5, capacity: 0 }),
  meter("custom thresholds (yellow at 50%)", { label: "Queue", value: 60, capacity: 100, "yellow-at": 0.5, "red-at": 0.9 }),
  cell("rtl", el("div", { dir: "rtl", lang: "ar" }, el("ml-meter", { label: "الذاكرة", value: 70, capacity: 100, unit: "GB" }))));

const chips = section("ml-chip",
  cell("verdicts", el("div", { class: "row" },
    el("ml-chip", { verdict: "green" }), el("ml-chip", { verdict: "yellow" }), el("ml-chip", { verdict: "red" }),
    el("ml-chip", { verdict: "none" }))),
  cell("custom text", el("div", { class: "row" },
    el("ml-chip", { verdict: "green", label: "Fits at 128k" }), el("ml-chip", { verdict: "yellow", label: "Tight" }),
    el("ml-chip", { verdict: "red", label: "Does not fit" }))),
  cell("plain status and icon only", el("div", { class: "row" },
    el("ml-chip", { label: "NVIDIA · CUDA" }), el("ml-chip", { verdict: "green", "icon-only": true, label: "ok" }),
    el("ml-chip", { verdict: "red", "icon-only": true, label: "failed" }))),
  cell("long and hostile", el("div", { class: "row narrow" },
    el("ml-chip", { verdict: "red", label: LONG }), el("ml-chip", { verdict: "yellow", label: HOSTILE }))));

const progress = (title, props) => cell(title, el("ml-progress", props));
const progresses = section("ml-progress",
  progress("downloading", { label: "thornfield-8B-Q4_K_M.gguf", format: "bytes", done: 1.7 * GB, total: 4.9 * GB, rate: 41e6, cancelable: true }),
  progress("indeterminate (size unknown)", { label: "Starting…", cancelable: true }),
  progress("done", { label: "marrowgate-A3B", format: "bytes", done: 4.9 * GB, total: 4.9 * GB, state: "done" }),
  progress("failed", { label: "greenhollow.gguf", format: "bytes", done: 1 * GB, total: 4.9 * GB, state: "failed", message: "The connection was reset." }),
  progress("paused", { label: "Copy from greenhollow", format: "bytes", done: 3 * GB, total: 8 * GB, state: "paused" }),
  progress("fraction only, eta given", { label: "Fitting", value: 0.35, eta: 125 }),
  progress("long label", { label: LONG, value: 0.6, cancelable: true }),
  progress("hostile label and message", { label: HOSTILE, value: 0.2, state: "failed", message: HOSTILE }),
  progress("zero total", { label: "Empty", done: 0, total: 0 }));

const spark = (title, props) => cell(title, el("ml-sparkline", props));
const sparklines = section("ml-sparkline",
  spark("rising", { values: [1, 2, 2, 3, 5, 4, 6, 8, 9, 12], label: "tokens per second", area: true }),
  spark("red verdict, wide", { values: [10, 30, 55, 80, 96, 99, 97], width: 220, height: 40, verdict: "red", min: 0, max: 100, area: true }),
  spark("green flat", { values: [5, 5, 5, 5, 5], verdict: "green" }),
  spark("single point", { values: [4] }),
  spark("empty", { values: [] }),
  spark("with gaps and junk", { values: [1, null, "x", 3, NaN, 2, Infinity, 5] }));

const models = [
  { id: "a", name: "thornfield-8B-Q4_K_M.gguf", nameSub: "dense · 8B", fit: "green", fitText: "Fits", size: 5 * GB, users: 22 },
  { id: "b", name: "marrowgate-A3B-UD-Q4_K_XL.gguf", nameSub: "mixture of experts · 3B active", fit: "yellow", fitText: "Tight", size: 60 * GB, users: 3 },
  { id: "c", name: LONG, nameSub: "does not fit with a long context", fit: "red", fitText: "No fit", size: 140 * GB, users: 0 },
  { id: "d", name: HOSTILE, nameSub: HOSTILE, fit: "none", size: 1, users: 1 },
];
const columns = [
  { key: "fit", label: "Fit", kind: "verdict", sortable: true },
  { key: "name", label: "Model", kind: "name", sortable: true },
  { key: "size", label: "Size", kind: "bytes", sortable: true },
  { key: "users", label: "Users", kind: "number", sortable: true },
];
const tables = section("ml-table",
  wide("models with verdicts, selectable, sortable", el("ml-table", { columns, rows: models, selectable: true, label: "Models that fit this machine", "sort-key": "size", "sort-dir": "descending" })),
  cell("empty", el("ml-table", { columns, rows: [], empty: "No models on this machine yet." })),
  cell("loading", el("ml-table", { columns, rows: [], loading: true })),
  cell("error", el("ml-table", { columns, rows: [], error: "The hub did not answer." })));

const steps = [{ id: "name", title: "Name" }, { id: "cluster", title: "Cluster" }, { id: "job", title: "Job" }, { id: "done", title: "Done" }];
const stepper = (id, current, validate) => {
  const s = el("ml-stepper", { id, steps, current, validate });
  steps.forEach((st) => s.append(el("div", { slot: `step-${st.id}`, text: `Content of step ${st.title}. ${HOSTILE}` })));
  return s;
};
const steppers = section("ml-stepper",
  cell("first step", stepper("st1", 0)),
  cell("middle step, validation refuses", stepper("st2", 1, (i) => (i === 1 ? "Pick at least one job first." : true))),
  cell("last step, finish", stepper("st3", 3)),
  cell("controls driven by the app (controls=none)", (() => { const s = stepper("st4", 2); s.setAttribute("controls", "none"); return s; })()));

const field = (title, e) => cell(title, e);
const fields = section("fields",
  field("select", el("ml-select", { label: "Cache type", hint: "Smaller caches fit more users.", value: "q8_0",
    options: [{ value: "f16", label: "f16 (largest)" }, { value: "q8_0", label: "q8_0" }, { value: "q4_0", label: "q4_0 (smallest)" }] })),
  field("select with error", el("ml-select", { label: "Model", error: "Choose a model.", placeholder: "Pick one…", value: "",
    options: [{ value: "a", label: "thornfield" }, { value: "h", label: HOSTILE }] })),
  field("slider with stops and tick labels", el("ml-slider", { label: "How much it reads", value: 8192, unit: "tokens",
    stops: [2048, 4096, 8192, 16384, 32768, 65536, 131072], hint: "Roughly a page of text per 500 tokens." })),
  field("slider with custom marks", el("ml-slider", { label: "Users", min: 1, max: 64, value: 12,
    marks: [{ value: 1 }, { value: 16 }, { value: 32 }, { value: 48 }, { value: 64 }] })),
  field("toggle on", el("ml-toggle", { label: "Run in the background", hint: "Stays reachable while the window is shut.", checked: true })),
  field("toggle off, disabled", el("ml-toggle", { label: "Share this machine", disabled: true })),
  field("path", el("ml-path", { label: "Model folder", value: "~/models", browse: "Browse…", verdict: "green", "verdict-text": "Folder exists",
    hint: "Models are read from here." })),
  field("path with error", el("ml-path", { label: "Model folder", value: "/no/such/place", error: "That folder does not exist.", verdict: "red", "verdict-text": "Missing" })),
  field("path hostile", el("ml-path", { label: HOSTILE, value: HOSTILE, hint: HOSTILE })),
  field("rtl", el("div", { dir: "rtl", lang: "ar" }, el("ml-select", { label: "اللغة", value: "a", options: [{ value: "a", label: "عربي" }] }))));

const buttons = ["green", "yellow", "red", "none"].map((v) => {
  const b = el("button", { class: "demo", type: "button", text: `${v} toast` });
  b.addEventListener("click", () => document.getElementById("toaster").show({
    verdict: v, title: v === "red" ? "Download failed" : "Saved", message: v === "none" ? HOSTILE : "Your changes were kept.",
    action: v === "green" ? { label: "Undo" } : undefined }));
  return b;
});
const sheetBtn = el("button", { class: "demo", type: "button", text: "Open sheet" });
const sheet = el("ml-sheet", { heading: `Close poolhouse? ${HOSTILE}`, id: "sheet" },
  el("p", { text: "Keeping it running lets this machine answer." }),
  el("button", { class: "demo", slot: "footer", type: "button", id: "sheet-cancel", text: "Cancel" }),
  el("button", { class: "demo", slot: "footer", type: "button", id: "sheet-ok", text: "Quit" }));
sheet.addEventListener("click", (e) => { if (e.target.id === "sheet-cancel" || e.target.id === "sheet-ok") sheet.close("button"); });
sheetBtn.addEventListener("click", () => sheet.setAttribute("open", ""));
const overlays = section("ml-sheet and ml-toaster",
  cell("sheet", sheetBtn, sheet), cell("toasts", el("div", { class: "row" }, ...buttons)));

document.getElementById("gallery").append(
  h("h1", {}, "ml-ui"), h("p", { style: { color: "var(--ml-muted)" } }, "Every primitive in every state."),
  ...meters, ...chips, ...progresses, ...sparklines, ...tables, ...steppers, ...fields, ...overlays);
