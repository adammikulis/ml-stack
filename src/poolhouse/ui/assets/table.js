/* <ml-table>: rows of records with verdict dots, sortable columns and an empty state. */
import { MlElement, define, h } from "./base.js";
import { fmt } from "./format.js";
import { LABELS, icon, normalize } from "./verdict.js";

const STYLES = `
.wrap { overflow-x: auto; border: 1px solid var(--ml-line); border-radius: var(--ml-radius);
  background: var(--ml-surface); }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
caption { text-align: left; padding: 0 0 8px; color: var(--ml-muted); font-size: 12px; caption-side: top; }
th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid var(--ml-line);
  vertical-align: middle; }
tbody tr:last-child td { border-bottom: 0; }
th { font-size: 11px; text-transform: uppercase; letter-spacing: .06em; color: var(--ml-muted);
  font-weight: 600; white-space: nowrap; background: var(--ml-sunken); }
th button { all: unset; cursor: pointer; display: inline-flex; gap: 4px; align-items: center; }
th button:focus-visible { outline: 2px solid var(--ml-focus); outline-offset: 2px; }
.num, th.num { text-align: right; font-family: var(--ml-mono); font-variant-numeric: tabular-nums; }
.mono { font-family: var(--ml-mono); }
.name { display: block; font-weight: 600; overflow-wrap: anywhere; }
.sub { display: block; color: var(--ml-muted); font-size: 12px; overflow-wrap: anywhere; }
.verdict { display: inline-flex; align-items: center; gap: 7px; color: var(--v-ink);
  font-weight: 600; white-space: nowrap; }
.dot { width: 18px; height: 18px; border-radius: 50%; display: inline-grid; place-items: center;
  background: color-mix(in srgb, var(--v) 18%, var(--ml-surface));
  border: 1.5px solid var(--v); color: var(--v-ink); flex: none; }
.dot svg { width: 11px; height: 11px; }
tbody tr[tabindex] { cursor: pointer; }
tbody tr[tabindex]:hover, tbody tr[aria-selected=true] {
  background: color-mix(in srgb, var(--ml-accent) 10%, transparent); }
.state { padding: 28px 16px; text-align: center; color: var(--ml-muted); }
.state.error { color: var(--ml-red-ink); }
`;

const ARROW = { ascending: " ↑", descending: " ↓" };

class MlTable extends MlElement {
  static props = { columns: "object", rows: "object", label: "string", empty: "string",
                   error: "string", loading: "bool", selectable: "bool", sortKey: "string",
                   sortDir: "string", selected: "string" };
  static styles = STYLES;

  build() {
    this.wrap = h("div", { class: "wrap" });
    this.root.append(this.wrap);
  }

  cell(col, row) {
    const v = row?.[col.key];
    const kind = col.kind || "text";
    if (kind === "verdict") {
      const name = normalize(v);
      return h("span", { class: `verdict v-${name}` }, h("span", { class: "dot" }, icon(name)),
        String(row[`${col.key}Text`] ?? LABELS[name]));
    }
    if (kind === "name") {
      return [h("span", { class: "name" }, v ?? ""), row[`${col.key}Sub`]
        ? h("span", { class: "sub" }, row[`${col.key}Sub`]) : null];
    }
    if (kind === "number" || kind === "bytes") {
      return fmt(v, kind === "bytes" ? "bytes" : "number", col.unit || "", col.digits ?? (kind === "bytes" ? 1 : 0));
    }
    return v === null || v === undefined ? "" : String(v);
  }

  sorted() {
    const rows = Array.isArray(this.rows) ? [...this.rows] : [];
    const key = this.sortKey;
    if (!key) return rows;
    const dir = this.sortDir === "descending" ? -1 : 1;
    return rows.sort((a, b) => {
      const x = a?.[key], y = b?.[key];
      return (typeof x === "number" && typeof y === "number"
        ? x - y : String(x ?? "").localeCompare(String(y ?? ""), undefined, { numeric: true })) * dir;
    });
  }

  sortBy(col) {
    const dir = this.sortKey === col.key && this.sortDir !== "descending" ? "descending" : "ascending";
    this.sortKey = col.key;
    this.sortDir = dir;
    this.emit("ml-sort", { key: col.key, direction: dir });
  }

  pick(row) {
    if (!this.selectable) return;
    this.selected = String(row.id ?? "");
    this.emit("ml-row", { id: row.id, row });
  }

  update() {
    const cols = Array.isArray(this.columns) ? this.columns : [];
    const rows = this.sorted();
    const state = this.error ? ["error", this.error]
      : this.loading ? ["loading", "Loading…"] : !rows.length ? ["empty", this.empty || "Nothing to show."] : null;
    const head = h("tr", {}, cols.map((c) => {
      const on = this.sortKey === c.key ? this.sortDir === "descending" ? "descending" : "ascending" : "none";
      return h("th", { scope: "col", class: c.kind === "number" || c.kind === "bytes" ? "num" : "",
        "aria-sort": c.sortable ? on : null },
        c.sortable ? h("button", { type: "button", onclick: () => this.sortBy(c) },
          c.label, ARROW[on] ?? "") : c.label);
    }));
    const body = state ? [h("tr", {}, h("td", { colspan: cols.length || 1 },
      h("div", { class: `state ${state[0]}`, role: state[0] === "error" ? "alert" : "status" }, state[1])))]
      : rows.map((r) => {
        const id = String(r.id ?? "");
        const tr = h("tr", { "aria-selected": this.selectable ? String(id !== "" && id === this.selected) : null,
          tabindex: this.selectable ? 0 : null,
          onclick: () => this.pick(r),
          onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); this.pick(r); } } },
          cols.map((c) => h("td", { class: `${c.kind === "number" || c.kind === "bytes" ? "num" : ""}${c.kind === "mono" ? " mono" : ""}` },
            this.cell(c, r))));
        return tr;
      });
    this.wrap.replaceChildren(h("table", { "aria-busy": String(!!this.loading) },
      this.label ? h("caption", {}, this.label) : null, h("thead", {}, head), h("tbody", {}, body)));
  }
}
define("ml-table", MlTable);
