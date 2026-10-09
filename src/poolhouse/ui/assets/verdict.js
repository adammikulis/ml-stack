/* The verdict vocabulary: green, yellow, red, none. verdict.json is the same definition
   for Python (poolhouse.ui.verdict); a test holds the two equal. */
import { s } from "./base.js";

export const THRESHOLDS = Object.freeze({ yellowAt: 0.8, redAt: 0.95 });
export const NAMES = Object.freeze(["green", "yellow", "red", "none"]);
export const LABELS = Object.freeze({ green: "OK", yellow: "Tight", red: "Over", none: "Unknown" });

/** The verdict named, or "none" when it is not one of the four. */
export function normalize(name) {
  const v = String(name ?? "").toLowerCase();
  return NAMES.includes(v) ? v : "none";
}

/** The verdict for `used` of `capacity`: green below yellowAt, red from redAt, yellow between. */
export function verdictOf(used, capacity, t = THRESHOLDS) {
  const u = Number(used), c = Number(capacity);
  if (!Number.isFinite(u) || !Number.isFinite(c) || c <= 0 || u < 0) return "none";
  const r = u / c;
  return r >= (t.redAt ?? THRESHOLDS.redAt) ? "red"
    : r >= (t.yellowAt ?? THRESHOLDS.yellowAt) ? "yellow" : "green";
}

const PATHS = {
  green: "M3.5 8.5l3 3 6-7",
  yellow: "M8 3.5v5.5M8 11.5v1",
  red: "M4 4l8 8M12 4l-8 8",
  none: "M4.5 8h7",
};

/** A 14px SVG mark for a verdict: a check, an exclamation, a cross, a dash. */
export function icon(verdict) {
  const v = normalize(verdict);
  const svg = s("svg", { viewBox: "0 0 16 16", width: 14, height: 14, "aria-hidden": "true",
                         class: "vicon" });
  svg.append(s("path", { d: PATHS[v], fill: "none", stroke: "currentColor",
                         "stroke-width": 2, "stroke-linecap": "round",
                         "stroke-linejoin": "round" }));
  return svg;
}
