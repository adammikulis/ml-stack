/* Number formatting shared by the meter, the progress bar and the table. */

const SI = ["B", "kB", "MB", "GB", "TB"];
const IEC = ["B", "KiB", "MiB", "GiB", "TiB"];

function scaled(n, base, units, digits) {
  let v = Math.abs(n), i = 0;
  while (v >= base && i < units.length - 1) { v /= base; i += 1; }
  const text = i === 0 ? String(Math.round(v)) : v.toFixed(digits);
  return `${n < 0 ? "-" : ""}${text} ${units[i]}`;
}

/** `n` as text in `format` (number, bytes, bytes-iec, percent), with an optional unit. */
export function fmt(n, format = "number", unit = "", digits = 1) {
  const v = Number(n);
  if (!Number.isFinite(v)) return "—";
  if (format === "bytes") return scaled(v, 1000, SI, digits);
  if (format === "bytes-iec") return scaled(v, 1024, IEC, digits);
  if (format === "percent") return `${(v * 100).toFixed(0)}%`;
  const text = Number.isInteger(v) && digits === 0 ? String(v) : v.toFixed(digits);
  return unit ? `${text} ${unit}` : text;
}

/** Seconds as `45s`, `3m 05s` or `1h 02m`; an em dash for anything that is not a time. */
export function duration(seconds) {
  const t = Number(seconds);
  if (!Number.isFinite(t) || t < 0) return "—";
  if (t < 60) return `${Math.round(t)}s`;
  const m = Math.floor(t / 60), r = Math.round(t % 60);
  if (m < 60) return `${m}m ${String(r).padStart(2, "0")}s`;
  return `${Math.floor(m / 60)}h ${String(m % 60).padStart(2, "0")}m`;
}

/** A finite number within [lo, hi], or `lo` when it is not a number. */
export const clamp = (v, lo, hi) => (Number.isFinite(v) ? Math.min(hi, Math.max(lo, v)) : lo);
