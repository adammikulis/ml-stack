"""Where is it, as an angle. Camera geometry for robots that have to drive at things."""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

__all__ = ["Bearing", "column_to_deg", "find_color_blob", "floor_boundary",
           "hfov_from_known_width", "nearest_obstacle", "to_gray"]


@dataclass(frozen=True)
class Bearing:
    """Which way to turn, and how much to believe it."""

    deg: float
    confidence: float
    column: int = 0
    width: int = 0
    reason: str = ""


def column_to_deg(column: float, frame_width: int, hfov_deg: float) -> float:
    """Pixel column -> signed angle off the optical axis."""
    if frame_width <= 1:
        return 0.0
    half = np.tan(np.radians(hfov_deg / 2.0))
    x = (2.0 * column / (frame_width - 1)) - 1.0        # -1 .. +1
    return float(np.degrees(np.arctan(x * half)))


def to_gray(frame: np.ndarray, *, bgr: bool = True) -> np.ndarray:
    """uint8 image -> float32 luma. Already-2D input passes through."""
    a = np.asarray(frame)
    if a.ndim == 2:
        return a.astype(np.float32)
    c0, c1, c2 = a[..., 0], a[..., 1], a[..., 2]
    b, g, r = (c0, c1, c2) if bgr else (c2, c1, c0)
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)


def floor_boundary(frame: np.ndarray, *, floor_frac: float = 0.25,
                   tol: float = 2.5, bgr: bool = True) -> np.ndarray:
    """Per column, the row where the floor stops. A free-space profile."""
    g = to_gray(frame, bgr=bgr)
    h, w = g.shape
    split = int(h * (1.0 - floor_frac))
    floor = g[split:]
    if floor.size == 0:
        return np.full(w, h, dtype=np.int32)

    mu = float(np.median(floor))
    mad = float(np.median(np.abs(floor - mu)))
    sigma = max(1.4826 * mad, 1.0)                    # MAD -> sigma, for normal data
    is_floor = np.abs(g - mu) <= tol * sigma          # (h, w) bool

    from_bottom = is_floor[::-1]
    run = np.argmin(from_bottom, axis=0)              # 0 if the bottom is not floor
    all_floor = from_bottom.all(axis=0)
    run = np.where(all_floor, h, run)
    return (h - run).astype(np.int32)


def nearest_obstacle(frame: np.ndarray, *, hfov_deg: float,
                     floor_frac: float = 0.25, tol: float = 2.5,
                     min_run: int = 3, near_frac: float = 0.06,
                     bgr: bool = True) -> Bearing:
    """Bearing to the thing standing CLOSEST to the camera, or no confidence."""
    g = to_gray(frame, bgr=bgr)
    h, w = g.shape
    if h < 8 or w < 8:
        return Bearing(0.0, 0.0, reason="frame too small")

    profile = floor_boundary(frame, floor_frac=floor_frac, tol=tol, bgr=bgr)
    background = float(np.median(profile))
    margin = max(2.0, near_frac * h)
    hot = profile > background + margin

    runs: list[tuple[int, int]] = []
    run_start = run_len = 0
    for i, v in enumerate(hot):
        if v:
            run_start = i if run_len == 0 else run_start
            run_len += 1
        else:
            if run_len >= min_run:
                runs.append((run_start, run_len))
            run_len = 0
    if run_len >= min_run:
        runs.append((run_start, run_len))

    if not runs:
        widest = 0
        run_len = 0
        for v in hot:
            run_len = run_len + 1 if v else 0
            widest = max(widest, run_len)
        return Bearing(0.0, 0.0, width=widest,
                       reason=f"nothing nearer than the background "
                              f"(widest run {widest}px, need {min_run})")

    def _nearness(run: tuple[int, int]) -> float:
        seg = profile[run[0]:run[0] + run[1]]
        return float(np.mean(seg) - background)

    best_start, best_len = max(runs, key=lambda r: (_nearness(r), r[1]))
    centre = best_start + best_len / 2.0
    nearness = _nearness((best_start, best_len)) / max(h, 1)
    confidence = float(min(1.0, (best_len / w) / 0.25) * min(1.0, nearness / 0.15))
    return Bearing(deg=column_to_deg(centre, w, hfov_deg), confidence=confidence,
                   column=int(round(centre)), width=best_len,
                   reason=f"obstacle {best_len}px wide, "
                          f"{nearness*100:.0f}% of frame nearer than background")


def find_color_blob(frame: np.ndarray, *, hue: str = "orange",
                    min_px: int = 200, bgr: bool = True) -> tuple[int, int, int]:
    """Find a strongly-coloured object. Returns (left_col, right_col, pixels)."""
    a = np.asarray(frame).astype(np.float32)
    if a.ndim != 3:
        return (0, 0, 0)
    b, g, r = (a[..., 0], a[..., 1], a[..., 2]) if bgr else (a[..., 2], a[..., 1], a[..., 0])
    total = r + g + b + 1e-6
    if hue == "orange":
        mask = (r > 60) & (r / total > 0.45) & (b / total < 0.28) & (r > g * 1.15)
    elif hue == "red":
        mask = (r > 60) & (r / total > 0.50) & (g / total < 0.30) & (b / total < 0.30)
    elif hue == "green":
        mask = (g > 50) & (g / total > 0.42) & (r / total < 0.35)
    elif hue == "blue":
        mask = (b > 50) & (b / total > 0.42) & (r / total < 0.33)
    else:
        raise ValueError(f"unknown hue {hue!r}")

    cols = mask.sum(axis=0)
    if int(mask.sum()) < min_px:
        return (0, 0, int(mask.sum()))
    hot = cols >= max(1, int(0.05 * cols.max()))
    best_s = best_l = run_s = run_l = 0
    for i, v in enumerate(hot):
        if v:
            run_s = i if run_l == 0 else run_s
            run_l += 1
            if run_l > best_l:
                best_l, best_s = run_l, run_s
        else:
            run_l = 0
    if best_l == 0:
        return (0, 0, int(mask.sum()))
    return (best_s, best_s + best_l - 1, int(mask.sum()))


def hfov_from_known_width(pixel_width: int, frame_width: int,
                          object_width_mm: float, distance_mm: float) -> float:
    """Solve the horizontal field of view from one frame of a known object."""
    if pixel_width <= 0 or frame_width <= 0 or distance_mm <= 0:
        return 0.0
    half_obj = math.atan((object_width_mm / 2.0) / distance_mm)
    frac = pixel_width / float(frame_width)
    if not 0 < frac < 1:
        return 0.0
    return 2.0 * math.degrees(math.atan(math.tan(half_obj) / frac))
