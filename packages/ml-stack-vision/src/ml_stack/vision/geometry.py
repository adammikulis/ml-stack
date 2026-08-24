"""Where is it, as an angle. Camera geometry for robots that have to drive at things.

A detector says "the interesting thing is at column 240". A drive layer needs
"turn 17 degrees clockwise". This module is the conversion, plus one detector
cheap enough for a CPU-only board.

Nothing here knows what a person looks like, and nothing here opens a camera.
Arrays in, numbers out -- so the interesting cases are testable without hardware.

``hfov_deg`` is deliberately REQUIRED rather than defaulted. Horizontal field of
view is the constant that converts pixels to angles, so a wrong value is a
proportional bearing error: 10% off is 10% of the turn, which at 2m is ~35cm of
miss. It is also the constant nobody measures, because a plausible number is
always available from a datasheet. Making it required means a caller has to
write it down somewhere it can be reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass

import math

import numpy as np

__all__ = ["Bearing", "column_to_deg", "find_color_blob", "floor_boundary",
           "hfov_from_known_width", "nearest_obstacle", "to_gray"]


@dataclass(frozen=True)
class Bearing:
    """Which way to turn, and how much to believe it.

    ``deg`` is signed clockwise-positive when the frame is the usual
    left-to-right raster: column 0 is to the viewer's left and yields a negative
    angle. ``confidence`` is 0..1; a caller should refuse to act below its own
    threshold rather than treat 0.0 as "straight ahead".
    """

    deg: float
    confidence: float
    column: int = 0
    width: int = 0
    reason: str = ""


def column_to_deg(column: float, frame_width: int, hfov_deg: float) -> float:
    """Pixel column -> signed angle off the optical axis.

    Uses the tangent relation, because pixel offset on a flat sensor is
    proportional to tan(angle) and not to angle.

    The error in the naive version has a direction worth knowing: arctan is
    concave, so a LINEAR map under-estimates every angle between the centre and
    the edge -- worst near mid-frame, where at a 62 deg FOV it reads 15.6 deg for
    a column that is really 16.9. A robot steering on a linear map therefore
    under-turns consistently and drifts past the target on the near side, which
    reads as a lazy detector rather than as a projection bug.
    """
    if frame_width <= 1:
        return 0.0
    half = np.tan(np.radians(hfov_deg / 2.0))
    x = (2.0 * column / (frame_width - 1)) - 1.0        # -1 .. +1
    return float(np.degrees(np.arctan(x * half)))


def to_gray(frame: np.ndarray, *, bgr: bool = True) -> np.ndarray:
    """uint8 image -> float32 luma. Already-2D input passes through.

    ``bgr`` defaults True because that is what OpenCV and most camera SDKs hand
    back. Getting it wrong swaps the red and blue weights, which does not raise
    and does not look obviously wrong -- it just quietly moves every threshold
    downstream.
    """
    a = np.asarray(frame)
    if a.ndim == 2:
        return a.astype(np.float32)
    c0, c1, c2 = a[..., 0], a[..., 1], a[..., 2]
    b, g, r = (c0, c1, c2) if bgr else (c2, c1, c0)
    return (0.114 * b + 0.587 * g + 0.299 * r).astype(np.float32)


def floor_boundary(frame: np.ndarray, *, floor_frac: float = 0.25,
                   tol: float = 2.5, bgr: bool = True) -> np.ndarray:
    """Per column, the row where the floor stops. A free-space profile.

    Scanning UP from the bottom of each column and stopping at the first pixel
    that no longer matches the floor model gives, for a forward-looking camera,
    a monotonic proxy for distance: a LARGER row index means the floor is
    interrupted lower in the frame, which means nearer to the camera.

    This is the useful primitive, and the reason is worth stating because the
    obvious alternative is wrong. Simply masking "everything that is not floor"
    finds the back wall of the room -- it is not floor, it spans every column,
    and it therefore dominates any widest-region search while carrying no
    information about what is standing in the room. The BOUNDARY separates them:
    a wall is a flat horizon, and anything standing in front of it is a bump.
    """
    g = to_gray(frame, bgr=bgr)
    h, w = g.shape
    split = int(h * (1.0 - floor_frac))
    floor = g[split:]
    if floor.size == 0:
        # Nothing seen, not "an obstacle at every column": this feeds a
        # nearness comparison, and the unsafe default is the one that reads as
        # something standing right in front of the robot.
        return np.full(w, h, dtype=np.int32)

    # MEDIAN AND MAD, NOT MEAN AND STD. The band used to model the floor is the
    # bottom of the frame, and anything standing ON the floor reaches into it --
    # that is what makes it detectable. A mean/std model is therefore
    # contaminated by the very object it is meant to find: a wide subject
    # inflates sigma, the threshold widens, everything reads as floor, and the
    # profile goes flat. Measured: a 35px subject was found and a 60px one
    # vanished. The median tolerates contamination up to half the band.
    mu = float(np.median(floor))
    mad = float(np.median(np.abs(floor - mu)))
    sigma = max(1.4826 * mad, 1.0)                    # MAD -> sigma, for normal data
    is_floor = np.abs(g - mu) <= tol * sigma          # (h, w) bool

    # Rows of floor counted up from the bottom, subtracted from the height:
    # the row index at which the floor stops. A column that is floor all the way
    # up yields h, i.e. "nothing interrupts it".
    from_bottom = is_floor[::-1]
    run = np.argmin(from_bottom, axis=0)              # 0 if the bottom is not floor
    all_floor = from_bottom.all(axis=0)
    run = np.where(all_floor, h, run)
    return (h - run).astype(np.int32)


def nearest_obstacle(frame: np.ndarray, *, hfov_deg: float,
                     floor_frac: float = 0.25, tol: float = 2.5,
                     min_run: int = 3, near_frac: float = 0.06,
                     bgr: bool = True) -> Bearing:
    """Bearing to the thing standing CLOSEST to the camera, or no confidence.

    Compares each column's floor boundary against the median across the frame.
    The median is the background -- the far wall, the horizon, whatever the room
    ends in -- so columns whose floor ends significantly nearer than that are
    objects standing in front of it. A room with nothing in it has a flat
    profile, no column beats the median, and this returns confidence 0.0, which
    is the honest answer and the one a naive "not floor" mask cannot give.

    ``near_frac`` is how much nearer than the background a column must be,
    as a fraction of frame height. Too small and floor texture becomes an
    obstacle; too large and a person at range is invisible.
    """
    g = to_gray(frame, bgr=bgr)
    h, w = g.shape
    if h < 8 or w < 8:
        return Bearing(0.0, 0.0, reason="frame too small")

    profile = floor_boundary(frame, floor_frac=floor_frac, tol=tol, bgr=bgr)
    background = float(np.median(profile))
    margin = max(2.0, near_frac * h)
    hot = profile > background + margin

    # Collect every candidate run, then rank by NEARNESS rather than width.
    # Ranking by width picks a wide far object over a narrow near one, which is
    # the wrong answer for "come here" and contradicts this function's name --
    # the wide thing at the back of the room is usually furniture.
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
    # Width says "this is an object, not noise"; nearness says "it is in the
    # room, not part of the wall". Requiring both is what rejects a textured
    # wall, which is wide but flat.
    confidence = float(min(1.0, (best_len / w) / 0.25) * min(1.0, nearness / 0.15))
    return Bearing(deg=column_to_deg(centre, w, hfov_deg), confidence=confidence,
                   column=int(round(centre)), width=best_len,
                   reason=f"obstacle {best_len}px wide, "
                          f"{nearness*100:.0f}% of frame nearer than background")


def find_color_blob(frame: np.ndarray, *, hue: str = "orange",
                    min_px: int = 200, bgr: bool = True) -> tuple[int, int, int]:
    """Find a strongly-coloured object. Returns (left_col, right_col, pixels).

    A saturated single-colour target is enormously easier to locate than a
    person: measured on real frames from this class of robot, a floor-geometry
    detector scored 0.01-0.03 confidence on a human standing 2m away, while a
    colour threshold finds a drinks can immediately. For CALIBRATION that is the
    right trade -- the target only has to be findable and of known size, and
    nobody has to hold still.

    Deliberately ratio-based rather than a fixed RGB box, so it survives the
    lighting actually available: a room measuring mean 20/255 and one measuring
    mean 100/255 both preserve the RATIO between channels even though every
    absolute value moves.
    """
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
    # The widest contiguous run of columns that contain the colour, so a
    # reflection somewhere else in the frame does not widen the answer.
    # `>=`, not `>`: a one-pixel-tall streak puts cols.max() at 1 and the floor at 1
    # too, and a strict test then excludes every column of the only blob there is --
    # reporting a zero-length run for a mask that plainly cleared min_px.
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
        # Nothing survived the run threshold. Say so as an empty span rather than as
        # (0, -1): a caller subtracting those gets a negative width, and a negative
        # width becomes a bearing pointing somewhere the target is not.
        return (0, 0, int(mask.sum()))
    return (best_s, best_s + best_l - 1, int(mask.sum()))


def hfov_from_known_width(pixel_width: int, frame_width: int,
                          object_width_mm: float, distance_mm: float) -> float:
    """Solve the horizontal field of view from one frame of a known object.

    No rotation, no second frame, no assumption about the commanded angle -- and
    critically, no need to know anything about the object except its WIDTH,
    which is why a drinks can works and a person does not.

    The object subtends 2*atan(W/2 / d). It covers `pixel_width` of
    `frame_width`, and the tangent relation maps that back to the full field:

        tan(half_object) / tan(HFOV/2) = pixel_width / frame_width

    exactly, for an object centred in frame; close enough off-centre that the
    error is far below the distance measurement's own.
    """
    if pixel_width <= 0 or frame_width <= 0 or distance_mm <= 0:
        return 0.0
    half_obj = math.atan((object_width_mm / 2.0) / distance_mm)
    frac = pixel_width / float(frame_width)
    if not 0 < frac < 1:
        return 0.0
    return 2.0 * math.degrees(math.atan(math.tan(half_obj) / frac))
