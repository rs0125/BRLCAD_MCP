"""Analytic CSG evaluation along a ray — the reference model for verification.

Instead of reasoning about what each cutter is *meant* to be (a through-hole? a
pocket? a slot?), we evaluate the spec's own boolean expression along a ray and
get the intervals of solid material it should cross.  Comparing that with what
the engine actually reports is one shape-agnostic check that subsumes every
special case: a missing subtraction, a wrong size or position, a blind pocket
and its depth, an internal void, even a hole of the wrong diameter.

Adding a primitive means adding its ray-intersection function here -- a small,
self-contained piece of maths -- and nothing else changes.

Everything in this module is pure: no socket, no engine, fully unit-testable.
"""

from __future__ import annotations

import math

from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part

Interval = tuple[float, float]
_EPS = 1e-9


# --- vector helpers -------------------------------------------------------

def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(a, k):
    return (a[0] * k, a[1] * k, a[2] * k)


def _norm(a):
    mag = math.sqrt(_dot(a, a))
    return _scale(a, 1.0 / mag) if mag > _EPS else (0.0, 0.0, 1.0)


# --- per-primitive ray intersection ---------------------------------------

def _box_intervals(part: Part, origin, direction) -> list[Interval]:
    """Slab test against the axis-aligned box (an ``rpp`` is exactly an AABB)."""
    sx, sy, sz = part.size  # type: ignore[misc]
    centre = part.center
    half = (sx / 2, sy / 2, sz / 2)
    t_lo, t_hi = -math.inf, math.inf
    for i in range(3):
        lo, hi = centre[i] - half[i], centre[i] + half[i]
        if abs(direction[i]) < _EPS:
            if origin[i] < lo or origin[i] > hi:
                return []
            continue
        t1 = (lo - origin[i]) / direction[i]
        t2 = (hi - origin[i]) / direction[i]
        t_lo = max(t_lo, min(t1, t2))
        t_hi = min(t_hi, max(t1, t2))
    return [(t_lo, t_hi)] if t_hi > t_lo else []


def _sphere_intervals(part: Part, origin, direction) -> list[Interval]:
    """Quadratic against a sphere of radius ``radius`` at ``center``."""
    r = part.radius or 0.0
    oc = _sub(origin, part.center)
    b = _dot(oc, direction)
    c = _dot(oc, oc) - r * r
    disc = b * b - c
    if disc <= 0:
        return []
    root = math.sqrt(disc)
    return [(-b - root, -b + root)]


def _cylinder_intervals(part: Part, origin, direction) -> list[Interval]:
    """Finite cylinder (``rcc``): side surface clipped by the two end caps."""
    r = part.radius or 0.0
    height = part.height or [0.0, 0.0, 0.0]
    length = math.sqrt(_dot(height, height))
    if length < _EPS or r <= 0:
        return []
    axis = _norm(height)
    w = _sub(origin, part.center)

    # Split ray and offset into components perpendicular to the axis.
    w_par, d_par = _dot(w, axis), _dot(direction, axis)
    w_perp = _sub(w, _scale(axis, w_par))
    d_perp = _sub(direction, _scale(axis, d_par))

    a = _dot(d_perp, d_perp)
    if a < _EPS:                     # ray parallel to the axis
        if _dot(w_perp, w_perp) > r * r:
            return []
        side: Interval = (-math.inf, math.inf)
    else:
        b = 2.0 * _dot(w_perp, d_perp)
        c = _dot(w_perp, w_perp) - r * r
        disc = b * b - 4 * a * c
        if disc <= 0:
            return []
        root = math.sqrt(disc)
        side = ((-b - root) / (2 * a), (-b + root) / (2 * a))

    # Clip to the axial extent 0..length measured along the axis.
    if abs(d_par) < _EPS:
        if w_par < 0.0 or w_par > length:
            return []
        caps: Interval = (-math.inf, math.inf)
    else:
        t1 = (0.0 - w_par) / d_par
        t2 = (length - w_par) / d_par
        caps = (min(t1, t2), max(t1, t2))

    lo, hi = max(side[0], caps[0]), min(side[1], caps[1])
    return [(lo, hi)] if hi > lo else []


_SHAPE_INTERVALS = {
    "box": _box_intervals,
    "cylinder": _cylinder_intervals,
    "sphere": _sphere_intervals,
}


def part_intervals(part: Part, origin, direction) -> list[Interval]:
    """Parameter intervals where the ray is inside this primitive."""
    fn = _SHAPE_INTERVALS.get(part.shape)
    return fn(part, origin, direction) if fn else []


# --- interval boolean algebra --------------------------------------------

def union(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Merged union of two interval lists."""
    merged: list[Interval] = []
    for lo, hi in sorted(a + b):
        if merged and lo <= merged[-1][1] + _EPS:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged


def subtract(a: list[Interval], b: list[Interval]) -> list[Interval]:
    """Whatever remains of *a* once every interval of *b* is removed."""
    out = list(a)
    for b_lo, b_hi in b:
        nxt: list[Interval] = []
        for lo, hi in out:
            if b_hi <= lo + _EPS or b_lo >= hi - _EPS:
                nxt.append((lo, hi))          # no overlap
                continue
            if b_lo > lo + _EPS:
                nxt.append((lo, b_lo))        # piece before the cut
            if b_hi < hi - _EPS:
                nxt.append((b_hi, hi))        # piece after the cut
        out = nxt
    return out


def model_intervals(spec: BuildSpec, origin, direction) -> list[Interval]:
    """Solid intervals along the ray, folding the spec LEFT TO RIGHT.

    The fold order mirrors how the region is actually built (each operator
    applies to the accumulation so far), so an interleaved spec such as
    add, subtract, add evaluates the same way here as in the database.
    """
    if not spec.parts:
        return []
    acc = part_intervals(spec.parts[0], origin, direction)
    for part in spec.parts[1:]:
        got = part_intervals(part, origin, direction)
        acc = union(acc, got) if part.op == "add" else subtract(acc, got)
    # Clip to t >= 0: material BEHIND the ray origin is not on the ray, and the
    # engine only reports what it travels through going forward.  Without this
    # the prediction could count solid the raytracer never sees.
    clipped = [(max(lo, 0.0), hi) for lo, hi in acc if hi > 0.0]
    return [(lo, hi) for lo, hi in clipped if hi > lo + _EPS]


def expected_thickness(spec: BuildSpec, origin, direction) -> float:
    """Total length of material the ray should cross (mm)."""
    return sum(hi - lo for lo, hi in model_intervals(spec, origin, direction))
