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


def _wedge_intervals(part: Part, origin, direction) -> list[Interval]:
    """Rectangular frustum: a box whose top face has its own footprint.

    Convex, so it is the intersection of six half-spaces and the same clip loop
    as the box works -- only the four side planes are tilted.  Written as a
    frustum rather than a raw eight-vertex ``arb8`` because the shapes the
    corpus needs (a 48 degree tapered flank, a triangular gusset, a chamfer) are
    all this, and a caller gets a taper right far more often than it gets eight
    ordered vertices right.  ``top_size`` of [x, 0] degenerates to a triangular
    prism, which is the gusset case.
    """
    sx, sy, sz = part.size  # type: ignore[misc]
    tx, ty = part.top_size if part.top_size else (sx, sy)
    cx, cy, cz = part.center
    z0, z1 = cz - sz / 2, cz + sz / 2
    if sz <= _EPS:
        return []

    t_lo, t_hi = -math.inf, math.inf
    # Bottom and top caps first: the plain slab test along Z.
    if abs(direction[2]) < _EPS:
        if origin[2] < z0 or origin[2] > z1:
            return []
    else:
        a = (z0 - origin[2]) / direction[2]
        b = (z1 - origin[2]) / direction[2]
        t_lo, t_hi = max(t_lo, min(a, b)), min(t_hi, max(a, b))

    # Each side plane interpolates its half-width with height, so the constraint
    # |p_i - c_i| <= half(z) is linear in t and stays a half-space.
    for i, (base, top, centre) in enumerate(
            ((sx / 2, tx / 2, cx), (sy / 2, ty / 2, cy))):
        slope = (top - base) / sz              # half-width change per unit Z
        # half(z) = base + slope * (z - z0); write p_i - centre = +/- half(z).
        for sign in (1.0, -1.0):
            # sign*(o_i + t*d_i - centre) - base - slope*(o_z + t*d_z - z0) <= 0
            coef = sign * direction[i] - slope * direction[2]
            const = (sign * (origin[i] - centre) - base
                     - slope * (origin[2] - z0))
            if abs(coef) < _EPS:
                if const > 0:
                    return []
                continue
            t = -const / coef
            if coef > 0:
                t_hi = min(t_hi, t)
            else:
                t_lo = max(t_lo, t)
    return [(t_lo, t_hi)] if t_hi > t_lo else []


def _cone_intervals(part: Part, origin, direction) -> list[Interval]:
    """Truncated cone (``tgc``): radius varies linearly from base to top.

    The same split into axial and perpendicular components as the cylinder; the
    only difference is that the radius being solved against is a function of how
    far along the axis you are, which keeps the side test quadratic.
    """
    r0 = part.radius or 0.0
    r1 = part.top_radius if part.top_radius is not None else r0
    height = part.height or [0.0, 0.0, 0.0]
    length = math.sqrt(_dot(height, height))
    if length < _EPS or (r0 <= 0 and r1 <= 0):
        return []
    axis = _norm(height)
    w = _sub(origin, part.center)
    w_par, d_par = _dot(w, axis), _dot(direction, axis)
    w_perp = _sub(w, _scale(axis, w_par))
    d_perp = _sub(direction, _scale(axis, d_par))

    # radius(t) = r0 + k * (w_par + t*d_par); solve |perp|^2 = radius^2.
    k = (r1 - r0) / length
    a = _dot(d_perp, d_perp) - (k * d_par) ** 2
    rad0 = r0 + k * w_par
    b = 2.0 * (_dot(w_perp, d_perp) - k * d_par * rad0)
    c = _dot(w_perp, w_perp) - rad0 * rad0
    # Solve a*t^2 + b*t + c <= 0.  The sign of *a* decides the SHAPE of the
    # solution set, and getting that wrong is not a rounding error: when the ray
    # leans inside the cone's own half-angle (|d_perp| < |k*d_par|, which every
    # axis-parallel ray does) *a* goes negative and the set becomes the OUTSIDE
    # of the roots -- two half-lines, not one interval.  Treating it as one
    # interval predicted zero material through a solid cone.
    if abs(a) < _EPS:                       # linear: ray parallel to the flank
        if abs(b) < _EPS:
            side = [(-math.inf, math.inf)] if c <= 0 else []
        elif b > 0:
            side = [(-math.inf, -c / b)]
        else:
            side = [(-c / b, math.inf)]
    else:
        disc = b * b - 4 * a * c
        if a > 0:
            if disc <= 0:
                return []
            root = math.sqrt(disc)
            lo, hi = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
            side = [(lo, hi)]
        elif disc <= 0:
            # Opens downward and never rises above zero: satisfied everywhere.
            side = [(-math.inf, math.inf)]
        else:
            root = math.sqrt(disc)
            lo, hi = sorted(((-b - root) / (2 * a), (-b + root) / (2 * a)))
            side = [(-math.inf, lo), (hi, math.inf)]

    # Clip to the axial span 0..length.  This is also what excludes the MIRROR
    # cone on the far side of the apex, which satisfies |perp| <= |radius| just
    # as well but is no part of the solid.
    if abs(d_par) < _EPS:
        if w_par < 0.0 or w_par > length:
            return []
        caps: Interval = (-math.inf, math.inf)
    else:
        t1 = (0.0 - w_par) / d_par
        t2 = (length - w_par) / d_par
        caps = (min(t1, t2), max(t1, t2))

    out = []
    for s_lo, s_hi in side:
        lo, hi = max(s_lo, caps[0]), min(s_hi, caps[1])
        if hi > lo:
            out.append((lo, hi))
    return out


_SHAPE_INTERVALS = {
    "box": _box_intervals,
    "cylinder": _cylinder_intervals,
    "sphere": _sphere_intervals,
    "wedge": _wedge_intervals,
    "cone": _cone_intervals,
}


def part_intervals(part: Part, origin, direction) -> list[Interval]:
    """Parameter intervals where the ray is inside this primitive.

    An unknown shape RAISES rather than returning nothing.  Returning [] reads
    as "the ray crosses no material here", so a primitive we cannot intersect
    would not make verification silent -- it would make it *wrong*, reporting
    every ray through that part as measuring more material than expected.  A
    shape reaching here unknown means the build-time whitelist and this table
    have drifted apart, and that must be loud.
    """
    fn = _SHAPE_INTERVALS.get(part.shape)
    if fn is None:
        raise ValueError(
            f"no ray-intersection function for shape '{part.shape}'; "
            f"verification cannot predict material for it")
    return fn(part, origin, direction)


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
