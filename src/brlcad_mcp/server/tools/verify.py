"""Engine-truth verification of a built model against its spec.

Answers "did the BUILD match the SPEC?" objectively using BRL-CAD's own
raytracer -- never a render or a vision model, which can be fooled by camera
angle, occlusion or lighting (the failure that let a missing hole pass as "looks
right").

The check is deliberately shape-agnostic.  Rather than reasoning about what each
cutter is meant to be -- a through-hole? a pocket? a slot? -- we sample the model
with rays and compare, for each ray, the material thickness the spec predicts
(:mod:`brlcad_mcp.server.tools.csg`, evaluated analytically) against the
thickness ``nirt`` reports.  One comparison subsumes every special case:

  * a subtraction that silently did not apply (the r-operator binding bug)
  * a cavity in the wrong place, or of the wrong size or diameter
  * a blind pocket, including whether its depth is right
  * wrong overall dimensions, dropped or duplicated parts

An earlier version reasoned per cutter (is this axis a through direction? where
is a witness ray safe?).  That needed new logic for every primitive and produced
two false failures on models it was not written for, so it was replaced by this.

Intent is a separate question: the sampling proves the build matches the spec,
but *not* that the spec matches what the user asked for.  ``Part.hole`` is a
cheap, optional assertion for that (see reconstruct._validate); the eval harness
covers it properly with ground-truth specs.

The geometry and parsing are pure and unit-tested; socket calls go through an
injected ``prober`` so a whole verdict is testable without a live listener.
"""

from __future__ import annotations

import math
import re

from pydantic import Field, ValidationError

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.csg import expected_thickness
from brlcad_mcp.server.tools.helpers import (
    is_error_response,
    ls_names,
    parse_json_arg,
    parse_response,
)
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part
from brlcad_mcp.transport import send_command

# Rays start well outside any plausible model and fire inward.
_RAY_STANDOFF = 10000.0
# Bounding-box tolerance: the larger of 1 mm or 2% of the expected length.
_BBOX_TOL_ABS = 1.0
_BBOX_TOL_FRAC = 0.02
# Thickness tolerance per ray: the larger of 0.5 mm or 2% of the expectation.
_LOS_TOL_ABS = 0.5
_LOS_TOL_FRAC = 0.02
# nirt leaves a "query_ray<hex>" object in the database for every ray it fires.
_RAY_ARTIFACT_PREFIX = "query_ray"
# Grid resolution per axis for the background sweep (n x n rays per axis).
_GRID = 3
# How far off a cutter's centre to place its targeted edge samples, as a
# fraction of its half-extent (inside the cavity, but near its wall).
_EDGE_FRAC = 0.6


# --- extents / bounding box ----------------------------------------------

def _part_extent(part: Part):
    """Axis-aligned (minx,miny,minz,maxx,maxy,maxz) bound of one part."""
    cx, cy, cz = part.center
    if part.shape == "box":
        sx, sy, sz = part.size  # type: ignore[misc]
        return (cx - sx / 2, cy - sy / 2, cz - sz / 2,
                cx + sx / 2, cy + sy / 2, cz + sz / 2)
    if part.shape == "cylinder":
        hx, hy, hz = part.height  # type: ignore[misc]
        r = part.radius or 0.0
        base, tip = (cx, cy, cz), (cx + hx, cy + hy, cz + hz)
        mag = math.sqrt(hx * hx + hy * hy + hz * hz) or 1.0
        axis = (hx / mag, hy / mag, hz / mag)
        # The radius widens the box only PERPENDICULAR to the axis; padding r on
        # every axis would over-report the length by 2r.
        pad = [r * math.sqrt(max(0.0, 1.0 - u * u)) for u in axis]
        lo = [min(base[i], tip[i]) - pad[i] for i in range(3)]
        hi = [max(base[i], tip[i]) + pad[i] for i in range(3)]
        return (*lo, *hi)
    r = part.radius or 0.0  # sphere
    return (cx - r, cy - r, cz - r, cx + r, cy + r, cz + r)


def add_parts_extent(spec: BuildSpec):
    """Combined extent of the ADD parts -- the material before any cutting."""
    exts = [_part_extent(p) for p in spec.parts if p.op == "add"]
    if not exts:
        return None
    lo = [min(e[i] for e in exts) for i in range(3)]
    hi = [max(e[i + 3] for e in exts) for i in range(3)]
    return (*lo, *hi)


def _overlaps(a, b) -> bool:
    """True if two extents overlap on all three axes."""
    if a is None or b is None:
        return False
    return all(a[i] < b[i + 3] and a[i + 3] > b[i] for i in range(3))


def local_material_extent(part: Part, spec: BuildSpec):
    """Extent of only the ADD parts this cutter actually passes through.

    Needed for spec-level reasoning about a cavity's intent: the whole model's
    extent would misjudge a multi-part model, because an L-bracket's union spans
    50 mm in X while a hole through its 2.5 mm upright only has to cross that
    upright.  Falls back to the whole model when nothing overlaps.
    """
    cutter = _part_extent(part)
    exts = [_part_extent(p) for p in spec.parts
            if p.op == "add" and _overlaps(cutter, _part_extent(p))]
    if not exts:
        return add_parts_extent(spec)
    lo = [min(e[i] for e in exts) for i in range(3)]
    hi = [max(e[i + 3] for e in exts) for i in range(3)]
    return (*lo, *hi)


def expected_bbox_lengths(spec: BuildSpec):
    """Expected outer (Lx,Ly,Lz): holes are interior, so only ADD parts count."""
    extent = add_parts_extent(spec)
    if extent is None:
        return None
    return (extent[3] - extent[0], extent[4] - extent[1], extent[5] - extent[2])


def parse_bb_lengths(output: str):
    """Pull (Lx,Ly,Lz) out of a ``bb`` report; None for any axis not found."""
    dims = []
    for axis in ("X", "Y", "Z"):
        found = re.search(rf"{axis} Length:\s*([\d.eE+-]+)", output)
        dims.append(float(found.group(1)) if found else None)
    return tuple(dims)


def bbox_matches(expected, actual) -> bool:
    """True if every axis length is within tolerance of the expected."""
    if expected is None or any(a is None for a in actual):
        return False
    return all(abs(e - a) <= max(_BBOX_TOL_ABS, e * _BBOX_TOL_FRAC)
               for e, a in zip(expected, actual))


# --- rays -----------------------------------------------------------------

def ray_cmd(start, direction) -> str:
    """A scriptable single-ray ``nirt`` command."""
    sx, sy, sz = start
    dx, dy, dz = direction
    return (f'nirt -e "xyz {sx:g} {sy:g} {sz:g}" '
            f'-e "dir {dx:g} {dy:g} {dz:g}" -e "s" -e "q"')


def ray_missed(nirt_output: str) -> bool:
    """True if the ray hit nothing at all."""
    return "missed the target" in nirt_output.lower()


def total_los(nirt_output: str) -> float:
    """Total material thickness a ray crossed, summed over ALL partitions.

    A ray can enter and leave the same region several times (either side of a
    hole, separate plates), and each crossing is its own line, so the sum is
    what compares against the analytic expectation.  On a hit line the closing
    paren is attached to the last coordinate, so LOS is the next field; the
    column header also contains a ``z)`` token but its next field is
    non-numeric and is skipped.
    """
    total = 0.0
    for line in nirt_output.splitlines():
        fields = line.split()
        for i, text in enumerate(fields[:-1]):
            if text.endswith(")"):
                try:
                    total += float(fields[i + 1])
                except ValueError:
                    pass
                break
    return total


def _ray_along(axis: int, point) -> tuple[tuple, tuple]:
    """A ray parallel to *axis* passing through *point*, fired inward."""
    start = list(point)
    start[axis] += _RAY_STANDOFF
    direction = [0.0, 0.0, 0.0]
    direction[axis] = -1.0
    return tuple(start), tuple(direction)


def sample_rays(spec: BuildSpec) -> list[tuple[str, tuple, tuple]]:
    """(label, start, direction) samples covering the model.

    Two families, both shape-agnostic:
      * a coarse grid across the model on each axis, which catches wrong
        dimensions and material that should or should not be there;
      * per subtracted part, a ray down the middle of the cavity on each axis
        plus samples offset toward its walls -- the offsets are what make a
        cavity of the wrong DIAMETER detectable rather than merely present.
    """
    extent = add_parts_extent(spec)
    if extent is None:
        return []
    rays: list[tuple[str, tuple, tuple]] = []

    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        for a in range(_GRID):
            for b in range(_GRID):
                point = [0.0, 0.0, 0.0]
                point[axis] = (extent[axis] + extent[axis + 3]) / 2.0
                for other, step in zip(others, (a, b)):
                    lo, hi = extent[other], extent[other + 3]
                    point[other] = lo + (hi - lo) * (step + 0.5) / _GRID
                start, direction = _ray_along(axis, point)
                rays.append((f"grid{axis}:{a}{b}", start, direction))

    for part in spec.parts:
        if part.op != "subtract":
            continue
        cut = _part_extent(part)
        centre = [(cut[i] + cut[i + 3]) / 2.0 for i in range(3)]
        for axis in range(3):
            start, direction = _ray_along(axis, centre)
            rays.append((f"{part.name}@centre.{axis}", start, direction))
            for other in (i for i in range(3) if i != axis):
                half = (cut[other + 3] - cut[other]) / 2.0
                for sign in (-1, 1):
                    point = list(centre)
                    point[other] = centre[other] + sign * half * _EDGE_FRAC
                    start, direction = _ray_along(axis, point)
                    rays.append(
                        (f"{part.name}@edge.{axis}{other}{'+' if sign > 0 else '-'}",
                         start, direction))
    return rays


# --- verdict --------------------------------------------------------------

def cleanup_ray_artifacts(prober) -> list[str]:
    """Remove the ``query_ray*`` objects MGED's nirt leaves in the database.

    Every ray we fire adds one, so verification -- the tool whose whole job is to
    tell the truth about a database -- was quietly littering it.  The leftovers
    cannot be read (``rt_db_get_internal(query_rayffff) failure``) and they make
    ``ls`` non-empty forever, so "did we delete everything?" stops having an
    honest answer.

    Best effort, and the return value is what we ATTEMPTED, not what went away:
    a leftover was observed surviving its own ``kill`` (the directory entry stays
    with no valid internal representation), so a caller must not read this as
    proof of a clean database.

    :func:`is_ray_artifact` filters the kill list by name, which is the safety
    property that matters here: if a listener ignores the ``query_ray*`` glob and
    returns the whole database, we must not kill model geometry.
    """
    try:
        listing = parse_response(prober(f"ls {_RAY_ARTIFACT_PREFIX}*"))
    except (ConnectionError, TimeoutError):
        return []
    names = sorted(n for n in ls_names(listing) if is_ray_artifact(n))
    if names:
        prober(f"kill {' '.join(names)}")
    return names


def is_ray_artifact(name: str) -> bool:
    """True for a nirt leftover, which is not model geometry."""
    return name.startswith(_RAY_ARTIFACT_PREFIX)


def _thickness_checks(spec: BuildSpec, prober, region: str):
    """Compare predicted vs measured thickness for every sample ray."""
    prober("zap")            # nirt sees the DISPLAYED objects, so isolate first
    prober(f"draw {region}")

    mismatches: list[str] = []
    total = 0
    for label, start, direction in sample_rays(spec):
        total += 1
        want = expected_thickness(spec, start, direction)
        out = parse_response(prober(ray_cmd(start, direction)))
        got = 0.0 if ray_missed(out) else total_los(out)
        if abs(want - got) > max(_LOS_TOL_ABS, want * _LOS_TOL_FRAC):
            mismatches.append(
                f"{label}: expected {want:.3g} mm of material, measured "
                f"{got:.3g} mm")
    # Tidy up after ourselves before returning a verdict on the database.
    cleanup_ray_artifacts(prober)
    return total, mismatches


def _describe(mismatches: list[str], total: int) -> str:
    shown = "; ".join(mismatches[:4])
    more = f" (+{len(mismatches) - 4} more)" if len(mismatches) > 4 else ""
    return (f"{len(mismatches)} of {total} sample rays disagree with the spec: "
            f"{shown}{more}. A ray measuring MORE material than expected means a "
            f"subtraction did not apply; LESS means material is missing or a "
            f"cavity is too large. Check the offending parts' centre and size.")


def _verify(spec: BuildSpec, prober):
    """Run the checks via *prober* (cmd -> response). Returns (passed, checks)."""
    region = f"{spec.name}.r"
    checks: list[tuple[str, bool, str]] = []

    # A missing object makes `l` return SUCCESS with an EMPTY payload rather than
    # an error, so an error-prefix check alone reports a phantom region as
    # existing and then blames the geometry for being absent.
    listing = prober(f"l {region}")
    exists = (not is_error_response(listing)
              and bool(parse_response(listing).strip()))
    checks.append(("exists", exists,
                   region if exists else f"{region} is not in the database"))
    if not exists:
        return False, checks

    expected = expected_bbox_lengths(spec)
    actual = parse_bb_lengths(parse_response(prober(f"bb {region}")))
    checks.append(("bbox", bbox_matches(expected, actual),
                   f"expected {expected} mm, got {actual} mm"))

    total, mismatches = _thickness_checks(spec, prober, region)
    checks.append(("geometry", not mismatches,
                   f"all {total} sample rays match the spec" if not mismatches
                   else _describe(mismatches, total)))

    return all(ok for _, ok, _ in checks), checks


def _format(region: str, passed: bool, checks) -> str:
    lines = [f"Verification of '{region}': {'PASS' if passed else 'FAIL'}"]
    for name, ok, detail in checks:
        lines.append(f"  [{'ok' if ok else 'x'}] {name}: {detail}")
    if not passed:
        lines.append("At least one engine-truth check failed; the build does "
                     "not match the spec.")
    return "\n".join(lines)


@mcp.tool()
def verify_model_dimensions(
    spec: str | dict = Field(
        ...,
        description=(
            "The SAME JSON spec used to build the model. Verifies the built "
            "region against it with BRL-CAD's own raytracer, NOT a render: it "
            "checks the region exists, its bounding box is right, and that "
            "along dozens of sample rays the model contains exactly the "
            "material the spec implies -- which catches missing subtractions, "
            "mis-placed or wrong-sized cavities, and pockets of the wrong "
            "depth, for any shape. Returns PASS/FAIL with per-check detail."
        ),
    ),
) -> str:
    """Verify a built model matches its spec using engine truth (bb + nirt)."""
    data, err = parse_json_arg(spec, "spec")
    if err:
        return err
    try:
        parsed = BuildSpec.model_validate(data)
    except ValidationError as exc:
        return f"Error: spec is not valid ({exc})."
    passed, checks = _verify(parsed, send_command)
    return _format(f"{parsed.name}.r", passed, checks)
