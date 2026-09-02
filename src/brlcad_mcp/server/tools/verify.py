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
import os
import re

from pydantic import Field, ValidationError

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.csg import expected_thickness
from brlcad_mcp.server.tools.helpers import (
    RAY_ARTIFACT_PREFIX,
    is_error_response,
    is_ray_artifact,
    ls_names,
    parse_json_arg,
    parse_response,
)
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part, _latest_spec
from brlcad_mcp.transport import send_command

# Rays start well outside any plausible model and fire inward.
_RAY_STANDOFF = 10000.0
# Bounding-box tolerance: the larger of 1 mm or 2% of the expected length.
_BBOX_TOL_ABS = 1.0
_BBOX_TOL_FRAC = 0.02
# Thickness tolerance per ray: the larger of 0.5 mm or 2% of the expectation.
_LOS_TOL_ABS = 0.5
_LOS_TOL_FRAC = 0.02
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
    if part.shape == "wedge":
        # The frustum's footprint is the WIDER of its two faces on each axis:
        # taking only ``size`` would under-report a wedge that flares outward.
        sx, sy, sz = part.size  # type: ignore[misc]
        tx, ty = part.top_size if part.top_size else (sx, sy)
        hx, hy = max(sx, tx) / 2, max(sy, ty) / 2
        return (cx - hx, cy - hy, cz - sz / 2, cx + hx, cy + hy, cz + sz / 2)
    if part.shape == "cone":
        hxv, hyv, hzv = part.height  # type: ignore[misc]
        r0 = part.radius or 0.0
        r1 = part.top_radius if part.top_radius is not None else r0
        r = max(r0, r1)                  # widest end bounds the whole frustum
        base, tip = (cx, cy, cz), (cx + hxv, cy + hyv, cz + hzv)
        mag = math.sqrt(hxv * hxv + hyv * hyv + hzv * hzv) or 1.0
        axis = (hxv / mag, hyv / mag, hzv / mag)
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


_EXTENT = re.compile(
    r"min\s*\{([^}]*)\}\s*max\s*\{([^}]*)\}", re.I)


def parse_bb_extent(output: str):
    """Absolute (minX,minY,minZ,maxX,maxY,maxZ) from a ``bb -e`` report.

    Plain ``bb`` reports LENGTHS only, which say nothing about where the model
    sits.  ``-e`` is what gives the corner, and the corner is what lets a check
    be written against a drawing that fixes distances from an edge without
    fixing the origin -- which is most drawings.
    """
    found = _EXTENT.search(output)
    if not found:
        return None
    try:
        lo = [float(v) for v in found.group(1).split()]
        hi = [float(v) for v in found.group(2).split()]
    except ValueError:
        return None
    if len(lo) != 3 or len(hi) != 3:
        return None
    return (*lo, *hi)


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


# --- ray checks: OFF for this release ------------------------------------
#
# The rays are fired by ``nirt``, which MGED runs as a SEPARATE executable, and
# in a plain build-tree launch that executable does not start at all:
#
#     nirt: error while loading shared libraries: libz_brl.so.1
#
# It needs libz_brl transitively via libpng_brl16, whose RUNPATH is empty, and
# DT_RUNPATH is not inherited by a dependency's own dependencies -- so mged
# starts fine and every ray it tries to fire dies at exec. An installed tree
# resolves it correctly and passes all of these checks, so this is about the
# launch environment, not the geometry.
#
# Left on, the failure was worse than useless: every ray reported nothing, that
# read as "the model is empty", and the agent went looking for a fault that was
# not there -- a search that reached ``nirt <object>``, which crashes MGED
# outright and takes the listener with it.
#
# NOTHING BELOW IS REMOVED. The sampling, the thickness prediction, the parsers
# and their tests are all intact; only the call site is gated. Turn them back on
# with BRLCAD_RAY_CHECKS=1 once nirt starts in the target environment -- verify
# it does with: nirt -h
RAY_CHECKS = os.getenv("BRLCAD_RAY_CHECKS", "").strip().lower() in (
    "1", "true", "yes", "on")


def nirt_ran(nirt_output: str) -> bool:
    """True if *nirt_output* is really nirt's report on a fired ray.

    Recognised by the three things only nirt prints: the firing state
    (``Origin (x y z) = ...``), the hit table header (``Region Name ... LOS``),
    or the miss phrase. Any one of them is enough, since which appear depends
    on the script the ray was fired with. None of them means nirt never
    reported: a timed-out command, an MGED error, a truncated or empty reply.

    This has to be checked before the numbers are read. :func:`total_los` sums
    the fields it recognises and returns 0.0 for anything it does not, and
    :func:`ray_missed` looks for one specific phrase and returns False for
    everything else. So an unrunnable ray otherwise reads as a confident "hit
    that crossed 0 mm of material", which is indistinguishable from real
    geometry being absent and blames the model for a transport failure.
    """
    lowered = nirt_output.lower()
    return ("origin (x y z)" in lowered
            or "region name" in lowered
            or ray_missed(nirt_output))


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
        listing = parse_response(prober(f"ls {RAY_ARTIFACT_PREFIX}*"))
    except (ConnectionError, TimeoutError):
        return []
    names = sorted(n for n in ls_names(listing) if is_ray_artifact(n))
    if names:
        prober(f"kill {' '.join(names)}")
    return names


def _thickness_checks(spec: BuildSpec, prober, region: str):
    """Compare predicted vs measured thickness for every sample ray."""
    prober("zap")            # nirt sees the DISPLAYED objects, so isolate first
    prober(f"draw {region}")

    mismatches: list[str] = []
    unmeasured: list[str] = []
    total = 0
    for label, start, direction in sample_rays(spec):
        total += 1
        want = expected_thickness(spec, start, direction)
        out = parse_response(prober(ray_cmd(start, direction)))
        if not nirt_ran(out):
            # Not a disagreement about geometry -- no measurement happened.
            unmeasured.append(f"{label}: {first_line(out)}")
            continue
        got = 0.0 if ray_missed(out) else total_los(out)
        if abs(want - got) > max(_LOS_TOL_ABS, want * _LOS_TOL_FRAC):
            mismatches.append(
                f"{label}: expected {want:.3g} mm of material, measured "
                f"{got:.3g} mm")
    # Tidy up after ourselves before returning a verdict on the database.
    cleanup_ray_artifacts(prober)
    return total, mismatches, unmeasured


def first_line(text: str, limit: int = 70) -> str:
    """The first non-blank line of *text*, for quoting a reply back concisely."""
    for line in text.splitlines():
        if line.strip():
            return line.strip()[:limit]
    return "<empty reply>"


def _describe_unmeasured(unmeasured: list[str], total: int) -> str:
    shown = "; ".join(unmeasured[:3])
    more = f" (+{len(unmeasured) - 3} more)" if len(unmeasured) > 3 else ""
    return (f"could not measure {len(unmeasured)} of {total} sample rays -- "
            f"nirt did not report on them: {shown}{more}. The geometry is "
            f"UNVERIFIED, not wrong: check that the listener is still up and "
            f"that BRLCAD_TIMEOUT is long enough for nirt, then verify again.")


def _describe(mismatches: list[str], total: int) -> str:
    shown = "; ".join(mismatches[:4])
    more = f" (+{len(mismatches) - 4} more)" if len(mismatches) > 4 else ""
    return (f"{len(mismatches)} of {total} sample rays disagree with the spec: "
            f"{shown}{more}. A ray measuring MORE material than expected means a "
            f"subtraction did not apply; LESS means material is missing or a "
            f"cavity is too large. Check the offending parts' centre and size.")


def _verify(spec: BuildSpec, prober, rays: bool | None = None):
    """Run the checks via *prober* (cmd -> response). Returns (passed, checks).

    *rays* overrides :data:`RAY_CHECKS` for one call, which is how the eval
    harness keeps its ray sweep while the agent-side tool ships without it. It
    is an argument rather than a global assignment so that one caller's choice
    cannot leak into another's.
    """
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

    if not (RAY_CHECKS if rays is None else rays):
        # Skipped entirely rather than reported as a third state: every consumer
        # of this list treats an entry as pass/fail, and a check that ran no
        # rays is neither. _format says so instead, so nothing reads as verified
        # that was not.
        return all(ok for _, ok, _ in checks), checks

    total, mismatches, unmeasured = _thickness_checks(spec, prober, region)
    if unmeasured:
        # Reported ahead of any mismatch: if some rays never ran, the ones that
        # did are not a basis for a verdict on the geometry either.
        checks.append(("geometry", False, _describe_unmeasured(unmeasured, total)))
    else:
        checks.append(("geometry", not mismatches,
                       f"all {total} sample rays match the spec" if not mismatches
                       else _describe(mismatches, total)))

    return all(ok for _, ok, _ in checks), checks


def _format(region: str, passed: bool, checks) -> str:
    lines = [f"Verification of '{region}': {'PASS' if passed else 'FAIL'}"]
    for name, ok, detail in checks:
        lines.append(f"  [{'ok' if ok else 'x'}] {name}: {detail}")
    ran_rays = any(name == "geometry" for name, _, _ in checks)
    built = any(name == "exists" and ok for name, ok, _ in checks)
    if not ran_rays and built:
        # Said plainly so a PASS is not mistaken for a full engine-truth check,
        # and so this does not get reported to a user as ray-verified.
        lines.append("  [--] geometry: NOT CHECKED. Ray verification is "
                     "disabled in this release, so only existence and the "
                     "bounding box were measured. Interior features (holes, "
                     "pockets, wall thickness) are unverified -- do not "
                     "describe them as confirmed.")
    if not passed:
        # "does not match" would be a false accusation when the reason is that a
        # check could not run, so say which of the two happened.
        unverified = any("UNVERIFIED" in detail for _, ok, detail in checks
                         if not ok)
        lines.append("A check could not be completed, so the build is "
                     "unverified; this is not evidence that it is wrong."
                     if unverified else
                     "At least one engine-truth check failed; the build does "
                     "not match the spec.")
    return "\n".join(lines)


def _resolve_target(name, spec) -> tuple[dict | None, str | None]:
    """The spec to verify against: stored (by *name*) or supplied. Pure-ish.

    Exactly one input, because they can disagree and silently preferring either
    one would make the check's meaning depend on an argument the caller may not
    know it sent. Tolerates a leaked ``FieldInfo`` sentinel from a direct
    (non-MCP) call by treating anything that is not str/dict as absent.
    """
    got_name = name.strip() if isinstance(name, str) else ""
    got_spec = spec.strip() if isinstance(spec, str) else (
        spec if isinstance(spec, dict) else "")
    if got_name and got_spec:
        return None, ("Error: pass 'name' OR 'spec', not both -- they can "
                      "disagree, and which one wins would change what is being "
                      "verified. Use 'name' for a model this server built.")
    if not got_name and not got_spec:
        return None, ("Error: pass 'name' (preferred -- the region's build name, "
                      "whose stored spec the server reads) or 'spec' (a full JSON "
                      "spec, for a model this server did not build).")
    if got_name:
        stored = _latest_spec(got_name)
        if stored is None:
            return None, (f"Error: no saved build named '{got_name}'. Use "
                          f"list_builds to see stored names, or pass 'spec'.")
        return stored, None
    return parse_json_arg(got_spec, "spec")


@mcp.tool()
def verify_model_dimensions(
    # NOTE: ``spec`` stays FIRST so existing positional callers keep working.
    # With ``name`` first, a positional dict landed in ``name``, was rejected as
    # not-a-string, and the call became a no-op -- a silent failure if the
    # both/neither guard below had not caught it.
    spec: str | dict = Field(
        default="",
        description=(
            "A full JSON spec, for a model this server did NOT build (or a "
            "hypothetical one). Prefer 'name' when the build is stored. Verifies "
            "the built region against engine truth, NOT a render. Returns "
            "PASS/FAIL with per-check detail."
            + (" Checks the region exists, its bounding box is right, and that "
               "along dozens of sample rays the model contains exactly the "
               "material the spec implies -- which catches missing "
               "subtractions, mis-placed or wrong-sized cavities, and pockets "
               "of the wrong depth, for any shape."
               if RAY_CHECKS else
               " In this release it checks EXISTENCE and the BOUNDING BOX only: "
               "ray checking is disabled, so interior features (holes, "
               "pockets, wall thickness) are NOT verified. A PASS here means "
               "the overall size is right; do not report interior features as "
               "confirmed by it.")
        ),
    ),
    name: str = Field(
        default="",
        description=(
            "PREFERRED whenever build_from_spec or edit_build made the model: the "
            "build name (no '.r'). The server reads the spec it ACTUALLY built "
            "from, so you do not resend it -- faster, and it cannot verify "
            "against a mistyped copy."
        ),
    ),
) -> str:
    """Verify a built model matches its spec using engine truth (bb + nirt).

    Ray checking is disabled in this release (see ``RAY_CHECKS`` above), so what
    actually runs is existence plus the bounding box. The ray machinery is still
    here and still tested; only the call site is gated.

    Two ways in, and they are NOT equivalent. ``name`` reads the stored spec --
    the one ``build_from_spec``/``edit_build`` actually used -- so the check is
    against ground truth. ``spec`` verifies against whatever is supplied, which is
    needed for a model this server did not build, but note that the sample rays
    and the expected bbox are derived from the ARGUMENT: a resent spec that
    drifted from the stored one moves the goalposts rather than failing, so the
    tool can report PASS against a spec that is not the one on disk.
    """
    data, err = _resolve_target(name, spec)
    if err:
        return err
    try:
        parsed = BuildSpec.model_validate(data)
    except ValidationError as exc:
        return f"Error: spec is not valid ({exc})."
    passed, checks = _verify(parsed, send_command)
    return _format(f"{parsed.name}.r", passed, checks)
