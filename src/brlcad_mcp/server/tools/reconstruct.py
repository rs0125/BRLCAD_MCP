"""MCP tool -- build a parametric model from a structured spec, then render it.

This is the DETERMINISTIC half of the image-to-model workflow.  The agent (which
can see a reference image) decides WHAT to build and fills in a spec; this tool
validates that spec, BUILDS it as BRL-CAD CSG over the socket, and RENDERS check
views so the result can be compared against the reference.  Same spec in -> same
geometry out, every time.  Re-running with the same names rebuilds cleanly
(existing objects are removed first), which is what the adjust loop needs.

Spec vocabulary: box / cylinder / sphere / wedge / cone primitives, unioned or
subtracted into a single region, with an optional overall colour.

The vocabulary is deliberately far smaller than BRL-CAD's 41 primitives, and the
constraint is NOT what the engine can build -- it builds all of them, and
``execute_command`` reaches any of them today.  It is what ``csg.py`` can
intersect a ray with analytically, because that prediction is what verification
compares against.  ``wedge`` (a rectangular frustum) and ``cone`` (a truncated
cone) were added because nothing else here can make a sloped or tapered face;
a torus was not, because its ray intersection is a quartic and fillets are
already reachable as box minus cylinder.
"""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, Field, ValidationError

from brlcad_mcp.config import settings
from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools import rendering as R
from brlcad_mcp.server.tools.helpers import (
    is_error_response,
    ls_names,
    parse_json_arg,
    parse_region_members,
    parse_response,
    region_fold_cmds,
)
from brlcad_mcp.transport import send_command

_SHAPES = ("box", "cylinder", "sphere", "wedge", "cone")
_OPS = ("add", "subtract")
_HOLE_KINDS = ("through", "pocket")
# Check views are diagnostic images: flat lighting keeps features countable.
_CHECK_LIGHTING = "ambient"


class Part(BaseModel):
    """One primitive in the model.  Which fields are required depends on shape:

    box:      size [x,y,z] and center [x,y,z]
    cylinder: center [x,y,z] (base), height [x,y,z] (axis vector), radius
    sphere:   center [x,y,z], radius
    wedge:    center [x,y,z], size [x,y,z], top_size [x,y] -- a box whose top
              face has its own footprint. The sloped-face primitive: tapered
              flanks, gussets (top_size [x,0]), chamfers.
    cone:     center [x,y,z] (base), height [x,y,z] (axis vector), radius
              (base), top_radius. Tapered bosses, countersinks, turned profiles.
    """

    name: str
    shape: str
    op: str = "add"
    center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: list[float] | None = None
    height: list[float] | None = None
    radius: float | None = None
    # wedge: footprint of the TOP face, [x, y].  Absent means the top matches
    # ``size``, i.e. the wedge degenerates to a plain box.
    top_size: list[float] | None = None
    # cone: radius at the far end of ``height``.  Absent means it equals
    # ``radius``, i.e. the cone degenerates to a plain cylinder.
    top_radius: float | None = None
    # Intent for a subtracted cylinder: "through" (a hole that exits the far
    # face) or "pocket" (a blind recess that deliberately stops inside).  A
    # verifier cannot infer this from geometry alone -- a cutter that stops
    # inside is either a bug or exactly what was wanted -- so state it.  Left
    # unset, it is inferred from the cutter's span and reported.
    hole: str | None = None


class BuildSpec(BaseModel):
    """A whole model: a named region built from parts, plus render settings."""

    name: str = "model"
    parts: list[Part]
    color: list[int] | None = None
    # The overall size the author BELIEVES this spec produces, [Lx, Ly, Lz] in mm.
    # Declaring it turns a silent placement error into a pre-build rejection:
    # prose alone did not stop parts being centred outside a stated range (an
    # L-bracket asked to span 0..50 mm kept coming out 52.5 mm), so the intent is
    # checked against the geometry instead of merely being requested.
    expect_bbox: list[float] | None = None
    views: list[str] = Field(default_factory=lambda: ["front", "side"])
    render_size: int = 256
    # Accepted so previously saved specs still load, but IGNORED: check views are
    # always rendered with flat ambient light so features stay countable (see
    # _CHECK_LIGHTING).  Use render_model / render_previews for lit renders.
    # It is deliberately NOT validated -- an ignored field must never be able to
    # reject a good build, which is what a membership check on it did.  Setting it
    # is reported back instead (see _report), so a caller learns it had no effect.
    lighting: str = "ambient"


def _validate(spec: BuildSpec) -> list[str]:
    """Semantic validation beyond pydantic's type checks.  Returns error lines."""
    errors: list[str] = []
    if not spec.parts:
        return ["spec has no parts"]
    for view in spec.views:
        if view not in R._VIEWS:
            errors.append(f"unknown view '{view}' "
                          f"(use {', '.join(sorted(R._VIEWS))})")
    if spec.parts[0].op != "add":
        errors.append("the first part must have op 'add' (a region must start "
                      "with a solid, not a subtraction)")
    seen: set[str] = set()
    for p in spec.parts:
        if not p.name or " " in p.name:
            errors.append(f"invalid part name '{p.name}' (no spaces, non-empty)")
        if p.name in seen:
            errors.append(f"duplicate part name '{p.name}'")
        seen.add(p.name)
        if p.shape not in _SHAPES:
            errors.append(f"'{p.name}': unknown shape '{p.shape}' "
                          f"(use {', '.join(_SHAPES)})")
            continue
        if p.op not in _OPS:
            errors.append(f"'{p.name}': op must be 'add' or 'subtract'")
        if len(p.center) != 3:
            errors.append(f"'{p.name}': center must be [x, y, z]")
        if p.shape == "box":
            if not p.size or len(p.size) != 3 or any(v <= 0 for v in p.size):
                errors.append(f"'{p.name}': box needs positive size [x, y, z]")
        elif p.shape == "cylinder":
            if not p.height or len(p.height) != 3 or not any(p.height):
                errors.append(f"'{p.name}': cylinder needs a height vector")
            if not p.radius or p.radius <= 0:
                errors.append(f"'{p.name}': cylinder needs a positive radius")
        elif p.shape == "sphere":
            if not p.radius or p.radius <= 0:
                errors.append(f"'{p.name}': sphere needs a positive radius")
        elif p.shape == "wedge":
            if not p.size or len(p.size) != 3 or any(v <= 0 for v in p.size):
                errors.append(f"'{p.name}': wedge needs positive size [x, y, z]")
            if p.top_size is not None and (
                    len(p.top_size) != 2 or any(v < 0 for v in p.top_size)):
                errors.append(f"'{p.name}': wedge top_size must be [x, y], "
                              f"each >= 0 (0 collapses that edge to a line, "
                              f"which is how you get a triangular gusset)")
            if p.top_size is not None and not any(p.top_size) and not any(
                    v for v in (p.size or [])[:2]):
                errors.append(f"'{p.name}': wedge has no width at either end")
        elif p.shape == "cone":
            if not p.height or len(p.height) != 3 or not any(p.height):
                errors.append(f"'{p.name}': cone needs a height vector")
            r0 = p.radius or 0.0
            r1 = p.top_radius if p.top_radius is not None else r0
            # BRL-CAD's `in ... trc` refuses a zero radius at either end, so a
            # true apex is not available. Say so and ask for a small positive
            # value rather than substituting one silently: a cone that is not
            # the cone the caller described would verify against the wrong
            # prediction, which is worse than a rejection.
            if r0 <= 0 or r1 <= 0:
                errors.append(
                    f"'{p.name}': cone needs a positive radius at BOTH ends "
                    f"(got base {r0:g}, top {r1:g}). A true point is not "
                    f"representable -- use a small radius such as 0.1 for a "
                    f"near-apex, or 'cylinder' if the ends are equal")
    errors.extend(_hole_intent_errors(spec))
    errors.extend(_bbox_intent_errors(spec))
    return errors


def _bbox_intent_errors(spec: BuildSpec) -> list[str]:
    """Check a declared overall size against what the parts actually produce."""
    if not spec.expect_bbox or len(spec.expect_bbox) != 3:
        return []
    from brlcad_mcp.server.tools.verify import expected_bbox_lengths

    actual = expected_bbox_lengths(spec)
    if actual is None:
        return []
    off = [(axis, want, got)
           for axis, want, got in zip("XYZ", spec.expect_bbox, actual)
           if abs(want - got) > max(0.01, want * 0.001)]
    if not off:
        return []
    detail = "; ".join(f"{axis}: declared {want:g} mm but the parts span {got:g} mm"
                       for axis, want, got in off)
    return [f"expect_bbox does not match the geometry -- {detail}. Move the parts "
            f"so the model occupies the intended range (check each centre and "
            f"size), or correct expect_bbox if the declaration was wrong."]


def _hole_intent_errors(spec: BuildSpec) -> list[str]:
    """Check declared hole intent against the cutter's actual geometry.

    ``hole`` states what a cavity is FOR, which geometry alone cannot tell us --
    a cutter stopping inside is either a bug or a deliberate pocket.  Declaring
    it lets us catch the mismatch here, before any geometry exists, instead of
    building something that quietly is not what was asked for.
    """
    # Imported lazily: verify imports this module, so a top-level import would
    # be circular.
    from brlcad_mcp.server.tools.verify import (
        _part_extent,
        add_parts_extent,
        local_material_extent,
    )

    errors: list[str] = []
    material = add_parts_extent(spec)
    if material is None:
        return errors
    for p in spec.parts:
        if p.op != "subtract":
            continue
        cut = _part_extent(p)
        # A cutter that does not reach the material removes nothing.  The build
        # would "succeed" and verification would agree (it faithfully matches
        # the spec), so this has to be caught as a SPEC error, here.
        if not all(cut[i] < material[i + 3] and cut[i + 3] > material[i]
                   for i in range(3)):
            errors.append(
                f"'{p.name}': this cutter does not overlap the material at all, "
                f"so it removes nothing -- check its centre and size")
            continue
        if p.hole is None:
            continue
        if p.hole not in _HOLE_KINDS:
            errors.append(f"'{p.name}': hole must be 'through' or 'pocket'")
            continue
        # Judge "through" against the material this cutter actually crosses, not
        # the whole model: an L-bracket's union spans 50 mm in X, so a hole
        # through its 2.5 mm upright would otherwise look like a blind pocket.
        local = local_material_extent(p, spec)
        spans = [i for i in range(3)
                 if cut[i] <= local[i] and cut[i + 3] >= local[i + 3]]
        if p.hole == "through" and not spans:
            # Pocket first, deliberately.  With the geometry advice leading and
            # 'pocket' parenthesised, models twice lengthened the cutter instead --
            # which punched through a face that was meant to stay solid, and took a
            # build+verify cycle to notice and revert.  A cutter that stops inside
            # the material on every axis is far more often a mislabelled blind
            # recess than a through-hole that needs growing.
            errors.append(
                f"'{p.name}': declared hole 'through' but the cutter stops "
                f"inside the material on every axis, so it would leave a blind "
                f"pocket -- if that recess is what you meant, declare hole "
                f"'pocket'; if it really must exit the far side, move its centre "
                f"outside the near face and enlarge it past the far face")
        if p.hole == "pocket" and spans:
            errors.append(
                f"'{p.name}': declared hole 'pocket' but the cutter crosses the "
                f"whole material, so it would cut straight through -- shorten it "
                f"to leave the intended depth (or declare hole 'through')")
    return errors


def _solid_name(region: str, part_name: str) -> str:
    """Solid name for a part, NAMESPACED under its region.

    A spec's parts are internal to its region, so they must not squat on global
    names: two models both having a part called ``body`` would otherwise collide
    (and a build would clobber the other's solid).  ``<region>_<part>.s`` keeps
    every build hermetic.
    """
    return f"{region}_{part_name}.s"


def _wedge_vertices(part: Part):
    """The eight arb8 vertices of a rectangular frustum, bottom face first.

    MGED wants the four bottom corners counter-clockwise then the four top ones
    in the SAME order; getting that order wrong yields a self-intersecting solid
    that still builds and then ray-traces as nonsense, so it is generated here
    rather than asked of a caller.
    """
    sx, sy, sz = part.size  # type: ignore[misc]
    tx, ty = part.top_size if part.top_size else (sx, sy)
    cx, cy, cz = part.center
    z0, z1 = cz - sz / 2, cz + sz / 2
    out = []
    for half_x, half_y, z in ((sx / 2, sy / 2, z0), (tx / 2, ty / 2, z1)):
        out += [(cx - half_x, cy - half_y, z), (cx + half_x, cy - half_y, z),
                (cx + half_x, cy + half_y, z), (cx - half_x, cy + half_y, z)]
    return out


def _solid_cmd(part: Part, region: str = "") -> str:
    """The MGED ``in`` command that creates this part's solid."""
    n = _solid_name(region, part.name) if region else f"{part.name}.s"
    cx, cy, cz = part.center
    if part.shape == "box":
        sx, sy, sz = part.size  # type: ignore[misc]
        return (f"in {n} rpp {cx - sx / 2:g} {cx + sx / 2:g} "
                f"{cy - sy / 2:g} {cy + sy / 2:g} "
                f"{cz - sz / 2:g} {cz + sz / 2:g}")
    if part.shape == "cylinder":
        hx, hy, hz = part.height  # type: ignore[misc]
        return f"in {n} rcc {cx:g} {cy:g} {cz:g} {hx:g} {hy:g} {hz:g} {part.radius:g}"
    if part.shape == "wedge":
        return f"in {n} arb8 " + " ".join(
            f"{v:g}" for pt in _wedge_vertices(part) for v in pt)
    if part.shape == "cone":
        hx, hy, hz = part.height  # type: ignore[misc]
        r0 = part.radius or 0.0
        r1 = part.top_radius if part.top_radius is not None else r0
        # trc: base point, height vector, base radius, top radius -- the right
        # circular case of tgc, which is all a frustum needs.
        return (f"in {n} trc {cx:g} {cy:g} {cz:g} "
                f"{hx:g} {hy:g} {hz:g} {r0:g} {r1:g}")
    return f"in {n} sph {cx:g} {cy:g} {cz:g} {part.radius:g}"  # sphere


def _op_char(part: Part) -> str:
    """The MGED set operator for a part's boolean op (add=u, subtract=-)."""
    return "u" if part.op == "add" else "-"


def _accum_name(region: str, i: int) -> str:
    """Name of the i-th intermediate accumulation combination for *region*."""
    return f"{region}.acc{i}"


def _region_build_cmds(name: str, parts: list[Part]) -> list[str]:
    """Commands that build the region as a STRICT left-to-right CSG accumulation.

    BRL-CAD's ``r`` command binds each ``-``/``+`` to only the *most recent*
    union operand, so a flat ``r x u a u b - h`` yields ``a u (b - h)`` -- the
    subtraction misses ``a`` entirely.  To get the intended
    ``(((a u b) - h) ...)`` we fold left to right through intermediate
    combinations (``<name>.accN``), each combining the running accumulator with
    the next part, so every operator applies to the whole solid so far.  The
    final fold is emitted as the region itself.
    """
    members = [(_op_char(p), _solid_name(name, p.name)) for p in parts]
    # accum_prefix is the bare model name so the combs stay `<name>.accN`, which
    # is exactly what _build_geometry kills on a rebuild.
    return region_fold_cmds(f"{name}.r", members, accum_prefix=name)


def _owns_by_structure(spec: BuildSpec, region_members) -> bool:
    """True if an existing region is built the way WE build regions.

    Every solid we create is namespaced ``<region>_<part>.s`` and every
    intermediate comb is ``<region>.accN``, so a region whose tree contains only
    those could not have been assembled by hand.  Checking the structure means
    ownership no longer depends on the saved-spec directory still being present --
    deleting it used to make our own geometry look foreign and block rebuilds.
    """
    if not region_members:
        return False
    solid_prefix, acc_prefix = f"{spec.name}_", f"{spec.name}.acc"
    return all(name.startswith(solid_prefix) or name.startswith(acc_prefix)
               for _, name in region_members)


def _collision_error(spec: BuildSpec, live_names: set[str],
                     region_members=()) -> str | None:
    """Refuse to overwrite geometry this workflow does not own.

    ``_build_geometry`` deliberately kills the names it is about to create, so a
    build could silently destroy a hand-made object that happens to share a
    name.  A name is ours if we have a saved spec for it OR the region on disk is
    structurally one of ours (see :func:`_owns_by_structure`).  Pure, so it is
    unit-tested without a socket.
    """
    if _latest_spec(spec.name) is not None:
        return None                       # our own model: rebuilding is fine
    if _owns_by_structure(spec, region_members):
        return None                       # our naming convention: also ours
    clashes = [n for n in
               [f"{spec.name}.r",
                *(_solid_name(spec.name, p.name) for p in spec.parts)]
               if n in live_names]
    if not clashes:
        return None
    return (f"refusing to overwrite existing object(s) not built from a spec: "
            f"{', '.join(sorted(clashes))}. Choose a different name, or remove "
            f"them deliberately first (they are snapshotted, so restore_backup "
            f"can undo that).")


def _live_names() -> set[str]:
    """Names currently in the open database (ls decorations stripped)."""
    return ls_names(parse_response(send_command("ls")))


def _build_geometry(spec: BuildSpec) -> str | None:
    """Create the solids and the region over the socket.  None on success."""
    send_command("units mm")
    # Idempotent rebuild: killtree drops the region plus its intermediate
    # accumulation combs, then killall clears the solids (and any stray accN
    # from a build whose part count later shrank).  Errors here are ignored --
    # nothing to remove on a first build.
    send_command(f"killtree {spec.name}.r")
    for p in spec.parts:
        send_command(f"killall {_solid_name(spec.name, p.name)}")
    for i in range(1, len(spec.parts)):
        send_command(f"killall {_accum_name(spec.name, i)}")
    for p in spec.parts:
        resp = send_command(_solid_cmd(p, spec.name))
        if is_error_response(resp):
            return f"failed to create '{p.name}' ({p.shape}): {parse_response(resp)}"
    for cmd in _region_build_cmds(spec.name, spec.parts):
        resp = send_command(cmd)
        if is_error_response(resp):
            return f"failed to build region '{spec.name}.r': {parse_response(resp)}"
    if spec.color and len(spec.color) == 3:
        r, g, b = spec.color
        send_command(f"comb_color {spec.name}.r {r} {g} {b}")
    return None


def _render_checks(region: str, views: list[str], size: int):
    """Render each check view into a fresh timestamped folder.

    Always uses AMBIENT lighting.  These images exist to be counted and compared
    against a reference, not admired: a three-point rig lights a stud or a boss
    the same colour as the face it stands on, so features wash out and become
    hard to count, while flat ambient keeps every edge legible.  Beauty renders
    are a separate job (render_model / render_previews).

    Returns (folder, [(view, png_path_or_error), ...]); folder is None when
    nothing was rendered.  The directory is created LAZILY, on the first actual
    render: creating it up front left an empty timestamped folder behind on every
    geometry-only build (``views: []``), which the eval does for each case.
    """
    folder = None
    out = []
    for view in views:
        if view not in R._VIEWS:
            out.append((view, f"unknown view '{view}'"))
            continue
        if folder is None:
            folder = os.path.join(settings.render.output_dir,
                                  "reconstruct_" + time.strftime("%Y%m%d_%H%M%S"))
            os.makedirs(folder, exist_ok=True)
        png = os.path.join(folder, f"{view}.png")
        err = R.render(R.check_spec(region, view, size, _CHECK_LIGHTING), png)
        out.append((view, err or png))
    return folder, out


# --- spec history (per region) -------------------------------------------
# Every accepted build/edit saves its full spec as a new version, so a model is
# an editable, revertable document -- not something you must fully re-specify.

def _specs_root() -> str:
    """Where saved build specs live.

    Defaults under the render folder for backwards compatibility -- moving it
    unconditionally would orphan an existing store -- but ``BRLCAD_SPEC_DIR``
    takes it elsewhere.  Worth setting: the render folder is a cache, and a saved
    spec is the only record of what a build actually was.
    """
    return settings.render.spec_dir or os.path.join(
        settings.render.output_dir, "specs")


def _name_dir(name: str) -> str:
    return os.path.join(_specs_root(), name)


def _versions(name: str) -> list[str]:
    """Saved spec files for *name*, oldest first (v001.json, v002.json, ...)."""
    d = _name_dir(name)
    if not os.path.isdir(d):
        return []
    files = sorted(f for f in os.listdir(d)
                   if f.startswith("v") and f.endswith(".json"))
    return [os.path.join(d, f) for f in files]


def _load_spec(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def _latest_spec(name: str) -> dict | None:
    v = _versions(name)
    return _load_spec(v[-1]) if v else None


def _save_spec(name: str, spec_dict: dict) -> str:
    d = _name_dir(name)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"v{len(_versions(name)) + 1:03d}.json")
    with open(path, "w") as fh:
        json.dump(spec_dict, fh, indent=2)
    return path


def _apply_edits(parts: list[dict], edits: list[dict]):
    """Apply edit ops to a list of part dicts.  Returns (new_parts, errors).

    Each edit's ``action`` is move / update / add / remove.  NOTE the action key
    is ``action``, NOT ``op`` -- a part's ``op`` field is its boolean role
    (add/subtract), so keeping the action under a separate key avoids clobbering
    it.  Pure -- no I/O -- so it is easy to test.
    """
    parts = [dict(p) for p in parts]
    by_name = {p.get("name"): p for p in parts}
    errors: list[str] = []
    for e in edits:
        action = e.get("action")
        if action == "add":
            part = e.get("part")
            if not isinstance(part, dict) or "name" not in part:
                errors.append("add: needs a 'part' object with a name")
            elif part["name"] in by_name:
                errors.append(f"add: part '{part['name']}' already exists")
            else:
                parts.append(dict(part))
                by_name[part["name"]] = parts[-1]
        elif action == "remove":
            nm = e.get("name")
            if nm not in by_name:
                errors.append(f"remove: no part '{nm}'")
            else:
                parts = [p for p in parts if p.get("name") != nm]
                del by_name[nm]
        elif action == "move":
            p, delta = by_name.get(e.get("name")), e.get("delta")
            if p is None:
                errors.append(f"move: no part '{e.get('name')}'")
            elif not (isinstance(delta, list) and len(delta) == 3):
                errors.append(f"move: '{e.get('name')}' needs delta [dx,dy,dz]")
            else:
                c = p.get("center", [0, 0, 0])
                p["center"] = [c[0] + delta[0], c[1] + delta[1], c[2] + delta[2]]
        elif action == "update":
            p = by_name.get(e.get("name"))
            if p is None:
                errors.append(f"update: no part '{e.get('name')}'")
            else:
                # 'op' here is the PART's boolean role (add/subtract), safe to
                # set because the edit action lives under 'action'.  'hole' is
                # updatable too: it is the through/pocket intent assertion the
                # verifier leans on, so a wrong annotation had to be fixable
                # without re-specifying the whole model.
                for k in ("center", "size", "height", "radius", "op", "shape",
                          "hole"):
                    if k in e:
                        p[k] = e[k]
        else:
            errors.append(f"unknown action '{action}' (use move/update/add/remove)")
    return parts, errors


def _report(parsed: BuildSpec, folder: str | None, results, header: str) -> str:
    lines = [f"{header} from {len(parsed.parts)} part(s):"]
    for p in parsed.parts:
        lines.append(f"  {'+' if p.op == 'add' else '-'} {p.name} ({p.shape})")
    if not parsed.expect_bbox:
        # The strongest size guard is opt-in, so say when it did not run rather
        # than letting a silent absence read as a clean check.
        lines.append("Note: expect_bbox was not declared, so the overall size "
                     "was not checked against an intended value. Set it when the "
                     "request states a size or a coordinate range.")
    if parsed.lighting != _CHECK_LIGHTING:
        # Say it plainly rather than letting the caller believe it took effect:
        # the tool's own example used to show "lighting": "studio", so a model
        # doing exactly what we documented got a silent no-op.
        lines.append(f"Note: 'lighting' was set to '{parsed.lighting}' and "
                     f"IGNORED -- check views are always rendered with flat "
                     f"{_CHECK_LIGHTING} light so repeated features stay "
                     f"countable. Use render_model for a lit render.")
    if folder:
        lines.append(f"Check renders in:\n  {folder}")
        for view, res in results:
            ok = (isinstance(res, str) and res.endswith(".png")
                  and os.path.exists(res))
            lines.append(f"  {view}: {res}" if ok else f"  {view}: FAILED - {res}")
        lines.append("These check renders will be shown back to you as images to "
                     "compare with the reference. To change the model, use "
                     "edit_build (a small list of ops) -- do NOT re-specify it; to "
                     "revert, use undo_build.")
    else:
        # No views requested: say so rather than pointing at a folder that was
        # deliberately never created.
        lines.append("No check views were requested, so nothing was rendered.")
        for view, res in results:
            lines.append(f"  {view}: FAILED - {res}")
    return "\n".join(lines)


@mcp.tool()
def build_from_spec(
    spec: str | dict = Field(
        ...,
        description=(
            "A JSON model spec. Shape:\n"
            '{\n'
            '  "name": "phone_case",\n'
            '  "color": [40, 40, 45],            // optional RGB 0-255\n'
            '  "views": ["front", "side"],       // check views to render\n'
            '  "render_size": 256,\n'
            '  "parts": [\n'
            '    {"name": "body", "shape": "box", "op": "add",\n'
            '     "center": [0,0,0], "size": [72,147,9]},\n'
            '    {"name": "hollow", "shape": "box", "op": "subtract",\n'
            '     "center": [0,0,1], "size": [68,143,9]},\n'
            '    {"name": "cam", "shape": "cylinder", "op": "subtract",\n'
            '     "center": [-20,55,0], "height": [0,0,10], "radius": 8}\n'
            '  ]\n'
            '}\n'
            "Shapes: box (size+center), cylinder (center=base, height vector, "
            "radius), sphere (center, radius), wedge (size+center plus "
            "top_size [x,y] -- a box whose top face has its own footprint: "
            "tapered flanks, chamfers, and gussets via top_size [x,0]), cone "
            "(center=base, height vector, radius, top_radius -- tapered "
            "bosses, countersinks, turned profiles; both radii must be > 0). "
            "op is 'add' or 'subtract'; the first part must be 'add'. All "
            "units are mm."
        ),
    ),
) -> str:
    """Build a parametric model from a JSON spec, then render check views.

    Deterministic: it validates the spec, creates the primitives and one region
    over the socket (rebuilding cleanly if the names already exist), applies an
    optional colour, and renders the requested views as small stamps so the
    result can be compared to a reference.  Use this for the BUILD + CHECK stage
    of modelling from a reference image: the agent decides the spec (from the
    image), this tool executes it reproducibly.  Returns the region name, what
    was built, and the render paths; the check renders are then shown back to
    you as images to compare against the reference.
    """
    data, err = parse_json_arg(spec, "spec")
    if err:
        return err
    try:
        parsed = BuildSpec.model_validate(data)
    except ValidationError as exc:
        return f"Error: spec does not match the schema.\n{exc}"

    errors = _validate(parsed)
    if errors:
        return "Error: invalid spec:\n" + "\n".join(f"  - {e}" for e in errors)

    # Guardrail-as-tooling: never clobber geometry we did not build.
    collision = _collision_error(
        parsed, _live_names(),
        parse_region_members(parse_response(send_command(f"l {parsed.name}.r"))))
    if collision:
        return f"Error: {collision}"

    build_err = _build_geometry(parsed)
    if build_err:
        return f"Error: {build_err}"

    _save_spec(parsed.name, parsed.model_dump())
    folder, results = _render_checks(
        parsed.name + ".r", parsed.views, parsed.render_size)
    return _report(parsed, folder, results, f"Built region '{parsed.name}.r'")


@mcp.tool()
def edit_build(
    name: str = Field(
        ...,
        description="Region name of an existing build to edit (e.g. 'lego_brick').",
    ),
    edits: str | list = Field(
        ...,
        description=(
            "A JSON list of edit ops applied to the CURRENT build -- send only "
            "the change, not the whole model. Each op's key is 'action' (NOT "
            "'op'; a part's own 'op' is its boolean role add/subtract):\n"
            '[{"action": "move", "name": "stud3", "delta": [0, -1, 0]},\n'
            ' {"action": "update", "name": "body", "size": [32, 16, 9.6]},\n'
            ' {"action": "add", "part": {"name": "stud7", "shape": "cylinder",\n'
            '    "op": "add", "center": [8,0,9.6], "height": [0,0,1.7], "radius": 2.4}},\n'
            ' {"action": "remove", "name": "stud6"}]\n'
            "Actions: move (relative delta on center), update (set any of center "
            "/ size / height / radius / op / shape), add (a new part), remove. "
            "Units mm."
        ),
    ),
) -> str:
    """Edit an existing build incrementally, WITHOUT re-specifying the model.

    Loads the current saved spec for <name>, applies the edit ops, regenerates
    the geometry and check renders, and saves the result as a new version (so it
    can be undone).  Use this for ANY change to an existing model -- move /
    resize / add / remove a part -- instead of rebuilding the whole thing with
    build_from_spec.  That way a small fix never loses the rest of the model.
    """
    current = _latest_spec(name)
    if current is None:
        return (f"Error: no saved build named '{name}'. Build it first with "
                f"build_from_spec.")
    ops, err = parse_json_arg(edits, "edits")
    if err:
        return err
    if not isinstance(ops, list):
        return "Error: edits must be a JSON list of ops."

    new_parts, edit_errors = _apply_edits(current.get("parts", []), ops)
    if edit_errors:
        return ("Error: could not apply edits:\n"
                + "\n".join(f"  - {e}" for e in edit_errors))
    current["parts"] = new_parts
    try:
        parsed = BuildSpec.model_validate(current)
    except ValidationError as exc:
        return f"Error: edited spec is invalid.\n{exc}"
    errors = _validate(parsed)
    if errors:
        return ("Error: edited spec is invalid:\n"
                + "\n".join(f"  - {e}" for e in errors))

    build_err = _build_geometry(parsed)
    if build_err:
        return f"Error: {build_err}"

    _save_spec(parsed.name, parsed.model_dump())
    folder, results = _render_checks(
        parsed.name + ".r", parsed.views, parsed.render_size)
    return _report(parsed, folder, results,
                   f"Edited '{parsed.name}.r' ({len(ops)} op(s))")


@mcp.tool()
def undo_build(
    name: str = Field(
        ..., description="Region name to revert to its previous saved version."),
) -> str:
    """Undo the last change to a build: revert to the previous version and
    rebuild it.  Repeated calls step further back through the history."""
    versions = _versions(name)
    if len(versions) < 2:
        return (f"Error: nothing to undo for '{name}' "
                f"({'no builds found' if not versions else 'only one version'}).")
    os.remove(versions[-1])  # drop the current version; previous becomes current
    parsed = BuildSpec.model_validate(_load_spec(versions[-2]))
    build_err = _build_geometry(parsed)
    if build_err:
        return f"Error: {build_err}"
    folder, results = _render_checks(
        parsed.name + ".r", parsed.views, parsed.render_size)
    return _report(parsed, folder, results,
                   f"Reverted '{parsed.name}.r' to the previous version")


@mcp.tool()
def list_builds(
    name: str = Field(..., description="Region name to list saved versions for."),
    show_spec: bool = Field(
        default=False,
        description=(
            "If true, also return the FULL JSON spec of the current version -- "
            "every part with its centre, size, radius and boolean role. Use this "
            "before edit_build when you do not already have the spec in front of "
            "you, so an edit can target real current values instead of guesses."
        ),
    ),
) -> str:
    """List the saved versions of a build, and optionally read the current spec.

    The saved spec is the source of truth for a build -- edit_build applies ops to
    it and regenerates the geometry -- so being able to READ it back matters: the
    part names are in the build report, but the current coordinates are not, and
    without them a revision after the spec has left the agent's context can only
    guess or re-specify the whole model.
    """
    versions = _versions(name)
    if not versions:
        return f"No saved builds named '{name}'."
    lines = [f"Saved versions of '{name}' (oldest first, last = current):"]
    for i, path in enumerate(versions, 1):
        try:
            n = len(_load_spec(path).get("parts", []))
        except (OSError, ValueError):
            n = "?"
        lines.append(f"  v{i:03d}: {n} part(s)")
    if show_spec:
        try:
            current = _load_spec(versions[-1])
        except (OSError, ValueError) as exc:
            lines.append(f"(could not read the current spec: {exc})")
        else:
            lines.append(f"\nCurrent spec (v{len(versions):03d}):")
            lines.append(json.dumps(current, indent=2))
            lines.append("Change it with edit_build (send only the ops, not this "
                         "whole spec).")
    return "\n".join(lines)
