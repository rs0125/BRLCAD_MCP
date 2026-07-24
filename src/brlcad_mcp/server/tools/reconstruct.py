"""MCP tool -- build a parametric model from a structured spec, then render it.

This is the DETERMINISTIC half of the image-to-model workflow.  The agent (which
can see a reference image) decides WHAT to build and fills in a spec; this tool
validates that spec, BUILDS it as BRL-CAD CSG over the socket, and RENDERS check
views so the result can be compared against the reference.  Same spec in -> same
geometry out, every time.  Re-running with the same names rebuilds cleanly
(existing objects are removed first), which is what the adjust loop needs.

Spec vocabulary (v1): box / cylinder / sphere primitives, unioned or subtracted
into a single region, with an optional overall colour.  Rounded-profile bodies
via sketch/extrude are a planned extension.
"""

from __future__ import annotations

import json
import os
import time

from pydantic import BaseModel, Field, ValidationError

from brlcad_mcp.config import settings
from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools import rendering as R
from brlcad_mcp.server.tools.helpers import is_error_response, parse_response
from brlcad_mcp.transport import send_command

_SHAPES = ("box", "cylinder", "sphere")
_OPS = ("add", "subtract")


class Part(BaseModel):
    """One primitive in the model.  Which fields are required depends on shape:

    box:      size [x,y,z] and center [x,y,z]
    cylinder: center [x,y,z] (base), height [x,y,z] (axis vector), radius
    sphere:   center [x,y,z], radius
    """

    name: str
    shape: str
    op: str = "add"
    center: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    size: list[float] | None = None
    height: list[float] | None = None
    radius: float | None = None


class BuildSpec(BaseModel):
    """A whole model: a named region built from parts, plus render settings."""

    name: str = "model"
    parts: list[Part]
    color: list[int] | None = None
    views: list[str] = Field(default_factory=lambda: ["front", "side"])
    render_size: int = 256
    lighting: str = "studio"


def _validate(spec: BuildSpec) -> list[str]:
    """Semantic validation beyond pydantic's type checks.  Returns error lines."""
    errors: list[str] = []
    if not spec.parts:
        return ["spec has no parts"]
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
        elif p.shape == "sphere" and (not p.radius or p.radius <= 0):
            errors.append(f"'{p.name}': sphere needs a positive radius")
    return errors


def _solid_cmd(part: Part) -> str:
    """The MGED ``in`` command that creates this part's solid (<name>.s)."""
    n = f"{part.name}.s"
    cx, cy, cz = part.center
    if part.shape == "box":
        sx, sy, sz = part.size  # type: ignore[misc]
        return (f"in {n} rpp {cx - sx / 2:g} {cx + sx / 2:g} "
                f"{cy - sy / 2:g} {cy + sy / 2:g} "
                f"{cz - sz / 2:g} {cz + sz / 2:g}")
    if part.shape == "cylinder":
        hx, hy, hz = part.height  # type: ignore[misc]
        return f"in {n} rcc {cx:g} {cy:g} {cz:g} {hx:g} {hy:g} {hz:g} {part.radius:g}"
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
    solids = [f"{p.name}.s" for p in parts]
    if len(parts) == 1:
        return [f"r {name}.r u {solids[0]}"]
    cmds: list[str] = []
    acc = solids[0]
    for i in range(1, len(parts) - 1):
        step = _accum_name(name, i)
        cmds.append(f"comb {step} u {acc} {_op_char(parts[i])} {solids[i]}")
        acc = step
    cmds.append(f"r {name}.r u {acc} {_op_char(parts[-1])} {solids[-1]}")
    return cmds


def _build_geometry(spec: BuildSpec) -> str | None:
    """Create the solids and the region over the socket.  None on success."""
    send_command("units mm")
    # Idempotent rebuild: killtree drops the region plus its intermediate
    # accumulation combs, then killall clears the solids (and any stray accN
    # from a build whose part count later shrank).  Errors here are ignored --
    # nothing to remove on a first build.
    send_command(f"killtree {spec.name}.r")
    for p in spec.parts:
        send_command(f"killall {p.name}.s")
    for i in range(1, len(spec.parts)):
        send_command(f"killall {_accum_name(spec.name, i)}")
    for p in spec.parts:
        resp = send_command(_solid_cmd(p))
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


def _render_checks(region: str, views: list[str], size: int, lighting: str):
    """Render each check view into a fresh timestamped folder.

    Returns (folder, [(view, png_path_or_error), ...]).
    """
    folder = os.path.join(settings.render.output_dir,
                          "reconstruct_" + time.strftime("%Y%m%d_%H%M%S"))
    os.makedirs(folder, exist_ok=True)
    out = []
    for view in views:
        if view not in R._VIEWS:
            out.append((view, f"unknown view '{view}'"))
            continue
        az, el = R._VIEWS[view]
        png = os.path.join(folder, f"{view}.png")
        err = R._dispatch_render(region, az, el, size,
                                 R._auto_ambient(lighting), 0, "draft",
                                 lighting, png)
        out.append((view, err or png))
    return folder, out


# --- spec history (per region) -------------------------------------------
# Every accepted build/edit saves its full spec as a new version, so a model is
# an editable, revertable document -- not something you must fully re-specify.

def _specs_root() -> str:
    return os.path.join(settings.render.output_dir, "specs")


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
                # set because the edit action lives under 'action'.
                for k in ("center", "size", "height", "radius", "op", "shape"):
                    if k in e:
                        p[k] = e[k]
        else:
            errors.append(f"unknown action '{action}' (use move/update/add/remove)")
    return parts, errors


def _report(parsed: BuildSpec, folder: str, results, header: str) -> str:
    lines = [f"{header} from {len(parsed.parts)} part(s):"]
    for p in parsed.parts:
        lines.append(f"  {'+' if p.op == 'add' else '-'} {p.name} ({p.shape})")
    lines.append(f"Check renders in:\n  {folder}")
    for view, res in results:
        ok = isinstance(res, str) and res.endswith(".png") and os.path.exists(res)
        lines.append(f"  {view}: {res}" if ok else f"  {view}: FAILED - {res}")
    lines.append("These check renders will be shown back to you as images to "
                 "compare with the reference. To change the model, use "
                 "edit_build (a small list of ops) -- do NOT re-specify it; to "
                 "revert, use undo_build.")
    return "\n".join(lines)


@mcp.tool()
def build_from_spec(
    spec: str = Field(
        ...,
        description=(
            "A JSON model spec. Shape:\n"
            '{\n'
            '  "name": "phone_case",\n'
            '  "color": [40, 40, 45],            // optional RGB 0-255\n'
            '  "views": ["front", "side"],       // check views to render\n'
            '  "render_size": 256,\n'
            '  "lighting": "studio",\n'
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
            "radius), sphere (center, radius). op is 'add' or 'subtract'; the "
            "first part must be 'add'. All units are mm."
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
    try:
        data = json.loads(spec)
    except json.JSONDecodeError as exc:
        return f"Error: spec is not valid JSON ({exc})."
    try:
        parsed = BuildSpec.model_validate(data)
    except ValidationError as exc:
        return f"Error: spec does not match the schema.\n{exc}"

    errors = _validate(parsed)
    if errors:
        return "Error: invalid spec:\n" + "\n".join(f"  - {e}" for e in errors)

    build_err = _build_geometry(parsed)
    if build_err:
        return f"Error: {build_err}"

    _save_spec(parsed.name, parsed.model_dump())
    folder, results = _render_checks(
        parsed.name + ".r", parsed.views, parsed.render_size, parsed.lighting)
    return _report(parsed, folder, results, f"Built region '{parsed.name}.r'")


@mcp.tool()
def edit_build(
    name: str = Field(
        ...,
        description="Region name of an existing build to edit (e.g. 'lego_brick').",
    ),
    edits: str = Field(
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
    try:
        ops = json.loads(edits)
    except json.JSONDecodeError as exc:
        return f"Error: edits is not valid JSON ({exc})."
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
        parsed.name + ".r", parsed.views, parsed.render_size, parsed.lighting)
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
        parsed.name + ".r", parsed.views, parsed.render_size, parsed.lighting)
    return _report(parsed, folder, results,
                   f"Reverted '{parsed.name}.r' to the previous version")


@mcp.tool()
def list_builds(
    name: str = Field(..., description="Region name to list saved versions for."),
) -> str:
    """List the saved versions of a build (oldest first), with part counts."""
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
    return "\n".join(lines)
