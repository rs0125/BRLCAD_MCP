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


def _region_expr(parts: list[Part]) -> str:
    """Boolean expression for the region: 'u a.s - b.s u c.s' (add=u, sub=-)."""
    toks = []
    for p in parts:
        toks.append(f"{'u' if p.op == 'add' else '-'} {p.name}.s")
    return " ".join(toks)


def _build_geometry(spec: BuildSpec) -> str | None:
    """Create the solids and the region over the socket.  None on success."""
    send_command("units mm")
    # Idempotent rebuild: drop the region (and its refs) and the solids first.
    send_command(f"killall {spec.name}.r")
    for p in spec.parts:
        send_command(f"killall {p.name}.s")
    for p in spec.parts:
        resp = send_command(_solid_cmd(p))
        if is_error_response(resp):
            return f"failed to create '{p.name}' ({p.shape}): {parse_response(resp)}"
    resp = send_command(f"r {spec.name}.r {_region_expr(spec.parts)}")
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

    folder, results = _render_checks(
        parsed.name + ".r", parsed.views, parsed.render_size, parsed.lighting)

    lines = [f"Built region '{parsed.name}.r' from {len(parsed.parts)} part(s):"]
    for p in parsed.parts:
        lines.append(f"  {'+' if p.op == 'add' else '-'} {p.name} ({p.shape})")
    lines.append(f"Check renders in:\n  {folder}")
    for view, res in results:
        ok = res.endswith(".png") and os.path.exists(res)
        lines.append(f"  {view}: {res}" if ok else f"  {view}: FAILED - {res}")
    lines.append("These check renders will be shown back to you as images to "
                 "compare with the reference; if the shape is off, call "
                 "build_from_spec again with a corrected spec.")
    return "\n".join(lines)
