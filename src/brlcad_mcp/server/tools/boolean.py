"""MCP tool definitions — CSG boolean operations."""

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import (
    check_mged_result,
    parse_region_members,
    parse_response,
    region_fold_cmds,
)
from brlcad_mcp.transport import send_command

_VALID_OPERATORS = {"u", "-", "+"}


def _append_cmds(region: str, operator: str, target: str) -> list[str]:
    """Commands appending ``operator target`` to an EXISTING region, correctly.

    ``r region - target`` looks like it subtracts from the whole region, but
    BRL-CAD binds the operator to only the most recently unioned member -- so on
    a region already holding several unioned solids the cut would apply to just
    the last one and silently miss the rest.  (Same gotcha that made a two-plate
    bracket lose one of its holes.)

    A union is unambiguous, and so is a region with a single member, so those
    keep the cheap one-command path.  Otherwise we re-fold the region's existing
    members plus the new one left to right, which makes the new operator apply
    to everything accumulated so far.
    """
    if operator == "u":
        return [f"r {region} {operator} {target}"]
    listing = parse_response(send_command(f"l {region}"))
    members = parse_region_members(listing)
    if len(members) < 2:
        return [f"r {region} {operator} {target}"]

    # Re-fold the existing members plus the new one.  `r` APPENDS to a region
    # that already exists rather than redefining it, so the old members must be
    # killed first -- otherwise they stay unioned at the top level alongside the
    # accumulator and put the removed material straight back.
    prefix = f"{region}.app"
    fold = region_fold_cmds(region, [*members, (operator, target)],
                            accum_prefix=prefix)
    combs = [c for c in fold if c.startswith("comb ")]
    define = [c for c in fold if c.startswith("r ")]
    cmds = [*combs, f"kill {region}", *define]

    colour = _region_colour(listing)
    if colour:                       # killing the region drops its colour
        cmds.append(f"comb_color {region} {colour}")
    return cmds


def _region_colour(l_output: str) -> str | None:
    """The ``Color R G B`` values from an ``l`` listing, space separated."""
    for line in l_output.splitlines():
        parts = line.split()
        if len(parts) == 4 and parts[0] == "Color":
            return " ".join(parts[1:])
    return None


@mcp.tool()
def boolean_combination(
    output_name: str = Field(
        ...,
        description=(
            "Name of the region for the result. "
            "To modify an existing region (e.g., subtract another object from it), "
            "pass the SAME name as base_object. "
            "To create a brand-new region, use a new name ending in '.r'."
        ),
    ),
    base_object: str = Field(..., description="The main object to start with"),
    operator: str = Field(
        ...,
        description="Must be 'u' (union), '-' (subtract), or '+' (intersect)",
    ),
    target_object: str = Field(
        ...,
        description="The object being added, subtracted, or intersected",
    ),
) -> str:
    """Performs Constructive Solid Geometry (CSG) boolean math on two objects.

    Creates a region (not just a combination) so the result is visible in raytrace.
    When output_name equals base_object, the operation is appended to the existing
    region instead of nesting it, which avoids overlap issues in raytrace.

    OVERLAP RESOLUTION GUARD: if you are calling this to resolve an overlap
    between two regions and the user has not explicitly chosen the subtract
    strategy (as opposed to moving a part), STOP - do not call this tool.
    Ask the user whether to subtract or move first.
    """
    if operator not in _VALID_OPERATORS:
        return f"Error: operator must be one of {_VALID_OPERATORS}, got '{operator}'."

    if output_name == base_object:
        cmds = _append_cmds(output_name, operator, target_object)
    else:
        cmds = [f"r {output_name} u {base_object} {operator} {target_object}"]

    for cmd in cmds:
        result = send_command(cmd)
        error = check_mged_result(result, command=cmd)
        if error:
            return error

    # Hide the individual pieces and show only the region
    if output_name != base_object:
        send_command(f"erase {base_object}")
    send_command(f"erase {target_object}")
    send_command(f"erase {output_name}")
    send_command(f"draw {output_name}")
    send_command("autoview")

    return (
        f"CSG result: {output_name} = {base_object} {operator} {target_object}. "
        f"Output: {parse_response(result)}"
    )
