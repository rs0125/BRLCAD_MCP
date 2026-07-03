"""MCP tool definitions — CSG boolean operations."""

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import check_mged_result, parse_response
from brlcad_mcp.transport import send_command

_VALID_OPERATORS = {"u", "-", "+"}


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
        cmd = f"r {output_name} {operator} {target_object}"
    else:
        cmd = f"r {output_name} u {base_object} {operator} {target_object}"

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
