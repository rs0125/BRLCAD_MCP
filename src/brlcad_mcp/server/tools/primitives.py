"""MCP tool definitions — primitive geometry creation."""

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import check_mged_result, parse_response
from brlcad_mcp.transport import send_command


@mcp.tool()
def create_sphere(
    name: str = Field(..., description="Name of the sphere, e.g., 'ball.s'"),
    x: float = Field(..., description="X coordinate of the center"),
    y: float = Field(..., description="Y coordinate of the center"),
    z: float = Field(..., description="Z coordinate of the center"),
    radius: float = Field(..., description="Radius of the sphere"),
) -> str:
    """Creates a perfect mathematical sphere in BRL-CAD."""
    cmd = f"in {name} sph {x} {y} {z} {radius}"
    result = send_command(cmd)
    error = check_mged_result(result, command=cmd)
    if error:
        return error
    send_command(f"draw {name}")
    send_command("autoview")
    return (
        f"Created sphere '{name}' at ({x}, {y}, {z}) with radius {radius}. "
        f"Output: {parse_response(result)}"
    )


@mcp.tool()
def create_cylinder(
    name: str = Field(..., description="Name of the cylinder, e.g., 'tube.s'"),
    base_x: float = Field(..., description="X coordinate of the base center"),
    base_y: float = Field(..., description="Y coordinate of the base center"),
    base_z: float = Field(..., description="Z coordinate of the base center"),
    height_x: float = Field(..., description="X component of the height vector"),
    height_y: float = Field(..., description="Y component of the height vector"),
    height_z: float = Field(..., description="Z component of the height vector"),
    radius: float = Field(..., description="Radius of the cylinder"),
) -> str:
    """Creates a right circular cylinder (RCC) in BRL-CAD."""
    cmd = f"in {name} rcc {base_x} {base_y} {base_z} {height_x} {height_y} {height_z} {radius}"
    result = send_command(cmd)
    error = check_mged_result(result, command=cmd)
    if error:
        return error
    send_command(f"draw {name}")
    send_command("autoview")
    return f"Created cylinder '{name}'. Output: {parse_response(result)}"


@mcp.tool()
def create_box(
    name: str = Field(..., description="Name of the box, e.g., 'block.s'"),
    x_min: float = Field(..., description="Minimum X coordinate"),
    y_min: float = Field(..., description="Minimum Y coordinate"),
    z_min: float = Field(..., description="Minimum Z coordinate"),
    x_max: float = Field(..., description="Maximum X coordinate"),
    y_max: float = Field(..., description="Maximum Y coordinate"),
    z_max: float = Field(..., description="Maximum Z coordinate"),
) -> str:
    """Creates an axis-aligned rectangular parallelepiped (box) in BRL-CAD."""
    cmd = f"in {name} rpp {x_min} {x_max} {y_min} {y_max} {z_min} {z_max}"
    result = send_command(cmd)
    error = check_mged_result(result, command=cmd)
    if error:
        return error
    send_command(f"draw {name}")
    send_command("autoview")
    return (
        f"Created box '{name}' from ({x_min}, {y_min}, {z_min}) "
        f"to ({x_max}, {y_max}, {z_max}). Output: {parse_response(result)}"
    )
