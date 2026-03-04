"""Server-side tool registry. Import sub-modules to register tools with MCP."""

# The act of importing these modules registers the @mcp.tool() decorated
# functions with the FastMCP instance created in app.py.

from brlcad_mcp.server.tools import boolean as boolean  # noqa: F401
from brlcad_mcp.server.tools import discovery as discovery  # noqa: F401
from brlcad_mcp.server.tools import primitives as primitives  # noqa: F401
