"""MCP tools — command discovery and help lookup.

Tools
-----
- ``list_commands``    — browse all available MGED commands with descriptions.
- ``get_command_help`` — fetch the man-page / usage for a specific command.
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.catalog import (
    ALL_KNOWN,
    COMMAND_CATALOG,
    categories_text,
)
from brlcad_mcp.transport import send_command


# ---------------------------------------------------------------------------
# Tool — list_commands
# ---------------------------------------------------------------------------

@mcp.tool()
def list_commands(
    category: Optional[str] = Field(
        default=None,
        description=(
            "Optional category filter. One of: "
            + ", ".join(COMMAND_CATALOG.keys())
            + ". Leave empty to list ALL commands."
        ),
    ),
    query_live: bool = Field(
        default=False,
        description=(
            "If true, also query MGED's '?' command at runtime and append "
            "any commands not already in the static catalog."
        ),
    ),
) -> str:
    """List available BRL-CAD / MGED commands with short descriptions.

    Use this when you need a geometry operation but none of the dedicated
    tools (create_sphere, create_box, etc.) cover it.  Browse the catalog,
    pick a command, then call ``get_command_help`` to learn its full syntax
    before executing it with ``execute_command``.
    """
    result = categories_text(category)

    if query_live:
        try:
            raw = send_command("?")
            # MGED's '?' returns a space/newline-separated list of command names.
            live_cmds = set(raw.replace("SUCCESS:", "").split())
            new_cmds = sorted(live_cmds - ALL_KNOWN - {"?"})
            if new_cmds:
                result += (
                    "\n\n=== ADDITIONAL COMMANDS (discovered live) ===\n  "
                    + "\n  ".join(new_cmds)
                )
        except (ConnectionError, TimeoutError) as exc:
            result += f"\n\n(Could not query MGED live: {exc})"

    return result


# ---------------------------------------------------------------------------
# Tool — get_command_help
# ---------------------------------------------------------------------------

@mcp.tool()
def get_command_help(
    command: str = Field(
        ...,
        description="The MGED command name to look up (e.g. 'in', 'rt', 'attr').",
    ),
) -> str:
    """Fetch detailed help / man-page text for an MGED command.

    Queries MGED's built-in ``help <command>`` to get the full usage
    description, argument syntax, and examples.  Call this after finding a
    promising command via ``list_commands``, before using ``execute_command``.
    """
    # Provide static description if available.
    static_desc = None
    for cat_cmds in COMMAND_CATALOG.values():
        if command in cat_cmds:
            static_desc = cat_cmds[command]
            break

    try:
        raw_help = send_command(f"help {command}")
    except (ConnectionError, TimeoutError) as exc:
        if static_desc:
            return (
                f"[offline] {command} — {static_desc}\n"
                f"(Could not reach MGED for full help: {exc})"
            )
        return f"Error: could not reach MGED and no static help for '{command}'."

    parts: list[str] = []
    if static_desc:
        parts.append(f"Summary: {static_desc}")
    parts.append(f"MGED help output:\n{raw_help}")
    return "\n\n".join(parts)
