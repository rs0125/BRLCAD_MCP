"""Shared constants and helpers for the tool modules.

All error detection relies on the ``SUCCESS: `` / ``ERROR: `` prefix.  The
transport layer (:mod:`brlcad_mcp.transport.socket_bridge`) produces these
by translating the libmcpcad listener's ``OK`` / ``ERR <code>`` frame status
into the prefix the tools below key off of.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

# Commands that must never be executed through the generic tools.
BLOCKED_COMMANDS: set[str] = {
    "quit",
    "exit",
    "q",
}

# Maximum number of retry attempts for analyze_command_error.
MAX_RETRY_ATTEMPTS = 5

# ---------------------------------------------------------------------------
# Response parsing — keyed off the Tcl listener's prefixes
# ---------------------------------------------------------------------------

_SUCCESS_PREFIX = "SUCCESS:"
_ERROR_PREFIX = "ERROR:"


def is_error_response(response: str) -> bool:
    """Return True if the MGED response starts with the ``ERROR:`` prefix."""
    return response.startswith(_ERROR_PREFIX)


def parse_response(response: str) -> str:
    """Strip the ``SUCCESS:`` or ``ERROR:`` prefix and return the payload."""
    if response.startswith(_SUCCESS_PREFIX):
        return response[len(_SUCCESS_PREFIX):].strip()
    if response.startswith(_ERROR_PREFIX):
        return response[len(_ERROR_PREFIX):].strip()
    # No recognised prefix — return as-is.
    return response.strip()


# ---------------------------------------------------------------------------
# Shared error handler — used by every tool that calls send_command
# ---------------------------------------------------------------------------

def check_mged_result(response: str, *, command: str) -> str | None:
    """If *response* is an MGED error, return a formatted error string.

    Returns ``None`` when the response indicates success, meaning the
    caller can proceed normally.  When it returns a non-None string the
    caller should return that string directly to the agent.
    """
    if not is_error_response(response):
        return None
    payload = parse_response(response)
    return (
        f"[MGED_ERROR] Command failed.\n"
        f"Command: {command}\n"
        f"Error output: {payload}\n\n"
        f"Tip: call analyze_command_error with this information to "
        f"diagnose the issue and retry with a corrected command."
    )
