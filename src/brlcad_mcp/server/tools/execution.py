"""MCP tools — generic command execution and error-recovery retry loop.

Tools
-----
- ``execute_command``        — run an arbitrary MGED command.
- ``analyze_command_error``  — diagnose a failure and retry with a corrected command.
"""

from __future__ import annotations

import logging

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import (
    BLOCKED_COMMANDS,
    MAX_RETRY_ATTEMPTS,
    check_mged_result,
    is_error_response,
    parse_response,
)
from brlcad_mcp.transport import send_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool — execute_command
# ---------------------------------------------------------------------------

@mcp.tool()
def execute_command(
    command: str = Field(
        ...,
        description=(
            "The full MGED command string to execute, "
            "e.g. 'in myshape.s tor 0 0 0 0 0 1 5 1' or 'attr set obj.r color 255/0/0'."
        ),
    ),
    auto_draw: bool = Field(
        default=False,
        description=(
            "If true, automatically call 'draw' and 'autoview' after the "
            "command to display the result. Use this when the command creates "
            "or modifies visible geometry."
        ),
    ),
    object_name: str | None = Field(
        default=None,
        description=(
            "Name of the object to draw after execution (only used when "
            "auto_draw is true). If omitted, auto_draw is skipped."
        ),
    ),
) -> str:
    """Execute an arbitrary MGED command that isn't covered by a dedicated tool.

    **Workflow:** call ``list_commands`` to find a relevant command, then
    ``get_command_help`` to learn its syntax, then this tool to run it.

    The command is sent directly to BRL-CAD's libmcpcad listener over the
    socket bridge.  A small set of destructive commands (quit, exit) are
    blocked for safety.

    If MGED returns an error, the response will be clearly tagged with
    ``[MGED_ERROR]`` so you can pass it to ``analyze_command_error`` for
    diagnosis and automatic retry.
    """
    # --- safety gate ---
    first_token = command.strip().split()[0] if command.strip() else ""
    if first_token.lower() in BLOCKED_COMMANDS:
        return f"Error: the command '{first_token}' is blocked for safety."

    logger.info("execute_command → %s", command)

    try:
        result = send_command(command)
    except (ConnectionError, TimeoutError) as exc:
        return f"Error: {exc}"

    # --- error detection (SUCCESS:/ERROR: prefix from the transport layer) ---
    error = check_mged_result(result, command=command)
    if error:
        return error

    payload = parse_response(result)

    if auto_draw and object_name:
        try:
            send_command(f"draw {object_name}")
            send_command("autoview")
        except (ConnectionError, TimeoutError):
            payload += " (Warning: auto-draw failed)"

    if not payload:
        payload = "(completed successfully — no output)"

    return f"Command: {command}\nResult: {payload}"


# ---------------------------------------------------------------------------
# Tool — analyze_command_error
# ---------------------------------------------------------------------------

@mcp.tool()
def analyze_command_error(
    failed_command: str = Field(
        ...,
        description="The exact MGED command string that failed.",
    ),
    error_output: str = Field(
        ...,
        description="The error message returned by MGED.",
    ),
    intent: str = Field(
        ...,
        description=(
            "A natural-language description of what the command was trying "
            "to achieve, so the analysis can suggest a correct alternative."
        ),
    ),
    corrected_command: str = Field(
        ...,
        description=(
            "Your best attempt at a corrected MGED command string based on "
            "the error output and help text. This will be executed if it "
            "passes validation."
        ),
    ),
    attempt: int = Field(
        default=1,
        description=(
            "Current retry attempt number (starts at 1). The tool refuses "
            "to execute if attempt > 5 to prevent infinite loops."
        ),
    ),
    auto_draw: bool = Field(
        default=False,
        description=(
            "If true, automatically call 'draw' and 'autoview' after the "
            "corrected command succeeds."
        ),
    ),
    object_name: str | None = Field(
        default=None,
        description=(
            "Name of the object to draw after execution (only used when "
            "auto_draw is true)."
        ),
    ),
) -> str:
    """Diagnose a failed MGED command, then execute a corrected version.

    Use this tool when ``execute_command`` returns a ``[MGED_ERROR]``.
    It performs the following steps:

    1. Extracts the base command name and fetches its ``help`` text from MGED.
    2. Checks whether referenced objects exist in the database.
    3. Executes the ``corrected_command`` you provide.
    4. If the corrected command also fails, returns a diagnostic so you can
       call this tool again with ``attempt + 1``.

    **Hard limit: 5 attempts.** After that the tool refuses to retry and
    returns all accumulated diagnostics for you to report to the user.
    """
    if attempt > MAX_RETRY_ATTEMPTS:
        return (
            f"[RETRY_LIMIT] Reached maximum of {MAX_RETRY_ATTEMPTS} attempts.\n"
            f"Last failed command: {failed_command}\n"
            f"Last error: {error_output}\n"
            f"Intent: {intent}\n\n"
            f"Please inform the user that this operation could not be "
            f"completed automatically and provide the diagnostics above."
        )

    diagnostics: list[str] = [
        f"=== Error Analysis (attempt {attempt}/{MAX_RETRY_ATTEMPTS}) ===",
        f"Failed command : {failed_command}",
        f"Error output   : {error_output}",
        f"Intent         : {intent}",
    ]

    # --- Step 1: fetch help for the base command ---
    base_cmd = failed_command.strip().split()[0] if failed_command.strip() else ""
    if base_cmd:
        try:
            help_text = send_command(f"help {base_cmd}")
            diagnostics.append(f"\nHelp for '{base_cmd}':\n{help_text}")
        except (ConnectionError, TimeoutError):
            diagnostics.append(f"\n(Could not fetch help for '{base_cmd}')")

    # --- Step 2: check if referenced objects exist ---
    tokens = failed_command.strip().split()
    obj_candidates = [
        t for t in tokens[1:]
        if any(t.endswith(ext) for ext in (".s", ".r", ".c", ".g"))
    ]
    if obj_candidates:
        for obj in obj_candidates:
            try:
                check = send_command(f"l {obj}")
                exists = not is_error_response(check)
                diagnostics.append(
                    f"  Object '{obj}': {'EXISTS' if exists else 'NOT FOUND'}"
                )
            except (ConnectionError, TimeoutError):
                diagnostics.append(f"  Object '{obj}': (could not verify)")

    # --- Step 3: safety-check the corrected command ---
    corrected_first = (
        corrected_command.strip().split()[0] if corrected_command.strip() else ""
    )
    if corrected_first.lower() in BLOCKED_COMMANDS:
        diagnostics.append(
            f"\n[BLOCKED] '{corrected_first}' is not allowed."
        )
        return "\n".join(diagnostics)

    # --- Step 4: execute the corrected command ---
    diagnostics.append(f"\nRetrying with: {corrected_command}")
    logger.info(
        "analyze_command_error attempt %d → %s", attempt, corrected_command
    )

    try:
        result = send_command(corrected_command)
    except (ConnectionError, TimeoutError) as exc:
        diagnostics.append(f"Connection error on retry: {exc}")
        return "\n".join(diagnostics)

    if is_error_response(result):
        diagnostics.append(
            f"Corrected command ALSO FAILED: {parse_response(result)}"
        )
        diagnostics.append(
            f"\nYou may call analyze_command_error again with "
            f"attempt={attempt + 1}, the new error output, and a "
            f"revised corrected_command."
        )
        return "\n".join(diagnostics)

    # --- Success! ---
    payload = parse_response(result)

    if auto_draw and object_name:
        try:
            send_command(f"draw {object_name}")
            send_command("autoview")
        except (ConnectionError, TimeoutError):
            payload += " (Warning: auto-draw failed)"

    diagnostics.append(f"SUCCESS on attempt {attempt}: {payload}")
    return "\n".join(diagnostics)
