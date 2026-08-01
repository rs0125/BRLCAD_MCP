"""Shared constants and helpers for the tool modules.

All error detection relies on the ``SUCCESS: `` / ``ERROR: `` prefix.  The
transport layer (:mod:`brlcad_mcp.transport.socket_bridge`) produces these
by translating the libmcpcad listener's ``OK`` / ``ERR <code>`` frame status
into the prefix the tools below key off of.
"""

from __future__ import annotations

import json
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
# Destructive-command detection (for auto-snapshot / restore)
# ---------------------------------------------------------------------------

# Verbs that delete, rename, or redefine EXISTING objects.  Before running one
# of these through execute_command we snapshot the objects it names (those that
# currently exist) so a raw edit that goes wrong can be rolled back.  Pure
# creation verbs (in/put/make) are intentionally NOT here: the name-conflict
# rule already stops the agent from overwriting, and snapshotting every
# primitive creation would be pointless churn.
DESTRUCTIVE_VERBS: set[str] = {
    "kill", "killall", "killtree", "rm",
    "mv", "mvall",
    "r", "c", "comb", "g",
}

# Tokens that appear in these commands but are never object names: boolean/set
# operators and combination flags.
_NON_OBJECT_TOKENS: set[str] = {"u", "-", "+", "n"}


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def destructive_targets(command: str) -> list[str]:
    """Object-name candidates a destructive command would delete/overwrite.

    Returns the de-duplicated, order-preserving list of tokens that could name
    existing objects (skipping the verb, flags, boolean operators, and numeric
    args).  Empty list when *command* is not a destructive verb.  The caller
    still intersects these with the live database before snapshotting, so
    over-inclusion here is harmless.
    """
    parts = command.strip().split()
    if not parts or parts[0].lower() not in DESTRUCTIVE_VERBS:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for tok in parts[1:]:
        if tok.startswith("-") and len(tok) > 1:   # a flag like -f (but not "-")
            continue
        if tok in _NON_OBJECT_TOKENS or _looks_numeric(tok):
            continue
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out

# ---------------------------------------------------------------------------
# Response parsing — keyed off the Tcl listener's prefixes
# ---------------------------------------------------------------------------

_SUCCESS_PREFIX = "SUCCESS:"
_ERROR_PREFIX = "ERROR:"


def parse_json_arg(value, what: str = "argument"):
    """Accept a JSON *string* or an already-decoded object; return (data, error).

    Models pass structured tool arguments either way -- as a JSON string or as a
    real object -- and a str-only annotation turns the latter into a hard
    ToolException that kills the turn.  Tools taking JSON payloads use this so
    both forms work and a bad payload becomes a readable message instead.
    """
    if isinstance(value, (dict, list)):
        return value, None
    if isinstance(value, str):
        try:
            return json.loads(value), None
        except json.JSONDecodeError as exc:
            return None, f"Error: {what} is not valid JSON ({exc})."
    return None, (f"Error: {what} must be a JSON object or a JSON string, "
                  f"got {type(value).__name__}.")


def region_fold_cmds(region: str, members: list[tuple[str, str]],
                     accum_prefix: str | None = None) -> list[str]:
    """Commands building *region* as a STRICT left-to-right CSG accumulation.

    BRL-CAD's ``r`` binds each ``-``/``+`` to only the *most recent union
    operand*, so a flat ``r x u a u b - h`` yields ``a u (b - h)`` and the
    subtraction misses ``a`` entirely.  Folding through intermediate
    combinations (``<accum_prefix>.accN``) makes every operator apply to the
    whole solid accumulated so far.

    *members* is ``[(op, name), ...]`` in order, where op is ``u``, ``-`` or
    ``+``; the first member's op is ignored (it seeds the accumulation).
    *accum_prefix* names the intermediate combs and defaults to *region*;
    callers that already clean up a particular naming pass their own so no
    orphan combs are left behind on a rebuild.
    """
    if not members:
        return []
    prefix = accum_prefix or region
    names = [name for _, name in members]
    if len(members) == 1:
        return [f"r {region} u {names[0]}"]
    cmds: list[str] = []
    acc = names[0]
    for i in range(1, len(members) - 1):
        step = f"{prefix}.acc{i}"
        cmds.append(f"comb {step} u {acc} {members[i][0]} {names[i]}")
        acc = step
    cmds.append(f"r {region} u {acc} {members[-1][0]} {names[-1]}")
    return cmds


def parse_region_members(l_output: str) -> list[tuple[str, str]]:
    """Top-level ``(op, name)`` members from an ``l <region>`` listing.

    ``l`` prints one indented ``<op> <name>`` line per member, e.g.::

        widget.r:  REGION id=1000 ...
           u body.s
           - hole.s
    """
    members: list[tuple[str, str]] = []
    for line in l_output.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in ("u", "-", "+"):
            members.append((parts[0], parts[1]))
    return members


def ls_names(ls_output: str) -> set[str]:
    """Bare object names from an ``ls`` payload, stripped of its decorations.

    ``ls`` marks combinations with a trailing ``/`` and regions with ``/R``
    (e.g. ``plate.r/R``), plus ``@``/``*`` flags.  Anything comparing names
    against the live database must strip these first -- otherwise a lookup for
    ``plate.r`` never matches the listed ``plate.r/R``.
    """
    return {tok.split("/")[0].rstrip("@*")
            for tok in ls_output.split()} - {""}


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
