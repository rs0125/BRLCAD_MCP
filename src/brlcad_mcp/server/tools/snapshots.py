"""Auto-snapshot and restore for raw destructive MGED commands.

A raw ``execute_command`` such as ``kill hole_x.s`` or ``r bracket.r u a.s``
can delete or redefine existing objects with no way back -- unlike the
spec-backed ``edit_build`` / ``undo_build`` flow, which keeps versioned specs.
To make raw edits recoverable we snapshot the objects a destructive command is
about to touch (those that currently exist) into a small ``.g`` backup right
before the command runs.  ``restore_backup`` rolls the last one back.

The snapshot uses ``keep`` (which recursively exports an object and everything
it references) and restore uses ``kill`` + ``dbconcat``.  Restore is exact when
the destructive op removed or renamed objects (the common case); if the op left
clashing sub-objects in place, ``dbconcat`` will report the clash rather than
silently corrupt the scene.
"""

from __future__ import annotations

import datetime
import json
import logging
import os

from pydantic import Field

from brlcad_mcp.config import settings
from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.helpers import (
    destructive_targets,
    is_error_response,
    parse_response,
)
from brlcad_mcp.transport import send_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filesystem layout (pure, socket-free -- unit tested)
# ---------------------------------------------------------------------------

def _backups_root() -> str:
    return os.path.join(settings.render.output_dir, "backups")


def _parse_ls_names(ls_output: str) -> set[str]:
    """Object names from an ``ls`` payload, stripped of MGED decorations.

    ``ls`` may append ``/`` (comb), ``@`` (global), ``*`` etc. and lays names
    out across whitespace/columns.  We only need the bare names.
    """
    names: set[str] = set()
    for tok in ls_output.split():
        names.add(tok.rstrip("/@*").strip())
    names.discard("")
    return names


def _sidecar_path(g_path: str) -> str:
    return g_path[:-2] + ".json" if g_path.endswith(".g") else g_path + ".json"


def _write_manifest(g_path: str, command: str, objects: list[str],
                    stamp: str) -> None:
    with open(_sidecar_path(g_path), "w") as fh:
        json.dump({"backup": g_path, "command": command,
                   "objects": objects, "created": stamp}, fh, indent=2)


def _list_manifests() -> list[dict]:
    """All backup manifests, newest first (backup filenames sort chronologically)."""
    root = _backups_root()
    if not os.path.isdir(root):
        return []
    out: list[dict] = []
    for name in sorted(os.listdir(root), reverse=True):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(root, name)) as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# ---------------------------------------------------------------------------
# Snapshot (called by execute_command before a destructive command runs)
# ---------------------------------------------------------------------------

def maybe_snapshot(command: str) -> str | None:
    """Snapshot the existing objects *command* would destroy, if any.

    Returns a short note to append to the tool output (so the agent/user knows
    a restore point exists), or ``None`` when nothing needed snapshotting.
    Never raises -- a snapshot failure must not block the user's command.
    """
    try:
        candidates = destructive_targets(command)
        if not candidates:
            return None
        live = _parse_ls_names(parse_response(send_command("ls")))
        objects = [c for c in candidates if c in live]
        if not objects:
            return None  # nothing to lose (fresh create / already-gone names)

        root = _backups_root()
        os.makedirs(root, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        g_path = os.path.join(root, f"snap_{stamp}.g")
        resp = send_command(f"keep {g_path} {' '.join(objects)}")
        if is_error_response(resp):
            logger.warning("snapshot keep failed: %s", parse_response(resp))
            return None
        _write_manifest(g_path, command, objects, stamp)
        logger.info("snapshot %s -> %s", objects, g_path)
        return (f"\n(auto-snapshot: saved {len(objects)} object(s) "
                f"[{', '.join(objects)}] before this destructive command; "
                f"restore with restore_backup)")
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.warning("snapshot skipped: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@mcp.tool()
def list_backups() -> str:
    """List restore points created before raw destructive commands (newest first).

    Each entry shows when it was made, the command that triggered it, and the
    objects it can restore.  Use ``restore_backup`` to roll one back.
    """
    manifests = _list_manifests()
    if not manifests:
        return ("No restore points yet.  One is created automatically before a "
                "raw destructive command (kill/rm/mv/r/...).")
    lines = [f"{len(manifests)} restore point(s), newest first:"]
    for i, m in enumerate(manifests):
        tag = "latest" if i == 0 else f"#{i + 1}"
        lines.append(f"  [{tag}] {m.get('created', '?')} -- "
                     f"'{m.get('command', '?')}' -> "
                     f"{', '.join(m.get('objects', []))}")
    return "\n".join(lines)


@mcp.tool()
def restore_backup(
    which: int = Field(
        default=0,
        description=(
            "Which restore point to roll back: 0 (default) is the most recent, "
            "1 the one before it, and so on.  Use list_backups to see them."
        ),
    ),
) -> str:
    """Roll back the objects saved by an auto-snapshot before a destructive command.

    Kills the current version of each saved object (if any) and re-imports the
    snapshot with ``dbconcat``, then refreshes the view.  This is the recovery
    path for raw ``execute_command`` edits -- spec-backed models built with
    build_from_spec should use ``undo_build`` instead.
    """
    manifests = _list_manifests()
    if not manifests:
        return "Error: no restore points exist."
    if which < 0 or which >= len(manifests):
        return (f"Error: no restore point #{which}; there are "
                f"{len(manifests)} (0 is newest).")

    m = manifests[which]
    g_path = m.get("backup", "")
    objects = m.get("objects", [])
    if not g_path or not os.path.isfile(g_path):
        return f"Error: backup file is missing on disk: {g_path}"

    # Clear any current versions so dbconcat re-imports cleanly, then bring the
    # snapshot back.  kill of an already-absent object is harmless.
    if objects:
        send_command(f"kill {' '.join(objects)}")
    resp = send_command(f"dbconcat {g_path}")
    if is_error_response(resp):
        return (f"Error: restore failed during dbconcat: {parse_response(resp)}. "
                "Some referenced sub-objects may still exist in the scene.")

    # Refresh the live view so the user sees the restored geometry.
    for obj in objects:
        send_command(f"draw {obj}")
    send_command("autoview")

    return (f"Restored {len(objects)} object(s) [{', '.join(objects)}] from "
            f"the snapshot taken before '{m.get('command', '?')}' "
            f"({m.get('created', '?')}). View refreshed.")
