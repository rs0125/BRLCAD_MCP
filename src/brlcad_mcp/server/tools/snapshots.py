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
    expand_targets,
    has_glob,
    is_error_response,
    is_ray_artifact,
    ls_names,
    parse_response,
    removes_objects,
)
from brlcad_mcp.transport import send_command

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Filesystem layout (pure, socket-free -- unit tested)
# ---------------------------------------------------------------------------

def _backups_root() -> str:
    """Where restore points live (BRLCAD_BACKUP_DIR).

    Its own directory, not a subfolder of the render output: clearing a render
    cache must not take the only way back from a bad raw edit with it.  Any
    pre-existing backups under the old location are still listed, below.
    """
    return settings.render.backup_dir


def _legacy_backups_root() -> str:
    """The previous location, still read so old restore points remain usable."""
    return os.path.join(settings.render.output_dir, "backups")


def _sidecar_path(g_path: str) -> str:
    return g_path[:-2] + ".json" if g_path.endswith(".g") else g_path + ".json"


def _write_manifest(g_path: str, command: str, objects: list[str],
                    stamp: str) -> None:
    with open(_sidecar_path(g_path), "w") as fh:
        json.dump({"backup": g_path, "command": command,
                   "objects": objects, "created": stamp}, fh, indent=2)


def _read_manifests(root: str) -> list[dict]:
    """Every manifest in one directory (unsorted); [] if it does not exist."""
    if not os.path.isdir(root):
        return []
    out: list[dict] = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(root, name)) as fh:
                out.append(json.load(fh))
        except (OSError, json.JSONDecodeError):
            continue
    return out


def _list_manifests() -> list[dict]:
    """All backup manifests, newest first.

    Reads the legacy location too, so restore points made before backups moved
    out of the render directory are still usable.  Each manifest records the
    absolute path of its ``.g``, so a restore works wherever it was written.
    Sorted on the recorded timestamp rather than filename, since two directories
    are being merged.
    """
    roots = [_backups_root()]
    legacy = _legacy_backups_root()
    if os.path.abspath(legacy) != os.path.abspath(roots[0]):
        roots.append(legacy)
    manifests = [m for root in roots for m in _read_manifests(root)]
    return sorted(manifests, key=lambda m: str(m.get("created", "")), reverse=True)


# ---------------------------------------------------------------------------
# Snapshot (called by execute_command before a destructive command runs)
# ---------------------------------------------------------------------------

def live_targets(command: str) -> list[str]:
    """The objects currently in the database that *command* would destroy.

    Globs are resolved against ``ls`` rather than compared to it -- see
    :func:`expand_targets`.  Returns [] for a non-destructive command, and never
    raises: this only ever informs a safety net.
    """
    try:
        candidates = destructive_targets(command)
        if not candidates:
            return []
        live = ls_names(parse_response(send_command("ls")))
        return expand_targets(candidates, live)
    except (ConnectionError, TimeoutError, OSError) as exc:
        logger.warning("could not resolve destructive targets: %s", exc)
        return []


def survivors(objects: list[str]) -> list[str]:
    """Which of *objects* are STILL in the database (for an after-the-fact check)."""
    if not objects:
        return []
    try:
        live = ls_names(parse_response(send_command("ls")))
    except (ConnectionError, TimeoutError, OSError):
        return []
    return [o for o in objects if o in live]


def maybe_snapshot(command: str) -> str | None:
    """Snapshot the existing objects *command* would destroy, if any.

    Returns a short note to append to the tool output (so the agent/user knows
    a restore point exists), or ``None`` when nothing needed snapshotting.
    Never raises -- a snapshot failure must not block the user's command.
    """
    try:
        # Ray leftovers are excluded: they hold nothing recoverable, `keep` on one
        # exports a broken record, and a turn spent trying to delete an
        # undeletable artifact otherwise leaves a restore point behind per
        # attempt (it made three in one turn).
        objects = [o for o in live_targets(command) if not is_ray_artifact(o)]
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


def describe_effect(objects: list[str], left: list[str],
                    command: str = "") -> str:
    """Note describing what a destructive command actually removed (pure).

    Empty when there was nothing to remove or everything went as asked.  MGED can
    return SUCCESS for a command that changed nothing -- `kill *` did exactly
    that -- so "no output" is not evidence of an effect.

    The ADVICE has to match the cause, or the note makes things worse.  The first
    version always suggested "try naming objects explicitly", which is nonsense
    when the command already did: faced with an undeletable `query_rayffff` the
    agent burned ten tool calls re-trying kill, killall, summary, get and put,
    because our own message implied a retry would help.  A note that sends the
    reader somewhere useless is worse than no note.
    """
    if not objects or not left:
        return ""
    partial = len(left) < len(objects)
    prefix = (f"\n(NOTE: {len(objects) - len(left)} of {len(objects)} object(s) "
              f"were removed, but "
              if partial else
              f"\n(WARNING: this command reported success but removed NOTHING -- "
              f"all {len(objects)} named object(s) are still present: ")
    shown = ", ".join(left[:8]) + (" ..." if len(left) > 8 else "")

    if all(is_ray_artifact(name) for name in left):
        return (f"{prefix}{shown}. These are nirt ray leftovers, not model "
                f"geometry: they cannot be read or deleted (kill and killall both "
                f"report success and leave them), and they are stale directory "
                f"entries rather than real objects. Treat the database as empty of "
                f"geometry and DO NOT keep trying to remove them.)")
    if has_glob(command):
        return (f"{prefix}{shown}. This MGED build does not appear to expand "
                f"wildcards for this command -- name the objects explicitly and "
                f"confirm with ls.)")
    return (f"{prefix}{shown}. They were named explicitly, so retrying the same "
            f"command will not help: the entries cannot be removed this way and "
            f"may be stale or corrupt. Report this instead of retrying.)")


def destructive_effect_note(command: str, targets: list[str]) -> str:
    """Check after the fact whether a destructive command did what it claimed.

    Only for verbs that actually delete objects -- see :data:`REMOVING_VERBS`.
    """
    if not removes_objects(command):
        return ""
    return describe_effect(targets, survivors(targets), command)


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
