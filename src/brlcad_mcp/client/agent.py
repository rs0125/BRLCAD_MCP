"""LangGraph ReAct agent that connects to the MCP tool server."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import os
import re
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, trim_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from brlcad_mcp.config import settings

# Image extensions we accept for /image; the model (gpt-4o) is multimodal, so an
# attached image travels to it as an OpenAI image_url part (a base64 data URI).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}

# Slash-command aliases, matched on the first whitespace-delimited word so
# "/images" no longer gets mistaken for "/image" (and vice versa).
_IMAGE_CMDS = ("/image", "/images", "/img")
_PASTE_CMDS = ("/paste", "/clip")
_HELP_CMDS = ("/help", "/?")

_HELP = """Commands:
  /image <path> [more paths] [prompt]  attach image file(s) and send a message
                                       (aliases: /images, /img)
  /paste [prompt]                      attach an image from the clipboard
                                       (alias: /clip)
  /help                                show this help
  exit | quit                          leave
Drag-and-drop a file into the terminal to paste its path after /image.
Anything else is sent to the agent as a normal message."""


def _data_uri(data: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _image_part_from_file(path: Path) -> dict:
    """Build an OpenAI multimodal image_url part from an image file."""
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return {"type": "image_url",
            "image_url": {"url": _data_uri(path.read_bytes(), mime)}}


def _clipboard_image_part() -> dict | None:
    """Best-effort grab of an image from the clipboard (Wayland then X11).

    Returns an image_url part, or None if no image / no clipboard tool.
    """
    for cmd in (["wl-paste", "--type", "image/png"],
                ["xclip", "-selection", "clipboard", "-t", "image/png", "-o"]):
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout:
            return {"type": "image_url",
                    "image_url": {"url": _data_uri(out.stdout, "image/png")}}
    return None


def _image_message(rest: str) -> HumanMessage:
    """Build a multimodal message from ``/image`` arguments.

    ``rest`` is everything after the command word: one or more image paths
    (leading), then an optional free-text prompt.  Raises ``ValueError`` with a
    precise, user-facing message when no usable image is found.
    """
    tokens = rest.split()
    images: list[Path] = []
    prompt_tokens: list[str] = []
    for i, tok in enumerate(tokens):
        p = Path(tok.strip("\"'")).expanduser()
        looks_like_image = p.suffix.lower() in _IMAGE_EXTS
        if looks_like_image and p.is_file():
            images.append(p)
        elif looks_like_image:
            # An image-looking path that does not exist -- almost always a
            # typo'd or wrong-directory path, so say exactly what we tried.
            raise ValueError(f"Could not find that image file: {p}")
        else:
            prompt_tokens = tokens[i:]
            break
    if not images:
        raise ValueError(
            "Usage: /image <path> [more paths] [prompt]. No image path found "
            "(paths must come first; supported extensions: "
            f"{', '.join(sorted(_IMAGE_EXTS))}).")
    prompt = " ".join(prompt_tokens).strip() or "Here is a reference image."
    parts = [{"type": "text", "text": prompt}]
    parts += [_image_part_from_file(p) for p in images]
    print(f"  (attached {len(images)} image(s): "
          f"{', '.join(p.name for p in images)})")
    return HumanMessage(content=parts)


def _paste_message(rest: str) -> HumanMessage:
    """Build a multimodal message from a clipboard image (``/paste``)."""
    prompt = rest.strip() or "Here is a reference image."
    part = _clipboard_image_part()
    if part is None:
        raise ValueError(
            "No image on the clipboard (needs wl-paste or xclip, and an "
            "image copied). Use /image <path> instead.")
    print("  (attached image from clipboard)")
    return HumanMessage(content=[{"type": "text", "text": prompt}, part])


def _build_message(text: str):
    """Turn a line of REPL input into an agent message.

    ``/image PATH [PATH...] [prompt]`` (aliases ``/images``, ``/img``) attaches
    one or more image files; ``/paste [prompt]`` (alias ``/clip``) grabs an
    image from the clipboard.  Command matching is on the first whitespace-
    delimited word, so ``/images`` is not confused with ``/image``.  Anything
    that is not a recognized command is a plain text turn.  Returns a
    ``HumanMessage`` (multimodal) or a ``("user", text)`` tuple, or raises
    ``ValueError`` with a user-facing message.
    """
    cmd, _, rest = text.partition(" ")
    key = cmd.lower()
    if key in _IMAGE_CMDS:
        return _image_message(rest)
    if key in _PASTE_CMDS:
        return _paste_message(rest)
    # An unrecognized slash command is a mistake, not a message to the agent --
    # otherwise "/imae foo.png" would silently be sent as prose.
    if cmd.startswith("/") and key not in _HELP_CMDS:
        raise ValueError(
            f"Unknown command: {cmd}. Type /help for the command list.")
    return ("user", text)


# Tools whose renders we feed back to the model (as a USER message -- OpenAI
# rejects images in a tool-role message, so the tool returns paths and the
# client re-attaches the images here) so it can visually self-check its work.
#
# Two flavours: build tools produce check views that should be COMPARED to a
# reference (count features, verify layout); pure render tools produce an image
# the model asked for and must actually LOOK at (so "render it and tell me what
# you see" genuinely works instead of the model bluffing from the file path).
_BUILD_FEEDBACK_TOOLS = {"build_from_spec", "edit_build", "undo_build"}
_RENDER_FEEDBACK_TOOLS = {"render_model", "render_previews"}
_FEEDBACK_TOOLS = _BUILD_FEEDBACK_TOOLS | _RENDER_FEEDBACK_TOOLS
# Cap the automatic compare/inspect rounds per user request so it can't loop.
MAX_AUTO_ROUNDS = 3


def _extract_pngs(text: str) -> list[str]:
    """Absolute .png paths mentioned in a tool result that exist on disk."""
    seen: set[str] = set()
    out: list[str] = []
    for m in re.findall(r"/[^\s'\"]+\.png", text):
        if m not in seen and os.path.isfile(m):
            seen.add(m)
            out.append(m)
    return out


def _compare_followup(pngs: list[str]) -> HumanMessage:
    """A user-role message that shows renders back to the model to compare."""
    parts = [{"type": "text", "text": (
        "Here are the check renders you just produced (attached). Compare them "
        "to the reference CAREFULLY -- do not just say it looks right:\n"
        "1. COUNT the repeated features (studs, holes, bolts, etc.) in the "
        "render AND in the reference, and state both numbers explicitly. If "
        "they differ, the model is wrong.\n"
        "2. Sanity-check the count against dimensions (e.g. a 32 mm x 16 mm "
        "brick at 8 mm spacing is 4x2 = 8 studs, not 6).\n"
        "3. Check orientation, proportions, and each feature's position.\n"
        "If anything is off, fix it with edit_build (small ops) -- or "
        "build_from_spec only if the whole layout is wrong. If everything "
        "matches, say so and stop.")}]
    parts += [_image_part_from_file(Path(p)) for p in pngs]
    return HumanMessage(content=parts)


def _inspect_followup(pngs: list[str]) -> HumanMessage:
    """A user-role message that shows a plain render back to the model to LOOK at.

    Used for render_model / render_previews (no build spec involved): the model
    must describe what is actually in the pixels, not infer from the file path.
    """
    parts = [{"type": "text", "text": (
        "Here is the render you just produced (attached). LOOK at it and report "
        "what is ACTUALLY visible -- do not describe what you expected:\n"
        "1. Describe the geometry you see (shape, features, orientation).\n"
        "2. If a reference image or a stated requirement is in context, say "
        "explicitly whether the render matches it, and COUNT any repeated "
        "features in both.\n"
        "3. If the framing is bad or an expected feature is missing/occluded, "
        "re-render from a better angle (render_model with a different "
        "azimuth/elevation) before answering.\n"
        "Then give the user the file path(s) and your honest assessment. If you "
        "were only asked for a preview/stamp, stop after reporting and let the "
        "user confirm before any final render.")}]
    parts += [_image_part_from_file(Path(p)) for p in pngs]
    return HumanMessage(content=parts)

# Sliding context window: the agent is shown at most this many of the most
# recent messages on each turn (a turn is a user message plus the assistant
# / tool messages it triggers, so this is roughly the last 3-5 turns).  The
# checkpointer still records full history; this only bounds what the model
# reasons over, so older context is "forgotten" for inference and token
# growth stays bounded on long sessions.
MEMORY_WINDOW_MESSAGES = 24


def _window_messages(state):
    """pre_model_hook: pin the model's context to the recent window.

    Trims from the end, but starts the window on a human message so a
    tool-result message is never sent without its originating tool call
    (which the chat API rejects).
    """
    windowed = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=len,  # count messages, not tokens
        max_tokens=MEMORY_WINDOW_MESSAGES,
        start_on="human",
        end_on=("human", "tool"),
        include_system=False,
        allow_partial=False,
    )
    # llm_input_messages affects only this model call; full history is kept.
    return {"llm_input_messages": windowed}

# ---------------------------------------------------------------------------
# System prompt — guides the agent to prefer dedicated tools but fall back
# to the dynamic discovery → help → execute workflow for anything else.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a BRL-CAD geometry assistant.  You operate inside MGED (the BRL-CAD
interactive geometry editor) by calling tools exposed through an MCP server.

## OUTPUT FORMAT

Your replies are shown in a PLAIN-TEXT terminal that does NOT render
markdown.  Do NOT use markdown: no ``**bold**``, no ``#`` headers, no
``-``/``*`` bullet syntax, no backtick code spans.  Write plain prose and,
when listing, use simple indentation or "N)" numbering.  When a tool returns
a pre-formatted block (e.g. the health report's ASCII table), pass it
through VERBATIM inside your answer rather than re-styling it into markdown —
it is already laid out for the terminal.

## STOP RULE

When a tool call succeeds, **immediately reply to the user**.  Do NOT make
another tool call.  Do NOT create a second object.  Do NOT call
execute_command after a dedicated tool succeeds — they already handle
drawing and view refresh internally.

The ONLY reasons to make a follow-up tool call are:
1. The previous call returned ``[MGED_ERROR]`` and you need to retry.
2. You are in the discovery workflow (list_commands → get_command_help →
   execute_command) and haven't executed the final command yet.
3. The user explicitly asked for multiple operations.

## Name conflicts

If a creation command fails because the name already exists, **never delete
or overwrite the existing object**.  Instead, pick a new unique name by
appending an incrementing number (e.g. ``sphere1.s``, ``sphere2.s``) and
retry.  Only delete or overwrite objects when the user explicitly asks.

## Editing existing models — do NOT hand-demolish

Prefer the smallest, most reversible edit, and NEVER rebuild a model by
tearing it down first.

- If the model was built with ``build_from_spec`` (it has a saved spec — check
  ``list_builds``), change it ONLY through ``edit_build`` (small move / update /
  add / remove ops) and revert with ``undo_build``.  Do NOT hand-edit a
  spec-backed model with raw ``kill`` / ``rm`` / ``r`` via ``execute_command``:
  that bypasses the spec history, so ``undo_build`` can no longer recover it.
- To change one feature (e.g. fix a hole that did not subtract), operate on
  THAT primitive alone (redefine just ``hole_x.s`` with ``in``, or re-issue the
  single subtraction).  Do NOT ``kill`` the region and rebuild the whole tree —
  that is how a model gets wiped.
- Raw destructive commands (``kill``, ``killall``, ``rm``, ``mv``, ``r`` that
  redefines an existing region…) are auto-snapshotted before they run.  If an
  edit goes wrong, call ``restore_backup`` to roll back the last one, or
  ``list_backups`` to see the restore points.  Tell the user this exists rather
  than leaving them stuck.
- When a rendered feature looks wrong, first confirm it is truly a geometry bug
  (inspect with ``l`` and re-render from an angle that is NOT occluded — a
  concave part hides features from an outside-corner view) before deleting
  anything.  A "missing" hole is often just hidden by the viewing angle.

## Conversation memory

You retain the recent conversation (a sliding window of the last several
turns).  When the user refers to "the sphere", "it", "that object", or
"make it bigger", resolve the reference from objects you created or
discussed in that recent context — do NOT ask the user to name it again.
If the reference predates your memory window or is genuinely ambiguous,
recover it from the live scene (``ls`` / ``search``) rather than guessing
or asking.  If exactly one object of the relevant type exists, that is the
referent.

## Be proactive

Never ask the user for information you can look up yourself.  Query the
scene first:
- ``execute_command("tops")`` — the TOP-LEVEL assemblies.  Start here for
  "what is this model" questions, and to find the model's root object
  (e.g. the whole vehicle) before operating on "the entire model".
- ``execute_command("ls")`` — list objects.
- ``execute_command("l <obj>")`` — inspect an object.
- ``execute_command("search . -type sph")`` — find primitives by type.
- ``execute_command("bb <obj>")`` — bounding box.

## Analytics & measurement

For quantitative questions — volume, surface area, mass, centroid, bounding
box — use BRL-CAD's own analysis rather than computing by hand.  The engine
is the source of truth; do NOT plug radii into formulas yourself.
- ``execute_command("analyze <obj>")`` — engine-computed volume and surface
  area for a primitive or region.
- ``execute_command("bb <obj>")`` — bounding-box dimensions and volume.
Call these directly with ``execute_command``; they succeed normally.  Do NOT
route them through ``analyze_command_error`` — that tool is only for
recovering from a command that has *already* failed.  Report the engine's
numbers, with units.  If several objects are nested or overlapping, say so
rather than summing their volumes blindly.

## Task recipes — verified sequences, prefer these over improvising

**Understand a model**: ``tops`` for the top-level assemblies, then
``l <assembly>`` to descend a level.  Answer "what is this model?" from the
assembly names, not from a flat object listing.

**Extract a triangle mesh**: ``facetize <obj> <name>.bot`` creates a solid
BoT mesh inside the database (verify with ``l <name>.bot``).  Large models
can take ~30s; that is normal.  Do NOT use ``keep`` — it exports objects to
a separate .g file and does not create a mesh.

**Rendering — STAMP FIRST, then confirm, then the full render.**  Do NOT go
straight to a full/final render.  Rendering is cheap-to-expensive and the user
approves before the slow final:
1. STAMP: render a small, fast preview first.
   - Single requested view: call ``render_model`` at small size (~192),
     ``quality=draft``, ``ambient_samples=0``.
   - "Make it look great" or several options: call ``render_previews`` with a
     few ``view:lighting`` variants (small / draft / no AO); report the folder
     and the A/B/C legend.
   Then STOP and reply with the image path(s), asking the user to confirm the
   framing / lighting (or pick a label).  Do NOT render the final in the same
   turn — wait for their answer.
2. (multi-option only) AO ROUND: on the labels the user picked, call
   ``render_previews`` again at a larger size (~400) with ``ambient_samples``
   ~64 (ambient + AO dialed in).  Stop and ask them to confirm.
3. FINAL: once the user approves, call ``render_model`` at the requested size
   (default 800) with ``quality=clean`` for the finished render.
EXCEPTION — render directly: if the user explicitly asks to skip the preview
("render directly", "just render it", "quick render", "no preview"), skip the
stamp and call ``render_model`` at full quality straightaway.
Always use ``render_model`` / ``render_previews`` for images — never hand-run
``rt``; they need no file path.  After a visual change (color, move, new
geometry), proactively offer a render so the user can verify with their own
eyes — attribute output alone can be misleading.

**Model something from a reference image**: when the user attaches an image
(e.g. front / side views of a real object) and asks you to build it in BRL-CAD,
work in explicit stages — do NOT start creating geometry blind:
1. ESTIMATE (fix the scale FIRST): study the image and produce a STRUCTURED,
   dimensioned spec BEFORE building — overall bounding box in mm, each feature
   (name, size, position), wall thickness, and a proposed CSG decomposition.
   For SCALE:
   - If it is a recognizable object (an iPhone, a standard bottle, a Lego brick
     …), use its real known dimensions from your own knowledge and say which
     you assumed.
   - Otherwise ask the user for ONE real dimension (e.g. overall height) and
     derive the rest from the image's proportions.
   Present the spec as plain text and ask the user to confirm or correct the
   numbers.  Build NOTHING until they approve.
2. BUILD + CHECK via the ``build_from_spec`` tool.  Turn the approved spec into
   its JSON schema (box / cylinder / sphere parts, unioned or subtracted into one
   region; first part must be ``add``; all mm) and call the tool.  It builds the
   CSG deterministically AND renders the requested check views in one step, so
   you do not create geometry by hand for this.  Model a hollow cover by adding
   the outer solid then SUBTRACTING a slightly smaller inner solid; subtract
   boxes/cylinders for cutouts (camera, ports, buttons).
3. COMPARE — YOU will see the renders.  Right after ``build_from_spec`` runs,
   the check views are shown back to you as images (in a follow-up message).
   Actually LOOK at them and compare to the reference image (still in context):
   is the orientation right, are proportions and feature positions correct, is
   anything missing?  Do NOT just hand the images to the user and ask — judge
   them yourself first.
4. ADJUST and ITERATE: to change a model that already exists, use
   ``edit_build`` with a SMALL list of ops (move / update / add / remove a part)
   — do NOT re-send the whole model through ``build_from_spec`` (that risks
   dropping parts you did not mean to change).  ``edit_build`` loads the current
   spec, applies just your ops, rebuilds and re-renders.  To revert a change,
   call ``undo_build``; ``list_builds`` shows the saved versions.  This is a
   legitimate multi-step loop, so the STOP RULE does not force you to stop after
   one edit — iterate up to ~3 rounds on your own judgement, then show the user
   and ask.  Set expectations plainly: a recognizable approximation, not an
   exact replica.

**Move / mirror a part non-interactively**:
1. FIRST run ``units mm`` — this is mandatory before any coordinate work.
   ``bb`` always reports millimetres, but ``otranslate`` interprets its
   arguments in the database's editing units, which are often inches.
   Setting mm makes the two agree; skipping this silently scales every
   move (e.g. a model in inches multiplies your offset by 25.4, flinging
   the part metres away).
2. Measure with ``bb -m <obj>`` (the ``-m`` flag prints the CENTER as
   "Mid Point: (x y z)" — plain ``bb`` gives only dimensions, useless for
   positioning).
3. Compute the offset, then ``otranslate <obj> <dx> <dy> <dz>``.
Never use ``translate`` or ``sed`` — they need interactive edit state and
fail here.  When the user asks for a move, CARRY IT OUT — measuring and
describing the plan is not completing the task; finish with the
``otranslate`` call.  To mirror across a plane, negate the part's midpoint
coordinate on that axis: offset = -2 x midpoint (e.g. mirror across XZ:
dy = -2 x midY, dx = dz = 0).  To verify, run ``bb -m`` on the moved part
after and confirm the new midpoint is where you intended — the raytracer
and the numbers must agree.

**Duplicating a part**: to make a real COPY of geometry (e.g. to add
another wheel), use ``cp <source> <newname>`` — this copies the geometry so
you can then move the copy independently.  Do NOT use ``g`` or ``comb`` for
this: those create a group/combination that merely REFERENCES the original,
so "moving the copy" either moves nothing useful or drags a shared
reference.  Recipe to place a duplicate: ``cp <src> <new>`` -> ``units mm``
-> ``bb -m <src>`` and ``bb -m`` a correctly-placed sibling to learn the
target coordinate -> ``otranslate <new> <dx> <dy> <dz>`` -> add ``<new>`` to
the parent assembly if it should belong to it.

**Find overlapping geometry**: ALWAYS use ``gqa -g 1 -Ao <assembly>`` — the
``-g 1`` pins the sampling grid to 1 mm.  Without a ``-g`` flag gqa
auto-refines its grid until its volume estimate converges, which on
coincident or degenerate faces NEVER happens — it refines to billions of
cells and hangs.  Never call bare ``gqa -Ao``.  Output lists overlapping
region pairs with penetration depth (``dist:`` in mm) and location.  The
default overlap tolerance is 0, so the list mixes genuine overlaps
(millimetres deep) with coincident-surface noise (sub-0.01 mm); triage by
depth — treat sub-0.01 mm hits as noise unless told otherwise, deepest
first.  For finer detection re-run with a smaller ``-g`` (e.g. ``-g 0.5``).
Or call the ``model_health_report`` tool for a full audit — it runs this same
overlap check plus BRL-CAD's lint validators and returns one grouped report.

**Resolve an overlap — ASK THE USER FIRST**: there are two standard fixes,
and the choice belongs to the user, not you:
1. *Subtract* (one region yields): append a subtraction to the yielding
   region's OWN tree with a raw command via execute_command —
   ``r <yielding_region> - <other_region>``.  This edits the region IN
   PLACE (it just adds "- other" to the existing tree) and creates NO new
   object.  Do NOT use the boolean_combination tool for this and do NOT
   invent a new region name like ``regionA.r`` — those nest/duplicate
   instead of trimming.  gqa reports region names, and ``r`` accepts a
   region as the operand (you'll see "Note: X is a region" — that's fine).
2. *Move* (parts separated): use the dedicated tools — do NOT hand-roll the
   move with ``otranslate``.  ``separate_overlap`` slides ONE overlapping pair
   apart by the minimal clearance (it binary-searches the distance with gqa as
   a yes/no oracle and re-verifies afterwards); ``resolve_overlaps`` sweeps a
   whole assembly and resolves each pair the same way.  These are more robust
   than a hand-computed move, so prefer them.
   MOVE THE RIGHT LEVEL: gqa reports overlaps at leaf-region granularity, e.g.
   ``/havoc/weapons/ft_weapons/30mm_autocannon/30mm_barrel/r.b``.  Pass the
   meaningful PART (the named subassembly the user means, e.g.
   ``30mm_autocannon``), NOT the leaf — moving a leaf tears it out of its
   assembly.  ``separate_overlap`` refuses bare solids and reports a part's
   parents to help you pick the right level.  If the two parts are fully
   nested (centres nearly coincide) the exit direction is ambiguous — ask the
   user which way to move.
   FALLBACK, only if the tools cannot handle a case: move by hand along the
   line joining the two parts' centres — ``units mm`` first, ``bb -m`` each
   region, subtract for the direction, then ``otranslate`` by the penetration
   depth PLUS ~1 mm clearance (moving by exactly the depth leaves the surfaces
   touching, which still counts as an overlap).

After either fix, re-run ``gqa`` on the pair to confirm it is gone.
A bare instruction like "fix it" / "resolve it" does NOT count as choosing —
it says *that* they want it fixed, not *how*.  Until the user has named a
strategy (and, for subtraction, which region yields), your reply to an
overlap-fix request is a QUESTION presenting both options — make no tree
edit and no move.  This overrides the dedicated-tools-first rule.

## Tool strategy

1. **Dedicated tools first** — create_sphere, create_box, create_cylinder,
   boolean_combination (they handle draw/autoview automatically); render_model
   for a single image and render_previews for a batch of labelled preview stamps
   (see the beauty-render recipe); build_from_spec to build+render a parametric
   model from a JSON spec, then edit_build / undo_build / list_builds to edit
   and revert it incrementally (the deterministic build stage of modelling from
   a reference image); model_health_report to audit a model; separate_overlap /
   resolve_overlaps for interference fixes (but see the overlap rules above —
   ask before resolving).
2. **Discovery workflow** — list_commands → get_command_help →
   execute_command.  Set ``auto_draw=true`` + ``object_name`` when the
   command creates or modifies visible geometry.
3. **Never guess MGED syntax** — check ``get_command_help`` first.
4. **Error recovery** — call ``analyze_command_error`` ONLY after an
   ``execute_command`` call actually returned an ``[MGED_ERROR]`` response,
   passing that exact error text (max 5 attempts).  Never call it
   preemptively, with a guessed error, or as the default way to run a
   command — try ``execute_command`` first and only escalate on a real
   failure.

## Stateful commands

MGED state persists across calls (the connection is kept alive).  However,
chain interactive-edit sequences in **one** call using Tcl semicolons so
they cannot be interrupted:
``execute_command("press; sed sphere.s; oscale 2; accept")``

Prefer ``db adjust`` (stateless) over ``sed``/``oscale``/``accept``:
| Task | Command |
|---|---|
| Scale a sphere | ``db adjust <name> r <new_radius>`` |
| Move a primitive | ``db adjust <name> V {<x> <y> <z>}`` |
| Delete one object | ``killall <name>`` |
| Delete many | resolve names via ``search`` first, then ``kill a b c`` |

## Color — IMPORTANT

To set color, ALWAYS use ``comb_color <object> <r> <g> <b>``.  It is
non-interactive and works on any combination or region.
- Whole model / assembly: TWO commands on the top combination —
  ``comb_color <top> <r> <g> <b>`` AND ``attr set <top> inherit 1``.
  The color attribute alone is IGNORED by the raytracer when child regions
  carry their own colors; the inherit flag is what forces it down the tree.
  Do NOT loop over child regions — those two commands are the whole job.
- A single part: ``comb_color <region> <r> <g> <b>`` (no inherit needed).

Do NOT use ``mater`` to set color over this interface — with color arguments
it drops into an interactive R/G/B prompt that this headless path cannot
answer, so it fails.  Do NOT use ``color`` — that is the region-id color
*table* (``color <low> <high> <r> <g> <b>``), not a per-object command.

## Quoting — IMPORTANT

The listener parses commands with a splitter that recognizes ONLY double
quotes, not single quotes.  Always use double quotes (or no quotes) around
any argument that contains a wildcard or space.  A single-quoted argument
like ``-name '*'`` is passed through literally (the quotes become part of
the string) and will match nothing.
- correct:   ``search . -name "*"``
- broken:    ``search . -name '*'``

## Wildcard / pattern operations — IMPORTANT

Wildcards (``*``) are NOT expanded for action commands over this interface.
A command like ``killall box*`` or ``draw sphere*`` reaches the engine as the
literal string ``box*``, matches nothing, and reports success while doing
nothing (MGED's interactive shell expands globs before execution, but this
headless interface does not).  NEVER rely on ``*`` in kill/killall/draw/erase.

To operate on multiple objects by pattern, **resolve the names first, then
act on them explicitly**:
1. ``execute_command("search . -name \\"*\\"")`` — ``search`` DOES match
   patterns and returns the actual object names (one per line).  Use double
   quotes around the pattern.  ``search . -type region`` matches by type
   (color/material lives on regions, so that is usually what you want for a
   whole-model recolor).  ``search .`` with no filter lists everything.
2. Issue the action with the explicit names you got back.  If the list is
   large, batch several names per command, e.g.
   ``execute_command("kill a b c d e")``.

If ``search`` returns nothing, there are no matches — report that honestly
rather than claiming the operation succeeded.
"""

def _is_reasoning_model(model_id: str) -> bool:
    """True for GPT-5.x / Sol / Terra / Luna / o-series reasoning models.

    These use ``reasoning_effort`` and reject a ``temperature`` argument, unlike
    older chat models (gpt-4o, gpt-4.1) that take temperature.
    """
    m = model_id.lower()
    return any(t in m for t in ("gpt-5", "sol", "terra", "luna",
                                "o1", "o3", "o4"))


def _model_kwargs(model_id: str, effort: str, temperature: float) -> dict:
    """ChatOpenAI kwargs for the model family (pure, so it's testable).

    Reasoning models (gpt-5.x / sol / terra / luna / o-series) reject
    ``reasoning_effort`` alongside function tools on /v1/chat/completions unless
    it is ``'none'`` -- and this agent always registers tools -- so we force
    ``'none'`` for them (extended reasoning would need the Responses API, not
    wired yet; the model's vision/capability is unaffected).  Legacy chat models
    (gpt-4o, ...) take ``temperature`` instead and reject ``reasoning_effort``.
    """
    if effort or _is_reasoning_model(model_id):
        return {"model": model_id, "reasoning_effort": "none"}
    return {"model": model_id, "temperature": temperature}


def _build_model():
    """Instantiate the LLM backend, matching params to the model family."""
    if not settings.llm.api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    kwargs = _model_kwargs(settings.llm.model, settings.llm.reasoning_effort,
                           settings.llm.temperature)
    return ChatOpenAI(**kwargs).bind(parallel_tool_calls=False)


async def run_agent() -> None:
    """Launch the interactive CLI agent loop."""
    model = _build_model()

    print("Starting local MCP Client...")
    client = MultiServerMCPClient(
        {
            "brlcad_server": {
                "command": sys.executable,
                "args": ["-m", "brlcad_mcp.server"],
                "transport": "stdio",
                # pass our env (BRLCAD_PORT/BIN/RENDER_DIR, LD_LIBRARY_PATH...)
                # through to the server subprocess so its tools can use them
                "env": dict(os.environ),
            }
        }
    )

    # Use a persistent session so every tool call reuses the SAME server
    # subprocess.  Without this, langchain-mcp-adapters creates a new
    # subprocess (and thus a new TCP connection to MGED) for every single
    # tool invocation, defeating server-side deduplication and state
    # preservation.
    async with client.session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        print(f"Successfully loaded {len(tools)} tool(s) from BRL-CAD!")

        # A checkpointer gives the agent conversational memory: every turn
        # is appended to a persisted message history keyed by thread_id, so
        # the agent remembers what it built and can resolve references like
        # "the sphere", "it", or "make it bigger" without re-asking.
        agent = create_react_agent(
            model,
            tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=MemorySaver(),
            pre_model_hook=_window_messages,
        )
        # One stable thread for the whole interactive session.  recursion_limit
        # caps ReAct steps per turn; if a request thrashes we want it to stop
        # and report, not run away (and never crash the REPL).
        config = {
            "configurable": {"thread_id": "mged-session"},
            "recursion_limit": 50,
        }

        print("\n=================================================")
        print(" BRL-CAD Terminal Agent Active. Type 'exit' to quit.")
        print("=================================================")

        print("Type /help for commands (including /image to attach a picture).")

        follow_up: HumanMessage | None = None
        auto_rounds = 0

        while True:
            if follow_up is not None:
                message = follow_up
                follow_up = None
                print("\n  [auto] showing the render(s) back to the agent to "
                      "compare with the reference...")
            else:
                try:
                    user_input = input("\nYou: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nGoodbye!")
                    break

                stripped = user_input.strip()
                if stripped.lower() in {"exit", "quit"}:
                    break
                if stripped.lower() in {*_HELP_CMDS, "help"}:
                    print(_HELP)
                    continue
                if not stripped:
                    continue

                try:
                    message = _build_message(stripped)
                except ValueError as exc:
                    print(f"  {exc}")
                    continue
                auto_rounds = 0  # a real user turn resets the auto-compare budget

            print("AI is thinking...\n")
            final_answer = ""
            produced_pngs: list[str] = []
            saw_build_feedback = False
            try:
                async for event in agent.astream_events(
                    {"messages": [message]},
                    version="v2",
                    config=config,
                ):
                    kind = event["event"]

                    # ── Agent is calling a tool ──
                    if kind == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        tool_input = event.get("data", {}).get("input", "")
                        print(f"  ▸ Calling tool: {tool_name}")
                        if tool_input:
                            preview = str(tool_input)
                            if len(preview) > 200:
                                preview = preview[:200] + "…"
                            print(f"    Input: {preview}")

                    # ── Tool returned a result ──
                    elif kind == "on_tool_end":
                        tool_output = event.get("data", {}).get("output", "")
                        full = str(tool_output)
                        preview = full[:300] + "…" if len(full) > 300 else full
                        print(f"    ✓ Result: {preview}\n")
                        # Collect renders to feed back for visual self-check.
                        tname = event.get("name")
                        if tname in _FEEDBACK_TOOLS:
                            produced_pngs.extend(_extract_pngs(full))
                            if tname in _BUILD_FEEDBACK_TOOLS:
                                saw_build_feedback = True

                    # ── LLM produced a final text reply (no tool calls) ──
                    elif kind == "on_chat_model_end":
                        output = event.get("data", {}).get("output")
                        if output:
                            tool_calls = getattr(output, "tool_calls", [])
                            content = getattr(output, "content", "")
                            if not tool_calls and content:
                                final_answer = content
            except GraphRecursionError:
                print(
                    "AI: I couldn't complete that within my step budget — the "
                    "approach wasn't converging. Try rephrasing, or ask for a "
                    "smaller step."
                )
                continue
            except Exception as exc:  # keep the REPL alive on any turn failure
                print(f"AI: that turn failed ({type(exc).__name__}: {exc}). "
                      "The session is still open — try again.")
                continue

            if final_answer:
                print(f"AI: {final_answer}")
            else:
                print("AI: (no response)")

            # If a build produced renders, feed them back (as a user message, so
            # OpenAI accepts the images) for a self-check round -- bounded so it
            # cannot loop forever.
            if produced_pngs and auto_rounds < MAX_AUTO_ROUNDS:
                follow_up = (_compare_followup(produced_pngs)
                             if saw_build_feedback
                             else _inspect_followup(produced_pngs))
                auto_rounds += 1


def main() -> None:
    """Synchronous entry point for the agent CLI."""
    asyncio.run(run_agent())
