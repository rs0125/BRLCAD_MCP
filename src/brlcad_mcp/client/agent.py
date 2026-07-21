"""LangGraph ReAct agent that connects to the MCP tool server."""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import trim_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

from brlcad_mcp.config import settings

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

**Render to an image**: use the ``render_model`` tool — do NOT hand-run ``rt``.
It renders the model the listener already has open (no file path needed),
handles drawing, view, lighting and PNG output, and returns the image path.
Pass ``obj`` plus a ``view`` preset (or ``azimuth``/``elevation``) and, if the
user wants it, a ``lighting`` mode (``studio`` default, ``model``, ``ambient``).
After completing a visual change (color, move, new geometry), offer the user a
render so they can verify the result with their own eyes — attribute output
alone can be misleading.

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
   for images; model_health_report to audit a model; separate_overlap /
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

def _build_model() -> ChatOpenAI:
    """Instantiate the LLM backend."""
    if not settings.llm.api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
    ).bind(parallel_tool_calls=False)


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

        while True:
            try:
                user_input = input("\nYou: ")
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if user_input.strip().lower() in {"exit", "quit"}:
                break

            print("AI is thinking...\n")
            final_answer = ""
            try:
                async for event in agent.astream_events(
                    {"messages": [("user", user_input)]},
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
                        output_str = str(tool_output)
                        if len(output_str) > 300:
                            output_str = output_str[:300] + "…"
                        print(f"    ✓ Result: {output_str}\n")

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


def main() -> None:
    """Synchronous entry point for the agent CLI."""
    asyncio.run(run_agent())
