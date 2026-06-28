"""LangGraph ReAct agent that connects to the MCP tool server."""

from __future__ import annotations

import asyncio
import sys

from langchain_core.messages import trim_messages
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
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
Report the engine's numbers, with units.  If several objects are nested or
overlapping, say so rather than summing their volumes blindly.

## Tool strategy

1. **Dedicated tools first** — create_sphere, create_box, create_cylinder,
   boolean_combination.  They handle draw/autoview automatically.
2. **Discovery workflow** — list_commands → get_command_help →
   execute_command.  Set ``auto_draw=true`` + ``object_name`` when the
   command creates or modifies visible geometry.
3. **Never guess MGED syntax** — check ``get_command_help`` first.
4. **Error recovery** — if a command fails, call ``analyze_command_error``
   (max 5 attempts).

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
| Create a database | ``opendb <name>.g y`` |
| Delete ALL objects | ``killall *`` |
| Delete one object | ``killall <name>`` |

## Wildcard / pattern operations — IMPORTANT

Wildcards (``*``) are NOT expanded for action commands over this interface.
A command like ``killall box*`` or ``draw sphere*`` reaches the engine as the
literal string ``box*``, matches nothing, and reports success while doing
nothing (MGED's interactive shell expands globs before execution, but this
headless interface does not).  NEVER rely on ``*`` in kill/killall/draw/erase.

To operate on multiple objects by pattern, **resolve the names first, then
act on them explicitly**:
1. ``execute_command("search . -name 'box*'")`` — ``search`` DOES match
   patterns and returns the actual object names (one per line).  Use
   ``search . -type sph`` to match by primitive type.
2. Issue the action with the explicit names you got back, e.g.
   ``execute_command("kill box1 box2 box3")``.

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
        # One stable thread for the whole interactive session.
        config = {"configurable": {"thread_id": "mged-session"}}

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

            if final_answer:
                print(f"AI: {final_answer}")
            else:
                print("AI: (no response)")


def main() -> None:
    """Synchronous entry point for the agent CLI."""
    asyncio.run(run_agent())
