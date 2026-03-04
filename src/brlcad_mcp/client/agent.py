"""LangGraph ReAct agent that connects to the MCP tool server."""

from __future__ import annotations

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from brlcad_mcp.config import settings

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

## Be proactive

Never ask the user for information you can look up yourself.  Query the
scene first:
- ``execute_command("ls")`` — list objects.
- ``execute_command("l <obj>")`` — inspect an object.
- ``execute_command("search . -type sph")`` — find primitives by type.
- ``execute_command("bb <obj>")`` — bounding box.

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

## MGED glob patterns

MGED commands handle ``*`` glob expansion internally on database object
names.  Unlike a Unix shell, Tcl does NOT expand ``*`` — it passes it
through to MGED, which matches against database objects.  Examples:
- ``killall *`` — delete every object
- ``draw *`` — draw every object
- ``killall sphere*`` — delete all objects starting with "sphere"
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

        agent = create_react_agent(
            model,
            tools,
            prompt=SYSTEM_PROMPT,
        )

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
