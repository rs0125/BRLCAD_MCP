"""LangGraph ReAct agent that connects to the MCP tool server."""

from __future__ import annotations

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
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

## Tool strategy

1. **Prefer dedicated tools first.**  For common operations (creating spheres,
   boxes, cylinders, boolean combinations) use the specialised tools
   (create_sphere, create_box, create_cylinder, boolean_combination).
   They handle drawing/autoview automatically and have strict parameter
   validation.

2. **If no dedicated tool fits, use the discovery workflow:**
   a. Call ``list_commands`` (optionally with a category filter) to browse
      all available MGED commands with one-liner descriptions.
   b. Identify a promising command, then call ``get_command_help`` to read
      its full man-page and learn the exact argument syntax.
   c. Finally, call ``execute_command`` with the correctly formed command
      string.  Set ``auto_draw=true`` and ``object_name`` when the command
      creates or modifies visible geometry.

3. **Never guess MGED syntax.**  Always verify via ``get_command_help``
   before calling ``execute_command`` with a command you haven't used before.
"""


def _build_model() -> ChatOpenAI:
    """Instantiate the LLM backend."""
    if not settings.llm.api_key:
        print("ERROR: OPENAI_API_KEY environment variable is not set.")
        sys.exit(1)
    return ChatOpenAI(
        model=settings.llm.model,
        temperature=settings.llm.temperature,
    )


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
    tools = await client.get_tools()
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

        print("AI is calculating geometry...")
        response = await agent.ainvoke(
            {"messages": [("user", user_input)]}
        )
        print(f"\nAI: {response['messages'][-1].content}")


def main() -> None:
    """Synchronous entry point for the agent CLI."""
    asyncio.run(run_agent())
