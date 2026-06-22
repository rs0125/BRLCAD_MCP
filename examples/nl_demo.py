"""Scripted natural-language walkthrough.

Runs a fixed list of plain-English prompts through the real LangGraph ReAct
agent against a live libmcpcad listener, printing each tool call and reply.
Reuses the production agent setup (MCP server subprocess + tools); only the
interactive REPL is replaced with a scripted prompt list.

Prerequisites:
  - a libmcpcad listener running (MGED `mcp_listen <port>`, or the standalone
    `mcpcad_test_server`), with BRLCAD_PORT pointing at it
  - OPENAI_API_KEY set (this calls a real LLM and costs money)

Usage:
  BRLCAD_PORT=5555 python examples/nl_demo.py
"""

import asyncio
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent

from brlcad_mcp.client.agent import SYSTEM_PROMPT, _build_model

PROMPTS = [
    "Make a sphere named ball with radius 7 at the origin.",
    "Create a box called crate that is 8 units on each side at the origin.",
    "What objects are in the database right now?",
    "Combine the ball and the crate into a new assembly called widget.",
]


async def main() -> None:
    model = _build_model()
    client = MultiServerMCPClient(
        {
            "brlcad_server": {
                "command": sys.executable,
                "args": ["-m", "brlcad_mcp.server"],
                "transport": "stdio",
            }
        }
    )

    async with client.session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        print(f"[loaded {len(tools)} tools]\n")
        agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)

        for prompt in PROMPTS:
            print(f"\n{'='*60}\nYou: {prompt}\n{'='*60}")
            final = ""
            async for event in agent.astream_events(
                {"messages": [("user", prompt)]}, version="v2"
            ):
                kind = event["event"]
                if kind == "on_tool_start":
                    print(f"  > tool: {event.get('name')}  "
                          f"input={event.get('data', {}).get('input', '')}")
                elif kind == "on_tool_end":
                    out = str(event.get("data", {}).get("output", ""))
                    print(f"    result: {out[:200]}")
                elif kind == "on_chat_model_end":
                    output = event.get("data", {}).get("output")
                    if output and not getattr(output, "tool_calls", []):
                        final = getattr(output, "content", "")
            print(f"AI: {final}")


if __name__ == "__main__":
    asyncio.run(main())
