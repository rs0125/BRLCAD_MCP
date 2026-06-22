"""Natural-language end-to-end tests through the real LLM agent.

These exercise the full stack: prompt -> LLM -> tool selection -> transport
-> mock listener -> fake database.  They cost money and require an API key,
so they are skipped unless ``--run-llm`` is passed:

    pytest --run-llm

The mock listener stands in for BRL-CAD, so no engine build is needed - we
assert on the resulting fake-database state and the tools the agent chose.
"""

from __future__ import annotations

import asyncio

import pytest

from brlcad_mcp.config import settings

pytestmark = pytest.mark.llm


def _require_key():
    if not settings.llm.api_key:
        pytest.skip("OPENAI_API_KEY not set")


async def _run_prompt(port: int, prompt: str) -> str:
    """Drive the real ReAct agent for one prompt against the mock listener."""
    import sys

    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.prebuilt import create_react_agent

    from brlcad_mcp.client.agent import SYSTEM_PROMPT, _build_model

    model = _build_model()
    # the spawned MCP server subprocess must dial the mock listener's port
    client = MultiServerMCPClient(
        {
            "brlcad_server": {
                "command": sys.executable,
                "args": ["-m", "brlcad_mcp.server"],
                "transport": "stdio",
                "env": {"BRLCAD_PORT": str(port), "BRLCAD_HOST": "127.0.0.1"},
            }
        }
    )
    async with client.session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        agent = create_react_agent(model, tools, prompt=SYSTEM_PROMPT)
        final = ""
        async for event in agent.astream_events(
            {"messages": [("user", prompt)]}, version="v2"
        ):
            if event["event"] == "on_chat_model_end":
                out = event.get("data", {}).get("output")
                if out and not getattr(out, "tool_calls", []):
                    final = getattr(out, "content", "")
        return final


def test_nl_create_sphere(listener):
    _require_key()
    prompt = "Make a sphere named ball with radius 7 at the origin."
    asyncio.run(_run_prompt(listener.port, prompt))
    # the agent must have created exactly the object the user asked for
    assert "ball" in listener.mged.db


def test_nl_query_database(listener):
    _require_key()
    listener.mged.db.update({"widget.r", "gizmo.s"})
    answer = asyncio.run(_run_prompt(listener.port, "What objects are in the database?"))
    # the agent should have run ls and reported the objects
    assert "ls" in listener.received
    assert "widget" in answer or "gizmo" in answer


def test_nl_boolean(listener):
    _require_key()
    listener.mged.db.update({"ball", "crate"})
    prompt = "Combine the ball and crate into a region called widget."
    asyncio.run(_run_prompt(listener.port, prompt))
    # some new combination object should now exist beyond the two inputs
    assert len(listener.mged.db) > 2
