"""client-v2 REPL entry point — run with ``python -m client_v2.main``.

Wires the production dependencies into the graph: the Responses-API model
(model.build_model), the MCP tools over a persistent stdio session, the skill
registry (loaded from client_v2/skills/), and the thin role prompts from
client_v2.prompts (the v1 monolith is no longer used).

Set ``CLIENT_V2_DEBUG=true`` to stream a per-node agent trace (routing, model
turns, tool calls, tool results) instead of just the final answer.

``/image`` and ``/paste`` attach a reference image -- the input the
model_from_dimensioned_sketch workflow works from.  ``/skills`` and ``/prompts``
list what is loaded, and ``/reload`` re-reads both from disk, so a skill or a
prompt file can be edited and retried without restarting the session.
"""

from __future__ import annotations

import asyncio
import os
import sys

from langchain_core.messages import AIMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from brlcad_mcp.config import settings
from client_v2.agents.conversational import message_text
from client_v2.graph import build_graph
from client_v2.model import build_model
from client_v2.prompts import PROMPTS
from client_v2.runlog import open_run_log
from client_v2.skills import SkillRegistry
from client_v2.terminal.attachments import (
    HELP,
    ReplCommand,
    attached_image_count,
    parse_input,
)
from client_v2.terminal.trace import format_update

_RECURSION_LIMIT = 60


def _last_ai_text(state) -> str:
    # responses/v1 AI content is a list of blocks (reasoning + text); extract
    # the human-readable text, not the raw list (which includes reasoning blobs).
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, AIMessage):
            text = message_text(msg).strip()
            if text:
                return text
    return "(no response)"


def _pending_question(result) -> str | None:
    """The question an authorization halt is waiting on, if the graph paused."""
    for pause in (result or {}).get("__interrupt__") or ():
        value = getattr(pause, "value", pause)
        if isinstance(value, dict) and value.get("authorize"):
            return str(value["authorize"])
    return None


async def _drive(graph, inputs, config, log=None):
    """Run a turn, answering any authorization halt from the user.

    A workflow that declares an `authorize` step genuinely stops the graph, so
    the REPL has to surface the question and resume with the answer instead of
    the turn silently ending.
    """
    result = await graph.ainvoke(inputs, config)
    while (question := _pending_question(result)) is not None:
        print(f"\nAI needs a decision: {question}")
        try:
            answer = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "cancel"
        if log:
            log.event("interrupt", question=question, answer=answer)
        result = await graph.ainvoke(Command(resume=answer or "approved"), config)
    return result


async def _run_turn(graph, inputs, config, debug: bool, log=None) -> None:
    """Invoke one turn; stream a node-by-node trace when debug is on."""
    if not debug:
        result = await _drive(graph, inputs, config, log)
        answer = _last_ai_text(result)
        if log:
            log.event("result", answer=answer)
        print(f"\nAI: {answer}")
        return

    print("\n--- trace ---")
    final = None
    payload = inputs
    while payload is not None:
        async for namespace, update in graph.astream(
                payload, config, stream_mode="updates", subgraphs=True):
            for node, node_update in update.items():
                print(format_update(node, node_update, namespace))
                final = node_update
        # Streaming ends at a halt too, so ask and resume the same way.
        state = await graph.aget_state(config)
        question = _pending_question({"__interrupt__": state.interrupts})
        if question is None:
            payload = None
            continue
        print(f"\nAI needs a decision: {question}")
        try:
            answer = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            answer = "cancel"
        if log:
            log.event("interrupt", question=question, answer=answer)
        payload = Command(resume=answer or "approved")
    print("--- end trace ---")
    # The final state is recoverable from the checkpointer for the answer line.
    snapshot = await graph.aget_state(config)
    answer = _last_ai_text(snapshot.values)
    if log:
        log.event("result", answer=answer)
    print(f"\nAI: {answer}")
    _ = final


async def run() -> None:
    worker_model = build_model()
    registry = SkillRegistry.from_dir()
    log = open_run_log()

    print("Starting client-v2 MCP client...")
    if settings.debug:
        print("(debug on: streaming agent trace)")
    client = MultiServerMCPClient({
        "brlcad_server": {
            "command": sys.executable,
            "args": ["-m", "brlcad_mcp.server"],
            "transport": "stdio",
            "env": dict(os.environ),
        }
    })

    async with client.session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        print(f"Loaded {len(tools)} tool(s) from BRL-CAD; "
              f"{len(registry.ids())} skill(s).")

        graph = build_graph(
            worker_model=worker_model,
            tools=tools,
            registry=registry,
            checkpointer=MemorySaver(),
            log=log,
        )
        # The callbacks capture every model call, including the ones the worker
        # makes inside its own tool loop.
        config = {"configurable": {"thread_id": "v2"},
                  "recursion_limit": _RECURSION_LIMIT,
                  "callbacks": log.callbacks()}
        log.event("session", tools=len(tools), skills=len(registry.ids()),
                  model=settings.llm.model)
        if log.path:
            print(f"run log: {log.path}")

        print("=" * 49)
        print(" client-v2 active. /help for commands, 'exit' to quit.")
        print("=" * 49)
        while True:
            try:
                text = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if not text:
                continue
            try:
                parsed = parse_input(text)
            except ValueError as exc:
                print(f"  {exc}")
                continue
            if isinstance(parsed, ReplCommand):
                if parsed.name == "quit":
                    break
                if parsed.name == "help":
                    print(HELP)
                elif parsed.name == "skills":
                    print(registry.catalog())
                elif parsed.name == "prompts":
                    print(PROMPTS.catalog())
                elif parsed.name == "reload":
                    # Both are edit-then-retry surfaces, so one command re-reads
                    # both rather than making you remember which needs which.
                    print(registry.reload())
                    print(PROMPTS.reload())
                continue
            message = (HumanMessage(content=parsed[1])
                       if isinstance(parsed, tuple) else parsed)
            attached = attached_image_count(message)
            if attached:
                print(f"  (attached {attached} image(s))")
            log.start_turn(text, images=attached)
            try:
                await _run_turn(graph, {"messages": [message]}, config,
                                settings.debug, log)
            except Exception as exc:  # keep the REPL alive on any turn failure
                print(f"\nAI: that turn failed ({type(exc).__name__}: {exc}).")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
