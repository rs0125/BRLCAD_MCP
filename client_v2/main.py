"""client-v2 REPL entry point — run with ``python -m client_v2.main``.

Wires the production dependencies into the graph: the Responses-API model
(model.build_model), the MCP tools over a persistent stdio session, the skill
registry (loaded from client_v2/skills/), and the thin role prompts from
client_v2.prompts (the v1 monolith is no longer used).

``/image`` and ``/paste`` attach a reference image -- the input the
model_from_dimensioned_sketch workflow works from.  ``/skills`` and ``/prompts``
list what is loaded, and ``/reload`` re-reads both from disk, so a skill or a
prompt file can be edited and retried without restarting the session.

The trace (``/trace``, or ``CLIENT_V2_DEBUG`` for the starting state)
--------------------------------------------------------------------
Shows model replies and tool calls/results live as they happen -- including inside
the worker's own tool loop -- plus one line per node as it finishes, carrying the
state it wrote: the route, the plan, the verdict.

* It is a **runtime** toggle, not launch-only.  The moment you want it is right
  after a turn that surprised you, and restarting to get it would throw away the
  conversation the model has been building on.
* The live lines come from a LangChain callback (``LiveTrace``), because node
  updates alone cannot show the worker's loop: that loop is a nested invocation
  which surfaces only once it has already finished.
* Streaming therefore covers OUTER nodes only, and prints them without their
  messages.  Including subgraphs added a 36-char UUID namespace per line and
  repeated activity the callback had already shown above it.
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
from client_v2.agents.authorize import describe_pause
from client_v2.agents.conversational import message_text
from client_v2.graph import build_graph
from client_v2.model import build_model, describe_backend
from client_v2.prompts import PROMPTS
from client_v2.runlog import open_run_log
from client_v2.skills import SkillRegistry
from client_v2.terminal.attachments import (
    HELP,
    ReplCommand,
    attached_image_count,
    parse_input,
)
from client_v2.terminal.trace import LiveTrace, format_update

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
    """The question an authorization halt is waiting on, if the graph paused.

    Rendered by describe_pause so the plan travels with the question: the bare
    skill text ("confirm the dimensioned plan with the user") asks the reader to
    approve something they cannot see.
    """
    for pause in (result or {}).get("__interrupt__") or ():
        value = getattr(pause, "value", pause)
        if isinstance(value, dict) and value.get("authorize"):
            return describe_pause(value)
    return None


def _ask_and_resume(question: str, log=None) -> Command:
    """Surface an authorization halt and build the resume command.

    Shared by both turn paths: streaming ends at a halt just as ainvoke does, so
    asking and resuming was written out twice before.
    """
    print(f"\nAI needs a decision: {question}")
    try:
        answer = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        answer = "cancel"
    if log:
        log.event("interrupt", question=question, answer=answer)
    return Command(resume=answer or "approved")


def _announce(answer: str, log=None) -> None:
    if log:
        log.event("result", answer=answer)
    print(f"\nAI: {answer}")


async def _run_plain(graph, inputs, config, log=None) -> None:
    """One turn, final answer only."""
    result = await graph.ainvoke(inputs, config)
    while (question := _pending_question(result)) is not None:
        result = await graph.ainvoke(_ask_and_resume(question, log), config)
    _announce(_last_ai_text(result), log)


async def _run_traced(graph, inputs, config, log=None) -> None:
    """One turn, printing each node as it finishes."""
    print("\n--- trace ---")
    payload = inputs
    while payload is not None:
        async for update in graph.astream(payload, config,
                                          stream_mode="updates"):
            for node, node_update in update.items():
                print(format_update(node, node_update, include_messages=False))
        state = await graph.aget_state(config)
        question = _pending_question({"__interrupt__": state.interrupts})
        payload = _ask_and_resume(question, log) if question else None
    print("--- end trace ---")
    # The final state comes back from the checkpointer for the answer line.
    snapshot = await graph.aget_state(config)
    _announce(_last_ai_text(snapshot.values), log)


async def _run_turn(graph, inputs, config, debug: bool, log=None) -> None:
    run_turn = _run_traced if debug else _run_plain
    await run_turn(graph, inputs, config, log)


def _mcp_client() -> MultiServerMCPClient:
    """The stdio MCP server subprocess -- importing its tools registers them."""
    return MultiServerMCPClient({
        "brlcad_server": {
            "command": sys.executable,
            "args": ["-m", "brlcad_mcp.server"],
            "transport": "stdio",
            "env": dict(os.environ),
        }
    })


async def _repl(graph, registry, log, base_callbacks, tracing: bool) -> None:
    """Read a line, run a turn, repeat.  Kept apart from the session setup."""
    live = LiveTrace()
    config = {"configurable": {"thread_id": "v2"},
              "recursion_limit": _RECURSION_LIMIT,
              "callbacks": list(base_callbacks)}

    # Local commands, keyed by name; each returns what to print.  `trace` is
    # toggled below before its line is rendered, and the lambda closes over the
    # variable so it reports the new state.
    commands = {
        "help": lambda: HELP,
        "skills": lambda: registry.catalog(),
        "prompts": lambda: PROMPTS.catalog(),
        # Both are edit-then-retry surfaces, so one command re-reads both.
        "reload": lambda: f"{registry.reload()}\n{PROMPTS.reload()}",
        "trace": lambda: (
            f"  trace {'on' if tracing else 'off'}"
            + ("  -- re-run your last request to watch it" if tracing else "")),
    }

    while True:
        try:
            text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not text:
            continue
        try:
            parsed = parse_input(text)
        except ValueError as exc:
            print(f"  {exc}")
            continue
        if isinstance(parsed, ReplCommand):
            if parsed.name == "quit":
                return
            if parsed.name == "trace":
                tracing = not tracing
            print(commands[parsed.name]())
            continue

        message = (HumanMessage(content=parsed[1])
                   if isinstance(parsed, tuple) else parsed)
        if attached := attached_image_count(message):
            print(f"  (attached {attached} image(s))")
        log.start_turn(text, images=attached)
        # Rebuilt per turn so /trace takes effect on the very next one.
        config["callbacks"] = base_callbacks + ([live] if tracing else [])
        try:
            await _run_turn(graph, {"messages": [message]}, config, tracing, log)
        except Exception as exc:   # keep the REPL alive on any turn failure
            print(f"\nAI: that turn failed ({type(exc).__name__}: {exc}).")


async def run() -> None:
    worker_model = build_model()
    registry = SkillRegistry.from_dir()
    log = open_run_log()
    tracing = settings.debug

    print("Starting client-v2 MCP client...")
    async with _mcp_client().session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        graph = build_graph(worker_model=worker_model, tools=tools,
                            registry=registry, checkpointer=MemorySaver(),
                            log=log)
        log.event("session", tools=len(tools), skills=len(registry.ids()),
                  model=settings.llm.model)

        print(f"Loaded {len(tools)} tool(s) from BRL-CAD; "
              f"{len(registry.ids())} skill(s).")
        # Name the backend: the endpoint is configurable, so a silent fall back
        # to the default would be the confusing failure.
        print(f"model: {describe_backend()}")
        if log.path:
            print(f"run log: {log.path}")
        print("=" * 49)
        print(" client-v2 active. /help for commands, 'exit' to quit.")
        # Stated rather than merely available: an opt-in trace nobody knows about
        # is a trace nobody uses.
        print(f" trace is {'ON' if tracing else 'off'} -- /trace to toggle.")
        print("=" * 49)

        await _repl(graph, registry, log, log.callbacks(), tracing)


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
