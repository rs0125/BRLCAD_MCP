"""The worker agent — the only agent with tool access.

Built on LangChain 1.x ``create_agent`` (the supported replacement for the
deprecated ``langgraph.prebuilt.create_react_agent``).  Middleware (e.g. the
SkillsMiddleware) is how dynamic behavior — like injecting skill definitions
into the system prompt — is added, per LangChain's v1 guidance.

The worker's subagent state carries the planner's ``plan`` and ``active_skill``
so the SkillsMiddleware can inject the plan plus the definitions of the skills it
names; the outer graph passes them in on each invocation.

Model, tools, and middleware are injected so the node is testable offline with
a fake tool-calling model and in-process tools.

Note: the old sliding-window ``pre_model_hook`` is intentionally dropped here;
context management moves to middleware (e.g. SummarizationMiddleware) in a later
increment rather than a bespoke hook.
"""

from __future__ import annotations

from typing import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState as _LCAgentState
from langchain.agents.middleware import SummarizationMiddleware

from client_v2.prompts import resolve
from client_v2.state import AgentState

# Context management.  A long modelling session accumulates large tool results
# (spec JSON, render paths, verification reports), so history has to be bounded
# or cost and latency grow without limit until the model's window overflows.
# Summarising beats the blunt sliding window v1 used: older turns are condensed
# rather than discarded, and the middleware keeps AI/Tool call pairs together.
SUMMARISE_AT_TOKENS = 60_000
KEEP_RECENT_MESSAGES = 12


class WorkerState(_LCAgentState):
    """create_agent's state, extended with what the planner decided.

    Both fields are read by the SkillsMiddleware: ``plan`` is the ordered,
    parameterised plan (preferred), ``active_skill`` the single-skill fallback
    used when no usable plan was produced.
    """

    active_skill: NotRequired[str | None]
    plan: NotRequired[dict | None]


def make_context_middleware(model, trigger_tokens: int = SUMMARISE_AT_TOKENS,
                            keep_messages: int = KEEP_RECENT_MESSAGES):
    """Summarise older history once it passes *trigger_tokens*."""
    return SummarizationMiddleware(
        model=model,
        trigger=("tokens", trigger_tokens),
        keep=("messages", keep_messages),
    )


def make_worker_node(model, tools, system_prompt=None, middleware=None):
    """Build a worker node that runs a tool loop over *tools*.

    Returns only the messages the agent newly produced, which add_messages
    appends to the shared graph state.

    ``system_prompt`` may be text, a callable, or None for the prompt library's
    ``worker`` entry.  It is resolved once, here, because create_agent wants a
    string -- and it only matters on the registry-less path, since the
    SkillsMiddleware supplies the prompt (re-read per call) whenever skills are
    loaded.
    """
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=resolve(system_prompt, "worker"),
        middleware=list(middleware or []),
        state_schema=WorkerState,
    )

    async def worker(state: AgentState) -> AgentState:
        # Async invocation is required: MCP tools are async-only StructuredTools,
        # and sync invoke raises "does not support sync invocation".  Pass the
        # planner-selected skill through so the SkillsMiddleware can inject it.
        msgs = state.get("messages", [])
        result = await agent.ainvoke({
            "messages": msgs,
            "active_skill": state.get("active_skill"),
            "plan": state.get("plan"),
        })
        new_messages = result["messages"][len(msgs):]
        return {"messages": new_messages}

    return worker
