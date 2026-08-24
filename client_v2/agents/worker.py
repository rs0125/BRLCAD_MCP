"""The worker agent -- the only agent with tool access.

Built on LangChain 1.x ``create_agent``, with middleware as the mechanism for
dynamic behaviour (the SkillsMiddleware injects skill definitions into the system
prompt).  Model, tools and middleware are injected, so the node runs offline
against a fake tool-calling model and in-process tools.

Notes
-----
* **Async only.**  MCP tools are async-only StructuredTools; a sync ``invoke``
  raises "does not support sync invocation".
* **History is summarised, not truncated.**  A long modelling session accumulates
  large tool results (spec JSON, render paths, verification reports), so it has to
  be bounded or cost and latency grow until the window overflows.  Summarising
  beats the blunt sliding window v1 used: older turns are condensed rather than
  discarded, and the middleware keeps AI/Tool call pairs together.
* **The prompt is resolved once here**, because ``create_agent`` wants a string.
  That only matters on the registry-less path -- whenever skills are loaded the
  SkillsMiddleware supplies the prompt instead, re-read on every model call.
"""

from __future__ import annotations

from typing import NotRequired

from langchain.agents import create_agent
from langchain.agents.middleware import AgentState as _LCAgentState
from langchain.agents.middleware import (
    ModelCallLimitMiddleware,
    SummarizationMiddleware,
    ToolCallLimitMiddleware,
)

from client_v2.prompts import resolve
from client_v2.state import AgentState

SUMMARISE_AT_TOKENS = 60_000
KEEP_RECENT_MESSAGES = 12

# Loop guards.  The worker's tool loop is otherwise unbounded: create_agent
# takes no max-iterations argument, and the graph's recursion_limit counts OUTER
# supersteps -- the whole tool loop is a single superstep, so it never fires.
# Seen live: a model built the part correctly, verified it against the
# raytracer, then called declare_assumption 367 times instead of reporting back.
# A stronger model knowing when to stop is luck, not a guard, and on a metered
# endpoint the bill is real.
#
# Both limits are far above what real work needs.  The successful runs of the
# same case finish in well under ten model calls.
MAX_MODEL_CALLS_PER_RUN = 50
# Declarations have no natural stopping point: each one succeeds and reports a
# growing total, which reads as encouragement.  An under-specified drawing
# legitimately produces a handful.
MAX_DECLARATIONS_PER_RUN = 30


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
    """
    # Guards go first so they see every model/tool call, and are built here
    # rather than passed in so a caller cannot forget them.
    guards = [
        ModelCallLimitMiddleware(run_limit=MAX_MODEL_CALLS_PER_RUN,
                                 exit_behavior="end"),
        ToolCallLimitMiddleware(tool_name="declare_assumption",
                                run_limit=MAX_DECLARATIONS_PER_RUN,
                                exit_behavior="continue"),
    ]
    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=resolve(system_prompt, "worker"),
        middleware=guards + list(middleware or []),
        state_schema=WorkerState,
    )

    async def worker(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        result = await agent.ainvoke({
            "messages": msgs,
            "active_skill": state.get("active_skill"),
            "plan": state.get("plan"),
        })
        return {"messages": result["messages"][len(msgs):]}

    return worker
