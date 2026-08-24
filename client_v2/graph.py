"""Assemble the client-v2 agent graph.

Shape (with a skill registry; without one it degrades to intake -> worker):

    intake --chat--> respond --------------------------------------> END
           --work--> planner -> authorize (may HALT for a decision)
                                  +--(deterministic plan)--> executor --+
                                  |                                  |
                                  +--(otherwise)-------> worker -----+
                                                                     |
                                          verifier <-----------------+
                                            |    |
                     revise (bounded) <-----+    +--> visual_check
                        back to planner               |     |     |
                              ^-----------------------+     |     +--> END
                               (visual mismatch, once)      +--> formatter --> END

verifier is engine truth (rays vs the spec); visual_check is a fidelity opinion
that only runs when renders AND a reference image are both present.

Everything is injected (models, tools, classifier, registry, checkpointer) so
the whole graph can be exercised offline with fakes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from client_v2.agents.authorize import make_authorize_node
from client_v2.agents.conversational import (
    ROUTE_CHAT,
    ROUTE_WORK,
    heuristic_classifier,
    make_intake_node,
    make_respond_node,
    route_after_intake,
)
from client_v2.agents.formatter import make_formatter_node, needs_formatting
from client_v2.agents.planner import make_planner_node
from client_v2.agents.verifier import (
    make_verifier_node,
    revision_budget_spent,
    route_after_verify,
)
from client_v2.agents.visual import make_visual_check_node, route_after_visual
from client_v2.agents.worker import make_context_middleware, make_worker_node
from client_v2.model import for_tool_loop
from client_v2.pipeline.executor import is_deterministic, make_executor_node
from client_v2.pipeline.plan import Plan
from client_v2.runlog import RunLog, null_log
from client_v2.skills import SkillRegistry
from client_v2.skills.middleware import make_skills_middleware
from client_v2.state import AgentState


def _logged(name: str, node, log: RunLog):
    """Wrap a node so what it wrote to state is recorded.

    Done here rather than inside each agent: one place, and every node gets it
    automatically -- including ones added later.
    """
    if inspect.iscoroutinefunction(node):
        async def wrapped_async(state):
            out = await node(state)
            log.event("node", node=name, wrote=out)
            return out
        return wrapped_async

    def wrapped(state):
        out = node(state)
        log.event("node", node=name, wrote=out)
        return out
    return wrapped


def route_after_verification(state) -> str:
    """Post-verify: retry an engine-truth failure, else go on to the visual check."""
    return "revise" if route_after_verify(state) == "revise" else "look"


def route_after_visual_check(state) -> str:
    """Post-visual: a mismatch may revise once; otherwise format or finish.

    The revision budget is checked HERE as well as on the verifier's edge: every
    path back to the planner answers to the same counter.  This edge used to be
    bounded only by the visual round counter, so when that stopped advancing
    nothing else could stop it -- ``revisions`` reached 12 while the only gate
    that could have halted the turn governed a different edge.
    """
    if route_after_visual(state) == "revise" and not revision_budget_spent(state):
        return "revise"
    return "format" if needs_formatting(state) else "done"


def make_planner_router(registry: SkillRegistry):
    """After planning: a deterministic plan -> executor, else -> worker."""
    def route_after_planner(state) -> str:
        raw = state.get("plan")
        if raw and is_deterministic(Plan.model_validate(raw), registry):
            return "executor"
        return "worker"
    return route_after_planner


def build_graph(
    *,
    worker_model,
    tools,
    worker_prompt=None,
    registry: SkillRegistry | None = None,
    chat_model=None,
    planner_model=None,
    formatter_model=None,
    visual_model=None,
    classifier: Callable[[str], str] = heuristic_classifier,
    checkpointer=None,
    log: RunLog | None = None,
):
    """Compile the intake -> (planner) -> {worker, respond} graph.

    ``worker_model`` runs the tool loop; ``chat_model`` answers the chat path
    (defaults to ``worker_model``).  When a ``registry`` is given, a planner
    node selects a skill (setting ``active_skill``) and the worker gets a
    SkillsMiddleware that injects the catalog plus the active skill's detail.
    ``planner_model`` defaults to ``worker_model`` (a higher-effort model can be
    passed here later).  Pass a ``checkpointer`` for cross-turn memory.

    ``worker_prompt`` may be text, a callable, or None -- the default -- for the
    prompt library's ``worker`` entry, re-read on every model call so an edited
    prompt file takes effect without a restart.  Tests pass a fixed string.
    """
    log = log or null_log()
    chat_model = chat_model or worker_model
    planner_model = planner_model or worker_model
    formatter_model = formatter_model or worker_model
    visual_model = visual_model or worker_model

    # Outermost first: bound the history before anything else looks at it, so a
    # long session cannot grow the worker's context without limit.
    middleware = [make_context_middleware(worker_model)]
    if registry is not None:
        middleware.append(make_skills_middleware(registry, worker_prompt))

    # Work goes through the planner first only when there are skills to pick.
    work_entry = "planner" if registry is not None else "worker"

    g = StateGraph(AgentState)

    def add(name: str, node) -> None:
        """Register a node under *name*, logged under the same name.

        One place, so a node can never end up logged under a label that does not
        match its edges -- and every node added later is recorded for free.
        """
        g.add_node(name, _logged(name, node, log))

    add("intake", make_intake_node(classifier))
    # Tool-call serialization belongs on the tool loop, not the model: the
    # middleware above and every other node invoke it with no tools.
    add("worker", make_worker_node(for_tool_loop(worker_model), tools,
                                   worker_prompt, middleware))
    add("respond", make_respond_node(chat_model))

    if registry is not None:
        add("planner", make_planner_node(planner_model, registry))
        add("authorize", make_authorize_node(registry))
        add("executor", make_executor_node(registry, tools))
        add("verifier", make_verifier_node())
        add("visual_check", make_visual_check_node(visual_model))
        add("formatter", make_formatter_node(formatter_model))

        g.add_edge("planner", "authorize")
        g.add_conditional_edges(
            "authorize", make_planner_router(registry),
            {"executor": "executor", "worker": "worker"})
        g.add_edge("executor", "verifier")
        g.add_edge("worker", "verifier")
        g.add_conditional_edges(
            "verifier", route_after_verification,
            {"revise": "planner", "look": "visual_check"})
        g.add_conditional_edges(
            "visual_check", route_after_visual_check,
            {"done": END, "revise": "planner", "format": "formatter"})
        g.add_edge("formatter", END)
    else:
        g.add_edge("worker", END)

    g.set_entry_point("intake")
    g.add_conditional_edges(
        "intake", route_after_intake,
        {ROUTE_WORK: work_entry, ROUTE_CHAT: "respond"})
    g.add_edge("respond", END)

    return g.compile(checkpointer=checkpointer)
