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
from client_v2.agents.verifier import make_verifier_node, route_after_verify
from client_v2.agents.visual import make_visual_check_node, route_after_visual
from client_v2.agents.worker import make_context_middleware, make_worker_node
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
    """Post-visual: a mismatch may revise once; otherwise format or finish."""
    if route_after_visual(state) == "revise":
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
    g.add_node("intake", _logged("intake", make_intake_node(classifier), log))
    g.add_node("worker", _logged("worker", make_worker_node(
        worker_model, tools, worker_prompt, middleware), log))
    g.add_node("respond", _logged("respond", make_respond_node(chat_model), log))
    if registry is not None:
        g.add_node("planner", _logged("planner", make_planner_node(planner_model, registry), log))
        g.add_node("executor", _logged("executor", make_executor_node(registry, tools), log))
        g.add_node("verifier", _logged("verifier", make_verifier_node(), log))
        # A workflow that declares an `authorize` step halts here for a real
        # decision before anything runs: prose asking the model to pause was
        # ignorable, an interrupt is not.
        g.add_node("authorize", _logged("authorize", make_authorize_node(registry), log))
        g.add_edge("planner", "authorize")
        # Then a deterministic plan runs in the executor; anything else (incl. no
        # usable plan) falls through to the model-driven worker.
        g.add_conditional_edges(
            "authorize", make_planner_router(registry),
            {"executor": "executor", "worker": "worker"})
        # Both work paths report to the verifier, which either finishes the turn
        # or kicks the work back to the planner for a bounded revision.
        g.add_edge("executor", "verifier")
        g.add_edge("worker", "verifier")
        g.add_node("formatter", _logged("formatter", make_formatter_node(formatter_model), log))
        # The visual check only costs a model call when renders AND a reference
        # image are both present; otherwise it passes straight through.
        g.add_node("visual_check", _logged(
            "visual_check", make_visual_check_node(visual_model), log))
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
