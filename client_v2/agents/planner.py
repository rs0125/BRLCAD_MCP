"""The planner agent — the "plan" phase.

Given a work request and the skill catalog, the planner produces an ORDERED,
PARAMETERIZED plan (a :class:`~client_v2.pipeline.plan.Plan`): which skills to
run, in what order, with which inputs.  The plan goes into ``state['plan']``,
which both consumers read: the executor runs it directly when every step is
callable, and otherwise the SkillsMiddleware injects it -- plus the definitions
of every skill it names -- into the worker's prompt.

If the model doesn't return a usable plan, it falls back to single-skill
selection (:func:`choose_skill`), which sets ``active_skill`` so the worker still
gets a hint rather than nothing.  Parsing/validation lives in
:mod:`client_v2.pipeline.plan` and is pure.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from client_v2.agents.conversational import last_human_text, message_text
from client_v2.agents.verifier import failure_context
from client_v2.agents.visual import visual_failure_context
from client_v2.pipeline.plan import parse_plan
from client_v2.skills import SkillRegistry

PLANNER_SYSTEM = (
    "You are the planner for a BRL-CAD geometry agent. Produce an ORDERED plan "
    "of skill steps that fulfills the user's request, then stop. Reply with "
    "ONLY a JSON object of the form:\n"
    '{"steps": [{"skill": "<id>", "params": {...}, "why": "..."}], '
    '"done_when": "..."}\n'
    "Use only the skill ids provided. For each step, supply that skill's "
    "required inputs (marked * in the brief) in params; reference an earlier "
    "step's output with ${step_id.output}. If no skill applies, reply "
    '{"steps": []}.'
)


def planning_brief(registry: SkillRegistry) -> str:
    """The skills the planner may use, each rendered for parameter binding.

    Delegates to ``SkillDef.planning_view`` -- the single canonical rendering of
    what a planner needs (inputs, preconditions, cautions, examples).  Keeping
    it there rather than formatting fields here is deliberate: a hand-curated
    format silently omits any field nobody remembered to add, which is exactly
    how a through-hole caution ended up invisible to the planner.
    """
    return "\n".join(registry.get(sid).planning_view()
                     for sid in registry.ids())


def choose_skill(text: str, known_ids: list[str]) -> str | None:
    """Resolve a plain reply to a known skill id, or None (pure fallback).

    Accepts an exact id, or an id mentioned inside a longer reply; anything
    else (including 'none') resolves to None.
    """
    t = (text or "").strip().lower()
    if not t or not known_ids:
        return None
    for skill_id in known_ids:              # exact reply wins
        if t == skill_id.lower():
            return skill_id
    for skill_id in known_ids:              # else, id mentioned in the reply
        if skill_id.lower() in t:
            return skill_id
    return None


def make_planner_node(model, registry: SkillRegistry):
    """Node that plans a skill sequence for the request."""

    async def planner(state):
        request = last_human_text(state)
        parts = [f"Skills:\n{planning_brief(registry)}", f"User request:\n{request}"]
        # On a kick-back, plan again knowing what failed -- either an
        # engine-truth check or the visual comparison with the reference.
        context = failure_context(state) or visual_failure_context(state)
        if context:
            parts.append(context)
        reply = await model.ainvoke(
            [SystemMessage(content=PLANNER_SYSTEM),
             HumanMessage(content="\n\n".join(parts))])
        text = message_text(reply)

        # Count this pass so the verify->replan loop is bounded.
        revisions = (state.get("revisions") or 0) + (1 if context else 0)

        plan, errors = parse_plan(text, registry)
        if plan is not None and plan.steps and not errors:
            # No active_skill when there is a plan: the worker is given the whole
            # plan and every skill it names.  This used to be the plan's TERMINAL
            # step, an arbitrary pick that hid the earlier steps from the worker.
            return {"plan": plan.model_dump(),
                    "active_skill": None,
                    "revisions": revisions}

        # Fallback: no usable plan -> single-skill hint (or clear a stale pick).
        return {"plan": None, "active_skill": choose_skill(text, registry.ids()),
                "revisions": revisions}

    return planner
