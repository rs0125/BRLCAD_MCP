"""The authorize phase -- a real pause, not a request to please pause.

Some workflows must stop and get a decision before acting: confirming the
dimensions read off a drawing, or picking a lighting variant before committing to
a slow render.  That was expressed only as prose ("stop and ask the user"), which
a model is free to ignore -- and did: `render_beauty` ran all three of its staged
renders without waiting.

A LangGraph ``interrupt`` cannot be ignored.  The graph genuinely halts, the
caller (REPL or harness) surfaces the question, and execution resumes with the
answer.  The trigger is data, not a new schema field: a skill that needs a
decision already says so with an ``{authorize: "..."}`` entry in its ``steps``.
"""

from __future__ import annotations

from langgraph.types import interrupt

from client_v2.skills import SkillRegistry
from client_v2.skills.middleware import plan_skill_ids


def authorization_request(plan, registry: SkillRegistry) -> str | None:
    """The question a planned skill wants answered first, or None.

    Reads the ``authorize`` step of the first planned skill that declares one,
    so the pause is defined by the skill definition rather than by the graph.
    """
    for skill_id in plan_skill_ids(plan):
        skill = registry.get(skill_id)
        if skill is None:
            continue
        for step in skill.steps:
            if isinstance(step, dict) and step.get("authorize"):
                return str(step["authorize"])
    return None


def make_authorize_node(registry: SkillRegistry):
    """Node that halts for a decision when the plan calls for one."""

    def authorize(state):
        # Only ever pause once per turn: the answer is recorded so a kick-back
        # through the planner does not re-ask the same question.
        if state.get("authorized"):
            return {}
        question = authorization_request(state.get("plan"), registry)
        if not question:
            return {"authorized": True}
        answer = interrupt({"authorize": question, "plan": state.get("plan")})
        return {"authorized": True, "authorization": str(answer)}

    return authorize
