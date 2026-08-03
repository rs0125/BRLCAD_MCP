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

import re

from langgraph.types import interrupt

from client_v2.agents.conversational import last_human_text
from client_v2.skills import SkillRegistry
from client_v2.skills.middleware import plan_skill_ids

# A request to destroy geometry WHOLESALE gets a gate even when no skill covers
# it.  "delete everything in this db" was planned as {"steps": []} -- no skill
# matched -- so no skill could declare an authorize step, and a wipe-the-database
# request ran with no confirmation at all.  Skill-declared gates only protect work
# we already modelled, which inverts the risk: the unmodelled requests are the
# ones going through raw execute_command with no preconditions and no verifier.
#
# Both halves must match, which is the whole point: "delete the mounting hole" is
# ordinary editing and must NOT nag, while "delete everything" must stop.
_DESTRUCTIVE_VERB = re.compile(
    r"\b(delete|remove|clear|wipe|erase|destroy|kill|purge|reset|nuke|drop)\b",
    re.IGNORECASE)
_BROAD_SCOPE = re.compile(
    r"(\beverything\b|\ball\b|\bentire\b|\bwhole\b|\bany(thing)?\b"
    r"|\bdatabase\b|\bthe db\b|\bscene\b|\bmodels\b|\bobjects\b|\*)",
    re.IGNORECASE)

WIPE_QUESTION = (
    "This will DESTROY geometry across the whole database, which is not "
    "reversible from the spec history. A snapshot is taken first, but say "
    "'proceed' only if you are sure -- or tell me which objects to remove "
    "instead."
)


def broad_destructive(text: str) -> bool:
    """True if *text* asks to destroy geometry wholesale rather than one feature."""
    return bool(text) and bool(_DESTRUCTIVE_VERB.search(text)) \
        and bool(_BROAD_SCOPE.search(text))


def authorization_request(plan, registry: SkillRegistry,
                          request: str = "") -> str | None:
    """The question that must be answered before this turn acts, or None.

    A planned skill's ``authorize`` step comes first, so a workflow defines its
    own pause.  Failing that, the raw request is checked for a wholesale
    destructive intent -- the case no skill covers.
    """
    for skill_id in plan_skill_ids(plan):
        skill = registry.get(skill_id)
        if skill is None:
            continue
        for step in skill.steps:
            if isinstance(step, dict) and step.get("authorize"):
                return str(step["authorize"])
    return WIPE_QUESTION if broad_destructive(request) else None


def make_authorize_node(registry: SkillRegistry):
    """Node that halts for a decision when the plan calls for one."""

    def authorize(state):
        # Only ever pause once per turn: the answer is recorded so a kick-back
        # through the planner does not re-ask the same question.
        if state.get("authorized"):
            return {}
        question = authorization_request(state.get("plan"), registry,
                                         last_human_text(state))
        if not question:
            return {"authorized": True}
        answer = interrupt({"authorize": question, "plan": state.get("plan")})
        return {"authorized": True, "authorization": str(answer)}

    return authorize
