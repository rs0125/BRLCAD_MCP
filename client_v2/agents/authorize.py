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
from client_v2.terminal.attachments import attached_image_count

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

# Tools that create or change geometry.  A planned skill that calls one of these
# is about to build, which is what makes the reference-image gate below apply.
# Keyed off the skill's own `call` steps rather than a list of skill ids, so a new
# building skill is covered without editing this module.
GEOMETRY_TOOLS = frozenset({"build_from_spec", "edit_build",
                            "boolean_combination"})

SKETCH_QUESTION = (
    "Before I build from that image: confirm the dimensions I read off it are "
    "right, or correct them. A drawing rarely states everything, so some values "
    "will be assumptions -- building on a misread number produces a model that "
    "is internally consistent and still wrong."
)


def turn_has_reference_image(state) -> bool:
    """True if the user attached an image on THIS turn.

    ``turn_start`` is set by intake to ``len(messages)`` once the new human
    message is already in state, so that message sits at ``turn_start - 1``.

    Scoped to the current turn on purpose, not the whole conversation: once the
    numbers have been confirmed, later turns must not keep re-asking about the
    same drawing.
    """
    messages = state.get("messages") or []
    start = state.get("turn_start") or 0
    if not (0 < start <= len(messages)):
        return False
    return bool(attached_image_count(messages[start - 1]))


def builds_geometry(skill) -> bool:
    """True if this skill's steps call a geometry-mutating tool."""
    for step in getattr(skill, "steps", []) or []:
        if isinstance(step, dict) and step.get("call") in GEOMETRY_TOOLS:
            return True
    return False


def plan_builds_geometry(plan, registry: SkillRegistry) -> bool:
    """True if any planned skill would create or change geometry."""
    return any(builds_geometry(skill) for skill in
               (registry.get(sid) for sid in plan_skill_ids(plan))
               if skill is not None)


def broad_destructive(text: str) -> bool:
    """True if *text* asks to destroy geometry wholesale rather than one feature."""
    return bool(text) and bool(_DESTRUCTIVE_VERB.search(text)) \
        and bool(_BROAD_SCOPE.search(text))


def authorization_request(plan, registry: SkillRegistry,
                          request: str = "",
                          new_reference_image: bool = False) -> str | None:
    """The question that must be answered before this turn acts, or None.

    Three sources, in order of specificity:

    1. A planned skill's ``authorize`` step -- a workflow defining its own pause.
    2. A request to destroy geometry wholesale, which no skill covers.
    3. Building geometry straight from a reference image the user just attached,
       with no chance to check the numbers first.  Confirming dimensions before
       modelling a drawing is the point of the workflow, and it was previously
       enforced by nothing: only ``model_from_dimensioned_sketch`` declares an
       authorize step, and a planner that picked its sub-skills directly -- as it
       did -- bypassed the gate entirely.  It happened to pause anyway, because
       the model chose to, which is not a safety property.
    """
    for skill_id in plan_skill_ids(plan):
        skill = registry.get(skill_id)
        if skill is None:
            continue
        for step in skill.steps:
            if isinstance(step, dict) and step.get("authorize"):
                return str(step["authorize"])
    if broad_destructive(request):
        return WIPE_QUESTION
    if new_reference_image and plan_builds_geometry(plan, registry):
        return SKETCH_QUESTION
    return None


def make_authorize_node(registry: SkillRegistry):
    """Node that halts for a decision when the plan calls for one."""

    def authorize(state):
        # Only ever pause once per turn: the answer is recorded so a kick-back
        # through the planner does not re-ask the same question.
        if state.get("authorized"):
            return {}
        question = authorization_request(
            state.get("plan"), registry, last_human_text(state),
            new_reference_image=turn_has_reference_image(state))
        if not question:
            return {"authorized": True}
        answer = interrupt({"authorize": question, "plan": state.get("plan")})
        return {"authorized": True, "authorization": str(answer)}

    return authorize
