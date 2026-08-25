"""The authorize phase -- a real pause, not a request to please pause.

A LangGraph ``interrupt`` cannot be ignored: the graph genuinely halts, the caller
(REPL or harness) surfaces the question, and execution resumes with the answer.
The trigger is data, not a new schema field -- a skill that needs a decision says
so with an ``{authorize: "..."}`` entry in its ``steps``.

When a halt is justified
------------------------
Only two things earn one: the action is HARD TO UNDO, or it needs information ONLY
THE USER HAS.  Wiping a database is the first.  Picking a lighting variant is the
second.  Everything else must not stop, because over-gating trains people to say
yes and that costs us the gates that matter.

Three findings behind the current shape:

* Prose is ignorable.  "Stop and ask the user" in a skill definition was ignored
  outright -- ``render_beauty`` ran all three of its staged renders without
  waiting.  Hence an interrupt rather than an instruction.
* Skill-declared gates only protect work we already modelled, which inverts the
  risk.  "delete everything in this db" planned as ``{"steps": []}``, so no skill
  could declare a gate, and a wipe ran unconfirmed -- while unmodelled requests
  are exactly the ones reaching raw ``execute_command`` with no preconditions and
  no verifier.  Hence the request-level check below.
* Building from a drawing earns NO gate, and one was briefly added by mistake.
  A build writes a versioned spec and ``undo_build`` reverts it, and the
  dimensions are printed on the drawing -- so it is neither irreversible nor
  privately known.  What actually catches a misread drawing is ``expect_bbox``
  (machine-checked, so the build is REJECTED rather than rubber-stamped) and a
  ground-truth eval case.
"""

from __future__ import annotations

import json
import re

from langgraph.types import interrupt

from client_v2.agents.conversational import last_human_text
from client_v2.skills import SkillRegistry
from client_v2.skills.middleware import plan_skill_ids

# Both halves must match: "delete the mounting hole" is ordinary editing and must
# NOT nag, while "delete everything" must stop.
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


def _declared_gate(registry: SkillRegistry, skill_ids) -> str | None:
    """The first ``authorize`` step declared by any of *skill_ids*."""
    for skill_id in skill_ids:
        skill = registry.get(skill_id)
        if skill is None:
            continue
        for step in skill.steps:
            if isinstance(step, dict) and step.get("authorize"):
                return str(step["authorize"])
    return None


def authorization_request(plan, registry: SkillRegistry,
                          request: str = "",
                          active_skill: str | None = None) -> str | None:
    """The question that must be answered before this turn acts, or None.

    A planned skill's ``authorize`` step first, then the fallback's single
    skill, then a wholesale-destructive request.  See the module docs for what
    does and does not justify a halt.

    ``active_skill`` matters because the planner has a fallback: when its reply
    yields no usable plan it returns ``plan=None`` with a single chosen skill
    instead.  Consulting only the plan meant that path skipped the skill's
    declared gate entirely -- and an unplanned turn is the one with the least
    scrutiny elsewhere, so skipping it there is exactly backwards.
    """
    gate = _declared_gate(registry, plan_skill_ids(plan))
    if gate:
        return gate
    if active_skill:
        gate = _declared_gate(registry, [active_skill])
        if gate:
            return gate
    return WIPE_QUESTION if broad_destructive(request) else None


def describe_pause(value) -> str:
    """The halt, phrased so it can actually be answered.

    An ``{authorize: "..."}`` entry in a skill is written for the agent, not
    for the person at the prompt: ``model_from_dimensioned_sketch`` says
    "confirm the dimensioned plan with the user", which read on its own asks
    someone to approve a plan they have not been shown.  The interrupt payload
    already carries the plan, so the numbers being approved are appended to
    the question rather than left in the state.
    """
    if not isinstance(value, dict):
        return str(value)
    question = str(value.get("authorize") or "").strip()
    steps = ((value.get("plan") or {}).get("steps") or [])
    if not steps:
        return question

    lines = [question, "", "Plan:"]
    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            lines.append(f"  {i}. {step}")
            continue
        skill = step.get("skill") or step.get("phase") or "step"
        lines.append(f"  {i}. {skill}")
        if step.get("why"):
            lines.append(f"       {step['why']}")
        params = step.get("params")
        if params:
            # Compact, and truncated: this is a prompt, not a dump.  The point
            # is that the dimensions are visible before they are committed to.
            rendered = json.dumps(params, separators=(", ", ": "))
            if len(rendered) > 300:
                rendered = rendered[:300] + " ..."
            lines.append(f"       {rendered}")
    return "\n".join(lines)


def make_authorize_node(registry: SkillRegistry):
    """Node that halts for a decision when the plan calls for one."""

    def authorize(state):
        # Pause once per turn: the answer is recorded so a kick-back through the
        # planner does not re-ask the same question.
        if state.get("authorized"):
            return {}
        question = authorization_request(state.get("plan"), registry,
                                         last_human_text(state),
                                         state.get("active_skill"))
        if not question:
            return {"authorized": True}
        answer = interrupt({"authorize": question, "plan": state.get("plan")})
        return {"authorized": True, "authorization": str(answer)}

    return authorize
