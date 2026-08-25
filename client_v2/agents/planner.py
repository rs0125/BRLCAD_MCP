"""The planner agent -- the "plan" phase.

Given a work request and the skill catalog, the planner produces an ORDERED,
PARAMETERIZED plan (a :class:`~client_v2.pipeline.plan.Plan`): which skills to
run, in what order, with which inputs.  The plan goes into ``state['plan']``,
which both consumers read: the executor runs it directly when every step is
callable, otherwise the SkillsMiddleware injects it -- plus the definitions of
every skill it names -- into the worker's prompt.  If no usable plan comes back,
:func:`choose_skill` sets a single ``active_skill`` so the worker gets a hint
rather than nothing.  Parsing and validation are pure, in
:mod:`client_v2.pipeline.plan`.

Two things the planner MUST see, both learned the hard way
---------------------------------------------------------
* **Everything that constrains a parameter value.**  The planner writes the
  params, so a caution living only in the worker's view cannot influence what
  gets built -- a through-hole caution was invisible to it for exactly that
  reason.  Hence the single canonical ``SkillDef.planning_view`` rather than
  fields formatted by hand here, which silently omits whatever nobody remembered
  to add.
* **A short tail of the conversation.**  With only the latest human line, a bare
  approval was unplannable: on "yeah go ahead" -- the turn that actually builds,
  after dimensions have been proposed -- the planner replied *"No actionable
  model, dimensions, reference, or approved constraints are present in the
  available conversation"* and returned an empty plan.  No plan meant no skill
  definition reached the worker, so ``build_model_spec``'s cautions were absent on
  the very turn that built the geometry; a missing ``expect_bbox`` was the visible
  symptom.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from client_v2.agents.conversational import last_human_text, message_text
from client_v2.agents.verifier import failure_context
from client_v2.agents.visual import visual_failure_context
from client_v2.pipeline.plan import parse_plan
from client_v2.prompts import PROMPTS
from client_v2.skills import SkillRegistry

MAX_CONTEXT_TURNS = 3
# Per message, kept from the TAIL (see ``_tail``).  1500 clipped a real 1,960-char
# proposal mid-list and lost the approval question at its end; 2400 covers that
# with headroom.  Deliberately not larger: this multiplies by MAX_CONTEXT_TURNS*2
# messages and the planner call is already the most expensive in a turn.
MAX_CONTEXT_CHARS = 2400


def planning_brief(registry: SkillRegistry) -> str:
    """The skills the planner may use, each rendered for parameter binding.

    Delegates to ``SkillDef.planning_view`` -- the canonical rendering of what a
    planner needs (inputs, preconditions, cautions, examples).  See module docs.
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


def conversation_context(state, turns: int = MAX_CONTEXT_TURNS,
                         limit: int = MAX_CONTEXT_CHARS) -> str:
    """The last few exchanges, so the planner can plan a FOLLOW-UP turn.

    A short tail, not the transcript: enough to know what was agreed, not enough
    to grow the planning call with the session.  Images are dropped --
    ``message_text`` keeps text blocks only -- so a reference image is never
    resent here.  See module docs for why this exists.
    """
    messages = state.get("messages") or []
    lines: list[str] = []
    for msg in messages[-turns * 2:]:
        role = ("User" if isinstance(msg, HumanMessage)
                else "Assistant" if isinstance(msg, AIMessage) else None)
        if role is None:
            continue                      # tool traffic is the worker's business
        text = message_text(msg).strip()
        if text:
            lines.append(f"{role}: {_tail(text, limit)}")
    return "\n".join(lines[-turns * 2:])


def _tail(text: str, limit: int) -> str:
    """The LAST *limit* characters of *text*, marked when clipped.

    Truncating from the head dropped the end of a long message, which is exactly
    where the decision lives.  Measured: a 1,960-char proposal lost the trailing
    460 chars carrying "may I use 1.6 mm as the underside perimeter-wall
    thickness?" -- the question the user had answered yes to.  The planner, seeing
    only the constraint list, then asserted the drawing's conflicting 1.2 mm, and
    nothing caught it because the plan is advice rather than a contract.
    """
    if len(text) <= limit:
        return text
    return "…" + text[-limit:]


def make_planner_node(model, registry: SkillRegistry):
    """Node that plans a skill sequence for the request."""

    async def planner(state):
        request = last_human_text(state)
        parts = [f"Skills:\n{planning_brief(registry)}"]
        # What was already said, so a bare approval ("yeah go ahead") is planable.
        history = conversation_context(state)
        if history:
            parts.append(f"Conversation so far:\n{history}")
        parts.append(f"User request:\n{request}")
        # On a kick-back, plan again knowing what failed -- either an
        # engine-truth check or the visual comparison with the reference.
        context = failure_context(state) or visual_failure_context(state)
        if context:
            parts.append(context)
        reply = await model.ainvoke(
            [SystemMessage(content=PROMPTS.text("planner")),
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
                    "revisions": revisions,
                    "plan_errors": []}

        # Fallback: no usable plan -> single-skill hint (or clear a stale pick).
        #
        # Record WHY.  Without this the fallback is indistinguishable from a
        # successful pass in the run log -- a deliberate {"steps":[]} and a
        # parse failure both showed up as a plan of null with no reason, and
        # the worker then ran unplanned with nothing saying so.
        why = list(errors) if errors else (
            ["planner returned no steps"] if plan is not None else
            ["planner reply did not parse as a plan"])
        return {"plan": None, "active_skill": choose_skill(text, registry.ids()),
                "revisions": revisions, "plan_errors": why}

    return planner
