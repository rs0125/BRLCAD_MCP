"""SkillsMiddleware — LangChain "Skills" pattern over our structured registry.

Progressive disclosure: the worker's system prompt always carries a lean
*catalog* of available skills (id + one-line description), and the *full* detail
of a skill is injected only when that skill is active (``state['active_skill']``).
This keeps the base prompt small while letting the agent pull in a skill's steps
/ preconditions / criteria exactly when a request calls for it.

Implemented with LangChain 1.x's ``@dynamic_prompt`` middleware so it composes
into ``create_agent``.  Nothing sets ``active_skill`` yet (the planner does, a
later increment); until then only the catalog is injected, which is the correct
Phase-3b behavior.
"""

from __future__ import annotations

import json

from langchain.agents.middleware import dynamic_prompt

from client_v2.prompts import resolve
from client_v2.skills import SkillRegistry


def render_plan(plan) -> str:
    """The planner's plan as an ordered, readable list of steps."""
    steps = (plan or {}).get("steps") or []
    lines = []
    for i, step in enumerate(steps, 1):
        skill = step.get("skill", "?")
        line = f"  {i}. {skill}"
        params = step.get("params") or {}
        if params:
            line += f"  params: {json.dumps(params, default=str)[:400]}"
        if step.get("why"):
            line += f"\n     why: {step['why']}"
        lines.append(line)
    done = (plan or {}).get("done_when")
    if done:
        lines.append(f"  Done when: {done}")
    return "\n".join(lines)


def plan_skill_ids(plan) -> list[str]:
    """Distinct skill ids named by the plan, in the order they first appear."""
    out: list[str] = []
    for step in (plan or {}).get("steps") or []:
        skill = step.get("skill")
        if skill and skill not in out:
            out.append(skill)
    return out


def compose_worker_prompt(base_prompt, registry: SkillRegistry,
                          active_skill: str | None = None,
                          plan=None) -> str:
    """Base worker prompt + skill catalog + the plan (or a single active skill).

    When the planner produced a plan, the worker is given the plan itself AND the
    full definition of every skill it names, in order.  Previously the plan was
    dropped here and only one skill's detail was injected -- the plan's terminal
    step, chosen arbitrarily -- so the planner's ordering and bound parameters
    were thrown away and the worker re-decided everything.

    ``base_prompt`` may be text, a callable, or None for the prompt library's
    ``worker`` entry.  Resolving it HERE, on every model call, is what lets a
    ``/reload`` of an edited prompt file take effect mid-session.

    Falls back to a single ``active_skill`` when there is no usable plan.  Pure
    apart from that lookup.
    """
    parts = [
        resolve(base_prompt, "worker").strip(),
        "",
        "## Available skills",
        "These are structured procedures you can follow. The catalog below is "
        "metadata only; when a request matches a skill, follow its definition.",
        registry.catalog(),
    ]
    planned = plan_skill_ids(plan)
    if planned:
        parts += ["", "## Plan to follow",
                  "The planner produced this plan for the current request. "
                  "Follow it in order, using the parameters it bound, unless "
                  "something in the results makes a step impossible.",
                  render_plan(plan)]
        details = [d for d in (registry.detail(sid) for sid in planned) if d]
        if details:
            parts += ["", "## Definitions of the planned skills"] + details
    elif active_skill:
        detail = registry.detail(active_skill)
        if detail:
            parts += ["", "## Active skill (follow this)", detail]
    return "\n".join(parts)


def make_skills_middleware(registry: SkillRegistry, base_prompt=None):
    """Build a dynamic-prompt middleware that injects skills via the registry."""

    @dynamic_prompt
    def _skills_prompt(request) -> str:
        state = getattr(request, "state", None) or {}
        if not hasattr(state, "get"):
            return compose_worker_prompt(base_prompt, registry)
        return compose_worker_prompt(base_prompt, registry,
                                     state.get("active_skill"),
                                     state.get("plan"))

    return _skills_prompt
