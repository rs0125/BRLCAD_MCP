"""Plan schema + validation/parsing for the parameterized planner.

A *plan* is an ordered list of skill invocations with bound parameters — the
planner's structured output.  This module is pure (no model, no graph): it
defines the schema, validates a plan against the skill registry (every step
names a known skill and supplies that skill's required inputs), and parses a
model reply into a plan.  The executor (next slice) consumes it.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from client_v2.skills import SkillRegistry


class PlanStep(BaseModel):
    """One planned skill invocation."""

    skill: str
    params: dict = Field(default_factory=dict)
    why: str = ""


class Plan(BaseModel):
    """An ordered sequence of steps plus a success summary."""

    steps: list[PlanStep] = Field(default_factory=list)
    done_when: str = ""


def validate_plan(plan: Plan, registry: SkillRegistry) -> list[str]:
    """Return a list of problems with *plan* (empty == valid).

    Checks each step names a known skill and provides that skill's REQUIRED
    inputs; unknown skills and missing params are the two failure modes that
    would otherwise blow up at execution time.
    """
    errors: list[str] = []
    if not plan.steps:
        errors.append("plan has no steps")
    for i, step in enumerate(plan.steps, 1):
        skill = registry.get(step.skill)
        if skill is None:
            errors.append(f"step {i}: unknown skill '{step.skill}'")
            continue
        required = [p.name for p in skill.io.inputs if p.required]
        missing = [r for r in required if r not in step.params]
        if missing:
            errors.append(
                f"step {i} ({step.skill}): missing required param(s): "
                f"{', '.join(missing)}")
    return errors


def _extract_json_object(text: str) -> str | None:
    """Slice the outermost ``{...}`` from *text* (tolerates prose / code fences)."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    return text[start:end + 1]


def parse_plan(text: str, registry: SkillRegistry) -> tuple[Plan | None, list[str]]:
    """Parse a model reply into a validated Plan.

    Returns ``(plan, errors)``.  ``plan`` is None if no JSON plan could be read;
    otherwise ``errors`` carries any registry-validation problems (the caller
    decides whether to run, replan, or fall back).
    """
    raw = _extract_json_object(text)
    if raw is None:
        return None, ["no JSON plan found in reply"]
    try:
        plan = Plan.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        return None, [f"plan is not valid JSON/schema: {exc}"]
    return plan, validate_plan(plan, registry)
