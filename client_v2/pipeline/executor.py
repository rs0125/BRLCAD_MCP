"""The executor — runs a plan's skills one after another (the "execute" phase).

For a plan whose skills are deterministically executable (their ``steps`` are
``call``/``bind`` directives, not model-judgment phases), the executor runs them
WITHOUT the LLM: it resolves each step's parameters (binding ``${ref}`` values
from prior step outputs), calls the named tool, and carries outputs forward.
That's the reliability win — deterministic where we can be.

Plans that aren't deterministically executable (e.g. the image workflow, whose
sub-skills need vision/judgment) fall through to the model-driven worker
instead; the router decides which path a plan takes.

Failure handling is intentionally a thin SEAM here (record the error and stop):
the diagnose/recover/retry loop is the verifier slice's job, and it drops in at
the marked point.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from client_v2.pipeline.plan import Plan
from client_v2.skills import SkillRegistry

_EXACT_REF = re.compile(r"^\$\{([^}]+)\}$")
_INLINE_REF = re.compile(r"\$\{([^}]+)\}")


def _lookup(context: dict, path: str):
    """Resolve a dotted path (``a.b``) against *context*; None if absent."""
    cur = context
    for part in path.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def resolve_params(value, context: dict):
    """Substitute ``${ref}`` templates in *value* from *context* (pure, recursive).

    A whole-string ``"${a.b}"`` yields the referenced value (any type); an
    inline ``"x=${a}"`` yields a string with the ref substituted.  Dicts/lists
    recurse; other values pass through unchanged.
    """
    if isinstance(value, str):
        exact = _EXACT_REF.match(value.strip())
        if exact:
            return _lookup(context, exact.group(1))
        return _INLINE_REF.sub(
            lambda m: str(_lookup(context, m.group(1)) or ""), value)
    if isinstance(value, dict):
        return {k: resolve_params(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve_params(v, context) for v in value]
    return value


def _is_executable_skill(skill) -> bool:
    """True if a skill's steps are all call/bind directives with at least one call."""
    steps = skill.steps
    if not steps:
        return False
    if not all(isinstance(s, dict) and ("call" in s or "bind" in s)
               for s in steps):
        return False
    return any("call" in s for s in steps)


def is_deterministic(plan: Plan, registry: SkillRegistry) -> bool:
    """True if every step's skill can be run without the model."""
    if not plan or not plan.steps:
        return False
    return all(
        (skill := registry.get(step.skill)) is not None
        and _is_executable_skill(skill)
        for step in plan.steps)


def _as_text(result) -> str:
    """Readable text from a tool result.

    MCP tools return a list of content blocks (``[{'type': 'text', 'text': ...}]``);
    stringifying that leaks Python reprs into the verifier and the user-facing
    answer, so pull the text out.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("text", result))
    if isinstance(result, list):
        parts = [_as_text(item) for item in result]
        return "\n".join(p for p in parts if p)
    return str(result)


def make_executor_node(registry: SkillRegistry, tools):
    """Node that runs a deterministic plan step-by-step over *tools*."""
    dispatch = {getattr(t, "name", None): t for t in tools}

    async def executor(state):
        plan = Plan.model_validate(state["plan"])
        outputs: dict = {}
        errors: list[str] = []
        lines: list[str] = []
        for step in plan.steps:
            skill = registry.get(step.skill)
            context = {**outputs, **step.params}
            last = None
            for directive in skill.steps:
                if "call" in directive:
                    tool_name = directive["call"]
                    tool = dispatch.get(tool_name)
                    if tool is None:
                        # Stop here and record it; the verifier turns this into
                        # a kick-back to the planner (bounded).
                        msg = f"{step.skill}: tool '{tool_name}' unavailable"
                        errors.append(f"Error: {msg}")
                        lines.append(f"[x] {msg}")
                        return {
                            "messages": [AIMessage(content="\n".join(lines))],
                            "step_outputs": outputs, "step_errors": errors}
                    args = resolve_params(directive.get("with", {}), context)
                    last = _as_text(await tool.ainvoke(args))
                    context = {**context, "result": last}
                    lines.append(f"[ok] {step.skill}: {tool_name}")
                elif "bind" in directive:
                    for name in directive["bind"]:
                        outputs[f"{step.skill}.{name}"] = last
            if last is not None:
                outputs[step.skill] = last
        lines.append(f"Plan complete ({len(plan.steps)} step(s)).")
        return {"messages": [AIMessage(content="\n".join(lines))],
                "step_outputs": outputs, "step_errors": errors}

    return executor
