"""The formatter agent — the "report" phase.

Turns the turn's raw results into the answer the user asked for, honouring the
output contract.  It has no tools, so it cannot invent new work.

It does not run on every turn: :func:`needs_formatting` is a deterministic
predicate.  Executor output is raw tool text and always needs shaping, a failed
verification always needs explaining, and an unresolved visual mismatch has to be
disclosed; a worker turn that already ended in prose with nothing outstanding does
not, so we skip the extra model call there.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from client_v2.agents.conversational import last_human_text, message_text
from client_v2.agents.verifier import turn_texts
from client_v2.prompts import PROMPTS


def needs_formatting(state) -> bool:
    """True when the turn's results still need shaping into a user answer."""
    if state.get("step_outputs") or state.get("step_errors"):
        return True                       # executor ran: raw tool text
    verification = state.get("verification") or {}
    if verification.get("passed") is False:
        return True                       # a failure must be explained
    # An unresolved visual mismatch MUST be reported.  Without this the turn ended
    # at the worker's own message -- written before the visual check ran -- so a
    # real finding ("0 of the reference's 8 underside recesses") was computed,
    # stored, logged and then silently dropped, and the user was shown PASS with a
    # feature list the render contradicted.  It is also exactly the defect engine
    # truth CANNOT catch: the build matched its spec, so every ray passed.
    visual = state.get("visual") or {}
    return visual.get("matched") is False


def turn_report(state) -> str:
    """What the worker itself said this turn, if anything.

    Included in the digest so the formatter can KEEP the concrete facts -- region
    name, dimensions, render paths -- instead of replacing a rich worker report
    with a thin summary built from state alone.  On the worker path
    ``step_outputs`` is empty, so without this the formatter would have almost
    nothing to work from.
    """
    start = state.get("turn_start") or 0
    for msg in reversed((state.get("messages") or [])[start:]):
        if isinstance(msg, AIMessage):
            text = message_text(msg).strip()
            if text:
                return text
    return ""


def results_digest(state, limit: int = 4000) -> str:
    """Compact, plain-text digest of the turn's results for the formatter."""
    lines: list[str] = []
    verification = state.get("verification") or {}
    if verification.get("checked"):
        lines.append("Verification: "
                     + ("PASSED" if verification.get("passed") else "FAILED"))
        for failure in verification.get("failures") or []:
            lines.append(f"  failure: {failure}")
    visual = state.get("visual") or {}
    if visual.get("matched") is False:
        lines.append("Visual comparison with the reference image: MISMATCH, and "
                     "it was NOT resolved.")
        lines.append(f"  difference: {visual.get('detail', '(unstated)')}")
        lines.append("  This is a shape/layout difference from the reference. "
                     "Engine verification only proves the build matches its spec, "
                     "so it cannot cover this. Report it as an outstanding "
                     "discrepancy -- do not describe the model as matching the "
                     "reference.")
    outputs = state.get("step_outputs") or {}
    for name, value in outputs.items():
        lines.append(f"[{name}] {value}")
    for err in state.get("step_errors") or []:
        lines.append(f"[error] {err}")
    report = turn_report(state)
    if report:
        lines.append("\nWhat was done (the agent's own report -- keep its concrete "
                     "values and file paths):")
        lines.append(report)
    return "\n".join(lines)[:limit] or "(no results recorded)"


def make_formatter_node(model):
    """Node that writes the final user-facing answer."""

    async def formatter(state):
        prompt = (f"User asked:\n{last_human_text(state)}\n\n"
                  f"Results:\n{results_digest(state)}")
        reply = await model.ainvoke(
            [SystemMessage(content=PROMPTS.text("formatter")),
             HumanMessage(content=prompt)])
        return {"messages": [reply]}

    return formatter


__all__ = ["make_formatter_node", "needs_formatting", "results_digest",
           "turn_report", "turn_texts"]
