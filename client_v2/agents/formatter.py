"""The formatter agent — the "report" phase.

Turns the turn's raw results into the answer the user asked for, honouring the
output contract.  It has no tools, so it cannot invent new work.

It does not run on every turn: :func:`needs_formatting` is a deterministic
predicate.  Executor output is raw tool text and always needs shaping, and a
failed verification always needs explaining; a worker turn that already ended in
prose does not, so we skip the extra model call there.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from client_v2.agents.conversational import last_human_text
from client_v2.agents.verifier import turn_texts
from client_v2.prompts import FORMATTER_SYSTEM


def needs_formatting(state) -> bool:
    """True when the turn's results still need shaping into a user answer."""
    if state.get("step_outputs") or state.get("step_errors"):
        return True                       # executor ran: raw tool text
    verification = state.get("verification") or {}
    return verification.get("passed") is False   # a failure must be explained


def route_after_verify_to_report(state) -> str:
    """Post-verify routing: format when needed, else finish."""
    return "format" if needs_formatting(state) else "done"


def results_digest(state, limit: int = 4000) -> str:
    """Compact, plain-text digest of the turn's results for the formatter."""
    lines: list[str] = []
    verification = state.get("verification") or {}
    if verification.get("checked"):
        lines.append("Verification: "
                     + ("PASSED" if verification.get("passed") else "FAILED"))
        for failure in verification.get("failures") or []:
            lines.append(f"  failure: {failure}")
    outputs = state.get("step_outputs") or {}
    for name, value in outputs.items():
        lines.append(f"[{name}] {value}")
    for err in state.get("step_errors") or []:
        lines.append(f"[error] {err}")
    return "\n".join(lines)[:limit] or "(no results recorded)"


def make_formatter_node(model):
    """Node that writes the final user-facing answer."""

    async def formatter(state):
        prompt = (f"User asked:\n{last_human_text(state)}\n\n"
                  f"Results:\n{results_digest(state)}")
        reply = await model.ainvoke(
            [SystemMessage(content=FORMATTER_SYSTEM),
             HumanMessage(content=prompt)])
        return {"messages": [reply]}

    return formatter


__all__ = ["make_formatter_node", "needs_formatting", "results_digest",
           "route_after_verify_to_report", "turn_texts"]
