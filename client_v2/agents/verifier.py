"""The verifier agent — the "verify" phase, plus the kick-back decision.

It does NOT re-derive truth itself: the objective checking lives in the
``verify_model_dimensions`` tool (bb + nirt engine truth).  This node *reads* the
outcomes already produced in the turn — engine-truth verdicts and tool errors —
and decides whether the work stands or goes back to the planner for revision.

Key rule: if nothing verifiable happened (no verdict, no error), the turn passes
through.  Only a *contradicted* outcome triggers a kick-back, so a turn that
never verified anything can't spin the loop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import ToolMessage

from client_v2.agents.conversational import message_text

# Max planner revisions per user turn, so a stubborn failure can't loop forever.
MAX_REVISIONS = 2

# "Verification of 'plate.r': FAIL" — emitted by verify_model_dimensions.
_VERDICT = re.compile(r"verification of ['\"]?([\w.]+)['\"]?\s*:\s*(pass|fail)",
                      re.IGNORECASE)
# An explicit failure TAG anywhere in the output means the tool failed.
_ERROR_TAGS = ("[mged_error]",)
# "Error:" only counts when the result STARTS with it -- that is how our tools
# report failure.  Matching it anywhere failed whole turns for a successful build
# that merely mentioned a problem in passing (e.g. one check render timing out).
_ERROR_PREFIXES = ("error:",)


def _is_tool_failure(text: str) -> bool:
    """True if this tool result reports a FAILURE, not merely mentions an error."""
    low = text.lower()
    if any(tag in low for tag in _ERROR_TAGS):
        return True
    first = next((ln.strip().lower() for ln in text.splitlines() if ln.strip()), "")
    return first.startswith(_ERROR_PREFIXES)


@dataclass
class Verdict:
    """Outcome of reading a turn's results."""

    passed: bool = True
    checked: bool = False        # did anything verifiable actually happen?
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "checked": self.checked,
                "failures": list(self.failures)}


def evaluate(texts: list[str]) -> Verdict:
    """Read engine-truth verdicts and error markers out of a turn's outputs.

    Pure.  A FAIL verdict or an error marker fails the turn; a PASS verdict
    marks it checked-and-good; nothing recognisable leaves it untouched
    (``checked=False``, ``passed=True``) so the graph does not loop.
    """
    verdict = Verdict()
    for text in texts:
        if not text:
            continue
        for region, outcome in _VERDICT.findall(text):
            verdict.checked = True
            if outcome.lower() == "fail":
                verdict.passed = False
                verdict.failures.append(
                    f"engine-truth verification failed for {region}")
        if _is_tool_failure(text):
            verdict.checked = True
            verdict.passed = False
            first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
            verdict.failures.append(f"tool reported an error: {first[:160]}")
    return verdict


def turn_texts(state, since: int | None = None) -> list[str]:
    """Result texts produced by THIS turn: step outputs + its tool messages.

    Bounded by ``turn_start`` (set by intake).  Scanning the whole history
    instead made a verdict permanent: one failed turn then failed every
    subsequent turn, because its FAIL was still sitting in the transcript.
    """
    texts: list[str] = []
    outputs = state.get("step_outputs") or {}
    texts.extend(str(v) for v in outputs.values() if v is not None)
    texts.extend(str(e) for e in (state.get("step_errors") or []))
    start = state.get("turn_start") if since is None else since
    for msg in (state.get("messages") or [])[start or 0:]:
        if isinstance(msg, ToolMessage):
            texts.append(message_text(msg))
    return texts


def make_verifier_node():
    """Node that records a verdict for the work just performed."""
    def verifier(state):
        return {"verification": evaluate(turn_texts(state)).as_dict()}
    return verifier


def route_after_verify(state) -> str:
    """Send failed-and-retryable work back to the planner; otherwise finish."""
    verification = state.get("verification") or {}
    if verification.get("passed", True):
        return "done"
    if (state.get("revisions") or 0) >= MAX_REVISIONS:
        return "done"          # out of budget: report what we have
    return "revise"


def failure_context(state) -> str:
    """Human-readable failure summary for the planner's revision prompt."""
    verification = state.get("verification") or {}
    failures = verification.get("failures") or []
    if not failures:
        return ""
    lines = ["The previous attempt FAILED these checks:"]
    lines += [f"  - {f}" for f in failures]
    lines.append("Revise the plan to fix them; do not repeat the same steps "
                 "unchanged.")
    return "\n".join(lines)
