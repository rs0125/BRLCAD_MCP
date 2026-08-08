"""The verifier agent — the "verify" phase, plus the kick-back decision.

It does NOT re-derive truth itself: the objective checking lives in the
``verify_model_dimensions`` tool (bb + nirt engine truth).  This node *reads* the
outcomes already produced in the turn — engine-truth verdicts and tool errors —
and decides whether the work stands or goes back to the planner for revision.

Key rule: if nothing verifiable happened (no verdict, no error), the turn passes
through.  Only a *contradicted* outcome triggers a kick-back, so a turn that
never verified anything can't spin the loop.

Why results are read IN ORDER
-----------------------------
An engine-truth PASS supersedes what went wrong before it -- every tool error so
far, plus an earlier FAIL for the same region.  Retry-then-succeed is the most
common shape of a successful turn: a build failed because no database was open,
the agent found ``opendb``, retried, and the region then verified PASS with an
exact bounding box.  Read without regard to order, that turn failed on its own
first attempt twelve times running while the proof of success sat in the same
transcript.

A PASS clears only what PRECEDED it, so an error after the last verdict still
fails the turn.  Known limit: one BuildSpec is one region, so a turn building two
regions could have the second one's PASS clear the first one's tool error.  FAIL
verdicts are cleared per-region; tool errors, which name no region, are not.

Two more scars worth keeping in mind: ``[MGED_ERROR]`` counts anywhere in a
result, but ``Error:`` only at the start of one -- matching it loosely failed
whole successful builds that merely mentioned a problem in passing, such as a
check render timing out.
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
# An explicit failure TAG anywhere in a result means the tool failed; a bare
# "Error:" only counts at the start of one (see module docs).
_ERROR_TAGS = ("[mged_error]",)
_ERROR_PREFIXES = ("error:",)
# Failure-message prefixes, so a later PASS can identify what it supersedes.
_TOOL_ERROR = "tool reported an error: "


def _fail_msg(region: str) -> str:
    return f"engine-truth verification failed for {region}"


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _is_tool_failure(text: str) -> bool:
    """True if this tool result reports a FAILURE, not merely mentions an error."""
    if any(tag in text.lower() for tag in _ERROR_TAGS):
        return True
    return _first_line(text).lower().startswith(_ERROR_PREFIXES)


@dataclass
class Verdict:
    """Outcome of reading a turn's results."""

    passed: bool = True
    checked: bool = False        # did anything verifiable actually happen?
    failures: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"passed": self.passed, "checked": self.checked,
                "failures": list(self.failures)}


def signals(texts: list[str]) -> list[tuple[str, str]]:
    """Ordered ``(kind, detail)`` outcomes read from a turn's results.

    *kind* is ``"pass"``/``"fail"`` (detail = the region) or ``"error"`` (detail =
    the offending first line).  Within one result the verdicts come before the
    error, matching how a tool writes them.  Pure reading, no judgement.
    """
    out: list[tuple[str, str]] = []
    for text in texts:
        if not text:
            continue
        out += [(outcome.lower(), region)
                for region, outcome in _VERDICT.findall(text)]
        if _is_tool_failure(text):
            out.append(("error", _first_line(text)[:160]))
    return out


def _surviving(failures: list[str], passed_region: str) -> list[str]:
    """The failures a PASS for *passed_region* does NOT supersede."""
    superseded = _fail_msg(passed_region)
    return [f for f in failures
            if not f.startswith(_TOOL_ERROR) and f != superseded]


def evaluate(texts: list[str]) -> Verdict:
    """Judge a turn from its ordered signals -- see module docs for why order matters.

    Pure.  A FAIL verdict or an error fails the turn; a PASS supersedes what came
    before it; nothing recognisable leaves the verdict untouched
    (``checked=False``, ``passed=True``) so the graph does not loop.
    """
    verdict = Verdict()
    for kind, detail in signals(texts):
        verdict.checked = True
        if kind == "fail":
            verdict.failures.append(_fail_msg(detail))
        elif kind == "error":
            verdict.failures.append(f"{_TOOL_ERROR}{detail}")
        else:                       # a pass proves the region is right NOW
            verdict.failures = _surviving(verdict.failures, detail)
    verdict.passed = not verdict.failures
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


def revision_budget_spent(state) -> bool:
    """True once this turn has used its planner revisions.

    EVERY edge that loops back to the planner must consult this.  The visual
    path did not, so its own counter was the only thing bounding it -- and when
    that counter stopped advancing the graph revised twelve times and died on the
    recursion limit, with ``revisions`` sitting at 12 and no authority to stop it.
    """
    return (state.get("revisions") or 0) >= MAX_REVISIONS


def route_after_verify(state) -> str:
    """Send failed-and-retryable work back to the planner; otherwise finish."""
    verification = state.get("verification") or {}
    if verification.get("passed", True):
        return "done"
    if revision_budget_spent(state):
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
