"""Visual check — show the agent its own renders and let it compare.

Engine truth (``verify_model_dimensions``) already proves the build matches the
spec.  What it cannot judge is whether the SPEC resembles the reference the user
attached: "is this the right shape at all" is inherently a looking-at-pictures
question.  So after a successful build this node attaches the renders that were
just produced and asks the model to compare them with the reference image still
in the conversation.

Deliberately narrow, because a vision check is the least reliable link in the
chain and must not become load-bearing:

* it runs ONLY when renders exist AND a reference image is in the history, so
  text-driven work never pays for it;
* a mismatch is reported and can trigger ONE bounded revision, not an open loop;
* it can never overturn an engine-truth PASS into silent success -- correctness
  stays with the rays, and this only adds a fidelity opinion.

Three rules that keep the loop honest
-------------------------------------
* **Only the latest RENDERS are judged.**  ``find_render_paths`` takes the newest
  render set, not the first one in the history.  Reading first-seen graded a
  pre-edit render: the check reported holes that a corrective ``edit_build`` had
  already removed, and the formatter passed that on as an unresolved discrepancy
  on a model that was correct.
* **Only the latest reference image is sent**, never the transcript.  Passing the
  history resent every image attached in the session on every check, growing
  without bound (the worker's summarisation middleware does not reach out here),
  and the older references are not what we are comparing against.
* **Routing turns on an explicit fact, not a counter.**  The router once compared
  ``visual_rounds`` with ``>`` where the node used ``>=``; because the node stops
  incrementing once its budget is spent, the counter froze one short of the
  router's test, which therefore never fired, and the graph revised on a stale
  verdict until it hit the recursion limit.  The node now stamps ``spent`` on a
  mismatch it cannot re-judge, and the router reads only that.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from client_v2.agents.conversational import message_text
from client_v2.prompts import PROMPTS
from client_v2.terminal.attachments import attached_image_count, image_part_from_file

# One correction round per turn: a vision judgement is the weakest signal here, so
# it gets far less budget than the engine-truth loop.
MAX_VISUAL_ROUNDS = 1
# More pictures crowd the comparison rather than helping.
MAX_RENDERS = 4

_PNG = re.compile(r"/[^\s'\"]+\.png")


def find_render_paths(state) -> list[str]:
    """PNG paths from the LATEST render in this turn that exist on disk.

    Newest-first, and only one render's worth.  Taking first-seen paths instead
    graded stale geometry: a build followed by a corrective ``edit_build`` leaves
    two render directories in the history, and the older set still shows the
    feature the edit removed -- reported as a mismatch against a model that was
    already correct.  One message is the right unit because a single build emits
    its whole view set in one tool result.
    """
    texts: list[str] = [str(v) for v in (state.get("step_outputs") or {}).values()]
    texts += [message_text(m) for m in (state.get("messages") or [])
              if isinstance(m, ToolMessage)]
    for text in reversed(texts):
        found = [p for p in dict.fromkeys(_PNG.findall(text)) if os.path.isfile(p)]
        if found:
            return found
    return []


def reference_message(state):
    """The most recent user message carrying an image, or None (see module docs)."""
    for msg in reversed(state.get("messages") or []):
        if attached_image_count(msg):
            return msg
    return None


def has_reference_image(state) -> bool:
    """True if the user attached an image somewhere in the conversation."""
    return reference_message(state) is not None


def parse_visual_verdict(reply: str) -> tuple[bool, str]:
    """(matched, detail) from a verdict reply.

    Anything that is not a clear MISMATCH counts as a match: an unparseable
    opinion must not be able to fail a build that engine truth already passed.
    """
    text = (reply or "").strip()
    first = text.splitlines()[0].strip() if text else ""
    if first.upper().startswith("MISMATCH"):
        detail = first.split(":", 1)[1].strip() if ":" in first else text
        return False, detail or "the render does not match the reference"
    return True, first or "matches the reference"


def make_visual_check_node(model):
    """Node that compares this turn's renders against the reference image."""

    async def visual_check(state):
        prior = state.get("visual")
        # One place decides whether a fresh opinion is possible; if it is not, a
        # prior mismatch is stamped `spent` so it cannot be routed on twice.
        can_judge = (bool(renders := find_render_paths(state))
                     and has_reference_image(state)
                     and (state.get("visual_rounds") or 0) < MAX_VISUAL_ROUNDS)
        if not can_judge:
            return {"visual": {**prior, "spent": True}} if prior else {}
        parts: list[dict] = [{"type": "text", "text": (
            "Here are the renders of what was just built. Compare them with the "
            "reference image above.")}]
        parts += [image_part_from_file(Path(p)) for p in renders[:MAX_RENDERS]]
        reply = await model.ainvoke([
            SystemMessage(content=PROMPTS.text("visual")),
            reference_message(state),         # just the reference, not the history
            HumanMessage(content=parts),
        ])
        matched, detail = parse_visual_verdict(message_text(reply))
        return {"visual": {"matched": matched, "detail": detail,
                           "renders": renders[:MAX_RENDERS]},
                "visual_rounds": (state.get("visual_rounds") or 0) + 1}

    return visual_check


def route_after_visual(state) -> str:
    """Send a LIVE visual mismatch back to the planner; otherwise carry on.

    Reads no counter, on purpose -- see module docs.
    """
    visual = state.get("visual") or {}
    if visual.get("matched", True) or visual.get("spent"):
        return "ok"                          # matched, or already acted on
    return "revise"


def visual_failure_context(state) -> str:
    """A revision instruction derived from the visual mismatch."""
    visual = state.get("visual") or {}
    if visual.get("matched", True):
        return ""
    return ("The build matches its spec, but comparing the renders with the "
            f"reference shows a difference: {visual.get('detail', '')}\n"
            "Revise the spec so the shape matches the reference.")
