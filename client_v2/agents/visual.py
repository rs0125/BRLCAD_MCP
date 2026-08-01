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
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from client_v2.agents.conversational import message_text
from client_v2.terminal.attachments import attached_image_count, image_part_from_file

# One visual correction round per user turn: a vision judgement is the weakest
# signal here, so it gets far less budget than the engine-truth loop.
MAX_VISUAL_ROUNDS = 1
# Renders to attach; more pictures crowd the comparison rather than helping.
MAX_RENDERS = 4

_PNG = re.compile(r"/[^\s'\"]+\.png")

VISUAL_SYSTEM = """\
You are checking whether a built 3D model matches the reference image the user
provided. You have no tools.

Compare the attached render(s) with the reference. Judge SHAPE and LAYOUT only --
overall form, feature positions, and how many of each repeated feature there are.
Ignore colour, lighting, materials and image quality; the renders are untextured
previews.

COUNT any repeated features (studs, holes, bolts, tiers) in the render AND in the
reference, and state both numbers. Do not say it looks right without counting.

Answer on the first line with exactly one of:
  MATCH
  MISMATCH: <the single most important difference, and what to change>
Then at most two short sentences of justification.
"""


def find_render_paths(state) -> list[str]:
    """PNG paths produced during this turn that exist on disk."""
    texts: list[str] = [str(v) for v in (state.get("step_outputs") or {}).values()]
    texts += [message_text(m) for m in (state.get("messages") or [])
              if isinstance(m, ToolMessage)]
    found: list[str] = []
    for text in texts:
        for path in _PNG.findall(text):
            if path not in found and os.path.isfile(path):
                found.append(path)
    return found


def reference_message(state):
    """The most recent user message carrying an image, or None.

    Only this one message is sent to the comparison call.  Passing the whole
    transcript would resend EVERY image attached in the session on every check,
    which grows without bound (the worker's summarisation middleware does not
    apply out here) -- and the older references are not what we are comparing
    against anyway.
    """
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
        renders = find_render_paths(state)
        if not renders or not has_reference_image(state):
            return {}                       # nothing to look at: no model call
        if (state.get("visual_rounds") or 0) >= MAX_VISUAL_ROUNDS:
            return {}
        parts: list[dict] = [{"type": "text", "text": (
            "Here are the renders of what was just built. Compare them with the "
            "reference image above.")}]
        parts += [image_part_from_file(Path(p)) for p in renders[:MAX_RENDERS]]
        reply = await model.ainvoke([
            SystemMessage(content=VISUAL_SYSTEM),
            reference_message(state),         # just the reference, not the history
            HumanMessage(content=parts),
        ])
        matched, detail = parse_visual_verdict(message_text(reply))
        return {"visual": {"matched": matched, "detail": detail,
                           "renders": renders[:MAX_RENDERS]},
                "visual_rounds": (state.get("visual_rounds") or 0) + 1}

    return visual_check


def route_after_visual(state) -> str:
    """Send a visual mismatch back to the planner once; otherwise carry on."""
    visual = state.get("visual") or {}
    if visual.get("matched", True):
        return "ok"
    if (state.get("visual_rounds") or 0) > MAX_VISUAL_ROUNDS:
        return "ok"                          # opinion noted, budget spent
    return "revise"


def visual_failure_context(state) -> str:
    """A revision instruction derived from the visual mismatch."""
    visual = state.get("visual") or {}
    if visual.get("matched", True):
        return ""
    return ("The build matches its spec, but comparing the renders with the "
            f"reference shows a difference: {visual.get('detail', '')}\n"
            "Revise the spec so the shape matches the reference.")
