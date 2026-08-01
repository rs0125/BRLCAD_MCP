"""The conversational agent (intake) — the primary/initial agent.

Its job (the "understand" phase): take the user's turn and decide whether
it's a work request (route to the worker, which has tools) or plain
conversation (answer directly).  It holds no tools itself.

The classifier is injectable so it's testable and swappable.  The default here
is a deterministic heuristic that biases toward "work" -- misrouting chat to the
worker is harmless (the worker answers fine), whereas misrouting work to chat
would skip the tools.  An LLM-backed classifier is a later increment.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage

from client_v2.prompts import PROMPTS
from client_v2.state import AgentState

ROUTE_WORK = "work"
ROUTE_CHAT = "chat"

# Inputs that clearly need no geometry tools.  Everything else -> work.
_CHAT_ONLY = {
    "hi", "hello", "hey", "yo", "sup", "thanks", "thank you", "ty",
    "bye", "goodbye", "ok", "okay", "cool", "nice", "great", "got it",
}


def message_text(msg: AnyMessage) -> str:
    """Plain text of a message, flattening multimodal (list) content."""
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
        elif isinstance(part, str):
            parts.append(part)
    return " ".join(parts)


def last_human_text(state: AgentState) -> str:
    """Text of the most recent human message in the state (empty if none)."""
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return message_text(msg)
    return ""


def heuristic_classifier(text: str) -> str:
    """Deterministic work/chat classifier; biased toward work (see module doc)."""
    t = text.strip().lower().rstrip("!.?")
    if not t or t in _CHAT_ONLY:
        return ROUTE_CHAT
    return ROUTE_WORK


def make_intake_node(classifier: Callable[[str], str]):
    """Node that classifies the latest user turn and resets turn-scoped state.

    Intake runs first on every turn, so it is where the per-turn bookkeeping is
    cleared.  This matters: the checkpointer deliberately keeps the whole
    conversation, but ``verification``, ``visual``, ``step_outputs`` and the two
    retry counters describe the CURRENT turn.  Left over, a failure from an
    earlier turn kept failing every later one and the revision budget never
    replenished, so the retry loop stopped working after one bad turn.
    """
    def intake(state: AgentState) -> AgentState:
        return {
            "route": classifier(last_human_text(state)),
            # Everything already in messages belongs to earlier turns.
            "turn_start": len(state.get("messages") or []),
            "verification": None,
            "visual": None,
            "step_outputs": {},
            "step_errors": [],
            "revisions": 0,
            "visual_rounds": 0,
            "plan": None,
            "active_skill": None,
            "authorized": False,
            "authorization": "",
        }
    return intake


def route_after_intake(state: AgentState) -> str:
    """Conditional-edge selector: read the route intake set (default work)."""
    return state.get("route", ROUTE_WORK)


def make_respond_node(model):
    """Node for the chat path: a brief conversational reply (no tools)."""
    def respond(state: AgentState) -> AgentState:
        msgs = state.get("messages", [])
        reply = model.invoke(
            [SystemMessage(content=PROMPTS.text("chat")), *msgs])
        return {"messages": [reply]}
    return respond
