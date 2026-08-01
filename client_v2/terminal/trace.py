"""Human-readable agent-trace formatting for debug mode.

When ``CLIENT_V2_DEBUG`` is on, the REPL streams the graph with
``stream_mode="updates"`` (subgraphs included) and prints each node's output
through :func:`format_update` -- so you can watch routing, model turns, tool
calls, and tool results as they happen.  Pure string formatting, so it is unit
tested without a live run.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from client_v2.agents.conversational import message_text

_MAX = 300


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX else text[:_MAX] + "…"


def describe_message(msg) -> str:
    """One-line description of a single message for the trace."""
    if isinstance(msg, AIMessage):
        calls = getattr(msg, "tool_calls", None) or []
        if calls:
            rendered = ", ".join(f"{c['name']}({c.get('args', {})})" for c in calls)
            return f"AI → tool call: {_clip(rendered)}"
        # responses/v1 content is a list of blocks (reasoning + text); pull text.
        text = message_text(msg)
        return f"AI: {_clip(text)}" if text else "AI: (reasoning)"
    if isinstance(msg, ToolMessage):
        name = getattr(msg, "name", None) or "tool"
        return f"TOOL[{name}]: {_clip(msg.content)}"
    if isinstance(msg, HumanMessage):
        return f"USER: {_clip(msg.content)}"
    role = getattr(msg, "type", type(msg).__name__)
    return f"{role}: {_clip(getattr(msg, 'content', ''))}"


def format_update(node: str, update, namespace: tuple = ()) -> str:
    """Render one node update (from stream_mode='updates') as trace lines."""
    prefix = f"[{':'.join(namespace)}] " if namespace else ""
    header = f"{prefix}▸ {node}"
    if not isinstance(update, dict):
        return f"{header}: {_clip(update)}"
    lines = [header]
    for key, value in update.items():
        if key == "messages":
            msgs = value if isinstance(value, list) else [value]
            lines.extend(f"    {describe_message(m)}" for m in msgs)
        else:
            lines.append(f"    {key}: {_clip(value)}")
    return "\n".join(lines)
