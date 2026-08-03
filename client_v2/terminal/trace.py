"""Human-readable agent-trace formatting for debug mode.

When ``CLIENT_V2_DEBUG`` is on the REPL shows what the agent is doing, from two
sources that answer different questions:

* :class:`LiveTrace` -- a callback handler printing model replies and tool
  activity **as they happen**.
* :func:`format_update` -- one line per node as it FINISHES, showing the state it
  wrote (route, plan, verification): the decisions, not the chatter.

Both are needed because the worker runs its tool loop in a NESTED invocation
(``agents/worker.py``), so the graph's own node updates cannot see inside it: a
long build printed nothing for minutes and then dumped every message at once.
Callbacks do propagate into that nested run -- the same reason the run log can
record the model calls made in there -- so the live view goes through a callback
while node updates stay for state.

Formatting is pure and the handler's output sink is injected, so all of it is unit
tested without a live run.
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from client_v2.agents.conversational import message_text

_MAX = 300


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= _MAX else text[:_MAX] + "…"


def origin(metadata) -> str:
    """The OUTER graph node a callback event came from, e.g. 'planner', 'worker'.

    Needed because a live line is printed while its node is still running, so the
    node's own summary line does not appear until after it -- read without a label
    the planner's reply looks like it belonged to intake.

    ``langgraph_node`` alone is not enough: inside the worker's nested agent it
    reports that agent's INTERNAL node ('model', 'tools'), which says nothing
    about whose loop it is.  ``langgraph_checkpoint_ns`` carries the whole path
    ('worker:<id>|model:<id>'), so the first segment is the role.
    """
    meta = metadata or {}
    namespace = meta.get("langgraph_checkpoint_ns") or ""
    outer = namespace.split("|")[0].split(":")[0]
    return outer or meta.get("langgraph_node") or ""


def reasoning_summary(msg) -> str:
    """Summary text from a responses/v1 reasoning block, if the model sent any.

    Reasoning arrives as ``{"type": "reasoning", "encrypted_content": ...,
    "summary": [...]}``.  The blob is opaque, but ``summary`` carries readable
    text WHEN the request asked for it (``reasoning={"summary": "auto"}``); it is
    an empty list otherwise, so this returns "" rather than inventing anything.
    """
    content = getattr(msg, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "reasoning":
            continue
        for item in block.get("summary") or ():
            text = item.get("text", "") if isinstance(item, dict) else str(item)
            if text:
                parts.append(text)
    return " ".join(parts)


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


# Pure bookkeeping: an index into the message list, never interesting to read.
_SKIP_KEYS = frozenset({"turn_start"})


def _is_empty(value) -> bool:
    """True for a state write that says nothing: None, empty, 0 or False.

    Intake resets nine fields on every turn, so without this the one line that
    matters -- the route it chose -- is buried in resets.  A counter back at 0
    carries no information; at 2 it does, and still prints.
    """
    return not value


def format_update(node: str, update, namespace: tuple = (),
                  include_messages: bool = True) -> str:
    """Render one node update (from stream_mode='updates') as trace lines.

    ``include_messages=False`` renders only the STATE the node wrote, for when a
    :class:`LiveTrace` has already printed the messages as they were produced;
    without it every message appears twice.
    """
    prefix = f"[{':'.join(namespace)}] " if namespace else ""
    header = f"{prefix}▸ {node}"
    if update is None:                    # a node that wrote nothing (e.g. a
        return header                     # visual check that had no work to do)
    if not isinstance(update, dict):
        return f"{header}: {_clip(update)}"
    lines = [header]
    for key, value in update.items():
        if key == "messages":
            msgs = value if isinstance(value, list) else [value]
            if include_messages:
                lines.extend(f"    {describe_message(m)}" for m in msgs)
            else:
                lines.append(f"    (+{len(msgs)} message(s))")
        elif key not in _SKIP_KEYS and not _is_empty(value):
            lines.append(f"    {key}: {_clip(value)}")
    return "\n".join(lines)


class LiveTrace(BaseCallbackHandler):
    """Prints model replies and tool activity the moment they happen.

    Registered as a callback (see the module docstring) so it also sees the
    worker's nested tool loop, which is where the interesting decisions are made.
    The output sink is injected so the whole handler is testable without a run.

    What it does NOT show is raw chain-of-thought: reasoning items come back
    encrypted, and only the model's own summary is readable -- and only when the
    request asked for one.  :func:`reasoning_summary` prints it when present.
    """

    def __init__(self, out=print):
        self._out = out
        # run_id -> which node it belongs to.  Needed because LangChain attaches
        # ``metadata`` to START events only: read on an end event it is absent, so
        # every result line would print unlabelled.  Entries are popped when the
        # run ends, so this cannot grow over a long session.
        self._origins: dict[str, str] = {}

    def _remember(self, kwargs) -> None:
        where = origin(kwargs.get("metadata"))
        run_id = str(kwargs.get("run_id", ""))
        if run_id and where:
            self._origins[run_id] = where

    def _emit(self, kwargs, text: str, *, ending: bool = False) -> None:
        """Print one line, tagged with the node that produced it."""
        run_id = str(kwargs.get("run_id", ""))
        where = origin(kwargs.get("metadata")) or (
            self._origins.pop(run_id, "") if ending
            else self._origins.get(run_id, ""))
        self._out(f"    [{where}] {text}" if where else f"    {text}")

    # --- model ------------------------------------------------------------

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        # Nothing to print yet -- only the origin, for the reply that follows.
        self._remember(kwargs)

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self._remember(kwargs)

    def on_llm_end(self, response, **kwargs) -> None:
        message = None
        try:
            generation = response.generations[0][0]
            message = getattr(generation, "message", None) or generation.text
        except (AttributeError, IndexError):
            return
        summary = reasoning_summary(message)
        if summary:
            self._emit(kwargs, f"· {_clip(summary)}")
        # Tool calls are announced by on_tool_start with their real arguments, so
        # only the model's TEXT is printed here -- otherwise every call is
        # reported twice, once as an intent and once as an invocation.
        text = message_text(message) if not isinstance(message, str) else message
        if text.strip():
            self._emit(kwargs, f"AI: {_clip(text)}", ending=True)
        else:
            # Nothing to print (a bare tool-call turn), but the run is over.
            self._origins.pop(str(kwargs.get("run_id", "")), None)

    def on_llm_error(self, error, **kwargs) -> None:
        self._emit(kwargs, f"✗ model error: {_clip(error)}", ending=True)

    # --- tools ------------------------------------------------------------

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self._remember(kwargs)
        name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
        args = kwargs.get("inputs")
        self._emit(kwargs,
                   f"▸ {name}({_clip(args if args is not None else input_str)})")

    def on_tool_end(self, output, **kwargs) -> None:
        text = output if isinstance(output, str) else message_text(output) or output
        self._emit(kwargs, f"  ✓ {_clip(text)}", ending=True)

    def on_tool_error(self, error, **kwargs) -> None:
        self._emit(kwargs, f"  ✗ {_clip(error)}", ending=True)
