"""Per-run JSONL logging -- so a failed run can be read, not re-guessed.

The reliability numbers told us *how often* things worked; they could not tell us
*why* a particular run went wrong.  Twice the diagnosis was "re-run it and look at
the artifacts", which happens to have worked but is detective work, not replay --
and in both cases the model was defensible and our expectation was wrong, exactly
when you most want the original inputs preserved.

So every run appends newline-delimited JSON events to one file:

* ``turn``        -- a user turn begins (its text, any attached image count)
* ``node``        -- a graph node finished, with the state it wrote
* ``model``       -- a MODEL CALL: the messages in, the reply out, token usage
* ``interrupt``   -- the graph halted for a decision, and what answered it
* ``result``      -- the turn's final answer

Two rules the implementation keeps:

1. **It never breaks a turn.** Every write is guarded; a logging failure is
   swallowed (and noted once), because losing a log line must never cost a build.
2. **Images are redacted.** A base64 data URI is replaced by its byte count.
   Logging them verbatim would produce multi-megabyte lines and make the log
   unreadable -- the opposite of the point.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

DEFAULT_LOG_DIR = os.path.expanduser(
    os.environ.get("CLIENT_V2_LOG_DIR", "~/brlcad_agent_logs"))

# Long strings are truncated: a log you cannot scroll through is not a log.
_MAX_STR = 2000
_DATA_URI = "data:image"
# Depth is a runaway/cycle guard, NOT a size control -- that is what _MAX_STR and
# _MAX_ITEMS are for.  It was 6, which is shallower than a real tool call: a spec
# sits at reply -> tool_calls -> [0] -> args -> spec -> parts -> part, so a
# 29-part build logged its geometry as ["<...>", ...].  The log recorded that a
# build happened and its verdict, but not what was built -- the one thing you
# open the log to find.
_MAX_DEPTH = 24
# Enough to hold a real parts list; a runaway sequence still cannot fill the disk.
_MAX_ITEMS = 400


def redact(value: Any, _depth: int = 0) -> Any:
    """JSON-safe copy of *value* with images redacted and long text truncated."""
    if _depth > _MAX_DEPTH:
        return "<...>"
    if isinstance(value, str):
        if _DATA_URI in value:
            return f"<image, {len(value)} b64 chars>"
        return value if len(value) <= _MAX_STR else value[:_MAX_STR] + "…<clipped>"
    if isinstance(value, dict):
        return {str(k): redact(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        out = [redact(v, _depth + 1) for v in value[:_MAX_ITEMS]]
        if len(value) > _MAX_ITEMS:
            out.append(f"<+{len(value) - _MAX_ITEMS} more items>")
        return out
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return redact(_describe(value), _depth + 1)


def _describe(obj: Any) -> Any:
    """Best-effort structured view of a LangChain message or arbitrary object."""
    content = getattr(obj, "content", None)
    if content is not None:
        out: dict[str, Any] = {"type": type(obj).__name__, "content": content}
        calls = getattr(obj, "tool_calls", None)
        if calls:
            out["tool_calls"] = [
                {"name": c.get("name"), "args": c.get("args")} for c in calls]
        name = getattr(obj, "name", None)
        if name:
            out["name"] = name
        return out
    return str(obj)


class RunLog:
    """Append-only JSONL sink for one run.  Safe to call from anywhere."""

    def __init__(self, path: str | None):
        self.path = path
        self.turn_index = 0
        self._broken = False

    def event(self, kind: str, **fields: Any) -> None:
        """Append one event.  Never raises."""
        if not self.path or self._broken:
            return
        record = {"t": round(time.time(), 3), "turn": self.turn_index,
                  "kind": kind, **{k: redact(v) for k, v in fields.items()}}
        try:
            with open(self.path, "a") as fh:
                fh.write(json.dumps(record, default=str) + "\n")
        except OSError as exc:            # a lost log line must not cost a build
            self._broken = True
            print(f"  (run log disabled: {exc})")

    def start_turn(self, text: str, images: int = 0) -> None:
        self.turn_index += 1
        self.event("turn", text=text, images=images)

    def callbacks(self) -> list[BaseCallbackHandler]:
        """Callback handlers that record every model call made under this run."""
        return [ModelCallLogger(self)] if self.path else []


class ModelCallLogger(BaseCallbackHandler):
    """Records each chat-model call: the messages in, the reply and token usage.

    A LangChain callback rather than wrappers around each agent, so it also
    captures the calls made *inside* the worker's own tool loop -- which is where
    the interesting decisions happen.
    """

    def __init__(self, log: RunLog):
        self.log = log

    def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
        flat = messages[0] if messages and isinstance(messages[0], list) else messages
        self.log.event("model_start", messages=flat,
                       run_id=str(kwargs.get("run_id", "")))

    def on_llm_end(self, response, **kwargs) -> None:
        reply, usage = None, None
        try:
            generation = response.generations[0][0]
            reply = getattr(generation, "message", None) or generation.text
            usage = getattr(reply, "usage_metadata", None)
        except (AttributeError, IndexError):
            pass
        self.log.event("model", reply=reply, usage=usage,
                       run_id=str(kwargs.get("run_id", "")))

    def on_llm_error(self, error, **kwargs) -> None:
        self.log.event("model_error", error=str(error),
                       run_id=str(kwargs.get("run_id", "")))


def open_run_log(directory: str = DEFAULT_LOG_DIR,
                 stamp: str | None = None) -> RunLog:
    """Start a run log in *directory*; falls back to a no-op if it cannot."""
    try:
        os.makedirs(directory, exist_ok=True)
        name = stamp or time.strftime("run_%Y%m%d_%H%M%S")
        return RunLog(os.path.join(directory, f"{name}.jsonl"))
    except OSError as exc:
        print(f"  (run log unavailable: {exc})")
        return RunLog(None)


def null_log() -> RunLog:
    """A log that discards everything -- the default for tests and libraries."""
    return RunLog(None)
