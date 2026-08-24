"""Fake chat models for client-v2 graph tests (no network).

Reused across increments as the graph grows (planner/verifier will need them
too).  Not a test module itself -- the name lacks the ``test_`` prefix so
pytest does not collect it.
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field


class FakeToolCallingModel(BaseChatModel):
    """Returns a scripted sequence of AIMessages; supports tool calling.

    Give it a list of AIMessages (some may carry ``tool_calls``); each model
    call pops the next one.  ``bind_tools`` returns self so it drops into
    ``create_agent``'s loop.  Each call's inbound messages are recorded in
    ``calls`` so tests can inspect the (middleware-composed) system prompt.
    """

    responses: list
    calls: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(list(messages))
        message = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=message)])

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools, **kwargs):
        return self


def ai_tool_call(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An assistant message that requests one tool call."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}])


class LoopingToolCallingModel(BaseChatModel):
    """Always requests the same tool call -- a model that never stops.

    Models the failure seen live: gpt-4.1 built and verified the part
    correctly, then called ``declare_assumption`` 367 times instead of
    finishing.  ``create_agent`` has no max-iterations option and the graph's
    ``recursion_limit`` counts outer supersteps, so nothing bounded it.
    """

    tool_name: str
    args: dict = Field(default_factory=dict)
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(
            message=ai_tool_call(self.tool_name, self.args,
                                 call_id=f"call_{self.calls}"))])

    @property
    def _llm_type(self) -> str:
        return "fake-looping"

    def bind_tools(self, tools, **kwargs):
        return self
