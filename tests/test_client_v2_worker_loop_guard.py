"""The worker's tool loop must terminate even when the model will not.

Observed live: a model built the part correctly, verified it, and then called
``declare_assumption`` 367 times rather than reporting back.  Nothing stopped
it -- ``create_agent`` takes no max-iterations argument, and the graph's
``recursion_limit`` counts OUTER supersteps, while the whole runaway happens
inside a single worker superstep.
"""

import pytest
from langchain_core.tools import tool

from client_v2.agents.worker import (
    MAX_DECLARATIONS_PER_RUN,
    MAX_MODEL_CALLS_PER_RUN,
    make_worker_node,
)
from tests.v2_fakes import LoopingToolCallingModel


@tool
def noop(value: str = "x") -> str:
    """A tool that always succeeds, so nothing else ends the loop."""
    return "ok"


@tool
def declare_assumption(topic: str = "t", chose: str = "a", over: str = "b",
                       reason: str = "r", region: str = "reg") -> str:
    """Stand-in for the real declaration tool."""
    return "Recorded."


@pytest.mark.asyncio
async def test_worker_stops_a_model_that_never_stops():
    model = LoopingToolCallingModel(tool_name="noop", args={"value": "x"})
    node = make_worker_node(model, [noop], system_prompt="go")
    await node({"messages": []})
    # Bounded, and bounded by OUR limit rather than by luck.
    assert model.calls <= MAX_MODEL_CALLS_PER_RUN + 1


@pytest.mark.asyncio
async def test_the_declaration_tool_is_capped_separately():
    # It has no natural stopping point: every call succeeds and reports a
    # growing count, which reads as encouragement.
    model = LoopingToolCallingModel(tool_name="declare_assumption", args={})
    node = make_worker_node(model, [declare_assumption], system_prompt="go")
    result = await node({"messages": []})
    msgs = [m for m in result["messages"]
            if getattr(m, "name", None) == "declare_assumption"]
    # Past the cap the tool is BLOCKED but still answers, telling the model to
    # stop -- so count real executions, not tool messages.
    executed = [m for m in msgs if "Recorded" in str(m.content)]
    blocked = [m for m in msgs if "Recorded" not in str(m.content)]
    assert len(executed) == MAX_DECLARATIONS_PER_RUN
    assert blocked, "over-limit calls should be answered with a refusal"
    assert "Do not call" in str(blocked[0].content)


@pytest.mark.asyncio
async def test_a_normal_short_run_is_untouched():
    # The guard must not fire on ordinary work.
    from langchain_core.messages import AIMessage

    from tests.v2_fakes import FakeToolCallingModel, ai_tool_call
    model = FakeToolCallingModel(responses=[
        ai_tool_call("noop", {"value": "x"}),
        AIMessage(content="done"),
    ])
    node = make_worker_node(model, [noop], system_prompt="go")
    result = await node({"messages": []})
    assert any(getattr(m, "content", "") == "done" for m in result["messages"])
