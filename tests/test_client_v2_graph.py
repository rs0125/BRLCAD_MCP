"""client-v2 graph wiring: routing and the worker tool loop, all offline.

Uses fake models + one in-process tool -- no MCP server, no API key, no
listener.  Proves the plumbing (routes to the right node, worker runs a tool
loop, results land in state), NOT model quality or geometry correctness (that
is the eval harness, a later phase).
"""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from client_v2.agents.conversational import ROUTE_CHAT, ROUTE_WORK
from client_v2.graph import build_graph
from tests.v2_fakes import FakeToolCallingModel, ai_tool_call


@tool
def echo(text: str) -> str:
    """Echo the given text back."""
    return f"echoed: {text}"


def _texts(state):
    return [getattr(m, "content", None) for m in state["messages"]]


async def test_chat_input_routes_to_respond_and_skips_worker():
    chat_model = FakeMessagesListChatModel(responses=[AIMessage(content="hey!")])
    # A worker model that would explode if the chat path ever reached it.
    boom = FakeToolCallingModel(responses=[])
    graph = build_graph(
        worker_model=boom, chat_model=chat_model, tools=[echo],
        worker_prompt="unused", classifier=lambda t: ROUTE_CHAT)

    result = await graph.ainvoke({"messages": [HumanMessage(content="hi")]})

    assert result["route"] == ROUTE_CHAT
    assert _texts(result)[-1] == "hey!"
    # Worker never ran -> no tool messages in the transcript.
    assert not any(isinstance(m, ToolMessage) for m in result["messages"])


async def test_work_input_runs_worker_tool_loop():
    # First model turn asks to call echo; second turn gives the final answer.
    worker_model = FakeToolCallingModel(responses=[
        ai_tool_call("echo", {"text": "hi"}),
        AIMessage(content="all done"),
    ])
    graph = build_graph(
        worker_model=worker_model, tools=[echo],
        worker_prompt="unused", classifier=lambda t: ROUTE_WORK)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="please echo hi")]})

    assert result["route"] == ROUTE_WORK
    # The tool actually executed, and the final assistant message is present.
    assert any(isinstance(m, ToolMessage) and m.content == "echoed: hi"
               for m in result["messages"])
    assert _texts(result)[-1] == "all done"


async def test_default_classifier_sends_geometry_request_to_worker():
    worker_model = FakeToolCallingModel(responses=[AIMessage(content="ok built")])
    graph = build_graph(
        worker_model=worker_model, tools=[echo], worker_prompt="unused")
    # No classifier override -> heuristic; "build ..." is work.
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="build a box")]})
    assert result["route"] == ROUTE_WORK
    assert _texts(result)[-1] == "ok built"
