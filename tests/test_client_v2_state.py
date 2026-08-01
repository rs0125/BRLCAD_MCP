"""client-v2 shared state: message reducer and initial-state helper."""

from typing import get_type_hints

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph.message import add_messages

from client_v2.state import AgentState, initial_state


def test_messages_uses_add_messages_reducer():
    # LangGraph reads the reducer from the Annotated metadata; the messages
    # channel must accumulate via add_messages, not overwrite.
    hints = get_type_hints(AgentState, include_extras=True)
    meta = getattr(hints["messages"], "__metadata__", ())
    assert add_messages in meta


def test_initial_state_from_text():
    state = initial_state("build a box")
    assert list(state) == ["messages"]
    assert isinstance(state["messages"][0], HumanMessage)
    assert state["messages"][0].content == "build a box"


def test_initial_state_passes_through_a_message():
    msg = AIMessage(content="hello")
    state = initial_state(msg)
    assert state["messages"] == [msg]
