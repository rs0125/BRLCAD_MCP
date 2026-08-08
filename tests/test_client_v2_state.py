"""client-v2 shared state: the message reducer."""

from typing import get_type_hints

from langgraph.graph.message import add_messages

from client_v2.state import AgentState


def test_messages_uses_add_messages_reducer():
    # LangGraph reads the reducer from the Annotated metadata; the messages
    # channel must accumulate via add_messages, not overwrite.
    hints = get_type_hints(AgentState, include_extras=True)
    meta = getattr(hints["messages"], "__metadata__", ())
    assert add_messages in meta
