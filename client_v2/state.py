"""Shared graph state for client-v2.

One state object flows through every node (intake -> worker/respond -> ...).
Nodes return partial updates; LangGraph merges them via the annotated reducers.
Fields beyond ``messages`` are placeholders the later agents (planner, verifier,
formatter) will populate -- declared now so the state shape is stable as the
graph grows.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    """The graph's working memory.

    ``messages`` is always present (the conversation, appended via add_messages).
    The rest are optional and filled in as the pipeline grows:
    ``route`` (intake's work/chat decision), ``plan`` (planner), ``artifacts``
    (worker outputs), ``verification`` (verifier verdict).
    """

    messages: Annotated[list[AnyMessage], add_messages]
    route: str
    # Index into ``messages`` where THIS turn's output starts.  The checkpointer
    # keeps the whole conversation, but verification is a statement about the
    # current turn -- without this boundary an old failure keeps failing every
    # later turn.
    turn_start: int
    active_skill: str
    plan: Any
    step_outputs: dict[str, Any]
    step_errors: list[str]
    artifacts: list[Any]
    verification: Any
    revisions: int
    visual: Any
    visual_rounds: int
    # Set once the turn's authorization gate has been passed (or was not needed),
    # with whatever the user answered.
    authorized: bool
    authorization: str


def initial_state(user_input: str | AnyMessage) -> AgentState:
    """Build a fresh state for one user turn from text or a prepared message."""
    msg = HumanMessage(content=user_input) if isinstance(user_input, str) \
        else user_input
    return {"messages": [msg]}
