"""client-v2 conversational agent: classification, routing, chat reply."""

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage

from client_v2.agents.conversational import (
    ROUTE_CHAT,
    ROUTE_WORK,
    heuristic_classifier,
    last_human_text,
    make_intake_node,
    make_respond_node,
    message_text,
    route_after_intake,
)


def test_heuristic_routes_chat_vs_work():
    for t in ("hi", "hello", "thanks", "ok", "bye", "  Cool!  ", ""):
        assert heuristic_classifier(t) == ROUTE_CHAT, t
    for t in ("build a box", "render the bracket", "what is this model?"):
        assert heuristic_classifier(t) == ROUTE_WORK, t


def test_message_text_flattens_multimodal():
    m = HumanMessage(content=[
        {"type": "text", "text": "make this"},
        {"type": "image_url", "image_url": {"url": "data:..."}},
    ])
    assert message_text(m) == "make this"


def test_last_human_text_picks_most_recent_human():
    state = {"messages": [
        HumanMessage(content="first"),
        AIMessage(content="ai reply"),
        HumanMessage(content="second"),
    ]}
    assert last_human_text(state) == "second"


def test_intake_node_records_route_and_selector_reads_it():
    intake = make_intake_node(lambda text: ROUTE_WORK)
    out = intake({"messages": [HumanMessage(content="do work")]})
    assert out["route"] == ROUTE_WORK
    assert route_after_intake(out) == ROUTE_WORK


def test_intake_resets_turn_scoped_state():
    # The checkpointer keeps the conversation, but a verdict, a plan and the
    # retry counters describe the CURRENT turn.  Left over, an earlier failure
    # kept failing every later turn and the revision budget never replenished.
    intake = make_intake_node(lambda text: ROUTE_WORK)
    stale = {
        "messages": [HumanMessage(content="old"), AIMessage(content="reply"),
                     HumanMessage(content="new")],
        "verification": {"passed": False}, "visual": {"matched": False},
        "step_outputs": {"old": "Error: boom"}, "step_errors": ["boom"],
        "revisions": 2, "visual_rounds": 1, "plan": {"steps": []},
        "active_skill": "something_else",
    }
    out = intake(stale)
    assert out["verification"] is None and out["visual"] is None
    assert out["step_outputs"] == {} and out["step_errors"] == []
    assert out["revisions"] == 0 and out["visual_rounds"] == 0
    assert out["plan"] is None and out["active_skill"] is None
    # Everything already in the transcript belongs to earlier turns.
    assert out["turn_start"] == 3


def test_route_selector_defaults_to_work_when_unset():
    assert route_after_intake({}) == ROUTE_WORK


def test_respond_node_produces_a_chat_reply():
    model = FakeMessagesListChatModel(responses=[AIMessage(content="hey there")])
    respond = make_respond_node(model)
    out = respond({"messages": [HumanMessage(content="hi")]})
    assert [m.content for m in out["messages"]] == ["hey there"]
