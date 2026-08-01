"""client-v2 authorize gate: a real halt, driven by the skill definition."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from client_v2.agents.authorize import authorization_request, make_authorize_node
from client_v2.graph import build_graph
from client_v2.skills import SkillDef, SkillRegistry
from tests.v2_fakes import FakeToolCallingModel


def _registry():
    return SkillRegistry({
        # Declares an authorize step -> must halt before anything runs.
        "gated": SkillDef.model_validate({
            "id": "gated", "kind": "workflow", "description": "needs a decision",
            "steps": [{"plan": ["think"]},
                      {"authorize": "confirm the numbers with the user"},
                      {"call": "act", "with": {}}]}),
        # No authorize step -> runs straight through.
        "ungated": SkillDef.model_validate({
            "id": "ungated", "description": "just does it",
            "steps": [{"call": "act", "with": {}}]}),
    })


@tool
def act() -> str:
    """Act."""
    return "acted"


# --- the trigger comes from the definition, not the graph -----------------

def test_the_question_is_read_from_the_skills_authorize_step():
    reg = _registry()
    assert authorization_request({"steps": [{"skill": "gated"}]}, reg) == (
        "confirm the numbers with the user")
    assert authorization_request({"steps": [{"skill": "ungated"}]}, reg) is None
    assert authorization_request({"steps": [{"skill": "unknown"}]}, reg) is None
    assert authorization_request(None, reg) is None


def test_the_shipped_workflow_that_needs_confirmation_declares_it():
    real = SkillRegistry.from_dir()
    plan = {"steps": [{"skill": "model_from_dimensioned_sketch"}]}
    assert authorization_request(plan, real)


async def test_node_passes_through_when_no_decision_is_needed():
    node = make_authorize_node(_registry())
    out = await _maybe_await(node({"plan": {"steps": [{"skill": "ungated"}]}}))
    assert out == {"authorized": True}


async def test_node_does_not_re_ask_once_answered():
    node = make_authorize_node(_registry())
    out = await _maybe_await(node(
        {"plan": {"steps": [{"skill": "gated"}]}, "authorized": True}))
    assert out == {}


async def _maybe_await(value):
    return await value if hasattr(value, "__await__") else value


# --- the halt is real, end to end ----------------------------------------

def _gated_graph():
    return build_graph(
        worker_model=FakeToolCallingModel(responses=[AIMessage(content="done")]),
        planner_model=FakeToolCallingModel(responses=[
            AIMessage(content='{"steps": [{"skill": "gated", "params": {}}]}')]),
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="summary")] * 3),
        tools=[act], worker_prompt="BASE", registry=_registry(),
        classifier=lambda t: "work", checkpointer=MemorySaver())


async def test_the_graph_actually_halts_and_resumes_with_the_answer():
    # THE POINT: prose telling the model to pause was ignorable (render_beauty
    # ran all three of its stages without waiting).  An interrupt is not.
    graph = _gated_graph()
    config = {"configurable": {"thread_id": "gate"}}

    paused = await graph.ainvoke(
        {"messages": [HumanMessage(content="do the gated thing")]}, config)
    interrupts = paused.get("__interrupt__") or ()
    assert interrupts, "the graph should have stopped for a decision"
    assert "confirm the numbers" in str(interrupts[0].value)
    assert not paused.get("authorized")        # nothing ran past the gate

    resumed = await graph.ainvoke(Command(resume="yes, approved"), config)
    assert resumed["authorized"] is True
    assert resumed["authorization"] == "yes, approved"


async def test_an_ungated_plan_never_halts():
    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[AIMessage(content="done")]),
        planner_model=FakeToolCallingModel(responses=[
            AIMessage(content='{"steps": [{"skill": "ungated", "params": {}}]}')]),
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="summary")] * 3),
        tools=[act], worker_prompt="BASE", registry=_registry(),
        classifier=lambda t: "work", checkpointer=MemorySaver())
    out = await graph.ainvoke(
        {"messages": [HumanMessage(content="go")]},
        {"configurable": {"thread_id": "nogate"}})
    assert not (out.get("__interrupt__") or ())
    assert out["authorized"] is True
