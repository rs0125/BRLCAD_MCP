"""client-v2 authorize gate: a real halt, driven by the skill definition."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from client_v2.agents.authorize import (
    WIPE_QUESTION,
    authorization_request,
    broad_destructive,
    make_authorize_node,
)
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


# --- a wholesale destructive request is gated even with no skill -----------

def test_wipe_the_database_requests_are_recognised():
    # THE BUG: "delete everything in this db" planned as {"steps": []}, so no
    # skill could declare an authorize step, and a wipe ran with no confirmation.
    for text in ("delete everything in this db",
                 "clear the whole database",
                 "wipe all objects",
                 "remove all the models",
                 "kill *",
                 "reset the entire scene"):
        assert broad_destructive(text), text


def test_ordinary_edits_are_not_gated():
    # Over-gating would nag on every edit, which trains people to say yes.
    for text in ("delete the mounting hole",
                 "remove the left stud",
                 "erase bracket.r from the display",
                 "build a 50 mm plate with two holes",
                 "make the bore 6 mm",
                 "render it from the top"):
        assert not broad_destructive(text), text


def test_a_destructive_verb_alone_is_not_enough():
    assert not broad_destructive("delete hole_x.s")
    assert not broad_destructive("show me everything in the db")   # no verb


def test_the_gate_fires_with_an_empty_plan():
    from client_v2.skills import SkillRegistry
    registry = SkillRegistry({})
    assert authorization_request({"steps": []}, registry,
                                 "delete everything in this db") == WIPE_QUESTION
    assert authorization_request({"steps": []}, registry, "build a plate") is None


async def test_the_graph_actually_halts_on_a_wipe_request():
    from langchain_core.messages import HumanMessage
    from langchain_core.tools import tool

    from client_v2.graph import build_graph
    from client_v2.skills import SkillRegistry
    from tests.v2_fakes import FakeToolCallingModel

    @tool
    def noop() -> str:
        """Does nothing."""
        return "ok"

    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=FakeToolCallingModel(
            responses=[AIMessage(content='{"steps": []}')] * 4),
        tools=[noop], worker_prompt="unused",
        registry=SkillRegistry({}), classifier=lambda t: "work",
        checkpointer=MemorySaver())

    cfg = {"configurable": {"thread_id": "wipe"}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="delete everything in this db")]}, cfg)

    pauses = result.get("__interrupt__") or ()
    assert pauses, "a wipe-the-database request must not run unconfirmed"
    assert "DESTROY" in str(pauses[0].value["authorize"])


# --- confirm dimensions before building from a drawing ---------------------

from client_v2.agents.authorize import (  # noqa: E402
    SKETCH_QUESTION,
    plan_builds_geometry,
    turn_has_reference_image,
)

_BUILDS = SkillRegistry({
    "build_model_spec": SkillDef.model_validate({
        "id": "build_model_spec", "description": "builds",
        "steps": [{"call": "build_from_spec", "with": {"spec": "${spec}"}}]}),
    "ingest_drawing": SkillDef.model_validate({
        "id": "ingest_drawing", "description": "just looks",
        "steps": [{"understand": ["describe the part"]}]}),
})
_BUILD_PLAN = {"steps": [{"skill": "build_model_spec", "params": {}}]}
_LOOK_PLAN = {"steps": [{"skill": "ingest_drawing", "params": {}}]}


def _img_turn(extra=0):
    msgs = [HumanMessage(content=[
        {"type": "text", "text": "model this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}])]
    msgs += [AIMessage(content="ok")] * extra
    return {"messages": msgs, "turn_start": 1}


def test_geometry_building_is_detected_from_the_skills_call_steps():
    assert plan_builds_geometry(_BUILD_PLAN, _BUILDS)
    assert not plan_builds_geometry(_LOOK_PLAN, _BUILDS)


def test_a_reference_image_this_turn_is_detected():
    assert turn_has_reference_image(_img_turn())
    assert not turn_has_reference_image(
        {"messages": [HumanMessage(content="text only")], "turn_start": 1})
    assert not turn_has_reference_image({})


def test_building_straight_from_a_fresh_drawing_is_gated():
    # THE BUG: only model_from_dimensioned_sketch declares an authorize step, so
    # a planner that picked its sub-skills directly bypassed the gate. The agent
    # paused anyway because it chose to -- that is not a safety property.
    assert authorization_request(_BUILD_PLAN, _BUILDS, "model this",
                                 new_reference_image=True) == SKETCH_QUESTION


def test_merely_reading_a_drawing_is_not_gated():
    assert authorization_request(_LOOK_PLAN, _BUILDS, "model this",
                                 new_reference_image=True) is None


def test_a_later_turn_does_not_re_ask_about_the_same_drawing():
    # The numbers were confirmed on the turn that carried the image; building on
    # a follow-up ("yeah go ahead") must not stop again.
    assert authorization_request(_BUILD_PLAN, _BUILDS, "yeah go ahead",
                                 new_reference_image=False) is None


def test_a_skill_declared_authorize_step_still_takes_precedence():
    registry = SkillRegistry({"w": SkillDef.model_validate({
        "id": "w", "description": "workflow",
        "steps": [{"authorize": "confirm the plan"},
                  {"call": "build_from_spec", "with": {}}]})})
    q = authorization_request({"steps": [{"skill": "w"}]}, registry, "go",
                              new_reference_image=True)
    assert q == "confirm the plan"


async def test_the_graph_halts_before_building_from_an_image():
    from langchain_core.tools import tool as _tool

    @_tool
    def build_from_spec(spec: str) -> str:
        """Builds."""
        raise AssertionError("must not build before the dimensions are confirmed")

    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=FakeToolCallingModel(responses=[
            AIMessage(content='{"steps": [{"skill": "build_model_spec", '
                              '"params": {"spec": "x"}}]}')] * 4),
        tools=[build_from_spec], worker_prompt="unused", registry=_BUILDS,
        classifier=lambda t: "work", checkpointer=MemorySaver())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=[
            {"type": "text", "text": "model this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA"}}])]},
        {"configurable": {"thread_id": "sketch"}})

    pauses = result.get("__interrupt__") or ()
    assert pauses, "building from a fresh drawing must stop for confirmation"
    assert "dimensions" in str(pauses[0].value["authorize"])
