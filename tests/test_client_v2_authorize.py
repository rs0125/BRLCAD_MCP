"""client-v2 authorize gate: a real halt, driven by the skill definition."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from client_v2.agents.authorize import (
    WIPE_QUESTION,
    authorization_request,
    broad_destructive,
    describe_pause,
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


# --- what does NOT justify a halt ------------------------------------------

def test_building_from_a_drawing_is_deliberately_not_gated():
    """A gate here was added and then removed on purpose. Do not re-add it.

    A halt has to earn its cost: the action is hard to undo, or it needs
    information only the user has.  A build is neither -- it writes a versioned
    spec and undo_build reverts it, and the dimensions are printed on the drawing.
    Asking "confirm these numbers?" while the user looks at the same drawing buys
    a rubber stamp, and over-gating trains people to say yes, which costs us the
    gates that do matter.  expect_bbox is the real guard: machine-checked, so a
    misread size REJECTS the build instead of being confirmed by a glance.
    """
    registry = SkillRegistry({"build_model_spec": SkillDef.model_validate({
        "id": "build_model_spec", "description": "builds",
        "steps": [{"call": "build_from_spec", "with": {"spec": "${spec}"}}]})})
    plan = {"steps": [{"skill": "build_model_spec", "params": {}}]}
    for request in ("model this drawing", "go ahead just draw this",
                    "build the bracket from the image"):
        assert authorization_request(plan, registry, request) is None, request


async def test_an_image_build_runs_without_stopping():
    @tool
    def build_from_spec(spec: str) -> str:
        """Builds."""
        return "Built region 'x.r' from 1 part(s)"

    registry = SkillRegistry({"build_model_spec": SkillDef.model_validate({
        "id": "build_model_spec", "description": "builds",
        "steps": [{"call": "build_from_spec", "with": {"spec": "${spec}"}}]})})
    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=FakeToolCallingModel(responses=[AIMessage(
            content='{"steps": [{"skill": "build_model_spec", '
                    '"params": {"spec": "x"}}]}')] * 4),
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="done")] * 4),
        tools=[build_from_spec], worker_prompt="unused", registry=registry,
        classifier=lambda t: "work", checkpointer=MemorySaver())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=[
            {"type": "text", "text": "model this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA"}}])]},
        {"configurable": {"thread_id": "img"}})

    assert not (result.get("__interrupt__") or ())      # no rubber-stamp pause
    assert result["step_outputs"]["build_model_spec"].startswith("Built region")


# --------------------------------------------------------------- describing a pause

def test_a_pause_shows_the_plan_it_is_asking_about():
    """The bare skill text is an instruction to the agent, not a question.

    Observed live: the halt read "confirm the dimensioned plan with the user",
    which is the literal {authorize: "..."} string from the skill YAML.  The
    user was asked to confirm a plan they had never been shown, and could only
    guess.  The interrupt payload already carries the plan, so describe it.
    """
    value = {
        "authorize": "confirm the dimensioned plan with the user",
        "plan": {"steps": [
            {"skill": "build_model_spec",
             "params": {"spec": {"name": "brick", "expect_bbox": [32, 16, 9.6]}},
             "why": "build the approved shape"},
            {"skill": "verify_model_dimensions", "params": {"name": "brick"}},
        ]},
    }
    out = describe_pause(value)
    assert "confirm the dimensioned plan" in out
    assert "build_model_spec" in out          # what it will do
    assert "32" in out and "9.6" in out       # the numbers being approved


def test_a_pause_with_no_plan_is_just_the_question():
    value = {"authorize": "This will DESTROY geometry across the whole database."}
    assert describe_pause(value) == "This will DESTROY geometry across the whole database."


def test_a_pause_with_an_empty_plan_is_just_the_question():
    # "delete everything" plans as {"steps": []} -- nothing to describe.
    value = {"authorize": "are you sure?", "plan": {"steps": []}}
    assert describe_pause(value) == "are you sure?"


def test_the_gate_is_not_skipped_when_the_fallback_picked_the_skill():
    """A null plan must not bypass a skill's authorize step.

    Observed live: the planner returned {"steps":[]}, so plan_skill_ids was
    empty and the gate found nothing to ask about even though the fallback had
    selected a skill that declares one.  A fallback path silently skipping a
    safety gate is the wrong way round: unplanned turns are the ones with least
    scrutiny elsewhere.
    """
    reg = SkillRegistry.from_dir()
    gated = [i for i in reg.ids()
             if any(isinstance(s, dict) and s.get("authorize")
                    for s in (reg.get(i).steps or []))]
    assert gated, "expected at least one skill to declare an authorize step"
    skill = gated[0]

    # With a plan naming it, the gate fires (existing behaviour).
    planned = authorization_request({"steps": [{"skill": skill}]}, reg, "")
    assert planned

    # With no plan but that skill active, it must still fire.
    fell_back = authorization_request(None, reg, "", active_skill=skill)
    assert fell_back == planned


def test_an_active_skill_without_a_gate_still_does_not_halt():
    reg = SkillRegistry.from_dir()
    ungated = [i for i in reg.ids()
               if not any(isinstance(s, dict) and s.get("authorize")
                          for s in (reg.get(i).steps or []))]
    assert ungated
    assert authorization_request(None, reg, "", active_skill=ungated[0]) is None
