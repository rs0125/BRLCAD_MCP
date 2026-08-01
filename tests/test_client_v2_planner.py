"""client-v2 planner: skill-id parsing, node selection, and the full
planner -> active_skill -> worker -> SkillsMiddleware injection chain."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from client_v2.agents.planner import (
    choose_skill,
    make_planner_node,
    planning_brief,
)
from client_v2.graph import build_graph
from client_v2.skills import SkillDef, SkillRegistry
from tests.v2_fakes import FakeToolCallingModel


def _registry():
    return SkillRegistry({
        "build_model_spec": SkillDef(
            id="build_model_spec", description="build a region",
            preconditions=["units == mm"]),
        "render_orthographic_checks": SkillDef(
            id="render_orthographic_checks", description="render checks"),
    })


# --- choose_skill (pure) --------------------------------------------------

def test_choose_skill_exact_and_embedded_and_none():
    ids = ["build_model_spec", "render_orthographic_checks"]
    assert choose_skill("build_model_spec", ids) == "build_model_spec"
    assert choose_skill("Use build_model_spec here.", ids) == "build_model_spec"
    assert choose_skill("none", ids) is None
    assert choose_skill("", ids) is None
    assert choose_skill("build_model_spec", []) is None


# --- planner node ---------------------------------------------------------

async def test_planner_falls_back_to_single_skill_on_plain_reply():
    # A plain id reply (not JSON) -> no plan, but the fallback still sets it.
    model = FakeToolCallingModel(responses=[AIMessage(content="build_model_spec")])
    planner = make_planner_node(model, _registry())
    out = await planner({"messages": [HumanMessage(content="build a washer")]})
    assert out["active_skill"] == "build_model_spec"
    assert out["plan"] is None


async def test_planner_clears_active_skill_when_none_matches():
    model = FakeToolCallingModel(responses=[AIMessage(content="none")])
    planner = make_planner_node(model, _registry())
    out = await planner({"messages": [HumanMessage(content="what is this?")]})
    assert out["active_skill"] is None       # written, so a stale pick is cleared


async def test_planner_emits_a_parameterized_plan_from_json():
    plan_json = (
        '{"steps": [{"skill": "build_model_spec", '
        '"params": {"spec": "<spec>"}, "why": "build it"}], '
        '"done_when": "region exists"}')
    model = FakeToolCallingModel(responses=[AIMessage(content=plan_json)])
    planner = make_planner_node(model, _registry())
    out = await planner({"messages": [HumanMessage(content="build a washer")]})
    assert out["plan"]["steps"][0]["skill"] == "build_model_spec"
    assert out["plan"]["steps"][0]["params"] == {"spec": "<spec>"}
    # No single "active" skill when a plan exists: the worker gets the whole plan
    # and the definitions of every skill in it, rather than one arbitrary step.
    assert out["active_skill"] is None


# --- full chain: planner -> worker -> middleware injection ----------------

@tool
def noop() -> str:
    """No-op."""
    return "ok"


async def test_selected_skill_detail_reaches_the_worker_model():
    registry = _registry()
    planner_model = FakeToolCallingModel(responses=[
        AIMessage(content="build_model_spec")])
    worker_model = FakeToolCallingModel(responses=[AIMessage(content="done")])
    graph = build_graph(
        worker_model=worker_model, planner_model=planner_model,
        tools=[noop], worker_prompt="BASE", registry=registry,
        classifier=lambda t: "work")

    await graph.ainvoke({"messages": [HumanMessage(content="build a washer")]})

    # The worker model was called with a system prompt carrying the ACTIVE
    # skill's full detail (not just the catalog) -- proving planner ->
    # active_skill -> worker subagent -> SkillsMiddleware all connect.
    system_text = str(worker_model.calls[0][0].content)
    assert "## Active skill" in system_text
    assert "units == mm" in system_text        # a precondition from the detail


def test_planning_brief_includes_cautions_that_constrain_params():
    # On the executor path the PLANNER writes the params, so modelling rules
    # must appear in ITS brief -- not only in the worker's prompt.
    reg = SkillRegistry({"s": SkillDef.model_validate({
        "id": "s", "description": "build",
        "io": {"inputs": [{"name": "spec", "type": "BuildSpec",
                           "required": True}]},
        "cautions": ["a through-hole cutter must start OUTSIDE the material"],
    })})
    brief = planning_brief(reg)
    assert "spec: BuildSpec" in brief
    assert "start OUTSIDE the material" in brief
    # The brief IS the skill's canonical planning view, not a parallel format
    # that could drift from it and drop fields.
    assert brief == reg.get("s").planning_view()


async def test_the_plan_itself_reaches_the_worker_model_through_the_graph():
    # End-to-end counterpart to the fallback test above: when the planner emits a
    # real plan, the WORKER's prompt must carry the plan and the definitions of
    # the skills it names.  Previously the plan stopped at the planner and only
    # its terminal step's detail was injected.
    registry = _registry()
    plan_json = (
        '{"steps": ['
        '{"skill": "build_model_spec", "params": {"spec": "<the spec>"},'
        ' "why": "build it first"},'
        '{"skill": "render_orthographic_checks", "params": {"region": "w.r"},'
        ' "why": "then check it"}], "done_when": "it verifies"}')
    planner_model = FakeToolCallingModel(responses=[AIMessage(content=plan_json)])
    worker_model = FakeToolCallingModel(responses=[AIMessage(content="done")])
    graph = build_graph(
        worker_model=worker_model, planner_model=planner_model,
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="summary")] * 3),
        tools=[noop], worker_prompt="BASE", registry=registry,
        classifier=lambda t: "work")

    await graph.ainvoke({"messages": [HumanMessage(content="build a washer")]})

    system_text = str(worker_model.calls[0][0].content)
    assert "## Plan to follow" in system_text
    assert "1. build_model_spec" in system_text
    assert "2. render_orthographic_checks" in system_text
    assert "<the spec>" in system_text          # the planner's bound parameters
    assert "build it first" in system_text      # and its reasoning
    assert "units == mm" in system_text         # definition of a NON-terminal step
