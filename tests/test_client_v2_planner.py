"""client-v2 planner: skill-id parsing, node selection, and the full
planner -> active_skill -> worker -> SkillsMiddleware injection chain."""

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
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


# --- the planner must be able to plan a FOLLOW-UP turn ---------------------

def test_the_planner_sees_what_was_already_agreed():
    """THE BUG: the planner got only the latest human line.

    On "yeah go ahead" -- the turn that actually builds, after the agent has
    proposed dimensions -- it had nothing to plan from and said so: "No
    actionable model, dimensions, reference, or approved constraints are present
    in the available conversation."  It returned an empty plan, so no skill
    definition reached the worker, so build_model_spec's cautions were absent on
    exactly the turn that built the geometry.
    """
    from client_v2.agents.planner import conversation_context
    state = {"messages": [
        HumanMessage(content="here is a drawing"),
        AIMessage(content="Body length 31.8 mm, width 15.8 mm, height 9.6 mm."),
        HumanMessage(content="yeah go ahead"),
    ]}
    ctx = conversation_context(state)
    assert "31.8 mm" in ctx                       # the agreed numbers survive
    assert "yeah go ahead" in ctx
    assert "User:" in ctx and "Assistant:" in ctx


def test_context_is_a_short_tail_not_the_whole_transcript():
    from client_v2.agents.planner import MAX_CONTEXT_TURNS, conversation_context
    msgs = []
    for i in range(20):
        msgs += [HumanMessage(content=f"ask {i}"), AIMessage(content=f"reply {i}")]
    ctx = conversation_context({"messages": msgs})
    assert "ask 19" in ctx and "ask 0" not in ctx
    assert len(ctx.splitlines()) <= MAX_CONTEXT_TURNS * 2


def test_a_reference_image_is_never_resent_to_the_planner():
    # Only text blocks are kept, so the planning call cannot carry base64.
    from client_v2.agents.planner import conversation_context
    img = HumanMessage(content=[
        {"type": "text", "text": "model this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])
    ctx = conversation_context({"messages": [img]})
    assert "model this" in ctx and "base64" not in ctx


def test_tool_traffic_is_left_out():
    from langchain_core.messages import ToolMessage

    from client_v2.agents.planner import conversation_context
    state = {"messages": [HumanMessage(content="build it"),
                          ToolMessage(content="SUCCESS: lots of output",
                                      name="build_from_spec", tool_call_id="1"),
                          AIMessage(content="done")]}
    ctx = conversation_context(state)
    assert "lots of output" not in ctx and "done" in ctx


def test_no_history_yields_no_context_block():
    from client_v2.agents.planner import conversation_context
    assert conversation_context({}) == ""


def test_context_keeps_the_tail_of_a_long_message_not_the_head():
    """Decisions live at the END of a proposal, so head-truncation drops them.

    Measured on a real run: a 1,960-char proposal lost the trailing 460 chars
    carrying the approval question, and the planner then asserted the drawing's
    conflicting value instead of the one the user had approved.
    """
    from client_v2.agents.planner import conversation_context
    proposal = ("Printed constraints:\n" + "x" * 3000
                + "\nMay I use 1.6 mm as the perimeter-wall thickness?")
    state = {"messages": [HumanMessage(content="draw this"),
                          AIMessage(content=proposal)]}
    out = conversation_context(state, turns=3, limit=2400)
    assert "1.6 mm as the perimeter-wall thickness" in out
    assert out.endswith("?")
    assert "…" in out                      # clipping is marked, not silent


def test_short_messages_are_passed_through_unclipped():
    from client_v2.agents.planner import conversation_context
    state = {"messages": [HumanMessage(content="the depth")]}
    assert conversation_context(state) == "User: the depth"


# ------------------------------------------------- the empty-plan fallback

@pytest.mark.asyncio
async def test_an_unusable_plan_records_why():
    """The fallback must not be indistinguishable from a successful pass.

    Observed live: the planner returned {"steps":[]} and the run log showed a
    plan of null with no reason, so diagnosing it meant reading five separate
    queries.  A parse failure and a deliberate empty plan looked identical.
    """
    model = FakeMessagesListChatModel(responses=[AIMessage(content='{"steps":[]}')])
    node = make_planner_node(model, SkillRegistry.from_dir())
    out = await node({"messages": [HumanMessage(content="do a thing")]})
    assert out["plan"] is None
    assert out.get("plan_errors"), "the reason the plan was unusable must be recorded"


@pytest.mark.asyncio
async def test_a_usable_plan_clears_any_earlier_reason():
    body = ('{"steps": [{"skill": "build_model_spec", '
            '"params": {"spec": "<spec>"}, "why": "build it"}]}')
    model = FakeToolCallingModel(responses=[AIMessage(content=body)])
    node = make_planner_node(model, SkillRegistry.from_dir())
    out = await node({"messages": [HumanMessage(content="do a thing")],
                      "plan_errors": ["stale reason from a previous pass"]})
    assert out["plan"] is not None
    assert not out.get("plan_errors")
