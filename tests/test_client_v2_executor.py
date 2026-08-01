"""client-v2 executor: param resolution, executability, step sequencing."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from client_v2.graph import build_graph
from client_v2.pipeline.executor import (
    is_deterministic,
    make_executor_node,
    resolve_params,
)
from client_v2.pipeline.plan import Plan, PlanStep
from client_v2.skills import SkillDef, SkillRegistry
from tests.v2_fakes import FakeToolCallingModel

# --- resolve_params (pure) ------------------------------------------------

def test_resolve_exact_ref_returns_raw_value():
    ctx = {"a": {"region": "washer.r"}}
    assert resolve_params("${a.region}", ctx) == "washer.r"
    assert resolve_params("${missing}", ctx) is None


def test_resolve_inline_and_nested_structures():
    ctx = {"n": "ball"}
    assert resolve_params("name=${n}.s", ctx) == "name=ball.s"
    out = resolve_params({"spec": "${n}", "views": ["iso", "${n}"]}, ctx)
    assert out == {"spec": "ball", "views": ["iso", "ball"]}


def test_resolve_passes_non_ref_values_through():
    assert resolve_params(8, {}) == 8
    assert resolve_params(["a", 2], {}) == ["a", 2]


# --- executability --------------------------------------------------------

def _exec_registry():
    return SkillRegistry({
        "make_a": SkillDef.model_validate({
            "id": "make_a", "description": "A",
            "steps": [{"call": "tool_a", "with": {"x": "${a}"}}]}),
        "make_b": SkillDef.model_validate({
            "id": "make_b", "description": "B",
            "steps": [{"call": "tool_b", "with": {"y": "${make_a}"}}]}),
        "judgey": SkillDef.model_validate({          # phase steps, not callable
            "id": "judgey", "description": "needs a model",
            "steps": [{"understand": ["look", "think"]}]}),
    })


def test_is_deterministic_true_for_call_skills_false_for_judgment():
    reg = _exec_registry()
    assert is_deterministic(
        Plan(steps=[PlanStep(skill="make_a", params={"a": "1"})]), reg)
    assert not is_deterministic(
        Plan(steps=[PlanStep(skill="judgey")]), reg)
    assert not is_deterministic(
        Plan(steps=[PlanStep(skill="ghost")]), reg)          # unknown
    assert not is_deterministic(Plan(steps=[]), reg)         # empty


# --- executor node: sequencing + carrying output forward ------------------

@tool
def tool_a(x: str) -> str:
    """A."""
    return f"A:{x}"


@tool
def tool_b(y: str) -> str:
    """B."""
    return f"B[{y}]"


async def test_executor_runs_steps_and_passes_output_forward():
    reg = _exec_registry()
    executor = make_executor_node(reg, [tool_a, tool_b])
    plan = Plan(steps=[
        PlanStep(skill="make_a", params={"a": "hello"}),
        PlanStep(skill="make_b", params={}),   # its ${make_a} comes from step 1
    ])
    out = await executor({"plan": plan.model_dump()})

    # make_a ran with x=hello; make_b received make_a's output.
    assert out["step_outputs"]["make_a"] == "A:hello"
    assert out["step_outputs"]["make_b"] == "B[A:hello]"
    summary = out["messages"][0].content
    assert "make_a: tool_a" in summary and "make_b: tool_b" in summary
    assert "Plan complete (2 step(s))" in summary


async def test_executor_stops_and_reports_on_missing_tool():
    reg = _exec_registry()
    executor = make_executor_node(reg, [tool_a])   # tool_b NOT provided
    plan = Plan(steps=[
        PlanStep(skill="make_a", params={"a": "hi"}),
        PlanStep(skill="make_b", params={}),
    ])
    out = await executor({"plan": plan.model_dump()})
    summary = out["messages"][0].content
    assert "tool 'tool_b' unavailable" in summary
    assert out["step_outputs"]["make_a"] == "A:hi"   # first step still ran


async def test_deterministic_plan_routes_through_graph_to_executor():
    # Full path: intake -> planner (emits a deterministic plan) -> router ->
    # executor.  The worker model would error if reached, proving it wasn't.
    reg = _exec_registry()
    planner_model = FakeToolCallingModel(responses=[AIMessage(
        content='{"steps": [{"skill": "make_a", "params": {"a": "hi"}}]}')])
    worker_model = FakeToolCallingModel(responses=[])   # explodes if used
    graph = build_graph(
        worker_model=worker_model, planner_model=planner_model,
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="done: A:hi")]),
        tools=[tool_a, tool_b], worker_prompt="unused", registry=reg,
        classifier=lambda t: "work")

    result = await graph.ainvoke({"messages": [HumanMessage(content="do it")]})
    assert result["step_outputs"]["make_a"] == "A:hi"   # executor ran the plan
