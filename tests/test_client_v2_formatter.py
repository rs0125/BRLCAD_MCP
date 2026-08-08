"""client-v2 formatter: when it runs, what it sees, and the final answer."""

from langchain_core.messages import AIMessage, HumanMessage

from client_v2.agents.conversational import message_text
from client_v2.agents.formatter import (
    make_formatter_node,
    needs_formatting,
    results_digest,
)
from tests.v2_fakes import FakeToolCallingModel


def test_executor_output_needs_formatting():
    assert needs_formatting({"step_outputs": {"make_a": "A:hi"}})
    assert needs_formatting({"step_errors": ["Error: boom"]})


def test_failed_verification_needs_explaining():
    assert needs_formatting({"verification": {"passed": False}})


def test_plain_worker_prose_is_left_alone():
    # The worker already wrote a user-facing answer -> skip the extra call.
    assert not needs_formatting({"verification": {"passed": True}})
    assert not needs_formatting({})


def test_results_digest_surfaces_verdict_outputs_and_errors():
    digest = results_digest({
        "verification": {"checked": True, "passed": False,
                         "failures": ["hole h1 not cut"]},
        "step_outputs": {"build": "Built region 'plate.r'"},
        "step_errors": ["Error: render failed"],
    })
    assert "Verification: FAILED" in digest
    assert "hole h1 not cut" in digest
    assert "Built region 'plate.r'" in digest
    assert "Error: render failed" in digest


def test_results_digest_handles_an_empty_turn():
    assert "no results recorded" in results_digest({})


async def test_formatter_writes_the_final_message():
    model = FakeToolCallingModel(responses=[AIMessage(content="Built plate.r")])
    formatter = make_formatter_node(model)
    out = await formatter({
        "messages": [HumanMessage(content="build a plate")],
        "step_outputs": {"build_model_spec": "Built region 'plate.r'"},
    })
    assert out["messages"][0].content == "Built plate.r"
    # It was shown the request and the results, not raw internals.
    prompt = str(model.calls[0][1].content)
    assert "build a plate" in prompt and "plate.r" in prompt


# --- an unresolved visual mismatch must be disclosed ------------------------

_LEGO_REPORT = (
    "1) Completed region lego_brick_sharp.r\n"
    "2) Engine verification PASS, bbox 31.8 x 15.8 x 11.4 mm, 87 rays matched.\n"
    "5) Check renders: /home/x/brlcad_renders/r_201603/bottom.png")
_MISMATCH = {"matched": False, "renders": ["/home/x/bottom.png"],
             "detail": "The underside is oversimplified; add the 8 smaller "
                       "circular recesses shown in the reference."}


def _visual_turn():
    return {
        "messages": [HumanMessage(content="draw this"),
                     AIMessage(content=_LEGO_REPORT)],
        "turn_start": 1,
        "verification": {"passed": True, "checked": True, "failures": []},
        "visual": _MISMATCH,
    }


def test_an_unresolved_visual_mismatch_needs_reporting():
    """THE BUG: the turn ended at the worker's own message.

    That message was written BEFORE the visual check ran, so a real finding --
    "0 of the reference's 8 underside recesses" -- was computed, stored, logged
    and silently dropped.  The user saw PASS plus a feature list the render
    contradicted.  It is also the one defect engine truth cannot catch: the build
    matched its spec, so all 87 rays passed.
    """
    assert needs_formatting(_visual_turn())


def test_a_matching_visual_on_a_clean_turn_still_skips_the_model_call():
    state = _visual_turn()
    state["visual"] = {"matched": True, "detail": "MATCH"}
    assert not needs_formatting(state)
    assert not needs_formatting({"verification": {"passed": True}})


def test_the_digest_states_the_discrepancy_and_that_rays_cannot_cover_it():
    digest = results_digest(_visual_turn())
    assert "MISMATCH" in digest and "NOT resolved" in digest
    assert "8 smaller circular recesses" in digest
    assert "cannot cover this" in digest
    assert "do not describe the model as matching" in digest


def test_the_digest_keeps_the_agents_own_concrete_values():
    # Otherwise routing to the formatter LOSES the region name, dimensions and
    # render paths: on the worker path step_outputs is empty.
    digest = results_digest(_visual_turn())
    assert "lego_brick_sharp.r" in digest
    assert "31.8 x 15.8 x 11.4" in digest
    assert "r_201603/bottom.png" in digest


def test_the_report_is_scoped_to_this_turn():
    from client_v2.agents.formatter import turn_report
    state = {"messages": [AIMessage(content="an older turn's answer"),
                          HumanMessage(content="new request"),
                          AIMessage(content="this turn's answer")],
             "turn_start": 2}
    assert turn_report(state) == "this turn's answer"
    assert turn_report({}) == ""


async def test_the_final_answer_carries_the_mismatch_through_the_graph():
    """End to end on the WORKER path -- the exact shape of the real failure.

    It has to be the worker path: on the executor path `step_outputs` is
    populated, so the formatter already ran and the finding was reported. The
    silent drop only happened when the worker did the work, because then
    `step_outputs` is empty and a passing verification routed straight to END --
    ending the turn at the worker's own message, written before the visual check.
    """
    import pathlib
    import tempfile

    from langchain_core.tools import tool
    from langgraph.checkpoint.memory import MemorySaver

    from client_v2.graph import build_graph
    from client_v2.skills import SkillDef, SkillRegistry
    from tests.v2_fakes import ai_tool_call

    png = pathlib.Path(tempfile.mkdtemp()) / "iso.png"
    png.write_bytes(b"x")

    @tool
    def draw_it() -> str:
        """Render."""
        return f"Built region 'brick.r'; render at {png}"

    # A judgment step, so the plan is NOT deterministically executable and the
    # work goes to the worker -- leaving step_outputs empty.
    registry = SkillRegistry({"shape_it": SkillDef.model_validate(
        {"id": "shape_it", "description": "needs judgement",
         "steps": [{"understand": ["look at the drawing"]}]})})
    plan = '{"steps": [{"skill": "shape_it", "params": {}}]}'

    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[
            ai_tool_call("draw_it", {}),
            AIMessage(content="Built brick.r. Verification PASS. "
                              f"Render: {png}"),
            AIMessage(content="Nothing further to change."),
        ]),
        planner_model=FakeToolCallingModel(
            responses=[AIMessage(content=plan)] * 6),
        visual_model=FakeToolCallingModel(responses=[
            AIMessage(content="MISMATCH: 8 underside recesses are missing")] * 6),
        formatter_model=FakeToolCallingModel(responses=[AIMessage(
            content="Built brick.r. Outstanding: 8 underside recesses are "
                    "missing versus the reference.")] * 6),
        tools=[draw_it], worker_prompt="unused", registry=registry,
        classifier=lambda t: "work", checkpointer=MemorySaver())

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=[
            {"type": "text", "text": "draw this"},
            {"type": "image_url",
             "image_url": {"url": "data:image/png;base64,AA"}}])]},
        {"configurable": {"thread_id": "vis"}})

    assert not result.get("step_outputs")        # worker path: nothing here
    assert result["visual"]["matched"] is False
    final = message_text(result["messages"][-1])
    assert "Outstanding" in final                # the finding reached the answer
