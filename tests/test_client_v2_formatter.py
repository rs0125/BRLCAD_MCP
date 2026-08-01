"""client-v2 formatter: when it runs, what it sees, and the final answer."""

from langchain_core.messages import AIMessage, HumanMessage

from client_v2.agents.formatter import (
    make_formatter_node,
    needs_formatting,
    results_digest,
    route_after_verify_to_report,
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


def test_routing_after_verify_picks_format_or_done():
    assert route_after_verify_to_report(
        {"step_outputs": {"a": "1"}}) == "format"
    assert route_after_verify_to_report({}) == "done"


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
