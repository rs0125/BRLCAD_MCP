"""client-v2 verifier: verdict reading, kick-back routing, and the bounded
verify -> replan loop through the graph."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

import client_v2.agents.verifier as VERIFIER
from client_v2.agents.verifier import (
    MAX_REVISIONS,
    evaluate,
    failure_context,
    route_after_verify,
    turn_texts,
)
from client_v2.graph import build_graph
from client_v2.skills import SkillDef, SkillRegistry
from tests.v2_fakes import FakeToolCallingModel

# --- evaluate (pure) ------------------------------------------------------

def test_pass_verdict_is_checked_and_passing():
    v = evaluate(["Verification of 'plate.r': PASS\n  [ok] bbox: ..."])
    assert v.checked and v.passed and v.failures == []


def test_fail_verdict_is_caught():
    v = evaluate(["Verification of 'plate.r': FAIL\n  [x] hole:h1: NO through-hole"])
    assert v.checked and not v.passed
    assert "plate.r" in v.failures[0]


def test_a_tagged_mged_rejection_is_seen_but_does_not_fail_the_turn():
    """One command MGED rejected is not a verdict on the model.

    Temporary while ray checks are off: with verification disabled there is
    often no PASS to supersede a rejection the agent already worked around, so
    it was surfacing to the user as "verification failed". Still RECOGNISED
    (checked stays True) -- just not counted.
    """
    v = evaluate(["[MGED_ERROR] Command failed.\nCommand: in x\n"])
    assert v.checked
    assert v.passed
    assert v.failures == []


def test_strict_mode_counts_a_tagged_rejection_again(monkeypatch):
    monkeypatch.setattr(VERIFIER, "COMMAND_ERRORS_FAIL_TURN", True)
    v = evaluate(["[MGED_ERROR] Command failed.\nCommand: in x\n"])
    assert v.checked and not v.passed


def test_a_prefix_error_still_fails_even_though_a_tag_does_not():
    """The two must not be conflated: this is how an unavailable tool and an
    unreachable listener report, and both mean nothing ran."""
    assert not evaluate(["Error: build_from_spec: tool 'x' unavailable"]).passed
    assert not evaluate(
        ["Error: Could not reach libmcpcad listener at /tmp/a.sock"]).passed


def test_unverifiable_turn_passes_through_without_looping():
    # Nothing recognisable -> checked False, passed True (so no kick-back).
    v = evaluate(["Built region 'plate.r' from 3 part(s)", "echoed: hi"])
    assert not v.checked and v.passed


def test_turn_texts_gathers_step_outputs_and_tool_messages():
    state = {
        "step_outputs": {"make_a": "A:hi"},
        "step_errors": ["Error: boom"],
        "messages": [HumanMessage(content="q"),
                     ToolMessage(content="tool said", name="t", tool_call_id="1")],
    }
    texts = turn_texts(state)
    assert "A:hi" in texts and "Error: boom" in texts and "tool said" in texts
    assert "q" not in texts        # human turns aren't results


# --- routing --------------------------------------------------------------

def test_route_done_on_pass_and_revise_on_fail():
    assert route_after_verify({"verification": {"passed": True}}) == "done"
    assert route_after_verify(
        {"verification": {"passed": False}, "revisions": 0}) == "revise"


def test_route_gives_up_when_revision_budget_is_spent():
    assert route_after_verify({"verification": {"passed": False},
                               "revisions": MAX_REVISIONS}) == "done"


def test_failure_context_lists_failures_for_the_planner():
    ctx = failure_context({"verification": {"failures": ["hole h1 not cut"]}})
    assert "FAILED" in ctx and "hole h1 not cut" in ctx
    assert failure_context({"verification": {"failures": []}}) == ""


# --- the loop, end to end through the graph -------------------------------

@tool
def tool_a(x: str) -> str:
    """A."""
    return f"A:{x}"


def _loop_registry():
    return SkillRegistry({
        "make_a": SkillDef.model_validate({
            "id": "make_a", "description": "works",
            "steps": [{"call": "tool_a", "with": {"x": "${a}"}}]}),
        "make_bad": SkillDef.model_validate({
            "id": "make_bad", "description": "calls a missing tool",
            "steps": [{"call": "absent_tool", "with": {}}]}),
    })


_BAD_PLAN = '{"steps": [{"skill": "make_bad", "params": {}}]}'
_GOOD_PLAN = '{"steps": [{"skill": "make_a", "params": {"a": "hi"}}]}'


async def test_failed_step_kicks_back_to_planner_and_succeeds_on_retry():
    # 1st plan hits a missing tool -> verifier fails -> planner revises -> 2nd
    # plan runs clean.  Proves the verify -> replan loop actually closes.
    planner_model = FakeToolCallingModel(responses=[
        AIMessage(content=_BAD_PLAN), AIMessage(content=_GOOD_PLAN)])
    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=planner_model, tools=[tool_a],
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="final summary")] * 4),
        worker_prompt="unused", registry=_loop_registry(),
        classifier=lambda t: "work")

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

    assert result["step_outputs"]["make_a"] == "A:hi"    # retry did the work
    assert result["verification"]["passed"] is True
    assert result["revisions"] == 1                      # exactly one revision


async def test_loop_terminates_when_every_attempt_fails():
    # Always-failing plan: the loop must stop at the revision budget, not spin.
    planner_model = FakeToolCallingModel(
        responses=[AIMessage(content=_BAD_PLAN)] * (MAX_REVISIONS + 2))
    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=planner_model, tools=[tool_a],
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="final summary")] * 4),
        worker_prompt="unused", registry=_loop_registry(),
        classifier=lambda t: "work")

    result = await graph.ainvoke({"messages": [HumanMessage(content="go")]})

    assert result["verification"]["passed"] is False     # reported, not hidden
    assert result["revisions"] == MAX_REVISIONS          # bounded


# --- turn scoping (regressions) -------------------------------------------

def test_an_earlier_turns_failure_does_not_fail_this_turn():
    # THE BUG: the verifier scanned the whole transcript, so one failed turn
    # then failed every later turn -- and the eval never caught it because each
    # case ran on a fresh thread.
    history = [
        HumanMessage(content="build a plate"),
        ToolMessage(content="Verification of 'plate.r': FAIL\n [x] geometry: bad",
                    name="verify_model_dimensions", tool_call_id="1"),
        AIMessage(content="that failed"),
        HumanMessage(content="now build a washer"),      # <- turn 2 starts
        ToolMessage(content="Verification of 'washer.r': PASS",
                    name="verify_model_dimensions", tool_call_id="2"),
    ]
    verdict = evaluate(turn_texts({"messages": history, "turn_start": 3}))
    assert verdict.passed and verdict.checked


def test_without_a_boundary_it_still_sees_everything():
    # Sanity check on the mechanism itself: turn_start is what scopes it.
    history = [
        ToolMessage(content="Verification of 'plate.r': FAIL",
                    name="v", tool_call_id="1"),
        HumanMessage(content="next"),
    ]
    assert not evaluate(turn_texts({"messages": history, "turn_start": 0})).passed
    assert evaluate(turn_texts({"messages": history, "turn_start": 2})).checked is False


# --- what counts as a tool failure ---------------------------------------

def test_a_successful_build_mentioning_an_error_is_not_a_failure():
    # A build that succeeded but whose check render timed out used to fail the
    # whole turn and trigger a pointless replan.
    verdict = evaluate([
        "Built region 'x.r' from 2 part(s):\n  front: FAILED - Error: rt timeout"])
    assert verdict.passed


def test_a_result_that_starts_with_error_is_a_failure():
    verdict = evaluate(["Error: refusing to overwrite existing object(s)"])
    assert verdict.checked and not verdict.passed


def test_the_explicit_mged_error_tag_is_found_anywhere_in_the_output(monkeypatch):
    # The tag is detected mid-result, not only at the start -- shown under
    # strict mode, since by default a tagged rejection does not fail the turn.
    monkeypatch.setattr(VERIFIER, "COMMAND_ERRORS_FAIL_TURN", True)
    verdict = evaluate(["Command: in x\n[MGED_ERROR] Command failed."])
    assert verdict.checked and not verdict.passed


async def test_a_bad_turn_does_not_poison_the_next_turn_on_the_same_thread():
    """The multi-turn regression the per-case eval could not see.

    Turn 1 fails and burns its whole revision budget.  Turn 2 is a clean request
    on the SAME thread, so it shares the checkpointed state.  Before turn-scoping
    it inherited turn 1's failure AND its spent budget, so it reported failure
    without ever getting a retry -- every session degraded permanently after one
    bad turn.
    """
    from langgraph.checkpoint.memory import MemorySaver

    planner_model = FakeToolCallingModel(responses=[
        AIMessage(content=_BAD_PLAN),        # turn 1: fails...
        AIMessage(content=_BAD_PLAN),        # ...and its two revisions also fail
        AIMessage(content=_BAD_PLAN),
        AIMessage(content=_GOOD_PLAN),       # turn 2: a clean plan
    ])
    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=planner_model, tools=[tool_a],
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="summary")] * 6),
        worker_prompt="unused", registry=_loop_registry(),
        classifier=lambda t: "work", checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "same-thread"}}

    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="build the broken thing")]}, config)
    assert first["verification"]["passed"] is False
    assert first["revisions"] == MAX_REVISIONS          # budget spent

    second = await graph.ainvoke(
        {"messages": [HumanMessage(content="now build the good thing")]}, config)
    assert second["step_outputs"]["make_a"] == "A:hi"   # it actually ran
    assert second["verification"]["passed"] is True     # not poisoned
    assert second["revisions"] == 0                     # budget replenished


# --- order-aware evaluation: a fixed error must not fail the turn ----------

_DB_ERR = "Error: failed to create 'base' (box): A database is not open!"
_PASS = ("Verification of 'p004_bearing.r': PASS\n"
         "  [ok] bbox: expected (100.0, 24.0, 55.0) mm, got (100.0, 24.0, 55.0) mm\n"
         "  [ok] geometry: all 72 sample rays match the spec")


def test_an_error_that_was_retried_and_verified_does_not_fail_the_turn():
    """The axlebearing turn, reduced.

    THE BUG: the first build failed (no database open), the agent found `opendb`,
    retried, and the region verified PASS with an exact bbox -- and the turn was
    still failed on its own first attempt, 12 times running, because the verdict
    was read without regard to order.  Retry-then-succeed is the most common
    shape of a successful turn.
    """
    verdict = evaluate([_DB_ERR, "opendb ok", "Built region 'p004_bearing.r'", _PASS])
    assert verdict.checked and verdict.passed
    assert verdict.failures == []


def test_an_error_after_the_last_verdict_still_fails():
    # A PASS only supersedes what came BEFORE it.
    verdict = evaluate([_PASS, "Error: kill failed"])
    assert verdict.checked and not verdict.passed


def test_a_pass_supersedes_an_earlier_fail_for_the_same_region():
    verdict = evaluate(["Verification of 'plate.r': FAIL\n [x] bbox: bad",
                        "edit_build applied",
                        "Verification of 'plate.r': PASS"])
    assert verdict.passed


def test_a_pass_for_one_region_does_not_clear_another_regions_fail():
    verdict = evaluate(["Verification of 'a.r': FAIL", "Verification of 'b.r': PASS"])
    assert not verdict.passed
    assert any("a.r" in f for f in verdict.failures)


def test_an_error_with_no_later_verdict_still_fails():
    assert not evaluate([_DB_ERR]).passed


# --- the loops terminate ---------------------------------------------------

def test_revision_budget_is_shared_by_every_loop_back():
    from client_v2.agents.verifier import revision_budget_spent
    assert not revision_budget_spent({"revisions": MAX_REVISIONS - 1})
    assert revision_budget_spent({"revisions": MAX_REVISIONS})


def test_a_spent_visual_mismatch_stops_routing_back_to_the_planner():
    from client_v2.agents.visual import route_after_visual
    from client_v2.graph import route_after_visual_check
    live = {"visual": {"matched": False}, "revisions": 0}
    spent = {"visual": {"matched": False, "spent": True}, "revisions": 0}
    assert route_after_visual(live) == "revise"       # one revision is allowed
    assert route_after_visual(spent) == "ok"          # ...and only one
    assert route_after_visual_check(spent) != "revise"


def test_the_visual_edge_also_obeys_the_revision_budget():
    from client_v2.graph import route_after_visual_check
    out_of_budget = {"visual": {"matched": False}, "revisions": MAX_REVISIONS,
                     "verification": {"passed": False}}
    assert route_after_visual_check(out_of_budget) != "revise"
