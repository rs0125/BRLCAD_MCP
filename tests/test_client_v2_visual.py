"""client-v2 visual check: gating, verdict parsing, and the bounded loop."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from client_v2.agents.visual import (
    MAX_VISUAL_ROUNDS,
    find_render_paths,
    has_reference_image,
    make_visual_check_node,
    parse_visual_verdict,
    route_after_visual,
    visual_failure_context,
)
from tests.v2_fakes import FakeToolCallingModel


def _image_message(text="here"):
    return HumanMessage(content=[
        {"type": "text", "text": text},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}])


# --- gating ---------------------------------------------------------------

def test_reference_image_is_detected_only_when_one_is_attached():
    assert has_reference_image({"messages": [_image_message()]})
    assert not has_reference_image({"messages": [HumanMessage(content="text")]})
    assert not has_reference_image({})


def test_render_paths_come_from_step_outputs_and_tool_messages(tmp_path):
    real = tmp_path / "iso.png"
    real.write_bytes(b"x")
    state = {
        "step_outputs": {"build": f"rendered to {real}"},
        "messages": [ToolMessage(content=f"also {real}", name="t",
                                 tool_call_id="1")],
    }
    # Deduplicated, and only paths that actually exist.
    assert find_render_paths(state) == [str(real)]


def test_nonexistent_png_paths_are_ignored():
    state = {"step_outputs": {"b": "wrote /no/such/file.png"}}
    assert find_render_paths(state) == []


async def test_no_model_call_without_renders_or_a_reference(tmp_path):
    model = FakeToolCallingModel(responses=[])      # would raise if called
    node = make_visual_check_node(model)
    # A reference but no renders.
    assert await node({"messages": [_image_message()]}) == {}
    # Renders but no reference.
    png = tmp_path / "a.png"
    png.write_bytes(b"x")
    assert await node({"step_outputs": {"b": str(png)},
                       "messages": [HumanMessage(content="text")]}) == {}


# --- verdict parsing ------------------------------------------------------

def test_match_and_mismatch_are_parsed():
    matched, _ = parse_visual_verdict("MATCH\nBoth show 8 studs.")
    assert matched
    matched, detail = parse_visual_verdict(
        "MISMATCH: the render has 6 studs, the reference has 8\nCount differs.")
    assert not matched and "6 studs" in detail


def test_unparseable_opinion_defaults_to_a_match():
    # A vision opinion must never fail a build that engine truth passed.
    assert parse_visual_verdict("hmm, hard to say")[0]
    assert parse_visual_verdict("")[0]


# --- routing and revision context ----------------------------------------

def test_routing_allows_one_revision_then_carries_on():
    # Routing turns on `spent`, which the NODE sets, not on a round counter.
    # This test used to assert termination at visual_rounds == MAX + 1 -- a state
    # the node can never produce, because it stops incrementing at MAX.  So it
    # proved termination for an unreachable state and passed while the real graph
    # revised until it hit the recursion limit.
    mismatch = {"visual": {"matched": False}, "visual_rounds": 1}
    assert route_after_visual(mismatch) == "revise"
    spent = {"visual": {"matched": False, "spent": True}, "visual_rounds": 1}
    assert route_after_visual(spent) == "ok"
    assert route_after_visual({"visual": {"matched": True}}) == "ok"
    assert route_after_visual({}) == "ok"


async def test_a_declined_round_marks_the_mismatch_spent(tmp_path):
    # The livelock: with its budget gone the node returned a bare {}, leaving the
    # previous round's mismatch live, so the router kept revising on it.
    png = tmp_path / "iso.png"
    png.write_bytes(b"x")
    node = make_visual_check_node(
        FakeToolCallingModel(responses=[AIMessage(content="MISMATCH: nope")]))
    state = {
        "messages": [_image_message(), ToolMessage(content=str(png), name="t",
                                                   tool_call_id="1")],
        "visual": {"matched": False, "detail": "nope"},
        "visual_rounds": MAX_VISUAL_ROUNDS,          # budget already spent
    }
    out = await node(state)
    assert out["visual"]["spent"] is True
    assert out["visual"]["detail"] == "nope"         # detail kept for the report
    assert route_after_visual({**state, **out}) == "ok"


def test_failure_context_tells_the_planner_what_to_change():
    ctx = visual_failure_context(
        {"visual": {"matched": False, "detail": "only 6 studs, expected 8"}})
    assert "only 6 studs" in ctx and "Revise the spec" in ctx
    assert visual_failure_context({"visual": {"matched": True}}) == ""


async def test_check_records_the_verdict_and_counts_the_round(tmp_path):
    png = tmp_path / "iso.png"
    png.write_bytes(b"x")
    model = FakeToolCallingModel(responses=[
        AIMessage(content="MISMATCH: the bracket is a cross, not an L")])
    node = make_visual_check_node(model)
    out = await node({"step_outputs": {"b": str(png)},
                      "messages": [_image_message()]})
    assert out["visual"]["matched"] is False
    assert "cross" in out["visual"]["detail"]
    assert out["visual_rounds"] == 1
    # The renders were actually attached to the comparison request.
    sent = model.calls[0][-1]
    assert sum(1 for p in sent.content
               if isinstance(p, dict) and p.get("type") == "image_url") == 1


def test_only_the_latest_reference_image_is_sent_not_the_whole_history(tmp_path):
    # Resending every image ever attached grew without bound on long sessions
    # (the worker's summarisation middleware does not apply out here).
    from client_v2.agents.visual import reference_message
    older, newer = _image_message("first ref"), _image_message("second ref")
    state = {"messages": [older, HumanMessage(content="chat"), newer]}
    assert reference_message(state) is newer
    assert reference_message({"messages": [HumanMessage(content="text")]}) is None


async def test_comparison_call_carries_exactly_the_reference_and_the_renders(
        tmp_path):
    png = tmp_path / "iso.png"
    png.write_bytes(b"x")
    model = FakeToolCallingModel(responses=[AIMessage(content="MATCH")])
    node = make_visual_check_node(model)
    old_ref = _image_message("an older reference")
    await node({"step_outputs": {"b": str(png)},
                "messages": [old_ref, AIMessage(content="chatter"),
                             _image_message("the reference")]})
    sent = model.calls[0]
    # system + one reference message + the renders message: no transcript.
    assert len(sent) == 3
    assert old_ref not in sent


# --- the crash, end to end -------------------------------------------------

async def test_a_persistent_visual_mismatch_terminates_instead_of_recursing(
        tmp_path):
    """The axlebearing crash, reproduced through the real graph.

    A build that keeps drawing a mismatching render used to loop
    visual_check -> planner -> worker -> verifier -> visual_check until
    GraphRecursionError at 60 supersteps.  With a low recursion limit this test
    fails loudly if the loop ever comes back.
    """
    from langchain_core.tools import tool

    from client_v2.graph import build_graph
    from client_v2.skills import SkillDef, SkillRegistry

    png = tmp_path / "iso.png"
    png.write_bytes(b"x")

    @tool
    def draw_it() -> str:
        """Render the model."""
        return f"rendered to {png}"

    registry = SkillRegistry({"render": SkillDef.model_validate(
        {"id": "render", "description": "renders",
         "steps": [{"call": "draw_it", "with": {}}]})})
    plan = '{"steps": [{"skill": "render", "params": {}}]}'

    graph = build_graph(
        worker_model=FakeToolCallingModel(responses=[]),
        planner_model=FakeToolCallingModel(
            responses=[AIMessage(content=plan)] * 20),
        # Always says the render is wrong -- the worst case for the visual loop.
        visual_model=FakeToolCallingModel(
            responses=[AIMessage(content="MISMATCH: still wrong")] * 20),
        formatter_model=FakeToolCallingModel(
            responses=[AIMessage(content="done")] * 20),
        tools=[draw_it], worker_prompt="unused", registry=registry,
        classifier=lambda t: "work")

    result = await graph.ainvoke(
        {"messages": [_image_message("build me this")]},
        {"recursion_limit": 25})       # far below the 60 that used to blow up

    assert result["visual"]["matched"] is False      # the opinion is reported
    assert result["visual_rounds"] == MAX_VISUAL_ROUNDS


def test_the_latest_render_set_is_judged_not_the_first(tmp_path):
    """A corrective edit_build leaves two render dirs; grading the older one
    reported a mismatch against geometry the edit had already fixed."""
    old, new = tmp_path / "old", tmp_path / "new"
    old.mkdir(), new.mkdir()
    stale = [old / "iso.png", old / "top.png"]
    fresh = [new / "iso.png", new / "top.png"]
    for p in stale + fresh:
        p.write_bytes(b"x")
    state = {"messages": [
        ToolMessage(content=f"Built region. renders:\n{stale[0]}\n{stale[1]}",
                    name="build_from_spec", tool_call_id="1"),
        ToolMessage(content=f"Edited region. renders:\n{fresh[0]}\n{fresh[1]}",
                    name="edit_build", tool_call_id="2"),
    ]}
    assert find_render_paths(state) == [str(fresh[0]), str(fresh[1])]


def test_render_paths_fall_back_to_an_earlier_message_when_the_last_has_none(tmp_path):
    """A trailing non-render tool result must not blank the comparison."""
    png = tmp_path / "iso.png"
    png.write_bytes(b"x")
    state = {"messages": [
        ToolMessage(content=f"renders: {png}", name="build_from_spec",
                    tool_call_id="1"),
        ToolMessage(content="Verification: PASS", name="verify_model_dimensions",
                    tool_call_id="2"),
    ]}
    assert find_render_paths(state) == [str(png)]


def test_renders_from_an_earlier_turn_are_not_judged(tmp_path):
    """A render is only evidence about the database it was made from.

    Observed live: turn 1 built and rendered a sphere, turn 2 deleted it, and
    turn 3 attached a cake sketch.  The visual check compared the sketch
    against turn 1's sphere renders and reported "the render has 0 tiers and 0
    studs" -- true of the picture, false of the database, which was empty.  The
    resulting MISMATCH then drove a real replan.  The verifier already bounds
    its scan by turn_start; this node did not.
    """
    old = tmp_path / "old_iso.png"
    old.write_bytes(b"\x89PNG stale")

    state = {
        # message 0 belongs to a previous turn
        "messages": [ToolMessage(content=f"Check renders in:\n  iso: {old}",
                                 tool_call_id="t0", name="build_from_spec")],
        "turn_start": 1,          # this turn starts after that message
    }
    assert find_render_paths(state) == []


def test_renders_from_the_current_turn_are_judged(tmp_path):
    new = tmp_path / "new_iso.png"
    new.write_bytes(b"\x89PNG fresh")
    state = {
        "messages": [ToolMessage(content=f"iso: {new}", tool_call_id="t1",
                                 name="build_from_spec")],
        "turn_start": 0,
    }
    assert find_render_paths(state) == [str(new)]


def test_a_missing_turn_start_still_works(tmp_path):
    # Defensive: state built by a test or an older checkpoint may not have it.
    p = tmp_path / "iso.png"
    p.write_bytes(b"\x89PNG")
    state = {"messages": [ToolMessage(content=f"iso: {p}", tool_call_id="t",
                                      name="build_from_spec")]}
    assert find_render_paths(state) == [str(p)]
