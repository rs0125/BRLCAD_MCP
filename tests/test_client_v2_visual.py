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
    mismatch = {"visual": {"matched": False}, "visual_rounds": 1}
    assert route_after_visual(mismatch) == "revise"
    spent = {"visual": {"matched": False},
             "visual_rounds": MAX_VISUAL_ROUNDS + 1}
    assert route_after_visual(spent) == "ok"
    assert route_after_visual({"visual": {"matched": True}}) == "ok"
    assert route_after_visual({}) == "ok"


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
