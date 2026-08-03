"""client-v2 debug trace: pure formatting, plus the live callback handler."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from client_v2.terminal.trace import (
    LiveTrace,
    describe_message,
    format_update,
    origin,
    reasoning_summary,
)


def _llm_result(message):
    return LLMResult(generations=[[ChatGeneration(message=message)]])


def test_describe_message_variants():
    assert describe_message(HumanMessage(content="hi there")).startswith("USER:")
    assert describe_message(AIMessage(content="done")).startswith("AI: done")
    tc = AIMessage(content="", tool_calls=[
        {"name": "echo", "args": {"text": "x"}, "id": "1"}])
    d = describe_message(tc)
    assert "tool call" in d and "echo" in d
    tm = ToolMessage(content="pong", name="ping", tool_call_id="1")
    assert describe_message(tm).startswith("TOOL[ping]: pong")


def test_describe_message_extracts_text_from_responses_v1_blocks():
    # responses/v1 AI content is a block list (reasoning + text); the trace
    # must show the text, not the raw list / encrypted reasoning blob.
    msg = AIMessage(content=[
        {"type": "reasoning", "encrypted_content": "BLOB", "summary": []},
        {"type": "text", "text": "Created ball.s."},
    ])
    out = describe_message(msg)
    assert out == "AI: Created ball.s."
    assert "BLOB" not in out


def test_describe_message_clips_long_content():
    long = "x" * 1000
    out = describe_message(AIMessage(content=long))
    assert out.endswith("…") and len(out) < 400


def test_format_update_renders_route_and_messages():
    out = format_update("intake", {"route": "work"})
    assert "▸ intake" in out and "route: work" in out

    out2 = format_update("worker", {"messages": [AIMessage(content="hello")]},
                         ("worker", "abc"))
    assert out2.startswith("[worker:abc]")
    assert "AI: hello" in out2


def test_format_update_handles_non_dict_payload():
    assert "raw value" in format_update("node", "raw value")


def test_format_update_drops_resets_and_keeps_what_carries_information():
    # Intake resets nine fields every turn.  Printing them buries the one line
    # that matters -- which way it routed -- so a reset is not shown; a counter
    # that has actually moved is.
    out = format_update("intake", {
        "route": "work", "turn_start": 3, "revisions": 0, "authorized": False,
        "verification": None, "step_outputs": {}, "step_errors": [],
        "authorization": "",
    })
    assert "route: work" in out
    for gone in ("turn_start", "revisions", "authorized", "verification",
                 "step_outputs", "step_errors", "authorization"):
        assert gone not in out

    assert "revisions: 2" in format_update("planner", {"revisions": 2})


def test_format_update_renders_a_node_that_wrote_nothing():
    # A pass-through (e.g. a visual check with no renders to compare) should read
    # as "ran, wrote nothing", not "wrote None".
    assert format_update("visual_check", None) == "▸ visual_check"


def test_format_update_can_summarise_messages_instead_of_listing_them():
    # With a LiveTrace already printing messages, listing them again doubles
    # every line -- so the node line only counts them.
    update = {"messages": [AIMessage(content="hello"), AIMessage(content="again")]}
    assert "AI: hello" in format_update("worker", update)
    terse = format_update("worker", update, include_messages=False)
    assert "(+2 message(s))" in terse
    assert "hello" not in terse


def test_reasoning_summary_reads_text_only_when_the_model_sent_one():
    assert reasoning_summary(AIMessage(content="plain text")) == ""
    # The usual case: encrypted blob, no summary requested -> nothing to show.
    assert reasoning_summary(AIMessage(content=[
        {"type": "reasoning", "encrypted_content": "BLOB", "summary": []},
        {"type": "text", "text": "done"},
    ])) == ""
    got = reasoning_summary(AIMessage(content=[
        {"type": "reasoning", "encrypted_content": "BLOB",
         "summary": [{"type": "summary_text", "text": "Measuring the bore."}]},
    ]))
    assert got == "Measuring the bore."
    assert "BLOB" not in got


def test_live_trace_prints_tool_calls_and_results():
    lines = []
    trace = LiveTrace(out=lines.append)
    trace.on_tool_start({"name": "build_from_spec"}, "", inputs={"name": "brk"})
    trace.on_tool_end("Built brk.r")
    assert "▸ build_from_spec" in lines[0] and "brk" in lines[0]
    assert "✓ Built brk.r" in lines[1]


def test_origin_names_the_outer_role_not_the_inner_agent_node():
    # Inside the worker's nested agent, langgraph_node is that agent's own node
    # ('model'/'tools'), which does not say whose loop it is; the checkpoint
    # namespace does.
    assert origin({"langgraph_node": "tools", "langgraph_checkpoint_ns":
                   "worker:71f947a4|tools:b8df1534"}) == "worker"
    assert origin({"langgraph_node": "planner",
                   "langgraph_checkpoint_ns": "planner:be0ea67a"}) == "planner"
    # Fall back to the node name, then to nothing, rather than inventing a label.
    assert origin({"langgraph_node": "model"}) == "model"
    assert origin(None) == ""


def test_live_trace_labels_lines_with_the_node_that_produced_them():
    # Without the label, a line printed while the planner is still running reads
    # as though it belonged to intake, whose summary line came just before it.
    lines = []
    trace = LiveTrace(out=lines.append)
    meta = {"metadata": {"langgraph_checkpoint_ns": "worker:abc|tools:def"}}
    trace.on_tool_start({"name": "echo"}, "", inputs={"text": "x"}, **meta)
    trace.on_llm_end(_llm_result(AIMessage(content="done")), **meta)
    assert lines[0].startswith("    [worker] ▸ echo")
    assert lines[1] == "    [worker] AI: done"


def test_live_trace_prints_model_text_but_not_its_tool_call_intent():
    # on_tool_start already reports the invocation with real arguments; echoing
    # the intent too reports every call twice.
    lines = []
    trace = LiveTrace(out=lines.append)
    trace.on_llm_end(_llm_result(AIMessage(content="Built the bracket.")))
    trace.on_llm_end(_llm_result(AIMessage(content="", tool_calls=[
        {"name": "render_model", "args": {"obj": "brk.r"}, "id": "1"}])))
    assert lines == ["    AI: Built the bracket."]


def test_live_trace_prints_a_reasoning_summary_when_present():
    lines = []
    LiveTrace(out=lines.append).on_llm_end(_llm_result(AIMessage(content=[
        {"type": "reasoning", "encrypted_content": "BLOB",
         "summary": [{"type": "summary_text", "text": "Checking the bore."}]},
        {"type": "text", "text": "Verified."},
    ])))
    assert lines == ["    · Checking the bore.", "    AI: Verified."]


def test_live_trace_labels_end_events_from_the_start_events_metadata():
    # LangChain attaches metadata to START events only, so an end event has to be
    # matched back to its run -- otherwise every result line prints unlabelled.
    lines = []
    trace = LiveTrace(out=lines.append)
    start = {"metadata": {"langgraph_checkpoint_ns": "worker:a|tools:b"},
             "run_id": "r1"}
    trace.on_tool_start({"name": "echo"}, "", inputs={"text": "x"}, **start)
    trace.on_tool_end("echoed: x", run_id="r1")          # no metadata here
    assert lines[1].startswith("    [worker]   ✓ echoed: x")

    trace.on_chat_model_start({}, [], metadata={
        "langgraph_checkpoint_ns": "planner:c"}, run_id="r2")
    trace.on_llm_end(_llm_result(AIMessage(content="planned")), run_id="r2")
    assert lines[2] == "    [planner] AI: planned"
    # Finished runs are forgotten, so a long session cannot accumulate them.
    assert trace._origins == {}


def test_live_trace_reports_failures():
    lines = []
    trace = LiveTrace(out=lines.append)
    trace.on_tool_error(RuntimeError("listener closed"))
    trace.on_llm_error(RuntimeError("rate limited"))
    assert "✗" in lines[0] and "listener closed" in lines[0]
    assert "✗" in lines[1] and "rate limited" in lines[1]


def test_live_trace_survives_an_unexpected_llm_payload():
    # A logging/display failure must never take down a turn.
    lines = []
    LiveTrace(out=lines.append).on_llm_end(LLMResult(generations=[]))
    assert lines == []


async def test_live_trace_sees_inside_the_workers_nested_tool_loop():
    """The whole reason this is a callback rather than a stream reader.

    The worker runs its tool loop in a nested ``agent.ainvoke``, so the graph's
    own node updates surface nothing until that loop has finished -- a long build
    printed nothing for minutes and then dumped everything at once.  Callbacks
    propagate into the nested run; if that ever stops being true, the live trace
    silently goes quiet and this fails.
    """
    from langchain_core.tools import tool

    from client_v2.agents.conversational import ROUTE_WORK
    from client_v2.graph import build_graph
    from tests.v2_fakes import FakeToolCallingModel, ai_tool_call

    @tool
    def echo(text: str) -> str:
        """Echo the given text back."""
        return f"echoed: {text}"

    worker_model = FakeToolCallingModel(responses=[
        ai_tool_call("echo", {"text": "hi"}),
        AIMessage(content="all done"),
    ])
    graph = build_graph(worker_model=worker_model, tools=[echo],
                        worker_prompt="unused", classifier=lambda t: ROUTE_WORK)

    lines = []
    await graph.ainvoke({"messages": [HumanMessage(content="echo hi")]},
                        {"callbacks": [LiveTrace(out=lines.append)]})

    printed = "\n".join(lines)
    assert "▸ echo" in printed          # the call, as it was made
    assert "✓ echoed: hi" in printed    # its result
    assert "AI: all done" in printed    # the reply that ended the loop
