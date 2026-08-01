"""client-v2 debug-trace formatting (pure)."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from client_v2.terminal.trace import describe_message, format_update


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
