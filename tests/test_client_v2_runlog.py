"""Per-run JSONL logging: redaction, safety, and what actually gets recorded."""

import json

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from client_v2.runlog import ModelCallLogger, RunLog, null_log, open_run_log, redact


def _read(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- redaction: the log has to stay readable -------------------------------

def test_base64_images_are_replaced_by_their_size():
    # Logging a data URI verbatim would produce multi-megabyte lines and make the
    # log unreadable -- the opposite of the point.
    uri = "data:image/png;base64," + "A" * 5000
    out = redact({"image_url": {"url": uri}})
    assert "AAAA" not in json.dumps(out)
    assert "b64 chars" in out["image_url"]["url"]


def test_long_text_is_clipped_not_dropped():
    out = redact("x" * 9000)
    assert out.endswith("<clipped>") and len(out) < 9000
    assert redact("short") == "short"


def test_messages_are_recorded_structurally():
    out = redact([HumanMessage(content="hi"),
                  AIMessage(content="", tool_calls=[
                      {"name": "build", "args": {"spec": "s"}, "id": "1"}])])
    assert out[0]["type"] == "HumanMessage" and out[0]["content"] == "hi"
    assert out[1]["tool_calls"][0]["name"] == "build"


def test_redaction_survives_odd_objects_and_deep_nesting():
    class Odd:
        pass
    assert isinstance(redact(Odd()), str)
    # Ordinary nesting -- a spec inside tool args -- must come through INTACT.
    # This used to assert that depth 7 was elided, which is precisely the bug:
    # it pinned a cap shallower than a real tool call, so a build's geometry was
    # logged as "<...>".  The cap is a cycle guard now, tested separately.
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    assert redact(deep) == deep


# --- the sink -------------------------------------------------------------

def test_events_are_appended_as_jsonl_with_turn_numbers(tmp_path):
    log = open_run_log(str(tmp_path))
    log.start_turn("build a washer")
    log.event("node", node="planner", wrote={"plan": {"steps": []}})
    log.start_turn("now render it")
    log.event("result", answer="done")
    events = _read(log.path)
    assert [e["kind"] for e in events] == ["turn", "node", "turn", "result"]
    assert [e["turn"] for e in events] == [1, 1, 2, 2]
    assert events[1]["wrote"]["plan"] == {"steps": []}


def test_a_broken_log_never_breaks_a_turn(tmp_path):
    # Losing a log line must not cost a build.
    log = RunLog(str(tmp_path / "no_such_dir" / "run.jsonl"))
    log.event("node", node="planner")      # must not raise
    log.event("node", node="worker")
    assert log._broken


def test_null_log_discards_everything_and_offers_no_callbacks():
    log = null_log()
    log.start_turn("anything")
    log.event("node", node="x")
    assert log.callbacks() == []


# --- model calls ----------------------------------------------------------

def test_model_calls_record_prompt_reply_and_usage(tmp_path):
    log = open_run_log(str(tmp_path))
    handler = ModelCallLogger(log)
    handler.on_chat_model_start({}, [[HumanMessage(content="plan this")]],
                                run_id="r1")
    reply = AIMessage(content="a plan")
    reply.usage_metadata = {"input_tokens": 10, "output_tokens": 4,
                            "total_tokens": 14}
    handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=reply)]]),
                       run_id="r1")
    handler.on_llm_error(ValueError("boom"), run_id="r1")

    kinds = [e["kind"] for e in _read(log.path)]
    assert kinds == ["model_start", "model", "model_error"]
    events = _read(log.path)
    assert events[0]["messages"][0]["content"] == "plan this"
    assert events[1]["reply"]["content"] == "a plan"
    assert events[1]["usage"]["total_tokens"] == 14
    assert "boom" in events[2]["error"]


# --- the spec must survive redaction ---------------------------------------

def test_a_nested_tool_spec_is_recorded_not_elided():
    """THE BUG: the depth cap was 6, shallower than a real tool call.

    A build spec sits at reply -> tool_calls -> [0] -> args -> spec -> parts ->
    part, so a 29-part LEGO brick logged its geometry as ["<...>", ...]: the log
    recorded that a build happened and its verdict, but not what was built.
    """
    from client_v2.runlog import redact
    spec = {"name": "lego_brick_2x4", "parts": [
        {"name": "body", "shape": "box", "op": "add",
         "center": [0, 0, 4.8], "size": [31.8, 15.8, 9.6]},
        {"name": "stud_1", "shape": "cylinder", "op": "add",
         "center": [-12, -4, 9.6], "height": [0, 0, 1.8], "radius": 2.4}]}
    reply = {"type": "AIMessage", "content": "x", "tool_calls": [
        {"name": "build_from_spec", "args": {"spec": spec}}]}

    out = redact(reply)

    parts = out["tool_calls"][0]["args"]["spec"]["parts"]
    assert parts[0]["name"] == "body"
    assert parts[0]["size"] == [31.8, 15.8, 9.6]      # the actual geometry
    assert parts[1]["radius"] == 2.4
    assert "<...>" not in json.dumps(out)


def test_a_real_parts_list_is_not_truncated():
    from client_v2.runlog import redact
    spec = {"parts": [{"name": f"p{i}"} for i in range(29)]}
    assert len(redact(spec)["parts"]) == 29


def test_a_runaway_sequence_is_still_bounded():
    from client_v2.runlog import _MAX_ITEMS, redact
    out = redact(list(range(_MAX_ITEMS + 500)))
    assert len(out) == _MAX_ITEMS + 1              # +1 for the "more items" note
    assert "more items" in out[-1]


def test_depth_is_still_capped_against_a_cycle():
    from client_v2.runlog import redact
    deep = cur = {}
    for _ in range(60):
        cur["next"] = {}
        cur = cur["next"]
    assert "<...>" in json.dumps(redact(deep))      # terminates, does not recurse away


def test_images_are_still_redacted_at_depth():
    from client_v2.runlog import redact
    nested = {"a": {"b": {"c": {"d": {"e": {"f": {"g": {
        "url": "data:image/png;base64," + "A" * 5000}}}}}}}}
    dumped = json.dumps(redact(nested))
    assert "<image," in dumped and "AAAA" not in dumped


def test_encrypted_reasoning_is_stubbed_but_its_id_survives():
    """The ciphertext is unreadable by anyone but OpenAI; the id is the trace signal."""
    from client_v2.runlog import redact
    item = {"id": "rs_abc123", "type": "reasoning", "summary": [],
            "encrypted_content": "gAAAAAB" + "x" * 4993}
    out = redact({"content": [item]})
    logged = out["content"][0]
    assert logged["encrypted_content"] == "<encrypted reasoning, 5000 chars>"
    assert logged["id"] == "rs_abc123"          # continuity across a tool call
    assert "xxxx" not in json.dumps(out)


def test_encrypted_reasoning_reports_its_true_size_not_the_clipped_one():
    """Measured before _MAX_STR truncation, so the number means something."""
    from client_v2.runlog import redact
    out = redact({"encrypted_content": "z" * 12345})
    assert out["encrypted_content"] == "<encrypted reasoning, 12345 chars>"


def test_a_plain_string_named_like_a_key_is_untouched():
    """Only the keyed field is opaque -- the word itself is not a redaction trigger."""
    from client_v2.runlog import redact
    assert redact({"note": "encrypted_content was large"}) == {
        "note": "encrypted_content was large"}
