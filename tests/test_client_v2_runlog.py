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
    deep = {"a": {"b": {"c": {"d": {"e": {"f": {"g": 1}}}}}}}
    assert "<...>" in json.dumps(redact(deep))


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
