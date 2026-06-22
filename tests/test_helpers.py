"""Tests for the shared response-parsing helpers."""

from brlcad_mcp.server.tools.helpers import (
    check_mged_result,
    is_error_response,
    parse_response,
)


def test_is_error_response():
    assert is_error_response("ERROR: boom")
    assert not is_error_response("SUCCESS: ok")
    assert not is_error_response("plain text")


def test_parse_response_strips_prefixes():
    assert parse_response("SUCCESS: ball.s") == "ball.s"
    assert parse_response("ERROR: not found") == "not found"
    assert parse_response("no prefix here") == "no prefix here"


def test_check_mged_result_passes_on_success():
    assert check_mged_result("SUCCESS: done", command="ls") is None


def test_check_mged_result_flags_errors():
    msg = check_mged_result("ERROR: unknown command: frob", command="frob")
    assert msg is not None
    assert "[MGED_ERROR]" in msg
    assert "frob" in msg
    assert "analyze_command_error" in msg  # nudges the recovery workflow
