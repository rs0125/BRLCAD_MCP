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


def test_unknown_command_says_so_instead_of_sending_the_agent_diagnosing():
    # A weaker model tried to run a *skill* id as an MGED command, got the
    # generic "call analyze_command_error" tip, and looped on it: diagnosing is
    # the wrong next move when the name simply is not a command.  Observed on
    # gpt-4o-mini, which retried `extract_dimension_constraints` repeatedly.
    out = check_mged_result("ERROR: unknown command: extract_dimension_constraints",
                            command="extract_dimension_constraints")
    assert out is not None
    assert "not an MGED command" in out
    # The loop is what has to be broken, not the mention of diagnosis: for a
    # genuine typo (kil -> kill) diagnosing IS the right move, so the tip stays
    # and the anti-loop instruction is added alongside it.
    assert "Do NOT send the same name again" in out
    assert "use the matching tool instead" in out


def test_other_errors_keep_the_diagnose_tip():
    out = check_mged_result("ERROR: bad argument count", command="in x sph")
    assert out is not None
    assert "analyze_command_error" in out
