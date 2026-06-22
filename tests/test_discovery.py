"""Tests for the discovery tools (command catalog + live help)."""

from brlcad_mcp.server.tools.discovery import get_command_help, list_commands


def test_list_commands_static_catalog(listener):
    # static catalog is always available, no live query
    out = list_commands(category=None, query_live=False)
    assert "in" in out
    assert "ls" in out
    assert listener.received == []  # purely static


def test_list_commands_category_filter(listener):
    out = list_commands(category="primitives", query_live=False)
    assert "in" in out


def test_list_commands_live_merge(listener):
    # query_live hits the listener's '?' command
    list_commands(category=None, query_live=True)
    assert listener.received[-1] == "?"


def test_get_command_help_queries_listener(listener):
    out = get_command_help(command="in")
    assert listener.received[-1] == "help in"
    assert "in" in out


def test_get_command_help_includes_static_summary(listener):
    out = get_command_help(command="ls")
    # ls is in the static catalog, so a summary is prepended
    assert "Summary" in out
