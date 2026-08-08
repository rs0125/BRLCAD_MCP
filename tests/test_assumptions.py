"""declare_assumption: the decisions a build rests on, recorded as data."""

import json
import os

import pytest

from brlcad_mcp.server.tools import assumptions as A


def declare(topic, chose, over="", reason="", region=""):
    """Call the tool the way FastMCP does: every argument supplied.

    Calling it as plain Python leaves the unfilled parameters holding pydantic
    ``FieldInfo`` objects -- the schema, not the default -- which is what the
    rest of the tool tests here do too.
    """
    return A.declare_assumption(topic, chose, over, reason, region)


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Point the declaration store at a temp dir.

    Patches ``_specs_root`` rather than ``settings.render.spec_dir``: settings
    is a frozen dataclass and assigning to it raises.
    """
    monkeypatch.setattr(A, "_specs_root", lambda: str(tmp_path / "specs"))
    return tmp_path / "specs" / A.DECLARATIONS_FILE


def test_a_declaration_is_appended_as_one_readable_row(store):
    declare("cavity depth", "6.3 mm", over="1.0 mm roof callout",
            reason="cannot both hold on a 9.6 mm body")
    row, = [json.loads(line) for line in open(store)]
    assert row["topic"] == "cavity depth"
    assert (row["chose"], row["over"]) == ("6.3 mm", "1.0 mm roof callout")


def test_declarations_accumulate_rather_than_replace(store):
    """A build makes several readings; the second must not erase the first."""
    declare("depth", "6.3 mm")
    declare("stud height", "1.8 mm")
    assert [r["topic"] for r in A.read_declarations()] == ["depth", "stud height"]


def test_nothing_declared_is_an_empty_list_not_an_error(store):
    # The normal state of a build that raised no questions.
    assert not os.path.exists(store)
    assert A.read_declarations() == []


def test_a_corrupt_line_is_skipped_rather_than_raising(store):
    declare("depth", "6.3 mm")
    with open(store, "a") as fh:
        fh.write("{ not json\n")
    declare("width", "9.6 mm")
    # A damaged record must not be able to break a build report or a scoring run.
    assert [r["topic"] for r in A.read_declarations()] == ["depth", "width"]


def test_filtering_by_region_keeps_the_unattributed_ones(store):
    """A reading is often settled BEFORE the region exists, so region is blank.

    Dropping those would hide exactly the early decisions that shaped the build.
    """
    declare("depth", "6.3 mm")                    # no region yet
    declare("length", "50 mm", region="bracket")
    declare("thickness", "4 mm", region="washer")
    assert [r["topic"] for r in A.read_declarations("bracket")] == \
        ["depth", "length"]


def test_the_reply_tells_the_agent_what_was_recorded(store):
    out = declare("depth", "6.3 mm", over="1.0 mm")
    assert "6.3 mm" in out and "1.0 mm" in out and "1 assumption" in out


def test_formatting_reads_as_lines_a_person_would_want(store):
    rows = [{"topic": "depth", "chose": "6.3 mm", "over": "1.0 mm",
             "reason": "cannot both hold"}]
    assert A.format_declarations(rows) == \
        "- depth: 6.3 mm (over 1.0 mm) -- cannot both hold"
    assert A.format_declarations([]) == "No assumptions were declared."
