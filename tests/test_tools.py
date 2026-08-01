"""Tests for the MCP tools, driven against the mock frame listener.

These run the real tool bodies through the real transport, so they assert
both the commands emitted to MGED *and* the resulting fake-database state.
"""

from brlcad_mcp.server.tools.boolean import boolean_combination
from brlcad_mcp.server.tools.execution import execute_command
from brlcad_mcp.server.tools.primitives import (
    create_box,
    create_cylinder,
    create_sphere,
)

# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def test_create_sphere(listener):
    result = create_sphere(name="ball.s", x=0, y=0, z=0, radius=10)
    assert "ball.s" in result
    assert "ball.s" in listener.mged.db
    # creation + draw + autoview
    assert listener.received == [
        "in ball.s sph 0 0 0 10",
        "draw ball.s",
        "autoview",
    ]


def test_create_box(listener):
    result = create_box(
        name="block.s", x_min=0, y_min=0, z_min=0, x_max=8, y_max=8, z_max=8
    )
    assert "block.s" in result
    assert listener.received[0] == "in block.s rpp 0 8 0 8 0 8"
    assert "block.s" in listener.mged.db


def test_create_cylinder(listener):
    create_cylinder(
        name="tube.s",
        base_x=0, base_y=0, base_z=0,
        height_x=0, height_y=0, height_z=10,
        radius=3,
    )
    assert listener.received[0] == "in tube.s rcc 0 0 0 0 0 10 3"
    assert "tube.s" in listener.mged.db


# ---------------------------------------------------------------------------
# boolean
# ---------------------------------------------------------------------------

def test_boolean_combination_new_region(listener):
    listener.mged.db.update({"a.s", "b.s"})
    result = boolean_combination(
        output_name="result.r", base_object="a.s", operator="-", target_object="b.s"
    )
    assert "result.r" in result
    assert listener.received[0] == "r result.r u a.s - b.s"
    assert "result.r" in listener.mged.db


def test_boolean_combination_append_to_existing(listener):
    listener.mged.db.update({"hull.r", "hole.s"})
    boolean_combination(
        output_name="hull.r", base_object="hull.r", operator="-", target_object="hole.s"
    )
    # When output == base the region is inspected first (its member count decides
    # whether a flat append is safe), then the op is appended.
    assert listener.received[0] == "l hull.r"
    assert "r hull.r - hole.s" in listener.received


def test_boolean_combination_invalid_operator(listener):
    result = boolean_combination(
        output_name="r.r", base_object="a.s", operator="x", target_object="b.s"
    )
    assert "Error" in result
    assert listener.received == []  # rejected before any command is sent


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------

def test_execute_command_success(listener):
    listener.mged.db.add("ball.s")
    result = execute_command(command="ls")
    assert "ball.s" in result


def test_execute_command_error_tagged(listener):
    result = execute_command(command="frobnicate now")
    assert "[MGED_ERROR]" in result
    assert "frobnicate" in result


def test_execute_command_blocked(listener):
    result = execute_command(command="quit")
    assert "blocked" in result
    assert listener.received == []  # safety gate fires before the wire


def test_execute_command_auto_draw(listener):
    listener.mged.db.add("widget.r")
    execute_command(command="ls", auto_draw=True, object_name="widget.r")
    assert "draw widget.r" in listener.received
    assert "autoview" in listener.received


# --- appending to a region: the r-operator binding gotcha -----------------

def _append_with(monkeypatch, listing):
    """Run _append_cmds with `l <region>` answering *listing*."""
    from brlcad_mcp.server.tools import boolean as B
    monkeypatch.setattr(
        B, "send_command",
        lambda cmd: f"SUCCESS: {listing}" if cmd.startswith("l ") else "SUCCESS:")
    return B._append_cmds("hull.r", "-", "hole.s")


def test_append_union_stays_a_single_command(monkeypatch):
    # A union is unambiguous however many members there are.
    from brlcad_mcp.server.tools import boolean as B
    monkeypatch.setattr(B, "send_command", lambda cmd: "SUCCESS:")
    assert B._append_cmds("hull.r", "u", "x.s") == ["r hull.r u x.s"]


def test_append_to_single_member_region_is_flat(monkeypatch):
    cmds = _append_with(monkeypatch, "hull.r: REGION id=1000\n   u body.s")
    assert cmds == ["r hull.r - hole.s"]


def test_append_to_multi_member_region_refolds_so_the_cut_applies_to_all(
        monkeypatch):
    # THE BUG: a flat `r hull.r - hole.s` would bind the subtraction to only
    # plate_b.s, silently leaving plate_a.s uncut.
    listing = ("hull.r: REGION id=1000\n"
               "   u plate_a.s\n"
               "   u plate_b.s")
    cmds = _append_with(monkeypatch, listing)
    assert cmds == [
        "comb hull.r.app.acc1 u plate_a.s u plate_b.s",
        "kill hull.r",
        "r hull.r u hull.r.app.acc1 - hole.s",
    ]


def test_append_kills_the_region_before_redefining_it(monkeypatch):
    # `r` APPENDS to an existing region, so without the kill the old members
    # stay unioned beside the accumulator and put the cut material back.
    listing = "hull.r: REGION\n   u a.s\n   u b.s"
    cmds = _append_with(monkeypatch, listing)
    assert cmds.index("kill hull.r") < cmds.index(
        "r hull.r u hull.r.app.acc1 - hole.s")
    assert all(not c.startswith("r ") for c in cmds[:cmds.index("kill hull.r")])


def test_append_restores_the_region_colour_after_the_kill(monkeypatch):
    listing = ("hull.r: REGION\nColor 200 100 50\n"
               "   u a.s\n   u b.s")
    cmds = _append_with(monkeypatch, listing)
    assert cmds[-1] == "comb_color hull.r 200 100 50"


def test_region_colour_parsing():
    from brlcad_mcp.server.tools.boolean import _region_colour
    assert _region_colour("x.r: REGION\nColor 1 2 3\n   u a.s") == "1 2 3"
    assert _region_colour("x.r: REGION\n   u a.s") is None


def test_parse_region_members_reads_ops_and_names():
    from brlcad_mcp.server.tools.helpers import parse_region_members
    listing = ("widget.r:  REGION id=1000 (air=0, los=100)\n"
               "Color 150 150 145\n"
               "   u body.s\n"
               "   - hole.s\n"
               "   + lug.s")
    assert parse_region_members(listing) == [
        ("u", "body.s"), ("-", "hole.s"), ("+", "lug.s")]
