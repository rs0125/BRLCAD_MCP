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
    # when output == base, the op is appended to the existing region
    assert listener.received[0] == "r hull.r - hole.s"


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
