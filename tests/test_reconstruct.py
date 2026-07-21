"""Tests for the build_from_spec tool's deterministic pieces.

These cover spec validation and the MGED command generation without a live
BRL-CAD build -- no socket, no rt.
"""

from brlcad_mcp.server.tools import reconstruct as RC
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part


def _spec(**kw):
    return BuildSpec(**kw)


def test_box_solid_command_centers_the_rpp():
    p = Part(name="body", shape="box", center=[0, 0, 0], size=[72, 148, 10])
    # rpp is min/max per axis, centred on `center`
    assert RC._solid_cmd(p) == "in body.s rpp -36 36 -74 74 -5 5"


def test_cylinder_and_sphere_commands():
    cyl = Part(name="cam", shape="cylinder", center=[1, 2, 3],
               height=[0, 0, 10], radius=4)
    assert RC._solid_cmd(cyl) == "in cam.s rcc 1 2 3 0 0 10 4"
    sph = Part(name="dot", shape="sphere", center=[0, 0, 0], radius=5)
    assert RC._solid_cmd(sph) == "in dot.s sph 0 0 0 5"


def test_region_expression_uses_u_and_minus():
    parts = [
        Part(name="body", shape="box", size=[10, 10, 10], op="add"),
        Part(name="hole", shape="sphere", radius=3, op="subtract"),
        Part(name="lug", shape="box", size=[2, 2, 2], op="add"),
    ]
    assert RC._region_expr(parts) == "u body.s - hole.s u lug.s"


def test_validate_accepts_a_good_spec():
    spec = _spec(name="case", parts=[
        Part(name="body", shape="box", size=[72, 148, 10]),
        Part(name="hollow", shape="box", size=[68, 144, 10], op="subtract",
             center=[0, 0, 1]),
    ])
    assert RC._validate(spec) == []


def test_validate_flags_leading_subtract_and_missing_params():
    spec = _spec(parts=[
        Part(name="hole", shape="sphere", op="subtract"),   # first op subtract
    ])
    errors = RC._validate(spec)
    assert any("first part must have op 'add'" in e for e in errors)
    assert any("sphere needs a positive radius" in e for e in errors)


def test_validate_flags_unknown_shape_and_duplicate_names():
    spec = _spec(parts=[
        Part(name="a", shape="box", size=[1, 1, 1]),
        Part(name="a", shape="blob"),
    ])
    errors = RC._validate(spec)
    assert any("duplicate part name 'a'" in e for e in errors)
    assert any("unknown shape 'blob'" in e for e in errors)
