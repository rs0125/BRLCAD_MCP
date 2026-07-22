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


# --- edit ops -------------------------------------------------------------

def _parts():
    return [
        {"name": "body", "shape": "box", "op": "add", "center": [0, 0, 0],
         "size": [10, 10, 5]},
        {"name": "stud", "shape": "cylinder", "op": "add", "center": [2, 0, 5],
         "height": [0, 0, 2], "radius": 1},
    ]


def test_edit_move_is_relative():
    new, errs = RC._apply_edits(_parts(), [{"action": "move", "name": "stud",
                                            "delta": [0, -1, 0]}])
    assert errs == []
    stud = next(p for p in new if p["name"] == "stud")
    assert stud["center"] == [2, -1, 5]


def test_edit_update_sets_fields_and_add_remove():
    edits = [
        {"action": "update", "name": "body", "size": [12, 12, 5]},
        {"action": "add", "part": {"name": "s2", "shape": "sphere",
                                   "center": [0, 0, 8], "radius": 2}},
        {"action": "remove", "name": "stud"},
    ]
    new, errs = RC._apply_edits(_parts(), edits)
    assert errs == []
    names = [p["name"] for p in new]
    assert names == ["body", "s2"]
    assert next(p for p in new if p["name"] == "body")["size"] == [12, 12, 5]


def test_update_does_not_clobber_part_boolean_op():
    # Regression: the edit action key is 'action', so a plain 'update' must NOT
    # set the part's boolean 'op' to "update".
    parts = [{"name": "t", "shape": "cylinder", "op": "subtract",
              "center": [0, 0, 0], "height": [0, 0, 1], "radius": 1}]
    new, errs = RC._apply_edits(parts, [{"action": "update", "name": "t",
                                         "center": [1, 0, 0]}])
    assert errs == []
    assert new[0]["op"] == "subtract"   # unchanged, not "update"
    assert new[0]["center"] == [1, 0, 0]


def test_edit_reports_errors_and_does_not_mutate_input():
    original = _parts()
    new, errs = RC._apply_edits(original, [
        {"action": "move", "name": "ghost", "delta": [1, 0, 0]},
        {"action": "add", "part": {"name": "body"}},   # duplicate
        {"action": "frobnicate"},                       # unknown action
    ])
    assert len(errs) == 3
    assert original == _parts()  # input untouched (ops copy)


def test_spec_history_save_list_and_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "_specs_root", lambda: str(tmp_path))
    assert RC._versions("thing") == []
    assert RC._latest_spec("thing") is None
    RC._save_spec("thing", {"name": "thing", "parts": [{"name": "a"}]})
    RC._save_spec("thing", {"name": "thing", "parts": [{"name": "a"}, {"name": "b"}]})
    versions = RC._versions("thing")
    assert len(versions) == 2
    assert versions[0].endswith("v001.json")
    assert versions[1].endswith("v002.json")
    assert len(RC._latest_spec("thing")["parts"]) == 2
