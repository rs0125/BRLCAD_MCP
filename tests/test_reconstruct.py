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


def test_region_build_folds_left_to_right():
    # BRL-CAD's flat 'r' would bind trailing subtractions to only the last
    # union operand; we fold through intermediate combs so each operator
    # applies to the whole accumulated solid.
    parts = [
        Part(name="body", shape="box", size=[10, 10, 10], op="add"),
        Part(name="hole", shape="sphere", radius=3, op="subtract"),
        Part(name="lug", shape="box", size=[2, 2, 2], op="add"),
    ]
    cmds = RC._region_build_cmds("widget", parts)
    # (body - hole) as an intermediate comb, then unioned with lug in the region.
    # Solids are namespaced under the region so parts never collide globally.
    assert cmds == [
        "comb widget.acc1 u widget_body.s - widget_hole.s",
        "r widget.r u widget.acc1 u widget_lug.s",
    ]


def test_region_build_subtractions_apply_to_whole_union():
    # The angle-bracket bug: two plates then two holes.  Both holes must be
    # subtracted from the union of BOTH plates, not just the last one.
    parts = [
        Part(name="left_plate", shape="box", size=[2.5, 50, 50]),
        Part(name="right_plate", shape="box", size=[50, 2.5, 50]),
        Part(name="left_hole", shape="cylinder", height=[4.5, 0, 0], radius=6,
             op="subtract"),
        Part(name="right_hole", shape="cylinder", height=[0, 12.5, 0], radius=6,
             op="subtract"),
    ]
    cmds = RC._region_build_cmds("angle_bracket", parts)
    assert cmds == [
        "comb angle_bracket.acc1 u angle_bracket_left_plate.s "
        "u angle_bracket_right_plate.s",
        "comb angle_bracket.acc2 u angle_bracket.acc1 "
        "- angle_bracket_left_hole.s",
        "r angle_bracket.r u angle_bracket.acc2 - angle_bracket_right_hole.s",
    ]


def test_region_build_single_and_pair():
    one = [Part(name="body", shape="box", size=[1, 1, 1])]
    assert RC._region_build_cmds("m", one) == ["r m.r u m_body.s"]
    pair = [Part(name="body", shape="box", size=[1, 1, 1]),
            Part(name="hole", shape="sphere", radius=1, op="subtract")]
    # A single add+subtract pair binds correctly even flat -- no comb needed.
    assert RC._region_build_cmds("m", pair) == ["r m.r u m_body.s - m_hole.s"]


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


def test_validate_flags_an_unknown_view():
    # Caught up front: otherwise the geometry builds and every check render
    # fails one-by-one with "unknown view".
    spec = BuildSpec(name="x", views=["iso", "bogus"],
                     parts=[Part(name="b", shape="box", size=[1, 1, 1])])
    errors = RC._validate(spec)
    assert any("unknown view 'bogus'" in e for e in errors)


def test_an_ignored_lighting_value_cannot_reject_a_build():
    # `lighting` is accepted for old saved specs but never used -- check views are
    # always ambient.  Validating it meant a DEAD field could block a perfectly
    # good build, so the membership check was removed.
    spec = BuildSpec(name="x", lighting="whatever_the_model_invented",
                     parts=[Part(name="b", shape="box", size=[1, 1, 1])])
    assert RC._validate(spec) == []


def test_setting_lighting_is_reported_as_ignored():
    # The tool's own example used to show "lighting": "studio", so a model doing
    # exactly what we documented got a silent no-op.  Say so instead.
    spec = BuildSpec(name="x", lighting="studio", expect_bbox=[1, 1, 1],
                     parts=[Part(name="b", shape="box", size=[1, 1, 1])])
    report = RC._report(spec, None, [], "Built region 'x.r'")
    assert "IGNORED" in report and "studio" in report
    quiet = BuildSpec(name="x", expect_bbox=[1, 1, 1],
                      parts=[Part(name="b", shape="box", size=[1, 1, 1])])
    assert "IGNORED" not in RC._report(quiet, None, [], "Built region 'x.r'")


def test_the_tool_example_no_longer_advertises_lighting():
    # Fixing the field but leaving the example in place would keep teaching it:
    # the model set lighting because our own spec example showed it.
    import inspect
    src = inspect.getsource(RC.build_from_spec)
    assert '"lighting"' not in src


# --- collision guard (guardrail as tooling) -------------------------------

def test_collision_guard_refuses_to_clobber_foreign_geometry(monkeypatch):
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)   # not ours
    spec = BuildSpec(name="widget", parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    err = RC._collision_error(spec, {"widget.r", "other.s"})
    assert err and "widget.r" in err and "restore_backup" in err


def test_collision_guard_allows_rebuilding_our_own_spec(monkeypatch):
    # A saved spec means we built it -> rebuilding in place is expected.
    monkeypatch.setattr(RC, "_latest_spec", lambda name: {"name": "widget"})
    spec = BuildSpec(name="widget", parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    assert RC._collision_error(spec, {"widget.r", "body.s"}) is None


def test_collision_guard_allows_fresh_names(monkeypatch):
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)
    spec = BuildSpec(name="fresh", parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    assert RC._collision_error(spec, {"unrelated.s"}) is None


def test_live_names_strips_ls_decorations(monkeypatch):
    # Regions are listed as 'a.r/R' -- the marker must be stripped so a name
    # comparison against the live database actually matches.
    monkeypatch.setattr(RC, "send_command",
                        lambda c: "SUCCESS: a.r/R  b.s  c.c/  _GLOBAL@")
    assert RC._live_names() == {"a.r", "b.s", "c.c", "_GLOBAL"}


def test_part_solids_are_namespaced_under_the_region():
    # Two models can both have a part called "body" without colliding.
    assert RC._solid_name("bushing", "body") == "bushing_body.s"
    p = Part(name="body", shape="box", center=[0, 0, 0], size=[2, 2, 2])
    assert RC._solid_cmd(p, "bushing").startswith("in bushing_body.s rpp")


def test_collision_guard_uses_namespaced_solid_names(monkeypatch):
    # A pre-existing generic 'body.s' must NOT block a namespaced build.
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)
    spec = BuildSpec(name="bushing", parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    assert RC._collision_error(spec, {"body.s"}) is None
    # ...but its own namespaced name still guards.
    assert RC._collision_error(spec, {"bushing_body.s"}) is not None


def test_through_hole_in_one_flange_of_a_multi_part_model_is_not_a_pocket():
    # Regression: judging "through" against the WHOLE model's extent made a hole
    # through an L-bracket's 2.5 mm upright look like a blind pocket, because the
    # union spans 50 mm in X.  It must be judged against the flange it crosses.
    spec = BuildSpec(name="br", parts=[
        Part(name="flange_yz", shape="box", center=[1.25, 25, 25],
             size=[2.5, 50, 50]),
        Part(name="flange_xz", shape="box", center=[25, 1.25, 25],
             size=[50, 2.5, 50]),
        Part(name="hole_yz", shape="cylinder", op="subtract",
             center=[-2, 25, 25], height=[8, 0, 0], radius=6, hole="through"),
    ])
    assert RC._validate(spec) == []


def test_no_views_creates_no_render_folder(monkeypatch):
    # Regression: the timestamped folder was created before checking whether
    # anything would be rendered, so every geometry-only build (views: []) left
    # an empty directory behind -- dozens of them across an eval run.
    def boom(*args, **kwargs):
        raise AssertionError("must not create a directory when nothing renders")
    monkeypatch.setattr(RC.os, "makedirs", boom)
    folder, results = RC._render_checks("x.r", [], 256)
    assert folder is None and results == []


def test_report_without_renders_says_so_instead_of_naming_a_folder():
    spec = BuildSpec(name="x", views=[], parts=[
        Part(name="b", shape="box", size=[1, 1, 1])])
    text = RC._report(spec, None, [], "Built region 'x.r'")
    assert "No check views were requested" in text
    assert "Check renders in" not in text


def test_declared_expect_bbox_catches_a_shifted_placement():
    # Prose alone did not stop this: an L-bracket asked to span 0..50 mm kept
    # being built at -2.5..50 (52.5 mm).  Declaring the intended size turns it
    # into a pre-build rejection.
    spec = BuildSpec(name="b", views=[], expect_bbox=[50, 50, 50], parts=[
        Part(name="x", shape="box", center=[25, -1.25, 25], size=[50, 2.5, 50]),
        Part(name="y", shape="box", center=[-1.25, 25, 25], size=[2.5, 50, 50])])
    errors = RC._validate(spec)
    assert any("expect_bbox does not match" in e for e in errors)
    assert any("52.5 mm" in e for e in errors)


def test_correct_placement_satisfies_the_declaration():
    spec = BuildSpec(name="b", views=[], expect_bbox=[50, 50, 50], parts=[
        Part(name="x", shape="box", center=[25, 1.25, 25], size=[50, 2.5, 50]),
        Part(name="y", shape="box", center=[1.25, 25, 25], size=[2.5, 50, 50])])
    assert RC._validate(spec) == []


def test_expect_bbox_is_optional():
    spec = BuildSpec(name="b", views=[], parts=[
        Part(name="x", shape="box", size=[1, 1, 1])])
    assert RC._validate(spec) == []


def test_check_views_render_flat_ambient_with_occlusion(monkeypatch):
    # Diagnostic images, not presentation.  Flat ambient keeps every face evenly
    # lit, but on its own a stud is the SAME colour as the face under it and
    # disappears on a head-on view -- occlusion supplies the contact shading that
    # makes repeated features countable.  A spec's `lighting` cannot change this.
    seen = []

    def fake_render(spec, png):
        seen.append((spec.lighting, spec.ambient_samples))
        return None
    monkeypatch.setattr(RC.R, "render", fake_render)
    monkeypatch.setattr(RC.os, "makedirs", lambda *a, **k: None)
    RC._render_checks("x.r", ["iso", "top"], 256)
    assert [light for light, _ in seen] == ["ambient", "ambient"]
    assert all(ao > 0 for _, ao in seen)


def test_ownership_survives_a_deleted_spec_directory(monkeypatch):
    # The guard used to treat "no saved spec" as "not ours", so wiping the specs
    # directory made our OWN geometry foreign and blocked every rebuild.
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)
    spec = BuildSpec(name="widget", views=[], parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    ours = [("u", "widget.acc1"), ("-", "widget_hole.s")]
    assert RC._collision_error(spec, {"widget.r"}, ours) is None


def test_a_hand_built_region_of_the_same_name_is_still_refused(monkeypatch):
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)
    spec = BuildSpec(name="widget", views=[], parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    # Members that are NOT in our namespace: a human assembled this.
    handmade = [("u", "some_solid.s"), ("-", "another.s")]
    err = RC._collision_error(spec, {"widget.r"}, handmade)
    assert err and "widget.r" in err


def test_structural_ownership_needs_actual_members(monkeypatch):
    monkeypatch.setattr(RC, "_latest_spec", lambda name: None)
    spec = BuildSpec(name="widget", views=[], parts=[
        Part(name="body", shape="box", size=[1, 1, 1])])
    # An empty listing proves nothing, so fall back to refusing.
    assert RC._collision_error(spec, {"widget.r"}, ()) is not None


def test_report_says_when_the_size_guard_did_not_run():
    plain = BuildSpec(name="x", views=[], parts=[
        Part(name="b", shape="box", size=[1, 1, 1])])
    assert "expect_bbox was not declared" in RC._report(plain, None, [], "Built")
    declared = BuildSpec(name="x", views=[], expect_bbox=[1, 1, 1], parts=[
        Part(name="b", shape="box", size=[1, 1, 1])])
    assert "expect_bbox was not declared" not in RC._report(
        declared, None, [], "Built")


# --- the saved spec has to be readable back -------------------------------

def test_hole_intent_can_be_corrected_without_respecifying():
    # `hole` is the through/pocket assertion the verifier leans on, so a wrong
    # annotation had to be fixable through edit_build rather than forcing a full
    # re-spec (which the prompt forbids and the collision guard resists).
    parts = [{"name": "bore", "shape": "cylinder", "op": "subtract",
              "center": [0, 0, 0], "height": [0, 0, 10], "radius": 2,
              "hole": "through"}]
    new, errors = RC._apply_edits(parts, [
        {"action": "update", "name": "bore", "hole": "pocket"}])
    assert errors == []
    assert new[0]["hole"] == "pocket"
    assert new[0]["radius"] == 2            # nothing else disturbed


def test_an_edit_action_never_clobbers_a_parts_boolean_role():
    parts = [{"name": "b", "shape": "box", "op": "add",
              "center": [0, 0, 0], "size": [1, 1, 1]}]
    new, errors = RC._apply_edits(parts, [
        {"action": "move", "name": "b", "delta": [1, 0, 0]}])
    assert errors == [] and new[0]["op"] == "add"
    assert new[0]["center"] == [1, 0, 0]


def test_list_builds_can_return_the_current_spec(tmp_path, monkeypatch):
    """The spec on disk is the source of truth, and nothing could read it.

    edit_build applies ops to the saved spec, but the build report gives only part
    NAMES -- not their coordinates.  Once the spec left the agent's context (the
    worker summarises at 60k tokens) a revision could only guess or re-specify the
    whole model.
    """
    monkeypatch.setattr(RC, "_specs_root", lambda: str(tmp_path))
    spec = {"name": "plate", "parts": [
        {"name": "body", "shape": "box", "op": "add",
         "center": [0, 0, 0], "size": [50, 20, 5]}]}
    RC._save_spec("plate", spec)

    listing = RC.list_builds(name="plate", show_spec=False)
    assert "v001: 1 part(s)" in listing
    assert "50" not in listing                    # counts only, by default

    full = RC.list_builds(name="plate", show_spec=True)
    assert '"size": [' in full and "50" in full    # the real values
    assert "edit_build" in full                    # points at the edit path


def test_reading_back_a_missing_build_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(RC, "_specs_root", lambda: str(tmp_path))
    assert "No saved builds" in RC.list_builds(name="ghost", show_spec=True)


def test_a_mislabelled_through_hole_is_offered_pocket_before_lengthening():
    """Message ORDER matters, not just its content.

    With the geometry advice leading and "(or declare hole 'pocket')" trailing,
    models twice lengthened the cutter instead -- punching through a face meant to
    stay solid, and costing a build+verify cycle to notice and revert.  A cutter
    contained on every axis is far more often a mislabelled blind recess.
    """
    spec = BuildSpec(name="brick", parts=[
        Part(name="body", shape="box", center=[0, 0, 5], size=[20, 20, 10]),
        Part(name="bore", shape="cylinder", op="subtract", center=[0, 0, 0],
             height=[0, 0, 6], radius=2, hole="through"),
    ])
    errors = RC._hole_intent_errors(spec)
    assert len(errors) == 1
    msg = errors[0]
    assert "pocket" in msg and "enlarge it" in msg
    assert msg.index("'pocket'") < msg.index("enlarge it"), msg
