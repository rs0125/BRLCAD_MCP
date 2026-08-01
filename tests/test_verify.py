"""Engine-truth verifier: extents, LOS parsing, sampling, and the verdict.

The verdict runs through an injected ``prober`` that answers from an analytic
model of a *hypothetical build*, so we can simulate a build that differs from
the spec (a missing subtraction, a wrong radius) and check it is caught -- with
no live listener.
"""

import math
import re

from brlcad_mcp.server.tools import verify as V
from brlcad_mcp.server.tools.csg import expected_thickness
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part


def _plate_spec(hole_radius=5.0, with_hole=True):
    parts = [Part(name="body", shape="box", center=[0, 0, 0], size=[40, 40, 6])]
    if with_hole:
        parts.append(Part(name="hole", shape="cylinder", op="subtract",
                          center=[0, 0, -5], height=[0, 0, 10],
                          radius=hole_radius))
    return BuildSpec(name="plate", parts=parts)


def _prober_for(built: BuildSpec, bb_lengths=None, exists=True):
    """A fake socket answering as if *built* were the geometry in the database.

    Ray answers come from the analytic model of ``built``, so verifying against
    a DIFFERENT spec simulates a bad build.
    """
    if bb_lengths is None:
        bb_lengths = V.expected_bbox_lengths(built)

    def probe(cmd: str) -> str:
        if cmd.startswith("l "):
            return "SUCCESS: plate.r: REGION" if exists else "ERROR: not found"
        if cmd.startswith("bb "):
            lx, ly, lz = bb_lengths
            return (f"SUCCESS: X Length: {lx} mm\nY Length: {ly} mm\n"
                    f"Z Length: {lz} mm")
        if cmd.startswith("nirt"):
            nums = [float(v) for v in
                    re.findall(r'-?\d+\.?\d*(?:e-?\d+)?', cmd)]
            origin, direction = tuple(nums[0:3]), tuple(nums[3:6])
            thickness = expected_thickness(built, origin, direction)
            if thickness <= 0:
                return "SUCCESS: You missed the target"
            return ("SUCCESS:     Region Name    Entry (x y z)   LOS  Obliq_in\n"
                    f"plate.r      (   0.0000    0.0000    0.0000)"
                    f"   {thickness:.4f}   0.0000")
        return "SUCCESS:"
    return probe


# --- extents / bbox -------------------------------------------------------

def test_expected_bbox_ignores_the_holes():
    assert V.expected_bbox_lengths(_plate_spec()) == (40, 40, 6)


def test_cylinder_extent_does_not_pad_radius_along_its_own_axis():
    # Regression: padding r on all three axes made a r=14 h=10 Z-cylinder
    # report a Z length of 38 instead of 10.
    spec = BuildSpec(name="s", parts=[
        Part(name="body", shape="cylinder", center=[0, 0, 0],
             height=[0, 0, 10], radius=14)])
    assert V.expected_bbox_lengths(spec) == (28.0, 28.0, 10.0)


def test_parse_bb_lengths_and_tolerance():
    out = "X Length: 40 mm\nY Length: 40 mm\nZ Length: 6 mm"
    assert V.parse_bb_lengths(out) == (40.0, 40.0, 6.0)
    assert V.bbox_matches((40, 40, 6), (40.0, 40.2, 6.0))
    assert not V.bbox_matches((40, 40, 6), (40, 40, 12))
    assert not V.bbox_matches((40, 40, 6), (40, None, 6))


# --- LOS parsing ----------------------------------------------------------

_TWO_PARTITIONS = (
    "    Region Name     Entry (x y z)         LOS  Obliq_in\n"
    "plate.r      (   50.0000  0.0000  0.0000)   7.0000   0.0000\n"
    "plate.r      (   -3.0000  0.0000  0.0000)   7.0000   0.0000")


def test_total_los_sums_every_partition():
    # A ray crossing either side of a hole reports two crossings; the total is
    # what compares against the analytic prediction.
    assert V.total_los(_TWO_PARTITIONS) == 14.0
    assert V.total_los("You missed the target") == 0.0


def test_ray_missed_detection():
    assert V.ray_missed("...You missed the target")
    assert not V.ray_missed(_TWO_PARTITIONS)


# --- sampling -------------------------------------------------------------

def test_sampling_covers_a_grid_on_every_axis_plus_each_cavity():
    rays = V.sample_rays(_plate_spec())
    labels = [label for label, _, _ in rays]
    for axis in range(3):
        assert sum(1 for lb in labels if lb.startswith(f"grid{axis}:")) == 9
    # Each cavity gets a centre ray per axis plus offsets toward its walls,
    # which is what makes a wrong DIAMETER detectable.
    assert sum(1 for lb in labels if lb.startswith("hole@centre.")) == 3
    assert any(lb.startswith("hole@edge.") for lb in labels)


def test_sampling_is_empty_without_material():
    assert V.sample_rays(BuildSpec(name="m", parts=[])) == []


def test_every_sample_ray_is_aimed_inward_from_outside():
    for _, start, direction in V.sample_rays(_plate_spec()):
        assert sum(1 for d in direction if d) == 1      # axis-aligned
        assert max(abs(v) for v in start) >= 1000       # well outside the model


# --- verdict --------------------------------------------------------------

def test_matching_build_passes():
    spec = _plate_spec()
    passed, checks = V._verify(spec, _prober_for(spec))
    assert passed
    assert {n for n, _, _ in checks} == {"exists", "bbox", "geometry"}
    detail = {n: d for n, _, d in checks}["geometry"]
    assert "sample rays match" in detail


def test_missing_subtraction_is_caught():
    # The r-operator binding class of bug: the spec has a hole, the build does
    # not, so rays through it measure MORE material than predicted.
    spec = _plate_spec()
    built = _plate_spec(with_hole=False)
    passed, checks = V._verify(spec, _prober_for(built))
    assert not passed
    detail = {n: d for n, _, d in checks}["geometry"]
    assert "disagree with the spec" in detail
    assert "did not apply" in detail          # explains what MORE material means


def test_wrong_hole_diameter_is_caught():
    # Impossible for the old single-axis probe: a hole of the wrong size is
    # still "present", but the offset samples measure the difference.
    spec = _plate_spec(hole_radius=6.0)
    built = _plate_spec(hole_radius=2.0)
    passed, _ = V._verify(spec, _prober_for(built))
    assert not passed


def _pocket_spec(base_z):
    """A 20x20x5 plate (top face z=2.5) with a recess cut from *base_z* upward.

    Depth is set by where the cutter STARTS, not by its height: it always
    over-runs the top face, so lengthening it removes no extra material.
    base_z=0.5 -> 2 mm deep (3 mm left); base_z=1.5 -> 1 mm deep (4 mm left).
    """
    return BuildSpec(name="plate", parts=[
        Part(name="body", shape="box", center=[0, 0, 0], size=[20, 20, 5]),
        Part(name="pk", shape="cylinder", op="subtract", center=[0, 0, base_z],
             height=[0, 0, 10], radius=3, hole="pocket")])


def test_correct_pocket_passes_and_leaves_the_right_thickness():
    spec = _pocket_spec(0.5)              # 2 mm deep -> 3 mm left behind
    assert math.isclose(
        expected_thickness(spec, (0.0, 0.0, 50.0), (0.0, 0.0, -1.0)), 3.0)
    passed, _ = V._verify(spec, _prober_for(spec))
    assert passed


def test_pocket_of_the_wrong_depth_is_caught():
    spec = _pocket_spec(0.5)              # wanted 2 mm deep (3 mm left)
    built = _pocket_spec(1.5)             # built only 1 mm deep (4 mm left)
    passed, checks = V._verify(spec, _prober_for(built))
    assert not passed
    assert "expected 3 mm" in {n: d for n, _, d in checks}["geometry"]


def test_wrong_bounding_box_is_caught():
    spec = _plate_spec()
    passed, checks = V._verify(spec, _prober_for(spec, bb_lengths=(40, 40, 20)))
    assert not passed
    assert any(n == "bbox" and not ok for n, ok, _ in checks)


def test_missing_region_short_circuits():
    spec = _plate_spec()
    passed, checks = V._verify(spec, _prober_for(spec, exists=False))
    assert not passed
    assert checks == [("exists", False, "plate.r is not in the database")]


def test_rays_are_scoped_to_the_region_before_probing():
    # ged nirt fires at the DISPLAYED objects, so the region must be isolated
    # first or unrelated geometry contributes phantom material.
    sent = []
    spec = _plate_spec()
    inner = _prober_for(spec)

    def probe(cmd):
        sent.append(cmd)
        return inner(cmd)
    V._verify(spec, probe)
    assert "zap" in sent and "draw plate.r" in sent
    first_ray = next(i for i, c in enumerate(sent) if c.startswith("nirt"))
    assert sent.index("zap") < first_ray


def test_tool_accepts_a_dict_spec(monkeypatch):
    # Regression: a dict spec used to raise a hard ToolException (str-only).
    spec = _plate_spec()
    monkeypatch.setattr(V, "send_command", _prober_for(spec))
    assert "PASS" in V.verify_model_dimensions(spec.model_dump())


def test_parse_json_arg_accepts_dict_string_and_reports_bad_input():
    from brlcad_mcp.server.tools.helpers import parse_json_arg
    assert parse_json_arg({"a": 1}, "spec") == ({"a": 1}, None)
    assert parse_json_arg('{"a": 1}', "spec") == ({"a": 1}, None)
    data, err = parse_json_arg("{not json", "spec")
    assert data is None and "not valid JSON" in err
    data, err = parse_json_arg(42, "spec")
    assert data is None and "must be a JSON object" in err


def test_missing_region_detected_when_l_returns_empty_success():
    # The real listener answers `l <missing>` with SUCCESS and an EMPTY payload,
    # not an error, so an error-prefix check alone would call it existing and
    # then blame the geometry for being absent.
    spec = _plate_spec()

    def probe(cmd):
        return "SUCCESS: " if cmd.startswith("l ") else "SUCCESS:"
    passed, checks = V._verify(spec, probe)
    assert not passed
    name, ok, detail = checks[0]
    assert name == "exists" and not ok
    assert "not in the database" in detail
