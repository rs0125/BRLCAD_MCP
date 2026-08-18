"""wedge and cone: the two primitives that unlock sloped and tapered faces.

Only these two were added, out of BRL-CAD's 41.  The constraint is not what the
engine can build -- it builds all of them, and `execute_command` can reach any
of them today -- but what ``csg.py`` can INTERSECT A RAY WITH analytically.  A
primitive without an interval function cannot be verified, and worse, used to
degrade to "crosses no material", which made verification wrong rather than
silent.  A torus was left out for exactly this reason: its ray intersection is a
quartic, and fillets are already reachable as box minus cylinder.
"""

import pytest

from brlcad_mcp.server.tools import verify as V
from brlcad_mcp.server.tools.csg import expected_thickness, part_intervals
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part, _solid_cmd

DOWN = (0.0, 0.0, -1.0)


def thickness(part_kwargs, origin, direction=DOWN):
    spec = BuildSpec.model_validate(
        {"name": "t", "views": [], "parts": [dict(
            name="p", op="add", **part_kwargs)]})
    return expected_thickness(spec, origin, direction)


# --- wedge ----------------------------------------------------------------

def test_wedge_with_no_taper_is_exactly_a_box():
    """The degenerate case has to agree with the box maths, or every wedge is
    suspect."""
    box = dict(shape="box", center=[0, 0, 10], size=[40, 30, 20])
    wedge = dict(shape="wedge", center=[0, 0, 10], size=[40, 30, 20])
    for x in (0, 10, 19, 21):
        assert thickness(box, (x, 0, 100)) == pytest.approx(
            thickness(wedge, (x, 0, 100))), x


def test_wedge_taper_thins_toward_the_top():
    """A ray down through the overhanging part crosses only the lower portion,
    and where it exits is set by the slope."""
    w = dict(shape="wedge", center=[0, 0, 10], size=[40, 30, 20],
             top_size=[20, 30])
    # Half-width runs 20 at z=0 to 10 at z=20, so at x=15 material ends at
    # z = 10: a downward ray crosses 10 mm.
    assert thickness(w, (15, 0, 100)) == pytest.approx(10.0)
    assert thickness(w, (5, 0, 100)) == pytest.approx(20.0)   # inside the top
    assert thickness(w, (25, 0, 100)) == pytest.approx(0.0)   # clear of it


def test_a_gusset_is_a_wedge_whose_top_collapses_to_a_line():
    w = dict(shape="wedge", center=[0, 0, 10], size=[40, 10, 20],
             top_size=[0, 10])
    assert thickness(w, (0, 0, 100)) == pytest.approx(20.0)    # under the ridge
    assert thickness(w, (10, 0, 100)) == pytest.approx(10.0)   # halfway out
    # half-width runs 20 -> 0 over 20 mm, so at x=19 material ends at z=1.
    assert thickness(w, (19, 0, 100)) == pytest.approx(1.0)


def test_a_wedge_that_flares_outward_is_bounded_by_its_top():
    """Extent must take the WIDER face; using ``size`` alone under-reports it
    and expect_bbox would then reject a correct build."""
    part = Part(name="f", shape="wedge", center=[0, 0, 10],
                size=[20, 20, 20], top_size=[40, 40])
    lo_x, _, _, hi_x, _, _ = V._part_extent(part)
    assert (lo_x, hi_x) == (-20.0, 20.0)


# --- cone -----------------------------------------------------------------

def test_cone_with_equal_radii_is_exactly_a_cylinder():
    cyl = dict(shape="cylinder", center=[0, 0, 0], height=[0, 0, 30], radius=12)
    cone = dict(shape="cone", center=[0, 0, 0], height=[0, 0, 30], radius=12,
                top_radius=12)
    for x in (0, 6, 11.5, 13):
        assert thickness(cyl, (x, 0, 100)) == pytest.approx(
            thickness(cone, (x, 0, 100))), x


def test_an_axis_parallel_ray_through_a_taper_is_not_predicted_as_empty():
    """The bug this pins: an axis-parallel ray makes the quadratic's leading
    coefficient NEGATIVE, so the solution set is the outside of the roots -- two
    half-lines.  Treating it as one interval predicted 0 mm through solid
    material and every such ray reported a mismatch."""
    c = dict(shape="cone", center=[0, 0, 0], height=[0, 0, 30], radius=12,
             top_radius=6)
    # Radius runs 12 -> 6, so at rho=9 material ends exactly halfway up.
    assert thickness(c, (9, 0, 100)) == pytest.approx(15.0)
    assert thickness(c, (0, 0, 100)) == pytest.approx(30.0)    # full height
    assert thickness(c, (5, 0, 100)) == pytest.approx(30.0)    # inside the top
    assert thickness(c, (13, 0, 100)) == pytest.approx(0.0)    # outside the base


def test_a_cone_is_bounded_by_its_wider_end():
    part = Part(name="c", shape="cone", center=[0, 0, 0], height=[0, 0, 30],
                radius=4, top_radius=10)
    lo_x, _, _, hi_x, _, _ = V._part_extent(part)
    assert (lo_x, hi_x) == (-10.0, 10.0)


def test_a_tilted_cone_still_measures_correctly():
    """Cylinders already supported an arbitrary axis; a cone must not quietly
    assume Z."""
    c = dict(shape="cone", center=[0, 0, 0], height=[30, 0, 0], radius=12,
             top_radius=6)
    # Same geometry rotated into X: a ray along -X down the axis sees 30 mm.
    spec = BuildSpec.model_validate(
        {"name": "t", "views": [], "parts": [dict(name="p", op="add", **c)]})
    assert expected_thickness(spec, (100, 0, 0), (-1, 0, 0)) == \
        pytest.approx(30.0)


# --- guards ---------------------------------------------------------------

def test_an_unknown_shape_raises_instead_of_predicting_nothing():
    """Returning [] reads as 'no material here', so a shape we cannot intersect
    would make verification WRONG, not merely absent."""
    with pytest.raises(ValueError, match="no ray-intersection function"):
        part_intervals(Part(name="x", shape="torus", center=[0, 0, 0]),
                       (0, 0, 0), DOWN)


def test_wedge_vertices_are_emitted_in_the_order_arb8_expects():
    """Bottom face counter-clockwise then the top in the SAME order. Wrong order
    builds a self-intersecting solid that still succeeds and then ray-traces as
    nonsense, so the order is generated, never asked of a caller."""
    cmd = _solid_cmd(Part(name="w", shape="wedge", center=[0, 0, 10],
                          size=[40, 30, 20], top_size=[20, 30]))
    nums = [float(v) for v in cmd.split()[3:]]
    pts = [tuple(nums[i:i + 3]) for i in range(0, 24, 3)]
    assert pts[:4] == [(-20, -15, 0), (20, -15, 0), (20, 15, 0), (-20, 15, 0)]
    assert pts[4:] == [(-10, -15, 20), (10, -15, 20), (10, 15, 20),
                       (-10, 15, 20)]


def test_a_cone_needs_a_positive_radius_at_both_ends():
    """BRL-CAD's `in trc` refuses zero, and substituting an epsilon silently
    would verify against geometry the caller never asked for."""
    from brlcad_mcp.server.tools.reconstruct import _validate
    spec = BuildSpec.model_validate({"name": "t", "views": [], "parts": [
        {"name": "c", "shape": "cone", "op": "add", "center": [0, 0, 0],
         "height": [0, 0, 25], "radius": 10, "top_radius": 0}]})
    errors = _validate(spec)
    assert any("positive radius at BOTH ends" in e for e in errors)
    assert any("0.1" in e for e in errors)          # says what to do instead


def test_cone_ray_maths_matches_a_numeric_sample():
    """Independent check of the interval maths: march along the ray and count
    how much of it falls inside the cone by the plain geometric definition."""
    c = dict(shape="cone", center=[0, 0, 0], height=[0, 0, 30], radius=12,
             top_radius=6)
    for rho in (0.0, 4.0, 9.0, 11.0):
        n, inside = 20000, 0
        for i in range(n):
            z = 30.0 * (i + 0.5) / n
            if rho <= 12.0 + (6.0 - 12.0) / 30.0 * z:
                inside += 1
        numeric = 30.0 * inside / n
        assert thickness(c, (rho, 0, 100)) == pytest.approx(numeric, abs=0.02)
