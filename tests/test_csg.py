"""Analytic CSG kernel: ray/primitive intersection and interval algebra.

Pure maths -- no engine, no socket.  This is the reference the verifier compares
the built geometry against, so it has to be right on its own terms.
"""

import math

import pytest

from brlcad_mcp.server.tools.csg import (
    expected_thickness,
    model_intervals,
    part_intervals,
    subtract,
    union,
)
from brlcad_mcp.server.tools.reconstruct import BuildSpec, Part

_DOWN = (0.0, 0.0, -1.0)


def _plate(size=(20, 20, 5)):
    return Part(name="body", shape="box", center=[0, 0, 0], size=list(size))


# --- primitives -----------------------------------------------------------

# One ray against one primitive: the intervals are hand-derived from the
# geometry, not read back from the implementation.  Parametrised rather than
# eight functions because every case is the same call with different numbers --
# and a table makes a missing case obvious in a way eight defs do not.
_SPHERE = Part(name="b", shape="sphere", center=[0, 0, 0], radius=4)
_CYL = Part(name="c", shape="cylinder", center=[0, 0, 0], height=[0, 0, 10],
            radius=3)


@pytest.mark.parametrize("part,origin,direction,expected", [
    # 5 mm plate centred on z=0: entered at z=2.5 (t=47.5), left at z=-2.5.
    pytest.param(_plate(), (0.0, 0.0, 50.0), _DOWN, (47.5, 52.5),
                 id="box-through-the-middle"),
    pytest.param(_plate(), (100.0, 0.0, 50.0), _DOWN, None,
                 id="box-passes-beside-it"),
    # Travelling in X at z=100: never inside the plate's z slab.
    pytest.param(_plate(), (-50.0, 0.0, 100.0), (1.0, 0.0, 0.0), None,
                 id="box-parallel-to-a-face-outside-the-slab"),
    # r=4 at the origin: enters z=4, leaves z=-4, so a chord of one diameter.
    pytest.param(_SPHERE, (0.0, 0.0, 50.0), _DOWN, (46.0, 54.0),
                 id="sphere-through-the-centre-spans-the-diameter"),
    pytest.param(_SPHERE, (10.0, 0.0, 50.0), _DOWN, None,
                 id="sphere-grazed-past"),
    # Down its own axis the end caps bound it -- capped, not infinite.
    pytest.param(_CYL, (0.0, 0.0, 50.0), _DOWN, (40.0, 50.0),
                 id="cylinder-along-its-axis-is-bounded-by-the-caps"),
    # Crossed sideways at mid-height: x=+3 in to x=-3 out.
    pytest.param(_CYL, (50.0, 0.0, 5.0), (-1.0, 0.0, 0.0), (47.0, 53.0),
                 id="cylinder-crossed-sideways-spans-its-diameter"),
    pytest.param(_CYL, (5.0, 0.0, 50.0), _DOWN, None,
                 id="cylinder-ray-outside-its-radius"),
])
def test_one_ray_against_one_primitive(part, origin, direction, expected):
    got = part_intervals(part, origin, direction)
    if expected is None:
        assert got == []
    else:
        (lo, hi), = got
        assert (lo, hi) == pytest.approx(expected)


def test_geometry_behind_the_ray_origin_is_not_counted():
    # part_intervals is unclipped maths (negative t = behind the origin), but the
    # model view must clip to t >= 0 because the raytracer only reports what it
    # travels through going forward.
    cyl = Part(name="c", shape="cylinder", center=[0, 0, 0],
               height=[0, 0, 10], radius=3)
    behind = part_intervals(cyl, (0.0, 0.0, -5.0), _DOWN)
    assert behind and behind[0][1] < 0            # entirely behind
    spec = BuildSpec(name="m", parts=[cyl])
    assert expected_thickness(spec, (0.0, 0.0, -5.0), _DOWN) == 0.0


def test_ray_starting_inside_counts_only_the_material_ahead():
    cyl = Part(name="c", shape="cylinder", center=[0, 0, 0],
               height=[0, 0, 10], radius=3)
    spec = BuildSpec(name="m", parts=[cyl])
    # Starting at z=4 inside the 0..10 cylinder, heading down: 4 mm ahead.
    assert math.isclose(expected_thickness(spec, (0.0, 0.0, 4.0), _DOWN), 4.0)


def test_unknown_shape_raises_rather_than_yielding_no_intervals():
    """It used to return [], which reads as "the ray crosses no material here".

    That does not make verification silent, it makes it WRONG: every ray through
    the unknown part reports measuring more material than predicted. A shape
    getting here means the build-time whitelist and the interval table have
    drifted apart, and that has to be loud.
    """
    with pytest.raises(ValueError, match="no ray-intersection function"):
        part_intervals(Part(name="x", shape="torus"), (0, 0, 1), _DOWN)


# --- interval algebra -----------------------------------------------------

def test_union_merges_touching_and_overlapping_runs():
    assert union([(0, 2)], [(1, 3)]) == [(0, 3)]
    assert union([(0, 1)], [(1, 2)]) == [(0, 2)]
    assert union([(0, 1)], [(5, 6)]) == [(0, 1), (5, 6)]


def test_subtract_splits_when_the_cut_is_interior():
    assert subtract([(0, 10)], [(4, 6)]) == [(0, 4), (6, 10)]


def test_subtract_trims_edges_and_can_remove_everything():
    assert subtract([(0, 10)], [(-5, 3)]) == [(3, 10)]
    assert subtract([(0, 10)], [(7, 20)]) == [(0, 7)]
    assert subtract([(0, 10)], [(-1, 11)]) == []
    assert subtract([(0, 10)], [(20, 30)]) == [(0, 10)]     # no overlap


# --- whole-model evaluation ----------------------------------------------

def test_through_hole_splits_the_material_into_two_runs():
    spec = BuildSpec(name="m", parts=[
        _plate(),
        Part(name="hole", shape="cylinder", op="subtract", center=[0, 0, -10],
             height=[0, 0, 20], radius=3),
    ])
    # Down the hole's axis: nothing left.
    assert expected_thickness(spec, (0.0, 0.0, 50.0), _DOWN) == 0.0
    # Beside it: the full 5 mm.
    assert math.isclose(expected_thickness(spec, (8.0, 0.0, 50.0), _DOWN), 5.0)
    # Across the plate through the hole: two 7 mm runs either side of Ø6.
    runs = model_intervals(spec, (50.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    assert len(runs) == 2
    assert math.isclose(sum(hi - lo for lo, hi in runs), 14.0)


def test_pocket_leaves_material_behind_it():
    spec = BuildSpec(name="m", parts=[
        _plate(),
        # 2 mm deep recess in the top face of a 5 mm plate.
        Part(name="pk", shape="cylinder", op="subtract", center=[0, 0, 0.5],
             height=[0, 0, 5.5], radius=3),
    ])
    assert math.isclose(expected_thickness(spec, (0.0, 0.0, 50.0), _DOWN), 3.0)


def test_fold_is_left_to_right_so_order_matters():
    # (a1 - s) then + a2 differs from a1 + a2 - s when a2 refills the cut.
    a1 = Part(name="a1", shape="box", center=[0, 0, 0], size=[10, 10, 10])
    cut = Part(name="s", shape="box", op="subtract", center=[0, 0, 0],
               size=[4, 4, 20])
    a2 = Part(name="a2", shape="box", center=[0, 0, 0], size=[2, 2, 10])
    after = BuildSpec(name="m", parts=[a1, cut, a2])     # a2 added AFTER the cut
    before = BuildSpec(name="m", parts=[a1, a2, cut])    # cut removes a2 too
    origin, down = (0.0, 0.0, 50.0), _DOWN
    assert math.isclose(expected_thickness(after, origin, down), 10.0)
    assert expected_thickness(before, origin, down) == 0.0


def test_empty_spec_has_no_material():
    assert expected_thickness(BuildSpec(name="m", parts=[]), (0, 0, 1), _DOWN) == 0.0


def test_kernel_covers_every_shape_the_builder_accepts():
    # Guard against divergence: if a primitive is added to the builder without a
    # ray-intersection function here, the verifier would silently predict NO
    # material for it and report confusing mismatches on valid models.
    from brlcad_mcp.server.tools.csg import _SHAPE_INTERVALS
    from brlcad_mcp.server.tools.reconstruct import _SHAPES
    assert set(_SHAPES) == set(_SHAPE_INTERVALS)
