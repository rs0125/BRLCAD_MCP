"""Tests for the render_model tool.

These cover the tool's logic (view presets, the light-rig geometry, graceful
handling of a missing database) without requiring an actual BRL-CAD build --
rt and pix-png are never invoked here.
"""

import math

from brlcad_mcp.server.tools import rendering as R

_render = getattr(R.render_model, "fn", R.render_model)


def test_view_presets_are_angle_pairs():
    for name in ("iso", "iso2", "front", "side", "top", "rear"):
        assert name in R._VIEWS
        az, el = R._VIEWS[name]
        assert isinstance(az, (int, float))
        assert isinstance(el, (int, float))


def test_rig_positions_count_and_finiteness():
    # camera on +X looking at the origin, view size 1000
    pos = R._rig_positions("1000 0 0", (0.0, 0.0, 0.0), "1000")
    assert len(pos) == 3
    assert {p[0] for p in pos} == {"key", "fill", "rim"}
    for _name, x, y, z, bright, shadows in pos:
        assert all(math.isfinite(c) for c in (x, y, z))
        assert bright > 0
        assert shadows >= 0  # shadow-ray count (0 = none, >1 = soft shadows)
    # the three lights are at three distinct positions
    assert len({(round(p[1]), round(p[2]), round(p[3])) for p in pos}) == 3


def test_missing_database_is_reported_gracefully():
    out = _render(
        obj="thing", view="iso", azimuth=None, elevation=None,
        lighting="ambient", size=64, ambient=None, ambient_samples=0,
        quality="draft", db_path="/no/such/file.g",
    )
    assert "not found" in out.lower()


def test_discover_db_returns_string(listener):
    # the mock listener has no 'opendb'; discovery must degrade to "" not crash
    assert isinstance(R._discover_db(), str)
