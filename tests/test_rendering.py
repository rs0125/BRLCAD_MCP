"""Tests for the render tools.

These cover the tools' logic (view presets, the light-rig geometry, variant
parsing, graceful handling of unknown modes) without requiring an actual
BRL-CAD build -- rt is never invoked here.
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


def test_parse_variants_valid_and_invalid():
    pairs, errors = R._parse_variants("iso:studio, iso:model , front:ambient")
    assert pairs == [("iso", "studio"), ("iso", "model"), ("front", "ambient")]
    assert errors == []


def test_parse_variants_rejects_bad_tokens():
    pairs, errors = R._parse_variants("iso:studio,bogus:studio,iso:neon,justview")
    assert pairs == [("iso", "studio")]
    assert len(errors) == 3  # unknown view, unknown lighting, missing colon


def test_unknown_lighting_mode_is_reported():
    # All modes now render over the socket (no db_path). An unknown lighting
    # mode is rejected before any socket traffic, so this stays hermetic.
    out = _render(
        obj="thing", view="iso", azimuth=None, elevation=None,
        lighting="bogus", size=64, ambient=None, ambient_samples=0,
        quality="draft",
    )
    assert "unknown lighting" in out.lower()


def test_render_core_rejects_unknown_lighting_before_any_socket_traffic():
    # render() validates lighting first, so this is hermetic (no send_command).
    out = R.render(R.RenderSpec(obj="thing", lighting="bogus"), "/tmp/x.png")
    assert out is not None and "unknown lighting" in out.lower()


def test_preview_and_check_presets_resolve_view_and_defaults():
    iso_az, iso_el = R._VIEWS["iso"]
    p = R.preview_spec("plate.r", "iso")
    assert (p.az, p.el) == (iso_az, iso_el)
    assert p.size == 192 and p.quality == "draft" and p.ambient_samples == 0
    assert p.lighting == "studio" and p.ambient is None   # auto at render time

    c = R.check_spec("plate.r", "front", 256, "ambient")
    assert (c.az, c.el) == R._VIEWS["front"]
    assert c.size == 256 and c.lighting == "ambient" and c.quality == "draft"
    # Occlusion is what makes a stud visible head-on; 0 would hide it.
    assert c.ambient_samples > 0


def test_beauty_preset_is_clean_with_occlusion():
    b = R.beauty_spec("plate.r", 35, 25)
    assert b.quality == "clean" and b.ambient_samples == 64 and b.size == 800


def test_draw_objects_zaps_before_drawing(monkeypatch):
    # Isolation: the display is cleared before drawing the target, so a render
    # never picks up unrelated objects left displayed from an earlier render.
    calls = []
    monkeypatch.setattr(R, "send_command", lambda c: calls.append(c) or "SUCCESS:")
    assert R._draw_objects("plate.r") is None
    assert calls == ["zap", "draw plate.r"]


def test_draw_objects_zaps_then_reports_bad_object(monkeypatch):
    def fake(cmd):
        return "ERROR: not found" if cmd.startswith("draw") else "SUCCESS:"
    monkeypatch.setattr(R, "send_command", fake)
    err = R._draw_objects("ghost.r")
    assert err and "ghost.r" in err


def test_render_model_treats_explicit_nulls_as_defaults(monkeypatch):
    # Regression: a model sending {"view": null, "lighting": null} -- meaning "no
    # opinion" -- used to fail the whole call with a validation error.
    captured = {}

    def fake_render(spec, png):
        captured["spec"] = spec
        return None
    monkeypatch.setattr(R, "render", fake_render)
    monkeypatch.setattr(R.os, "makedirs", lambda *a, **k: None)
    out = R.render_model(obj="thing.r", view=None, azimuth=None, elevation=None,
                         lighting=None, size=256, ambient=None,
                         ambient_samples=0, quality=None)
    assert "Rendered" in out
    spec = captured["spec"]
    assert spec.lighting == "studio" and spec.quality == "clean"
    assert (spec.az, spec.el) == R._VIEWS["iso"]
