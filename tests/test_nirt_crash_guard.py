"""The `nirt <object>` crash guard, and telling an unrunnable ray from an empty one.

`nirt` with a positional object argument frees and then reuses its argument
table inside ged_nirt_core, so MGED calls bu_bomb and the process dies, taking
the mcp_listen listener with it.  Both defences below exist because of a real
session that lost its listener that way.
"""

import brlcad_mcp.server.tools.verify as V
from brlcad_mcp.server.tools.helpers import unsafe_nirt

# Captured from a live listener: nirt echoes its firing state either way.
HIT = """Origin (x y z) = (0.00 0.00 -34.64)  (h v d) = (0.00 -0.00 -50.00)
Direction (x y z) = (0.0000 0.0000 1.0000)  (az el) = (0.00 -90.00)
Region Name                   Entry (x y z)              LOS   Obliq_in  Attrib
rr.r                 (    0.000     0.000   -10.000)    20.00    0.000
"""
MISS = """Origin (x y z) = (500.00 500.00 -34.64)  (h v d) = (500.00 -500.00 -50.00)
Direction (x y z) = (0.0000 0.0000 1.0000)  (az el) = (0.00 -90.00)
You missed the target
"""


# --- the guard ------------------------------------------------------------

def test_object_form_is_refused():
    assert unsafe_nirt("nirt sphere.r") == "sphere.r"
    assert unsafe_nirt("nirt  plate.r  ") == "plate.r"


def test_object_after_a_script_flag_is_still_refused():
    # bu_opt consumes the -e value, so the trailing name is left positional
    # and reaches the freed table exactly as in the bare form.
    assert unsafe_nirt('nirt -e "xyz 0 0 0" sphere.r') == "sphere.r"


def test_object_after_a_valueless_flag_is_refused():
    # Conservative on purpose: a flag that takes no value would otherwise let
    # the object through, and being over-strict only costs the caller a turn.
    assert unsafe_nirt("nirt -b sphere.r") == "sphere.r"


def test_the_verifiers_scripted_form_is_allowed():
    safe = V.ray_cmd((0, 0, -50), (0, 0, 1))
    assert unsafe_nirt(safe) is None


def test_bare_nirt_is_allowed():
    # No positional arguments, so it uses the drawn list and does not crash.
    assert unsafe_nirt("nirt") is None


def test_other_commands_are_untouched():
    assert unsafe_nirt("ls") is None
    assert unsafe_nirt("in box rpp 0 1 0 1 0 1") is None
    assert unsafe_nirt("") is None


def test_unbalanced_quotes_are_left_to_mged():
    assert unsafe_nirt('nirt -e "xyz 0 0') is None


# --- telling "did not run" from "measured nothing" ------------------------

def test_real_nirt_output_is_recognised():
    assert V.nirt_ran(HIT)
    assert V.nirt_ran(MISS)


def test_a_hit_table_alone_is_recognised():
    # Which parts nirt prints depends on the script it was given, so the hit
    # table on its own has to count as a report.
    assert V.nirt_ran("    Region Name     Entry (x y z)         LOS  Obliq_in\n"
                      "plate.r      (   50.0000  0.0000  0.0000)   7.0000   0.0000")


def test_non_nirt_replies_are_not_mistaken_for_measurements():
    for reply in ("", "   ",
                  "ERROR: libmcpcad listener timed out after 5.0s",
                  "Command failed.",
                  "unknown command: nirt"):
        assert not V.nirt_ran(reply), reply


def test_a_failed_ray_would_otherwise_read_as_zero_material():
    """Why nirt_ran has to be checked first, stated as an assertion."""
    dud = "ERROR: listener timed out"
    assert V.total_los(dud) == 0.0      # numbers: indistinguishable from a void
    assert not V.ray_missed(dud)        # phrase:  indistinguishable from a hit
    assert not V.nirt_ran(dud)          # so only this catches it


def test_measurements_still_parse():
    assert V.total_los(HIT) == 20.0
    assert V.ray_missed(MISS)


def test_first_line_quotes_a_reply_briefly():
    assert V.first_line("\n\n  boom happened  \n next") == "boom happened"
    assert V.first_line("") == "<empty reply>"
    assert len(V.first_line("x" * 200)) == 70


# --- ray checks disabled for this release ---------------------------------

def test_ray_checks_are_off_by_default():
    """The release ships with them off; nirt does not start in a build tree."""
    assert V.RAY_CHECKS is False


def test_no_ray_is_fired_while_disabled(monkeypatch):
    """The point of the switch: nirt must not be invoked at all."""
    from brlcad_mcp.server.tools.reconstruct import BuildSpec

    spec = BuildSpec.model_validate({
        "name": "b", "expect_bbox": [10, 10, 10],
        "parts": [{"name": "body", "shape": "box", "op": "add",
                   "center": [5, 5, 5], "size": [10, 10, 10]}]})
    sent = []

    def prober(cmd):
        sent.append(cmd)
        if cmd.startswith("l "):
            return "SUCCESS: b.r: REGION id=1000 -- u b_body.s"
        if cmd.startswith("bb "):
            return ("SUCCESS: Bounding Box Dimensions, Object(s) b.r:\n"
                    "X Length: 10 mm\nY Length: 10 mm\nZ Length: 10 mm")
        return "SUCCESS: "

    monkeypatch.setattr(V, "RAY_CHECKS", False)
    passed, checks = V._verify(spec, prober)
    assert passed
    assert not any(c.startswith("nirt") for c in sent), sent
    # and no zap/draw either: those exist only to isolate the region for rays
    assert not any(c.startswith(("zap", "draw")) for c in sent), sent
    assert [name for name, _, _ in checks] == ["exists", "bbox"]


def test_the_report_says_geometry_was_not_checked():
    """A PASS must not be mistakable for a full engine-truth check."""
    out = V._format("b.r", True, [("exists", True, "b.r"),
                                  ("bbox", True, "expected (10,10,10)")])
    assert "NOT CHECKED" in out
    assert "unverified" in out
    assert "do not describe them as confirmed" in out


def test_a_missing_region_does_not_get_the_disabled_note():
    """Nothing was built, so the ray note would only be noise."""
    out = V._format("b.r", False, [("exists", False, "b.r is not in the database")])
    assert "NOT CHECKED" not in out


def test_enabling_restores_the_rays(monkeypatch):
    """The machinery is gated, not deleted."""
    from brlcad_mcp.server.tools.reconstruct import BuildSpec

    spec = BuildSpec.model_validate({
        "name": "b", "expect_bbox": [10, 10, 10],
        "parts": [{"name": "body", "shape": "box", "op": "add",
                   "center": [5, 5, 5], "size": [10, 10, 10]}]})
    sent = []

    def prober(cmd):
        sent.append(cmd)
        if cmd.startswith("l "):
            return "SUCCESS: b.r: REGION id=1000 -- u b_body.s"
        if cmd.startswith("bb "):
            return ("SUCCESS: Bounding Box Dimensions, Object(s) b.r:\n"
                    "X Length: 10 mm\nY Length: 10 mm\nZ Length: 10 mm")
        if cmd.startswith("nirt"):
            return ("SUCCESS:     Region Name    Entry (x y z)   LOS  Obliq_in\n"
                    "b.r      (   0.0000  5.0000  5.0000)  10.0000   0.0000")
        return "SUCCESS: "

    passed, checks = V._verify(spec, prober, rays=True)   # per-call override
    assert any(c.startswith("nirt") for c in sent)
    assert "geometry" in [name for name, _, _ in checks]
    assert passed
