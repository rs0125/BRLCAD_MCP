"""Eval harness: case loading, LOS parsing, ray scoring, and the report."""

from evals.harness import (
    Case,
    CaseResult,
    RayCheck,
    _parse_los,
    load_cases,
    report,
    score_case,
    score_rays,
)

_HIT = ("SUCCESS: "
        "    Region Name               Entry (x y z)              LOS  Obliq_in\n"
        "ev_bushing.r         (   8.0000    0.0000   24.0000)   24.0000   0.0000")
_MISS = "SUCCESS: You missed the target"


def test_parse_los_reads_the_hit_line():
    # The ')' is attached to the last coordinate, and the header also has a 'z)'.
    assert _parse_los(_HIT) == 24.0
    assert _parse_los(_MISS) is None


def test_shipped_cases_load_and_declare_expectations():
    cases = load_cases()
    ids = {c.id for c in cases}
    assert {"washer", "l_bracket", "plate_two_holes"} <= ids
    bracket = next(c for c in cases if c.id == "l_bracket")
    # The regression guard must assert BOTH plates' holes (the binding bug).
    assert sum(1 for r in bracket.rays if r.expect == "miss") == 2
    assert bracket.bbox == [50, 50, 50]


def _prober(ray_response):
    def probe(cmd):
        if cmd.startswith("nirt"):
            return ray_response
        return "SUCCESS:"
    return probe


def test_score_rays_checks_expect_and_los():
    rays = [RayCheck(desc="wall", start=[8, 0, 60], dir=[0, 0, -1],
                     expect="hit", los=24)]
    (name, ok, detail), = score_rays(rays, "r.r", _prober(_HIT))
    assert name == "ray:wall" and ok and "LOS 24.0" in detail

    # Same ray, wrong expected LOS -> fails.
    rays[0].los = 6
    (_, ok, detail), = score_rays(rays, "r.r", _prober(_HIT))
    assert not ok and "expected LOS 6" in detail


def test_score_rays_detects_a_hole_that_did_not_cut_through():
    rays = [RayCheck(desc="bore open", start=[60, 0, 0], dir=[-1, 0, 0],
                     expect="miss")]
    (_, ok, detail), = score_rays(rays, "r.r", _prober(_HIT))
    assert not ok and "expected miss, got hit" in detail
    (_, ok, _), = score_rays(rays, "r.r", _prober(_MISS))
    assert ok


def test_score_rays_scopes_the_rays_to_the_region():
    # ged nirt fires at DISPLAYED objects, so the region must be drawn alone.
    sent = []

    def probe(cmd):
        sent.append(cmd)
        return _MISS
    score_rays([RayCheck(desc="d", start=[0, 0, 1], dir=[0, 0, -1],
                         expect="miss")], "target.r", probe)
    assert sent[:2] == ["zap", "draw target.r"]


def test_score_case_uses_ground_truth_not_the_agents_spec():
    # A case whose ground truth says 30 mm thick, but bb reports 12 -> FAIL.
    case = Case(id="c", prompt="p", spec={
        "name": "blk",
        "parts": [{"name": "b", "shape": "box", "center": [0, 0, 0],
                   "size": [30, 30, 30]}]})

    def probe(cmd):
        if cmd.startswith("l "):
            return "SUCCESS: blk.r: REGION"       # a non-empty listing = exists
        if cmd.startswith("bb"):
            return ("SUCCESS: X Length: 30 mm\nY Length: 30 mm\n"
                    "Z Length: 12 mm")
        return "SUCCESS:"
    result = score_case(case, probe)
    assert not result.passed
    assert any(n == "bbox" and not ok for n, ok, _ in result.checks)


def test_report_gives_the_aggregate_reliability_number():
    text = report([CaseResult("a", True), CaseResult("b", True),
                   CaseResult("c", False,
                              checks=[("hole:x", False, "no through-hole")])])
    assert "2/3 cases passed" in text and "67% reliability" in text
    assert "no through-hole" in text        # failures are itemised


def test_report_includes_first_attempt_stats_when_rounds_recorded():
    text = report([CaseResult("a", True, rounds=0),
                   CaseResult("b", True, rounds=2)])
    assert "first-attempt passes: 1/2" in text


def test_report_still_shows_first_attempt_when_every_case_needed_no_revision():
    # All-zero rounds is the BEST outcome, so the line must not be hidden.
    text = report([CaseResult("a", True, rounds=0),
                   CaseResult("b", True, rounds=0)])
    assert "first-attempt passes: 2/2" in text
    assert "mean revision rounds: 0.0" in text


def test_report_omits_round_stats_in_tool_mode():
    # rounds=None means "not applicable", distinct from 0.
    text = report([CaseResult("a", True), CaseResult("b", True)])
    assert "first-attempt" not in text


def test_dimension_check_reads_the_agents_proposal():
    from evals.harness import score_dimensions
    case = Case(id="c", prompt="p", spec={"name": "x", "parts": []},
                dimensions=["50", "2.5", "12"])
    (name, ok, _), = score_dimensions(case, "50 mm flanges, 2.5 mm thick, 12 mm holes")
    assert name == "dimensions" and ok
    (_, ok, detail), = score_dimensions(case, "50 mm flanges, 3 mm thick")
    assert not ok and "2.5" in detail and "12" in detail


def test_dimension_check_is_skipped_without_a_proposal():
    # Tool mode builds from ground truth, so there is nothing to read.
    from evals.harness import score_dimensions
    case = Case(id="c", prompt="p", spec={"name": "x", "parts": []},
                dimensions=["50"])
    assert score_dimensions(case, "") == []
