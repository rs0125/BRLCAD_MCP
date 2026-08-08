"""Eval harness: case loading, LOS parsing, ray scoring, and the report."""

import os

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


def test_conflict_check_wants_the_contradiction_named_not_resolved_our_way():
    """There is no right side to a contradiction, so we score the declaration.

    The dangerous failure is silently clamping a value that cannot hold: the
    build then looks clean and the conflict is invisible.
    """
    from evals.harness import score_conflicts
    case = Case(id="c", prompt="p", spec={"name": "x", "parts": []},
                conflicts=["6.3", "1.0"])
    declared = [{"topic": "cavity depth", "chose": "6.3 mm",
                 "over": "1.0 mm roof callout",
                 "reason": "cannot both hold on a 9.6 mm body"}]
    (name, ok, _), = score_conflicts(case, declared)
    assert name == "conflict-declared" and ok
    # Picking the OTHER side is equally acceptable -- it still named both.
    other = [{"topic": "roof", "chose": "1.0 mm callout", "over": "6.3 mm cavity"}]
    (_, ok, _), = score_conflicts(case, other)
    assert ok
    # Silently building one reading without declaring the clash is the failure.
    (_, ok, detail), = score_conflicts(
        case, [{"topic": "cavity depth", "chose": "6.3 mm", "over": ""}])
    assert not ok and "1.0" in detail


def test_conflict_check_is_skipped_for_an_unambiguous_case():
    from evals.harness import score_conflicts
    case = Case(id="c", prompt="p", spec={"name": "x", "parts": []})
    assert score_conflicts(case, [{"topic": "anything", "chose": "1"}]) == []


def test_a_contradictory_case_can_be_scored_with_nothing_built():
    """Declaring the conflict is a real partial result even if no region exists."""
    from evals.harness import score_case
    case = Case(id="c", prompt="p",
                spec={"name": "ghost", "parts": [
                    {"name": "b", "shape": "box", "size": [1, 1, 1]}]},
                conflicts=["6.3"])
    result = score_case(case, lambda cmd: "",
                        declarations=[{"topic": "depth", "chose": "6.3 mm"}])
    assert not result.passed                        # nothing was built
    assert ("conflict-declared", True) in \
        [(n, ok) for n, ok, _ in result.checks]


def test_unattended_prompt_is_the_shipped_prompt_plus_the_delta():
    """A callable, so the baseline tracks prompt edits instead of pinning a copy."""
    from client_v2.prompts import PROMPTS
    from evals.harness import UNATTENDED_SUFFIX, unattended_worker_prompt
    text = unattended_worker_prompt()
    assert text.startswith(PROMPTS.text("worker"))
    assert text.endswith(UNATTENDED_SUFFIX)
    # The delta must actually forbid asking, or auto-approve scores builds that
    # were founded on an unresolved conflict.
    assert "Never end your turn with a question" in text
    # The scorer reads declare_assumption rows, so the delta must ask for them
    # by name -- a prose ASSUMPTIONS block is no longer what gets measured.
    assert "declare_assumption" in text


def test_halts_are_recorded_on_the_result_and_shown():
    from evals.harness import CaseResult
    assert "halts=2" in CaseResult("c", True, halts=2).summary
    assert "halts" not in CaseResult("c", True, halts=0).summary


def test_a_run_writes_one_self_contained_directory(tmp_path, monkeypatch):
    """The whole point: audit a result without collecting pieces from three
    home-directory roots."""
    from evals import harness as H
    monkeypatch.setattr(H, "RUNS_DIR", str(tmp_path))
    run = H.new_run_dir("agent", auto_approve=True)
    assert os.path.basename(run).endswith("_agent-unattended")
    for sub in ("renders", "models", "backups"):
        assert os.path.isdir(os.path.join(run, sub))
    assert os.path.realpath(os.path.join(tmp_path, "latest")) == \
        os.path.realpath(run)


def test_a_second_run_moves_the_latest_symlink(tmp_path, monkeypatch):
    from evals import harness as H
    monkeypatch.setattr(H, "RUNS_DIR", str(tmp_path))
    H.new_run_dir("tool")
    second = H.new_run_dir("agent")
    assert os.path.realpath(os.path.join(tmp_path, "latest")) == \
        os.path.realpath(second)


def test_renders_are_routed_by_env_because_the_server_is_a_subprocess(monkeypatch):
    """settings is frozen at import in THIS process; the renders happen in the
    MCP server subprocess, which inherits our environment."""
    from evals.harness import route_artifacts_into
    monkeypatch.delenv("BRLCAD_RENDER_DIR", raising=False)
    monkeypatch.delenv("BRLCAD_BACKUP_DIR", raising=False)
    route_artifacts_into("/tmp/run42")
    assert os.environ["BRLCAD_RENDER_DIR"] == "/tmp/run42/renders"
    assert os.environ["BRLCAD_BACKUP_DIR"] == "/tmp/run42/backups"


def test_exported_models_land_in_the_run_dir(tmp_path):
    from evals.harness import _models
    assert _models(str(tmp_path)) == os.path.join(str(tmp_path), "models")
    assert _models(None) is None            # legacy path when no run dir


# --- declared assumptions -------------------------------------------------
#
# The check these replace substring-matched the whole transcript and produced a
# FALSE PASS: both conflicting values appeared, three replies apart, in prose
# that had nothing to do with the conflict.  These tests pin that shut.

def _conflict_case():
    return Case(id="c", prompt="p", spec=None, conflicts=["6.3", "1.0"])


def test_a_declaration_naming_both_values_passes():
    from evals.harness import score_conflicts
    rows = [{"topic": "cavity depth", "chose": "6.3 mm",
             "over": "1.0 mm roof callout"}]
    (name, ok, _), = score_conflicts(_conflict_case(), rows)
    assert (name, ok) == ("conflict-declared", True)


def test_values_split_across_two_declarations_do_not_pass():
    """The claim being scored is that the two readings CONFLICT, and that claim
    lives in one row.  Mentioning each separately is not making it."""
    from evals.harness import score_conflicts
    rows = [{"topic": "cavity depth", "chose": "6.3 mm", "over": ""},
            {"topic": "roof", "chose": "1.0 mm", "over": ""}]
    (_, ok, detail), = score_conflicts(_conflict_case(), rows)
    assert not ok and "6.3 + 1.0" in detail


def test_declaring_nothing_fails_and_says_so():
    from evals.harness import score_conflicts
    (_, ok, detail), = score_conflicts(_conflict_case(), [])
    assert not ok and "no assumptions at all" in detail


def test_a_case_with_no_conflicts_is_not_scored_on_declarations():
    from evals.harness import score_conflicts
    assert score_conflicts(Case(id="c", prompt="p", spec=None), []) == []


def test_declarations_are_read_from_the_run_directory(tmp_path):
    """Off disk, not through settings: settings froze the spec dir at import,
    before route_artifacts_into ran, so only the subprocess sees the new one."""
    import json

    from evals.harness import DECLARATIONS_FILE, read_declarations
    specs = tmp_path / "specs"
    specs.mkdir()
    (specs / DECLARATIONS_FILE).write_text(
        json.dumps({"topic": "depth", "chose": "6.3"}) + "\n"
        "{ this line is corrupt\n"
        + json.dumps({"topic": "width", "chose": "9.6"}) + "\n")
    rows = read_declarations(str(tmp_path))
    assert [r["topic"] for r in rows] == ["depth", "width"]   # corrupt skipped
    assert read_declarations(str(tmp_path / "nope")) == []
    assert read_declarations(None) == []


def test_the_report_shows_what_was_assumed_even_on_a_pass():
    rows = [{"topic": "cavity depth", "chose": "6.3 mm", "over": "1.0 mm"}]
    text = report([CaseResult("c", True, declaration_only=True,
                              declarations=rows)])
    assert "assumed cavity depth='6.3 mm' over '1.0 mm'" in text


# --- geometry on a case with no hand-authored spec ------------------------
#
# Every image case scored "declaration only" because ground truth meant a full
# spec.  A bbox and a few rays read off the drawing are far cheaper to author
# and still independent of the agent, which is the only property that matters.

def _lego():
    from evals.harness import load_cases
    return next(c for c in load_cases()
                if c.id == "img_lego_brick_conflict")


def test_a_region_plus_read_dimensions_is_enough_to_score_geometry():
    case = _lego()
    assert not case.has_spec              # nothing to BUILD from -- agent only
    assert case.has_ground_truth          # but plenty to MEASURE against
    assert case.region == "img_lego_brick.r"


def test_a_case_with_neither_spec_nor_expectations_stays_declaration_only():
    from evals.harness import case_from_dict
    case = case_from_dict({"id": "x", "prompt": "p", "region": "thing"})
    assert not case.has_ground_truth      # a name alone asserts nothing


def test_an_unsettled_bbox_axis_is_skipped_not_failed():
    """The lego brick's height turns on a 1.7-vs-1.8 discrepancy the DRAWING
    has, so asserting Z would fail a defensible reading."""
    from evals.harness import score_bbox

    def probe(cmd):
        return "SUCCESS: X Length: 31.8 mm\nY Length: 15.8 mm\nZ Length: 99 mm"
    names = [n for n, _, _ in score_bbox([31.8, 15.8, None], "r", probe)]
    assert names == ["bbox:X", "bbox:Y"]          # Z absent, not failed
    assert all(ok for _, ok, _ in score_bbox([31.8, 15.8, None], "r", probe))


def test_a_wrong_axis_is_named_in_the_failure():
    from evals.harness import score_bbox

    def probe(cmd):
        return "SUCCESS: X Length: 31.8 mm\nY Length: 40 mm\nZ Length: 11.4 mm"
    bad = [(n, d) for n, ok, d in score_bbox([31.8, 15.8, 11.4], "r", probe)
           if not ok]
    assert bad == [("bbox:Y", "expected 15.8 mm, got 40.0 mm")]


def test_relative_rays_are_anchored_to_the_measured_corner():
    """A terse prompt does not dictate placement, so the same correct brick can
    sit anywhere; an absolute ray would fail on position rather than shape."""
    from evals.harness import RayCheck, score_rays
    fired = []

    def probe(cmd):
        if cmd.startswith("bb -e"):
            return "SUCCESS: min {100.0 200.0 300.0} max {131.8 215.8 311.4}"
        if cmd.startswith("nirt"):
            fired.append(cmd)
            return _HIT
        return "SUCCESS:"

    ray = RayCheck(desc="stud", start=[3.9, 3.9, 10.45], dir=[0, 0, -1],
                   expect="hit", relative=True)
    score_rays([ray], "brick.r", probe)
    assert "103.9" in fired[0] and "203.9" in fired[0] and "310.45" in fired[0]


def test_an_absolute_ray_is_left_alone(monkeypatch):
    from evals.harness import RayCheck, score_rays
    fired = []

    def probe(cmd):
        if cmd.startswith("nirt"):
            fired.append(cmd)
            return _HIT
        return "SUCCESS:"

    ray = RayCheck(desc="hole", start=[8, 0, 24], dir=[0, 0, -1], expect="hit")
    score_rays([ray], "brick.r", probe)
    # No bb -e needed at all: nothing to anchor.
    assert "8" in fired[0] and not any(c.startswith("bb -e") for c in fired)


def test_relative_rays_fail_loudly_when_the_extent_cannot_be_read():
    """Silently treating the offsets as absolute would fire rays into empty
    space and report every one as a miss -- a wrong answer, not a failed run."""
    from evals.harness import RayCheck, score_rays
    ray = RayCheck(desc="stud", start=[3.9, 3.9, 10.45], dir=[0, 0, -1],
                   expect="hit", relative=True)
    (name, ok, detail), = score_rays([ray], "brick.r", lambda cmd: "SUCCESS:")
    assert name == "ray:setup" and not ok and "anchor" in detail


def test_a_spec_less_case_that_built_nothing_reports_the_absence():
    """`l <missing>` answers SUCCESS with an empty payload, so an existence
    check that only looked for an error string would call a missing region fine."""
    from evals.harness import score_case
    result = score_case(_lego(), lambda cmd: "SUCCESS:")
    assert not result.passed
    assert ("exists", False) in [(n, ok) for n, ok, _ in result.checks]
    # And it stops there rather than burying it under bbox/ray noise.
    assert not any(n.startswith("bbox:") for n, _, _ in result.checks)


def test_tool_mode_skips_a_case_it_cannot_build():
    """has_ground_truth is weaker than has_spec: enough to score a build, not
    enough to produce one."""
    case = _lego()
    assert case.has_ground_truth and not case.has_spec


def test_reset_clears_a_spec_less_case_by_its_dictated_name():
    """A region left over from an earlier run would be measured as if this run
    had built it -- so the reset cannot be skipped just because there is no spec."""
    from evals.harness import reset_case
    sent = []
    reset_case(_lego(), lambda cmd: sent.append(cmd) or "SUCCESS:")
    assert "killtree img_lego_brick.r" in sent


# --- proportions, for a reference with no printed units -------------------

def test_ratio_check_ignores_scale():
    """A sketch dimensioned 10/6/5/2 with no unit makes mm and cm both
    defensible, so absolute lengths would fail a legitimate reading."""
    from evals.harness import score_bbox_ratio

    def at(scale):
        def probe(cmd):
            return (f"SUCCESS: X Length: {10 * scale} mm\n"
                    f"Y Length: {10 * scale} mm\nZ Length: {12 * scale} mm")
        return probe
    for scale in (1, 10, 0.5):
        (_, ok, _), = score_bbox_ratio([10, 10, 12], "cake.r", at(scale))
        assert ok, f"scale {scale} should pass"


def test_ratio_check_still_catches_the_wrong_shape():
    from evals.harness import score_bbox_ratio

    def probe(cmd):
        return "SUCCESS: X Length: 30 mm\nY Length: 30 mm\nZ Length: 3 mm"
    (_, ok, detail), = score_bbox_ratio([1, 1, 1], "cube.r", probe)
    assert not ok and "1.00:1.00:0.10" in detail     # a slab, not a cube


def test_ratio_check_reports_an_unmeasurable_region():
    from evals.harness import score_bbox_ratio
    (_, ok, detail), = score_bbox_ratio([1, 1, 1], "gone.r", lambda c: "SUCCESS:")
    assert not ok and "could not measure" in detail


def test_every_image_case_now_carries_some_ground_truth():
    """The corpus item: nothing was scoreable on geometry before this."""
    from evals.harness import load_cases
    images = [c for c in load_cases() if c.image and c.id.startswith("img_")]
    assert len(images) == 10
    unscoreable = [c.id for c in images if not c.has_ground_truth]
    assert unscoreable == [], f"still declaration-only: {unscoreable}"


def test_the_two_undimensioned_cases_assert_shape_not_size():
    """A photo and a unit-less sketch fix proportions and nothing else; giving
    them an absolute bbox would be inventing numbers they do not carry."""
    from evals.harness import load_cases
    by_id = {c.id: c for c in load_cases()}
    for case_id in ("img_rubiks_photo", "img_cake_sketch"):
        case = by_id[case_id]
        assert case.bbox_ratio and case.bbox is None


# --- terse/guided pairs ---------------------------------------------------

def test_every_guided_case_pairs_with_a_terse_one_on_the_same_drawing():
    """The gap between the pair's scores is the measurement -- it says what
    disambiguation is worth. A guided case with no twin measures nothing."""
    from evals.harness import load_cases
    cases = load_cases()
    guided = [c for c in cases if c.id.endswith("_guided")]
    assert len(guided) == 4
    for g in guided:
        twins = [c for c in cases
                 if c.image == g.image and c.id != g.id and not c.has_spec]
        assert twins, f"{g.id} has no terse twin on {g.image}"


def test_guided_cases_buy_absolute_geometry():
    """Each guided prompt exists to supply what the image cannot carry, so each
    must assert all three axes -- otherwise the extra sentences bought nothing."""
    from evals.harness import load_cases
    for case in load_cases():
        if not case.id.endswith("_guided"):
            continue
        assert case.bbox and all(v is not None for v in case.bbox), case.id
        assert case.bbox_ratio is None, f"{case.id} should not need ratios"


def test_a_guided_case_no_longer_scores_the_conflict():
    """Its prompt resolves the contradiction, so there is nothing left to
    declare -- asserting one would penalise following the instruction."""
    from evals.harness import load_cases
    by_id = {c.id: c for c in load_cases()}
    assert by_id["img_lego_brick_conflict"].conflicts == [["6.3"], ["1.0", "8.6"]]
    assert by_id["img_lego_brick_guided"].conflicts == []


# --- pass^k ---------------------------------------------------------------

def test_pass_k_separates_solid_from_flaky_from_broken():
    """pass@1 is the number a one-shot demo shows you; pass^k is what "can I
    rely on this" means. Reporting only the first is how a flaky agent looks
    finished."""
    from evals.harness import report_repeats
    rows = ([{"case": "solid", "passed": True, "failed_checks": []}] * 3
            + [{"case": "flaky", "passed": True, "failed_checks": []}]
            + [{"case": "flaky", "passed": False, "failed_checks": ["bbox:X"]}] * 2
            + [{"case": "broken", "passed": False,
                "failed_checks": ["exists"]}] * 3)
    text = report_repeats(rows)
    assert "PASS^k solid" in text
    assert "FLAKY  flaky" in text and "1/3" in text
    assert "FAIL   broken" in text
    assert "pass^k : 1/3 cases passed EVERY run" in text
    assert "pass@1 : 4/9 runs passed" in text
    # The dominant cause is named, not just counted -- a number without a lead
    # sends you back to the logs.
    assert "x bbox:X  (2/3 runs)" in text


def test_results_are_flushed_after_every_pass(tmp_path):
    """Hours-long job: the file is the record, not the process."""
    from evals.harness import CaseResult, append_results, read_results
    append_results(str(tmp_path), 0, [CaseResult("a", True),
                                      CaseResult("b", False)])
    append_results(str(tmp_path), 1, [CaseResult("a", False, error="boom")])
    rows = read_results(str(tmp_path))
    assert [(r["rep"], r["case"], r["passed"]) for r in rows] == [
        (0, "a", True), (0, "b", False), (1, "a", False)]
    assert rows[2]["error"] == "boom"


def test_reading_results_survives_a_truncated_final_line(tmp_path):
    """A job killed mid-write must not lose the passes that completed."""
    from evals.harness import CaseResult, append_results, read_results
    append_results(str(tmp_path), 0, [CaseResult("a", True)])
    with open(os.path.join(str(tmp_path), "results.jsonl"), "a") as fh:
        fh.write('{"rep": 1, "case": "b"')          # killed mid-write
    assert [r["case"] for r in read_results(str(tmp_path))] == ["a"]


def test_a_run_dir_can_be_reused_so_the_listener_can_restart(tmp_path, monkeypatch):
    """The render path leaks fds, so a long job must restart the server between
    passes -- which means appending to one run dir rather than making a new one."""
    from evals import harness as H
    monkeypatch.setattr(H, "RUNS_DIR", str(tmp_path))
    first = H.new_run_dir("agent", auto_approve=True)
    again = H.new_run_dir("agent", auto_approve=True, path=first)
    assert again == first
    for sub in ("renders", "models", "backups", "specs"):
        assert os.path.isdir(os.path.join(again, sub))


def test_the_next_pass_index_comes_off_disk(tmp_path):
    """An externally-driven loop should not have to track where it got to, and a
    resumed job must continue numbering rather than overwrite pass 0."""
    from evals.harness import CaseResult, append_results, next_rep
    assert next_rep(str(tmp_path)) == 0
    append_results(str(tmp_path), 0, [CaseResult("a", True)])
    assert next_rep(str(tmp_path)) == 1
    append_results(str(tmp_path), 1, [CaseResult("a", True)])
    assert next_rep(str(tmp_path)) == 2


# --- strengthening: equivalent framings, and scale-free probes ------------

def test_a_contradiction_may_be_declared_in_either_equivalent_framing():
    """6.3 cavity vs 1.0 roof on a 9.6 body is the same clash as 6.3 vs 8.6,
    since 8.6 + 1.0 = 9.6. Demanding the literal '1.0' failed two runs that had
    named it perfectly well -- a false FAIL as misleading as a false PASS."""
    from evals.harness import score_conflicts
    case = Case(id="c", prompt="p", spec=None,
                conflicts=[["6.3"], ["1.0", "8.6"]])
    for chose in ("8.6 mm main cavity", "1.0 mm roof"):
        (_, ok, _), = score_conflicts(
            case, [{"topic": "t", "chose": chose, "over": "6.3 mm callout"}])
        assert ok, chose
    # Still catches naming only one side.
    (_, ok, _), = score_conflicts(
        case, [{"topic": "t", "chose": "8.6 mm", "over": ""}])
    assert not ok


def test_a_bare_token_list_still_works():
    """Old single-token form must keep meaning what it meant."""
    from evals.harness import conflict_groups
    assert conflict_groups(["6.3", "1.0"]) == [["6.3"], ["1.0"]]
    assert conflict_groups([["a", "b"], "c"]) == [["a", "b"], ["c"]]


def test_fractional_rays_probe_the_same_place_at_any_scale():
    """A photo fixes that a cube is divided in thirds but not how big it is, so
    the probe has to be a fraction of what was measured."""
    from evals.harness import RayCheck, score_rays
    fired = []

    def probe_at(side):
        def probe(cmd):
            if cmd.startswith("bb -e"):
                return (f"SUCCESS: min {{0 0 0}} max "
                        f"{{{side} {side} {side}}}")
            if cmd.startswith("nirt"):
                fired.append(cmd)
            return _HIT
        return probe

    ray = RayCheck(desc="cubie", start=[1/6, 1/6, 1.5], dir=[0, 0, -1],
                   expect="hit", fraction=True)
    score_rays([ray], "c.r", probe_at(60))
    score_rays([ray], "c.r", probe_at(600))
    assert "10" in fired[0] and "100" in fired[1]     # 1/6 of each side


def test_expected_los_can_be_a_fraction_of_the_model():
    """Same reason: 'the full height' is assertable without a unit."""
    from evals.harness import RayCheck, score_rays

    def probe(cmd):
        if cmd.startswith("bb -e"):
            return "SUCCESS: min {0 0 0} max {24 24 24}"
        return _HIT                                   # LOS 24.0 in the fixture

    full = RayCheck(desc="full", start=[0.5, 0.5, 2], dir=[0, 0, -1],
                    expect="hit", fraction=True, los_frac=1.0)
    (_, ok, _), = score_rays([full], "c.r", probe)
    assert ok                                          # 1.0 * 24 == 24
    short = RayCheck(desc="short", start=[0.5, 0.5, 2], dir=[0, 0, -1],
                     expect="hit", fraction=True, los_frac=0.5)
    (_, ok, detail), = score_rays([short], "c.r", probe)
    assert not ok and "expected LOS 12" in detail


def test_no_image_case_passes_on_fewer_than_three_checks():
    """Two cases were passing on 'exists' plus one assertion, contributing to a
    96% that they had not earned."""
    from evals.harness import load_cases
    for case in load_cases():
        if not case.id.startswith("img_"):
            continue
        n = (1 + sum(1 for v in (case.bbox or []) if v is not None)
             + (1 if case.bbox_ratio else 0) + len(case.rays)
             + (1 if case.dimensions else 0) + (1 if case.conflicts else 0))
        assert n >= 4, f"{case.id} asserts only {n} checks"


def test_recovered_tool_errors_are_recorded_and_shown():
    """55 rejected builds were invisible while the report printed
    'mean revision rounds: 0.0' -- which reads as first-time-right."""
    from evals.harness import CaseResult, report_repeats
    assert "recovered from 3 tool error" in CaseResult(
        "c", True, tool_errors=3).summary
    text = report_repeats([{"case": "a", "passed": True, "failed_checks": [],
                            "tool_errors": 4, "checks": [["exists", True]]}])
    assert "server rejections the agent recovered from: 4" in text
    assert "not raw first-attempt accuracy" in text


def test_thinly_checked_cases_are_called_out_in_the_aggregate():
    from evals.harness import report_repeats
    text = report_repeats([{"case": "thin", "passed": True,
                            "failed_checks": [],
                            "checks": [["exists", True], ["bbox:ratio", True]]}])
    assert "thinly checked" in text and "thin (2)" in text
