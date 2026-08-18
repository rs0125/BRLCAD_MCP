"""Objective reliability harness for BRL-CAD model building.

Answers the question we previously had no number for: *how often does this
actually produce correct geometry?*

Two modes:

* ``tool``  — build each case from its GROUND-TRUTH spec, then score.  No API
  key, deterministic; measures the tool pipeline (this is what would have caught
  the boolean-binding bug automatically).
* ``agent`` — give the agent the case's natural-language prompt and score
  whatever it built.  This is the reliability metric: first-attempt accuracy,
  rounds to converge, and failures.

Scoring is always against the case's ground-truth spec and its explicit ray
assertions — never against the spec the agent invented, so the agent cannot
grade its own homework.  All checks are engine truth (``bb`` + ``nirt``); no
render, no vision model.

Agent mode has two shapes, and comparing them is the point:

* **scripted** — the case's ``approval`` answers the confirmation the workflow
  asks for, so the human-in-the-loop halt is exercised as a user would exercise it.
* **unattended** (``--auto-approve``) — the BASELINE.  No human, no simulated
  user; the worker prompt gains a delta telling it to resolve ambiguity itself
  and declare it.  The agent is then the only stochastic thing in the loop, so a
  failure is attributable to it and nothing else.

Every run writes ONE self-contained directory, ``evals/runs/<stamp>_<shape>/``
(``latest`` symlinks to the newest): ``log.jsonl`` (every model call, node write
and interrupt), ``renders/``, ``models/`` (per-case standalone ``.g``), and
``report.txt``.

Usage:
    ./evals/run.sh                                # tool mode, starts the listener
    ./evals/run.sh --mode agent --auto-approve    # the unattended baseline
    ./evals/run.sh --mode agent --case l_bracket  # one case, scripted approval

    # ...or drive it directly against a listener you already have up:
    BRLCAD_PORT=5555 python -m evals.harness --mode agent --auto-approve
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from brlcad_mcp.config import settings
from brlcad_mcp.server.tools import verify as V
from brlcad_mcp.server.tools.assumptions import DECLARATIONS_FILE
from brlcad_mcp.server.tools.helpers import parse_response
from brlcad_mcp.server.tools.reconstruct import BuildSpec

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
# Reference drawings live in the repo so a case is runnable from a fresh clone.
IMAGES_DIR = os.path.join(os.path.dirname(__file__), "images")

# Everything one run produces, under ONE directory, so a result can be audited or
# handed to someone else without collecting pieces from three home-directory
# roots (~/brlcad_agent_logs, ~/brlcad_renders, ~/brlcad_eval_models).  Renders
# and backups are redirected by ENV because the MCP server runs as a subprocess
# and reads them at ITS import; the harness passes its own environment through.
RUNS_DIR = os.path.join(os.path.dirname(__file__), "runs")


def new_run_dir(mode: str, auto_approve: bool = False,
                path: str | None = None) -> str:
    """Create ``evals/runs/<stamp>_<shape>/`` and point ``latest`` at it.

    *path* reuses an existing directory instead of starting a new one, so a
    multi-pass job can restart its listener between passes and still accumulate
    into one record.  That restart is not cosmetic: the render path leaks two
    pipe fds per call, and a long-lived listener eventually hands a >1024 fd to
    ``bu_process_pending``, whose stack ``fd_set`` holds exactly 1024 -- which
    aborts the server mid-run.  A fresh process resets the counter.
    """
    shape = mode if not auto_approve else f"{mode}-unattended"
    path = path or os.path.join(
        RUNS_DIR, f"{time.strftime('%Y%m%d_%H%M%S')}_{shape}")
    for sub in ("", "renders", "models", "backups", "specs"):
        os.makedirs(os.path.join(path, sub), exist_ok=True)
    link = os.path.join(RUNS_DIR, "latest")
    try:                                    # convenience only -- never fatal
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(os.path.basename(path), link)
    except OSError:
        pass
    return path


def _models(run_dir: str | None) -> str | None:
    """Where exported .g files go for this run (None keeps the legacy path)."""
    return os.path.join(run_dir, "models") if run_dir else None


def route_artifacts_into(run_dir: str) -> None:
    """Point the SERVER's render and backup output at this run's directory.

    Set on ``os.environ`` rather than on ``settings``: the MCP server runs as a
    subprocess that the harness spawns with ``env=dict(os.environ)``, and it
    reads these at its own import.  ``settings`` in *this* process is a frozen
    dataclass built at import time, so mutating it here would be both impossible
    and pointless -- the renders are produced over there.
    """
    os.environ["BRLCAD_RENDER_DIR"] = os.path.join(run_dir, "renders")
    os.environ["BRLCAD_BACKUP_DIR"] = os.path.join(run_dir, "backups")
    # Specs default to living UNDER the render dir, which put them inside
    # renders/ here -- they are not renders, and they are the record verify-by-
    # name reads back, so give them their own place in the run.
    os.environ["BRLCAD_SPEC_DIR"] = os.path.join(run_dir, "specs")


def read_declarations(run_dir: str | None) -> list[dict]:
    """Every ``declare_assumption`` row this run has written so far.

    Read off DISK rather than through ``assumptions.read_declarations``: that
    resolves the path through ``settings``, which was frozen from the
    environment at import -- before ``route_artifacts_into`` ran.  The server
    subprocess sees the new value, this process would not.  Here the run
    directory is known outright, so there is nothing to resolve.
    """
    if not run_dir:
        return []
    path = os.path.join(run_dir, "specs", DECLARATIONS_FILE)
    if not os.path.isfile(path):
        return []
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows

# --- unattended (auto-approve) mode ---------------------------------------
#
# The baseline run has NO human and NO simulated user, which creates a trap: the
# workflow is built to stop and ask when a drawing is ambiguous, so blindly
# approving whatever it asks would let it build on an unresolved conflict and be
# scored as if the conflict had been settled.  Garbage in, a number out.
#
# So auto-approve is deliberately COUPLED to this prompt delta (see
# ``run_agent_mode``): if nobody will answer, the agent must decide for itself
# and say what it decided.  That converts "halted for a human" into "declared a
# resolution", which is scoreable without a human in the loop.
#
# Kept minimal on purpose.  It does NOT say how to resolve a conflict (prefer the
# section view, prefer the larger value, ...) because whether the agent resolves
# sensibly is exactly what the corpus is supposed to measure -- precedence rules
# here would measure OUR rules instead.  It only removes the option of asking.
UNATTENDED_SUFFIX = """

## Unattended run
You are running with no human available. Nobody will answer a question, and any
request for confirmation is auto-acknowledged without being read -- so asking
buys you nothing and silently loses the decision.

Therefore:
- Never end your turn with a question. Decide, then act.
- If the reference is ambiguous, or two of its callouts contradict each other,
  choose the reading you judge most likely and BUILD it. Do not stall, and do not
  quietly average or clamp the values.
- Record every such decision with the `declare_assumption` tool, giving the
  VALUES: `chose` the reading you took, `over` the one you rejected when two
  printed values conflicted. Do this even when the reading seems obvious -- a
  decision that exists only in your reply is indistinguishable from a misread.
"""


def unattended_worker_prompt():
    """The shipped worker prompt plus ``UNATTENDED_SUFFIX``.

    A CALLABLE rather than a string so ``resolve()`` re-reads the prompt library
    on every model call -- the baseline then tracks edits to the real prompt file
    instead of pinning a copy of it, and what the eval measures stays the shipped
    prompt plus one documented delta.
    """
    from client_v2.prompts import PROMPTS
    return PROMPTS.text("worker") + UNATTENDED_SUFFIX


# What a halt is answered with when no reviewer exists.  Deliberately not "yes,
# those numbers are right" -- that would be answering a question we never read.
# It restates the policy instead, so a structural authorize gate cannot be
# mistaken for approval of specific values.
UNATTENDED_ACK = (
    "Proceeding unattended: no reviewer is available to check these values. "
    "Apply your own stated interpretation, build it, and record every assumption "
    "with declare_assumption."
)


# --- cases ----------------------------------------------------------------

@dataclass
class RayCheck:
    """One explicit ray assertion: fired from *start* along *dir*."""

    desc: str
    start: list[float]
    dir: list[float]
    expect: str            # "miss" (a void/hole) or "hit" (material)
    los: float | None = None   # optional expected line-of-sight length (mm)
    # When true, ``start`` is an offset from the model's MEASURED minimum corner
    # rather than an absolute point.  A terse prompt does not dictate where the
    # part sits, so two defensible runs can place the same correct geometry at
    # different coordinates; an absolute ray would then fail on placement rather
    # than on shape.  Drawings dimension features from an EDGE ("first stud 3.9
    # from the end"), which is exactly what an offset from the corner expresses.
    relative: bool = False
    # Like ``relative``, but the offsets are FRACTIONS of the measured bbox
    # rather than millimetres -- (0.5, 0.5) is the middle of the footprint
    # whatever size the model came out.  This is the only way to probe a
    # reference carrying no scale at all: a photo fixes that a cube is divided
    # in thirds without fixing how big the cube is.
    fraction: bool = False
    # Expected line-of-sight as a fraction of the model's extent along the ray,
    # for the same reason: ``los`` in millimetres cannot be asserted when the
    # reference never says how big the thing is, and ``los_frac`` can.
    los_frac: float | None = None


@dataclass
class Case:
    id: str
    prompt: str
    spec: dict                       # ground truth
    rays: list[RayCheck] = field(default_factory=list)
    # Expected outer lengths (Lx,Ly,Lz).  An axis may be None, meaning "the
    # drawing does not settle this one" -- the lego brick's height is 9.6 plus a
    # stud that section A-A calls 1.7 and the side view calls 1.8, so asserting
    # Z would fail a defensible reading over an ambiguity the drawing itself has.
    bbox: list[float | None] | None = None
    # Expected PROPORTIONS (Lx:Ly:Lz), compared after normalising by the largest
    # axis.  The only ground truth a reference without printed numbers can offer:
    # a photo of a Rubik's cube fixes nothing about size but does fix that the
    # thing is cubic, and a sketch dimensioned "10 / 6 / 5 / 2" with no unit
    # fixes the shape while leaving mm-versus-cm a judgement call.  Scoring those
    # in absolute mm would fail a defensible reading of an unstated unit.
    bbox_ratio: list[float] | None = None
    # Whether the case DICTATES which drawing dimension lies on which axis.
    # False by default, and that default is the honest one: a drawing fixes the
    # part's three lengths but almost never which way up you build it, so an
    # axis-ordered bbox fails a correct model for its pose.  It did exactly
    # that twice in one run -- an angle bracket built 103x120x103 was scored
    # against 120x100x100 by ground truth that contradicted the very prompt
    # telling it where to put the fold.  Set True only when the prompt really
    # does pin the axes AND the expectation was written to match it.
    oriented: bool = False
    # Optional reference image attached in agent mode.  For a DIMENSIONED
    # drawing the ground-truth spec comes from the numbers printed on it, so the
    # case measures whether the agent reads the drawing correctly -- not whether
    # it guesses the same numbers we did.
    image: str | None = None
    # Values that must appear in the agent's dimension proposal (the reply it
    # gives BEFORE building).  This is how we check it read the drawing right,
    # separately from whether it then built the geometry right.
    dimensions: list[str] = field(default_factory=list)
    # What a user would say to approve the proposal.  The workflow deliberately
    # pauses for confirmation, so the eval has to answer instead of skipping it.
    approval: str = ("Yes, those dimensions are correct. Build the model now, "
                     "then verify it.")
    # Values that CANNOT both hold in this drawing.  Set for a deliberately
    # contradictory case: the assertion is then that the agent NAMED the conflict,
    # not which side it picked -- there is no right side, and scoring one would be
    # scoring our reading rather than its judgement.  Geometry is still scored
    # only if ``spec``/``rays`` are given, so a purely-ambiguous case can assert
    # the declaration alone.
    # Either a flat list of tokens, or a list of GROUPS of equivalent
    # tokens -- see ``conflict_groups``.
    conflicts: list = field(default_factory=list)
    # Region the agent was TOLD to produce ("named img_lego_brick" in the
    # prompt), for a case that asserts bbox/rays without a full spec.  Only a
    # spec can drive tool mode, but agent mode just needs to know what to measure
    # -- and measuring a hand-read bbox is far cheaper to author than a spec, so
    # requiring one was keeping geometry unscored on every image case.
    region_name: str = ""
    # A case deliberately shipped WITHOUT ground truth, to answer a question
    # about behaviour before investing in scoring it.  Marked rather than
    # silently untested: the corpus guards assert that every other image case
    # carries real assertions, and an unmarked gap should trip them.
    provisional: bool = False
    # Fewest ``declare_assumption`` rows this drawing ought to produce.  For an
    # under-dimensioned sheet, declaring NOTHING is the failure: the part can
    # only have been built by inventing numbers, and inventing them silently is
    # the behaviour that makes such a drawing dangerous to work from.  This is
    # the ambiguous tier's real assertion -- it has no envelope to check.
    min_declarations: int = 0

    @property
    def has_spec(self) -> bool:
        """Whether the case can be BUILT from ground truth (tool mode)."""
        return bool(self.spec.get("name"))

    @property
    def has_ground_truth(self) -> bool:
        """Whether any geometry can be scored.

        Two tiers, and the cheap one matters: a full hand-authored spec, or just
        a region name plus expectations read off the drawing.  A case with
        neither still scores what the agent SAID (dimensions read, conflicts
        declared), which is real signal.  What is never allowed is taking the
        expectation from the agent's own output -- that is it grading itself.
        """
        return self.has_spec or bool(
            self.region_name and (self.bbox or self.rays or self.bbox_ratio))

    @property
    def region(self) -> str:
        if self.has_spec:
            return f"{self.spec['name']}.r"
        return f"{self.region_name}.r" if self.region_name else ""

    @property
    def image_path(self) -> str | None:
        """Absolute path to the reference image, if the case has one.

        A bare filename resolves against ``evals/images/`` -- the corpus images
        live IN the repo so a case is runnable by anyone who clones it, rather
        than depending on one machine's ~/Downloads.  Absolute and ``~`` paths
        still work, for a one-off image not worth committing.
        """
        if not self.image:
            return None
        if os.path.isabs(self.image) or self.image.startswith("~"):
            return os.path.expanduser(self.image)
        return os.path.join(IMAGES_DIR, self.image)


def case_from_dict(data: dict) -> Case:
    """Build a Case from the on-disk shape.

    ONE construction path for both sources -- YAML files and the Python image
    corpus -- so the two cannot drift into different schemas.  ``spec`` is
    optional: a case may assert only that the agent DECLARED its reading (see
    ``Case.conflicts``), which is the only thing scoreable for a drawing whose
    ground truth has not been hand-authored yet.
    """
    expect = data.get("expect") or {}
    return Case(
        id=data["id"], prompt=data["prompt"],
        spec=data.get("spec") or {},
        bbox=expect.get("bbox"), bbox_ratio=expect.get("bbox_ratio"),
        image=data.get("image"),
        dimensions=[str(d) for d in expect.get("dimensions", [])],
        conflicts=expect.get("conflicts") or [],
        approval=data.get("approval") or Case.approval,
        region_name=data.get("region") or "",
        provisional=bool(data.get("provisional")),
        oriented=bool(data.get("oriented")),
        min_declarations=int(expect.get("min_declarations") or 0),
        rays=[RayCheck(**r) for r in expect.get("rays", [])])


def load_cases(path: str = CASES_DIR, include_images: bool = True) -> list[Case]:
    """Every case: the YAML files, plus the Python image corpus."""
    cases: list[Case] = []
    for name in sorted(os.listdir(path)):
        if not name.endswith((".yaml", ".yml")):
            continue
        with open(os.path.join(path, name)) as fh:
            cases.append(case_from_dict(yaml.safe_load(fh) or {}))
    if include_images:
        from evals.image_cases import IMAGE_CASES
        cases.extend(case_from_dict(d) for d in IMAGE_CASES)
    ids = [c.id for c in cases]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:                       # two sources, so collisions are now possible
        raise ValueError(f"duplicate case id(s): {', '.join(sorted(dupes))}")
    return sorted(cases, key=lambda c: c.id)


# --- scoring (pure apart from the injected prober) ------------------------

@dataclass
class CaseResult:
    case_id: str
    passed: bool
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    error: str | None = None
    # Planner revisions used.  None means "not applicable" (tool mode); 0 is a
    # real, meaningful value (passed first try) and must not be conflated with it.
    rounds: int | None = None
    model_file: str | None = None      # standalone .g written for inspection
    # How many times the graph halted for a human.  RECORDED, not asserted: the
    # agent is stochastic, so the same drawing can produce a different number of
    # halts run to run, and a strict expected count would flake.  In unattended
    # mode any halt at all is worth seeing -- the prompt told it not to ask.
    halts: int = 0
    # True when the case has no hand-authored spec, so only what the agent SAID
    # was scored.  Reported separately: rolling these into the headline would
    # inflate it with cases where no geometry was checked at all.
    declaration_only: bool = False
    # The ``declare_assumption`` rows this case produced.  Reported whether or
    # not the case passed: a run where the agent declared nothing and a run
    # where it declared the wrong thing both score FAIL, and only these tell
    # them apart when someone audits the run afterwards.
    declarations: list[dict] = field(default_factory=list)
    # Server rejections the agent hit and recovered from INSIDE a turn -- an
    # invalid spec sent back by build_from_spec, say.  ``rounds`` counts planner
    # revisions only, so without this the report can print "mean revision
    # rounds: 0.0" for a run that took 55 rejected builds to get there, and read
    # as first-time-right when it was nothing of the sort.
    tool_errors: int = 0

    @property
    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        extra = f" (rounds={self.rounds})" if self.rounds else ""
        if self.halts:
            extra += f" (halts={self.halts})"
        if self.tool_errors:
            extra += f" (recovered from {self.tool_errors} tool error(s))"
        if self.declaration_only:
            extra += " (declaration only -- no geometry checked)"
        return f"{state:4} {self.case_id}{extra}"


def _parse_los(nirt_output: str) -> float | None:
    """Line-of-sight length from a nirt hit line, if present.

    A hit line looks like::

        ev_bushing.r         (   8.0000    0.0000   24.0000)   24.0000   0.0000

    The closing paren is attached to the last coordinate, so LOS is the token
    right after the one ending in ``)``.  The column header also contains a
    ``z)`` token, but its next field is non-numeric and gets skipped.
    """
    for line in nirt_output.splitlines():
        fields = line.split()
        for i, field_text in enumerate(fields[:-1]):
            if field_text.endswith(")"):
                try:
                    return float(fields[i + 1])
                except ValueError:
                    break        # header row -> try the next line
    return None


def score_bbox(expected, region: str, prober, oriented: bool = False) -> list:
    """Compare measured outer lengths against the drawing's.

    Two modes, and the default matters.  When the case pins the axes
    (``oriented``), compare axis by axis and name the offending one -- that is
    most of the diagnosis.  Otherwise assert only that each expected length
    APPEARS among the measured three, matched greedily and each measurement used
    once.  A drawing gives a part its lengths; it rarely says which way up to
    build it, and scoring the pose as if it were the shape reports a correct
    model as wrong.

    Multiset containment rather than sorting both sides because an expectation
    may leave an axis open (see ``Case.bbox``), and a partial triple cannot be
    sorted against a full one.
    """
    if not expected:
        return []
    actual = V.parse_bb_lengths(parse_response(prober(f"bb {region}")))
    if oriented:
        checks = []
        for axis, want, got in zip("XYZ", expected, actual):
            if want is None:                   # not settled by the drawing
                continue
            checks.append((f"bbox:{axis}", V.bbox_matches((want,), (got,)),
                           f"expected {want} mm, got {got} mm"))
        return checks
    remaining = [a for a in actual if a is not None]
    missing = []
    for want in [w for w in expected if w is not None]:
        hit = next((a for a in remaining if V.bbox_matches((want,), (a,))), None)
        if hit is None:
            missing.append(want)
        else:
            remaining.remove(hit)
    return [("bbox", not missing,
             f"measured {tuple(actual)} mm contains every expected length"
             if not missing else
             f"expected {missing} mm among {tuple(actual)} mm (any axis order)")]


_RATIO_TOL = 0.03


def score_bbox_ratio(expected, region: str, prober) -> list:
    """Compare the model's PROPORTIONS, ignoring its overall scale.

    Both sides are normalised by their own largest axis, so this asks "is it the
    right shape" and never "is it the right size".  That is the whole point: for
    a reference with no printed units, size is the agent's judgement call and
    shape is not.
    """
    if not expected:
        return []
    actual = V.parse_bb_lengths(parse_response(prober(f"bb {region}")))
    if any(a is None for a in actual) or not any(actual):
        return [("bbox:ratio", False, f"could not measure {region}")]
    want = sorted(e / max(expected) for e in expected)
    got = sorted(a / max(actual) for a in actual)
    # Sorted for the same reason score_bbox uses containment: a proportion is a
    # statement about SHAPE, and a part laid on a different face has the same
    # shape.  Comparing in axis order failed a correct sheet-metal bracket
    # purely for having X and Y the other way round.
    ok = all(abs(w - g) <= _RATIO_TOL for w, g in zip(want, got))
    shape = ":".join(f"{v:.2f}" for v in got)
    return [("bbox:ratio", ok,
             f"expected {':'.join(f'{v:.2f}' for v in want)}, got {shape} "
             f"(from {actual} mm)")]


def score_rays(rays: list[RayCheck], region: str, prober) -> list:
    """Run each explicit ray assertion; return (name, ok, detail) triples.

    Scopes the rays to *region* first: ged ``nirt`` fires at the DISPLAYED
    objects, so unrelated geometry would otherwise register false hits.
    """
    checks = []
    if rays:
        prober("zap")
        prober(f"draw {region}")
    corner = lengths = None
    if any(r.relative or r.fraction or r.los_frac is not None for r in rays):
        extent = V.parse_bb_extent(parse_response(prober(f"bb -e {region}")))
        if extent is None:
            return [("ray:setup", False,
                     f"could not read the extent of {region}; relative rays "
                     f"have nothing to anchor to")]
        corner = extent[:3]
        lengths = [extent[i + 3] - extent[i] for i in range(3)]
    for ray in rays:
        if ray.fraction:
            start = [c + f * L for c, f, L in zip(corner, ray.start, lengths)]
        elif ray.relative:
            start = [c + o for c, o in zip(corner, ray.start)]
        else:
            start = ray.start
        want_los = ray.los
        if ray.los_frac is not None:
            # Along whichever axis the ray travels: the ray is axis-aligned in
            # every case authored so far, and a diagonal has no single extent
            # to be a fraction OF.
            axis = max(range(3), key=lambda i: abs(ray.dir[i]))
            want_los = ray.los_frac * lengths[axis]
        cmd = V.ray_cmd(tuple(start), tuple(ray.dir))
        out = parse_response(prober(cmd))
        missed = V.ray_missed(out)
        ok = missed if ray.expect == "miss" else not missed
        detail = f"expected {ray.expect}, got {'miss' if missed else 'hit'}"
        if ok and want_los is not None and not missed:
            los = _parse_los(out)
            if los is None or abs(los - want_los) > max(0.5, want_los * 0.02):
                ok = False
                detail = f"expected LOS {want_los:.4g} mm, got {los}"
            else:
                detail = f"hit, LOS {los} mm"
        checks.append((f"ray:{ray.desc}", ok, detail))
    return checks


# How many of a case's printed dimensions must appear in the agent's own words.
# Not all of them: an agent that builds a part correctly routinely narrates only
# the numbers it found notable, and demanding a full recital made this the single
# largest source of failures in a run -- three of seven, every one on a model
# that was geometrically fine.  A floor still catches the failure that matters,
# which is reading almost nothing and inventing the rest.
_DIMENSION_FRACTION = 0.7


def score_dimensions(case: Case, proposal: str):
    """Check the agent's stated dimensions against the drawing's real numbers.

    Separate from the geometry checks on purpose: reading the drawing and
    building from what you read are different skills, and a case can fail one
    while passing the other.

    Weakest check in the harness, and knowingly so -- it substring-matches
    prose, the same technique that had to be torn out of the conflict check.
    It survives because it catches a real failure (numbers invented wholesale)
    that engine truth cannot see, but it is scored as a FLOOR so that ordinary
    variation in how much an agent narrates does not read as a defect.
    """
    if not case.dimensions or not proposal:
        return []
    found = [d for d in case.dimensions if d in proposal]
    need = max(1, round(len(case.dimensions) * _DIMENSION_FRACTION))
    missing = [d for d in case.dimensions if d not in proposal]
    return [("dimensions", len(found) >= need,
             f"stated {len(found)}/{len(case.dimensions)} "
             f"(need {need}); missing: {', '.join(missing) or 'none'}")]


def conflict_groups(conflicts) -> list[list[str]]:
    """Normalise ``conflicts`` to one group of equivalent tokens per side.

    A bare string is its own group, so the old single-token form still works.
    """
    return [[c] if isinstance(c, str) else [str(v) for v in c]
            for c in (conflicts or [])]


def score_conflicts(case: Case, declarations: list[dict]):
    """Did the agent DECLARE the drawing's contradiction rather than silently pick?

    The failure this catches is the dangerous one: quietly clamping a value that
    cannot hold, so the build looks clean and the contradiction is invisible.

    Reads ``declare_assumption`` rows, not the transcript.  Substring-matching
    free text produced a FALSE PASS on the first contradictory case run: both
    conflicting values appeared -- three replies apart, in unrelated prose --
    while the agent had in fact resolved the contradiction silently.  Here the
    values have to occur TOGETHER in one declaration's ``chose``/``over``, which
    is a claim the agent has to actually make rather than one we infer.

    Each side is a GROUP of equivalent tokens, because one contradiction can be
    stated in more than one frame and all of them are correct.  On the lego
    brick, a 6.3 mm cavity and a 1.0 mm roof cannot both hold on a 9.6 mm body;
    an agent may say it took 1.0 over 6.3, or equivalently 8.6 of cavity over
    6.3, since 8.6 + 1.0 = 9.6.  Demanding the literal "1.0" failed two runs
    that had named the contradiction perfectly well -- a false FAIL exactly as
    misleading as the false PASS above.
    """
    groups = conflict_groups(case.conflicts)
    if not groups:
        return []
    for row in declarations:
        stated = f"{row.get('chose', '')} {row.get('over', '')}"
        if all(any(tok in stated for tok in group) for group in groups):
            return [("conflict-declared", True,
                     f"declared: {row.get('topic')} = {row.get('chose')!r} "
                     f"over {row.get('over')!r}")]
    if not declarations:
        return [("conflict-declared", False, "declared no assumptions at all")]
    got = "; ".join(f"{r.get('chose')}/{r.get('over')}" for r in declarations[:4])
    want = " + ".join("|".join(g) for g in groups)
    return [("conflict-declared", False,
             f"no single declaration names {want} (declared: {got})")]


def score_declarations(case: Case, declarations: list[dict]):
    """Did an under-dimensioned drawing produce the declarations it demands?

    Deliberately a floor, not an exact count: how many separate assumptions a
    reading breaks into is a judgement, but ZERO on a sheet that cannot
    determine the part is not.
    """
    if not case.min_declarations:
        return []
    got = len(declarations)
    return [("declarations", got >= case.min_declarations,
             f"{got} declared, expected at least {case.min_declarations} "
             f"for a drawing this under-dimensioned")]


def score_case(case: Case, prober, rounds: int = 0,
               proposal: str = "", halts: int = 0,
               declarations: list[dict] | None = None) -> CaseResult:
    """Score the built region against the case's GROUND TRUTH."""
    declarations = declarations or []
    if not case.has_ground_truth:
        # Declaration-only case: score what the agent stated, not the geometry.
        # Passing here means "read the drawing and declared its reading", NOT
        # "built the right thing" -- the report marks these so the two are never
        # added up as if they measured the same thing.
        checks = (score_dimensions(case, proposal)
                  + score_conflicts(case, declarations)
                  + score_declarations(case, declarations))
        return CaseResult(case.id, all(ok for _, ok, _ in checks) and bool(checks),
                          checks, rounds=rounds, halts=halts,
                          declaration_only=True, declarations=declarations)
    if case.has_spec:
        # Full ground truth: the spec drives the whole engine-truth sweep --
        # existence, derived bbox, and a ray per feature.
        checks = V._verify(BuildSpec.model_validate(case.spec), prober)[1]
    else:
        # Region name + hand-read expectations, no spec.  Existence has to be
        # asserted here rather than left implicit: without it a region the agent
        # never created reports a bbox of None and reads as a measurement
        # mismatch, which hides that nothing was built at all.
        found = parse_response(prober(f"l {case.region}")).strip()
        checks = [("exists", bool(found) and "not found" not in found.lower(),
                   f"region {case.region}")]
    if any(name == "exists" and not ok for name, ok, _ in checks):
        # Nothing was built: further bbox/ray failures are noise that hides the
        # real result, which is simply that the region is missing.  The
        # declaration check still runs -- on a contradictory case, naming the
        # conflict and declining to guess is a meaningful partial result.
        checks.extend(score_conflicts(case, declarations))
        return CaseResult(case.id, False, checks, rounds=rounds, halts=halts,
                          declarations=declarations)
    checks.extend(score_bbox(case.bbox, case.region, prober, case.oriented))
    checks.extend(score_bbox_ratio(case.bbox_ratio, case.region, prober))
    checks.extend(score_rays(case.rays, case.region, prober))
    checks.extend(score_dimensions(case, proposal))
    checks.extend(score_conflicts(case, declarations))
    checks.extend(score_declarations(case, declarations))
    passed = all(ok for _, ok, _ in checks)
    return CaseResult(case.id, passed, checks, rounds=rounds, halts=halts,
                      declarations=declarations)


def report(results: list[CaseResult]) -> str:
    """Per-case lines plus the aggregate reliability number."""
    lines = ["", "=" * 62, "BRL-CAD model-building reliability", "=" * 62]
    for res in results:
        lines.append(res.summary)
        if res.model_file:
            lines.append(f"       model: {res.model_file}")
        for row in res.declarations:
            over = f" over {row.get('over')!r}" if row.get("over") else ""
            lines.append(f"       ~ assumed {row.get('topic')}="
                         f"{row.get('chose')!r}{over}")
        if not res.passed:
            for name, ok, detail in res.checks:
                if not ok:
                    lines.append(f"       x {name}: {detail}")
            if res.error:
                lines.append(f"       x error: {res.error}")
    built = [r for r in results if not r.declaration_only]
    declared = [r for r in results if r.declaration_only]
    lines += ["-" * 62]
    if built:
        ok = sum(1 for r in built if r.passed)
        lines.append(f"{ok}/{len(built)} cases passed "
                     f"({100.0 * ok / len(built):.0f}% reliability)")
    if declared:
        # Kept apart on purpose: these check what the agent SAID, not what it
        # built, so folding them into the headline would inflate it with cases
        # where no geometry was verified at all.
        ok = sum(1 for r in declared if r.passed)
        lines.append(f"{ok}/{len(declared)} declaration-only cases passed "
                     f"(read + declared; geometry NOT checked)")
    # Always report first-attempt accuracy when rounds were tracked: "all zero"
    # is the best possible result, so hiding the line then would bury the news.
    rounds = [r.rounds for r in results if r.rounds is not None]
    if rounds:
        first_try = sum(1 for r in results
                        if r.passed and r.rounds == 0)
        lines.append(f"first-attempt passes: {first_try}/{len(results)}; "
                     f"mean revision rounds: {sum(rounds) / len(rounds):.1f}")
    return "\n".join(lines)


# --- runners --------------------------------------------------------------

def case_message(case: Case):
    """The agent's input for a case: its prompt, plus a reference image if any."""
    from langchain_core.messages import HumanMessage

    from client_v2.terminal.attachments import image_part_from_file

    path = case.image_path
    if not path:
        return HumanMessage(content=case.prompt)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"case {case.id}: missing image {path}")
    return HumanMessage(content=[{"type": "text", "text": case.prompt},
                                 image_part_from_file(Path(path))])


def release_socket() -> None:
    """Close our persistent listener connection so another client can be served."""
    from brlcad_mcp.transport import socket_bridge
    socket_bridge._connection._disconnect()


MODELS_DIR = os.path.expanduser("~/brlcad_eval_models")


def export_case(case: Case, prober, models_dir: str | None = None) -> str | None:
    """Export a case's region to its OWN .g file so it can be opened directly.

    Every case is built in one shared scratch database, which is awkward to
    inspect; ``keep`` copies the region and everything it references into a
    standalone file (openable with ``mged <run>/models/<case>.g``).
    """
    if not case.has_ground_truth:
        return None             # no known region to export
    models_dir = models_dir or MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f"{case.id}.g")
    try:
        os.remove(path)          # keep refuses to overwrite
    except OSError:
        pass
    out = prober(f"keep {path} {case.region}")
    return path if os.path.isfile(path) else f"export failed: {out[:80]}"


def reset_case(case: Case, prober) -> None:
    """Delete a case's geometry and saved spec so each run starts clean.

    The harness owns the ``ev_*`` names, and a run must not be influenced by
    what a previous run left behind -- stale objects would otherwise trip the
    build tool's collision guard (it refuses to overwrite geometry that has no
    saved spec) and every case would fail for the wrong reason.
    """
    import shutil

    if not case.has_ground_truth:
        return                  # no known region name; nothing of ours to clear

    from brlcad_mcp.server.tools.reconstruct import (
        _accum_name,
        _name_dir,
        _solid_name,
    )

    # A spec-less case still knows the name the prompt dictates, and clearing it
    # still matters: a region left over from an earlier run would be measured as
    # if this run had built it.  ``killtree`` takes the comb and its children, so
    # the per-part sweep below is only the extra precision a spec buys.
    name = case.spec.get("name") or case.region_name
    prober(f"killtree {name}.r")
    for part in case.spec.get("parts", []):
        prober(f"killall {_solid_name(name, part['name'])}")
    for i in range(1, len(case.spec.get("parts", [])) + 1):
        prober(f"killall {_accum_name(name, i)}")
    shutil.rmtree(_name_dir(name), ignore_errors=True)


def run_tool_mode(cases: list[Case], run_dir: str | None = None) -> list[CaseResult]:
    """Build each case from its ground-truth spec, then score it."""
    from brlcad_mcp.server.tools.reconstruct import build_from_spec
    from brlcad_mcp.transport import send_command

    results = []
    for case in cases:
        if not case.has_spec:
            # Tool mode builds FROM the ground-truth spec, so a case that has
            # none is not applicable here -- it is an agent-mode case only.
            # (``has_ground_truth`` is weaker: a bbox read off the drawing is
            # enough to SCORE a build, but not enough to produce one.)
            continue
        reset_case(case, send_command)          # each case starts from nothing
        spec = dict(case.spec)
        spec["views"] = []                     # geometry only: skip renders
        out = build_from_spec(json.dumps(spec))
        if out.startswith("Error"):
            results.append(CaseResult(case.id, False, error=out.strip()))
            continue
        result = score_case(case, send_command)
        result.model_file = export_case(case, send_command, _models(run_dir))
        results.append(result)
    return results


RESULTS_FILE = "results.jsonl"


def append_results(run_dir: str | None, rep: int,
                   results: list[CaseResult]) -> None:
    """Append one pass's results, flushed, before the next pass starts.

    A pass^k run is hours long, so the results file -- not memory -- is the
    source of truth: a crash in pass 7 must not cost the six that finished.
    """
    if not run_dir:
        return
    with open(os.path.join(run_dir, RESULTS_FILE), "a") as fh:
        for res in results:
            fh.write(json.dumps({
                "rep": rep, "case": res.case_id, "passed": res.passed,
                "rounds": res.rounds, "halts": res.halts,
                "declaration_only": res.declaration_only,
                "error": res.error,
                "tool_errors": res.tool_errors,
                # EVERY check, not just the failures.  Recording only failures
                # made a pass unauditable: two cases were passing on two checks
                # apiece and nothing in the record said so.
                "checks": [[n, ok] for n, ok, _ in res.checks],
                "failed_checks": [n for n, ok, _ in res.checks if not ok],
                "declarations": res.declarations,
            }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def next_rep(run_dir: str) -> int:
    """The pass index to use next, from what is already on disk.

    Derived rather than passed in, so an externally-driven loop -- one that
    restarts the listener between passes -- does not have to track where it got
    to, and a resumed job continues numbering instead of overwriting.
    """
    reps = [r.get("rep", 0) for r in read_results(run_dir)]
    return max(reps) + 1 if reps else 0


def read_results(run_dir: str) -> list[dict]:
    path = os.path.join(run_dir, RESULTS_FILE)
    if not os.path.isfile(path):
        return []
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue
    return rows


def report_repeats(rows: list[dict]) -> str:
    """pass^k and pass@1 per case, from the incremental results file.

    ``pass@1`` is the average single-run success rate -- the number a one-shot
    demo shows you.  ``pass^k`` is whether ALL k runs passed, which is what
    "can I rely on this" actually means.  The gap between them IS the
    reliability signal, and reporting only the first is how a flaky agent looks
    finished.
    """
    by_case: dict[str, list[dict]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    lines = ["", "=" * 72,
             "RELIABILITY  (pass^k = every run passed; pass@1 = average run)",
             "=" * 72]
    solid = flaky = broken = 0
    for case_id in sorted(by_case):
        runs = by_case[case_id]
        ok = sum(1 for r in runs if r["passed"])
        k = len(runs)
        mark = "PASS^k" if ok == k else ("FAIL  " if ok == 0 else "FLAKY ")
        solid += ok == k
        broken += ok == 0
        flaky += 0 < ok < k
        note = " (declaration only)" if runs[0].get("declaration_only") else ""
        lines.append(f"{mark} {case_id:28} {ok}/{k}{note}")
        # Why it failed, most common first -- one flaky case usually has one
        # cause, and naming it is the difference between a number and a lead.
        tally: dict[str, int] = {}
        for r in runs:
            for name in r.get("failed_checks") or []:
                tally[name] = tally.get(name, 0) + 1
            if r.get("error"):
                tally[f"error: {r['error'][:60]}"] = \
                    tally.get(f"error: {r['error'][:60]}", 0) + 1
        for name, n in sorted(tally.items(), key=lambda kv: -kv[1])[:4]:
            lines.append(f"         x {name}  ({n}/{k} runs)")
    total_runs = len(rows)
    total_ok = sum(1 for r in rows if r["passed"])
    cases = len(by_case)
    lines += ["-" * 72]
    if cases:
        lines.append(f"pass^k : {solid}/{cases} cases passed EVERY run "
                     f"({100.0 * solid / cases:.0f}%)")
    if total_runs:
        lines.append(f"pass@1 : {total_ok}/{total_runs} runs passed "
                     f"({100.0 * total_ok / total_runs:.0f}%)")
    lines.append(f"solid {solid} | flaky {flaky} | always-failing {broken}")
    halts = sum(r.get("halts") or 0 for r in rows)
    if halts:
        lines.append(f"halts: {halts} across {total_runs} runs "
                     f"(unattended mode told the agent not to ask)")
    # Say plainly that the guardrails did work, or the pass rate reads as
    # first-attempt accuracy when it is nothing of the kind.
    errs = sum(r.get("tool_errors") or 0 for r in rows)
    if errs:
        touched = sum(1 for r in rows if r.get("tool_errors"))
        lines.append(f"server rejections the agent recovered from: {errs} "
                     f"across {touched}/{total_runs} runs -- this is reliability "
                     f"WITH the spec validator, not raw first-attempt accuracy")
    # How much each case was actually checked.  Two cases were passing on two
    # checks apiece; without this the aggregate hides which ones are load
    # bearing and which are nearly free.
    thin = sorted((len(r.get("checks") or []), r["case"]) for r in rows)
    thin = {c: n for n, c in thin if n and n <= 3}
    if thin:
        lines.append("thinly checked (few assertions -- a pass here means "
                     "little): " + ", ".join(f"{c} ({n})"
                                             for c, n in thin.items()))
    return "\n".join(lines)


async def run_agent_mode(cases: list[Case], auto_approve: bool = False,
                         run_dir: str | None = None,
                         rep: int = 0) -> list[CaseResult]:
    """Give the agent each case's prompt, then score what it built.

    Two shapes, and the difference is the point of comparing them:

    * **scripted** (default) -- the case's ``approval`` answers the confirmation
      the workflow asks for, so the halt is exercised as a user would exercise it.
    * **unattended** (``auto_approve``) -- the BASELINE.  No human, no simulated
      user, nothing stochastic besides the agent itself, so a failure is
      attributable to the agent and nothing else.  The worker prompt gains
      ``UNATTENDED_SUFFIX`` (they are coupled deliberately: auto-approving without
      telling the agent nobody is listening would score builds founded on
      unresolved conflicts), and no scripted turn is sent -- one turn, decide,
      build, declare.
    """
    from langchain_core.messages import (
        AIMessage,
        HumanMessage,
        ToolMessage,
    )
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from langchain_mcp_adapters.tools import load_mcp_tools
    from langgraph.checkpoint.memory import MemorySaver
    from langgraph.types import Command

    from brlcad_mcp.transport import send_command
    from client_v2.agents.conversational import message_text
    from client_v2.graph import build_graph
    from client_v2.model import build_model
    from client_v2.runlog import open_run_log
    from client_v2.skills import SkillRegistry

    model = build_model()
    # Judge model, pinned independently of the worker.  When comparing two
    # models on the same corpus, letting the visual check switch with the
    # worker would confound the comparison: a difference in score could be
    # either model building better or judging more leniently, and there would
    # be no way to tell which.
    judge = None
    if os.getenv("EVAL_JUDGE_MODEL"):
        from langchain_openai import ChatOpenAI

        from client_v2.model import model_config
        judge = ChatOpenAI(**model_config(
            os.environ["EVAL_JUDGE_MODEL"],
            settings.llm.reasoning_effort, settings.llm.temperature))
    registry = SkillRegistry.from_dir()
    # One log for the whole eval run: when a case fails, the reason is readable
    # afterwards instead of needing the run repeated.
    stamp = "log" if not rep else f"log{rep:02d}"
    log = (open_run_log(directory=run_dir, stamp=stamp) if run_dir
           else open_run_log(stamp=time.strftime("eval_%Y%m%d_%H%M%S")))
    client = MultiServerMCPClient({"brlcad_server": {
        "command": sys.executable, "args": ["-m", "brlcad_mcp.server"],
        "transport": "stdio", "env": dict(os.environ)}})

    # Run every agent turn first, then score AFTER the MCP session closes: the
    # listener serves one client at a time, so scoring on our own socket while
    # the MCP server subprocess still holds its connection would deadlock.
    # Reset BEFORE opening the MCP session, then RELEASE our socket.  The
    # listener serves one client at a time from a serial accept loop, and the
    # transport keeps a persistent connection -- so an idle-but-open socket of
    # ours starves the MCP server subprocess and its first build times out.
    for case in cases:
        reset_case(case, send_command)
    release_socket()

    outcomes: list[tuple[Case, int, str | None, str]] = []
    async with client.session("brlcad_server") as session:
        tools = await load_mcp_tools(session)
        # A checkpointer + per-case thread makes the approval turn a genuine
        # continuation of the proposal turn.
        graph = build_graph(worker_model=model, tools=tools, registry=registry,
                            worker_prompt=(unattended_worker_prompt
                                           if auto_approve else None),
                            visual_model=judge,
                            checkpointer=MemorySaver(), log=log)

        async def drive(payload, cfg, answer):
            """Run a turn, answering any authorization halt. Returns (state, halts).

            A workflow that declares an `authorize` step genuinely halts the
            graph, so an unattended run has to answer it rather than mistaking
            the pause for a finished turn.  The halt COUNT is returned because it
            is a result, not plumbing: in unattended mode the prompt told the
            agent not to ask, so a halt means the structural gate fired anyway
            (or the agent ignored the instruction) and that is worth seeing.
            """
            state = await graph.ainvoke(payload, cfg)
            halts = 0
            for _ in range(4):                      # bounded: never loop forever
                pauses = state.get("__interrupt__") or ()
                if not pauses:
                    break
                halts += 1
                log.event("interrupt", question=str(pauses[0]), answer=answer)
                state = await graph.ainvoke(Command(resume=answer), cfg)
            return state, halts

        def all_ai_text(state) -> str:
            """Everything the agent said this turn.

            The dimension check reads this rather than only the final message:
            when the prompt pre-approves the numbers the agent builds straight
            away, so the values may be stated in a build summary or an earlier
            message rather than in a separate proposal.
            """
            return "\n".join(message_text(m).strip()
                              for m in state.get("messages", [])
                              if isinstance(m, AIMessage))

        def count_tool_errors(state) -> int:
            """Server rejections the agent recovered from within this turn.

            These are invisible to ``rounds`` -- they happen inside one worker
            turn, below the granularity a planner revision measures -- yet they
            are most of the self-correction actually going on.  Counting them is
            what stops the report implying a first-attempt success it did not
            have.
            """
            return sum(1 for m in state.get("messages", []) or []
                       if isinstance(m, ToolMessage)
                       and message_text(m).lstrip().startswith("Error"))

        for case in cases:
            print(f"  running {case.id} ...", flush=True)
            # The thread carries the repeat index: without it, pass 2 of a case
            # would RESUME pass 1's conversation from the checkpointer and see
            # the model it already built, which is not a fresh sample.
            cfg = {"configurable": {"thread_id": f"{case.id}#{rep}"},
                   "recursion_limit": 60, "callbacks": log.callbacks()}
            log.start_turn(f"[case {case.id}] {case.prompt}",
                           images=1 if case.image_path else 0)
            answer = UNATTENDED_ACK if auto_approve else case.approval
            # Declarations accumulate in one file across the run, and cases run
            # one at a time -- so this case's are whatever appears after here.
            before = len(read_declarations(run_dir))
            try:
                # Turn 1: the agent reads the reference and either builds (in
                # unattended mode) or proposes dimensions for confirmation.
                state, halts = await drive(
                    {"messages": [case_message(case)]}, cfg, answer)
                proposal = all_ai_text(state)
                tool_errors = count_tool_errors(state)
                rounds = state.get("revisions") or 0
                # Turn 2 exists only for the scripted shape: answer the
                # confirmation so the authorize step is exercised rather than
                # bypassed.  In unattended mode there is no second turn by
                # design -- the prompt told the agent to decide and proceed, so
                # sending one would hand it the human it was told it did not have.
                if not auto_approve and case.approval:
                    state, more = await drive(
                        {"messages": [HumanMessage(content=case.approval)]},
                        cfg, answer)
                    halts += more
                    rounds = max(rounds, state.get("revisions") or 0)
                    # Keep BOTH turns' text: the dimension proposal is in turn 1
                    # and the build summary in turn 2, and either can carry the
                    # values the scorer looks for.
                    proposal = f"{proposal}\n{all_ai_text(state)}"
                    tool_errors = count_tool_errors(state)
                outcomes.append((case, rounds, None, proposal, halts,
                                 read_declarations(run_dir)[before:],
                                 tool_errors))
            except Exception as exc:            # a crashed turn is a failure
                outcomes.append((case, 0, f"{type(exc).__name__}: {exc}",
                                 "", 0, [], 0))

    results = []
    for case, rounds, error, proposal, halts, declared, errs in outcomes:
        if error:
            results.append(CaseResult(case.id, False, error=error,
                                      rounds=rounds, halts=halts,
                                      tool_errors=errs))
            continue
        result = score_case(case, send_command, rounds=rounds,
                            proposal=proposal, halts=halts,
                            declarations=declared)
        result.tool_errors = errs
        result.model_file = export_case(case, send_command, _models(run_dir))
        results.append(result)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tool", "agent"), default="tool")
    ap.add_argument(
        "--case",
        help=("run only these cases: an id, a comma-separated list, or a glob "
              "such as '*_guided' to run one tier across the whole corpus"))
    ap.add_argument(
        "--auto-approve", action="store_true",
        help=("agent mode BASELINE: no human and no simulated user. Halts are "
              "acknowledged without being read, and the worker prompt gains the "
              "unattended delta telling it to resolve ambiguity itself and "
              "record it with declare_assumption. One turn per case, so a "
              "failure is attributable to the agent alone."))
    ap.add_argument(
        "--repeat", type=int, default=1, metavar="K",
        help=("run the whole suite K times and report pass^k (every run passed) "
              "alongside pass@1. The suite repeats as a whole rather than each "
              "case K times in a row, so an interrupted job still has one "
              "sample of everything instead of ten samples of the first case."))
    ap.add_argument(
        "--run-dir", metavar="PATH",
        help=("append to an existing run directory instead of starting a new "
              "one. Lets a driver restart the listener between passes -- which "
              "it must, for a long job: the render path leaks two pipe fds per "
              "call and the server aborts once an fd exceeds 1024."))
    args = ap.parse_args()
    if args.auto_approve and args.mode != "agent":
        sys.exit("--auto-approve only applies to --mode agent")
    if args.repeat > 1 and args.mode != "agent":
        sys.exit("--repeat only makes sense for --mode agent (tool mode is "
                 "deterministic: the same spec builds the same geometry)")

    cases = load_cases()
    if args.case:
        import fnmatch
        wanted = [w.strip() for w in args.case.split(",") if w.strip()]
        cases = [c for c in cases
                 if any(c.id == w or fnmatch.fnmatch(c.id, w) for w in wanted)]
        if not cases:
            sys.exit(f"no case matches: {args.case}")

    run_dir = new_run_dir(args.mode, args.auto_approve, args.run_dir)
    route_artifacts_into(run_dir)           # before the MCP subprocess is spawned
    first_rep = next_rep(run_dir)

    shape = " (unattended baseline)" if args.auto_approve else ""
    print(f"{len(cases)} case(s), mode={args.mode}{shape}")
    print(f"run dir: {run_dir}")
    header = (f"mode={args.mode}{shape}\ncases={len(cases)}\n"
              f"stamp={os.path.basename(run_dir)}\n")
    if args.mode == "agent":
        for i in range(args.repeat):
            rep = first_rep + i
            print(f"\n--- pass {rep + 1} ({time.strftime('%H:%M:%S')}) ---",
                  flush=True)
            results = asyncio.run(
                run_agent_mode(cases, args.auto_approve, run_dir, rep))
            # Write BEFORE the next pass: this job runs for hours and the file
            # is the record, not the process.
            append_results(run_dir, rep, results)
            text = report(results)
            print(text, flush=True)
            rows = read_results(run_dir)
            if len({r.get("rep", 0) for r in rows}) > 1:
                text = report_repeats(rows)
                print(text, flush=True)
            Path(run_dir, "report.txt").write_text(
                f"{header}passes={rep + 1}\n{text}\n")
    else:
        results = run_tool_mode(cases, run_dir)
        text = report(results)
        print(text)
        # The report goes in the run dir too, so the directory is the whole
        # record: what was asked, what happened, what was built, the verdict.
        Path(run_dir, "report.txt").write_text(header + text + "\n")
    print(f"\nartifacts: {run_dir}")
    for entry, what in (
        ("report.txt", "this verdict"),
        ("log.jsonl", "every model call, node write and interrupt"),
        (RESULTS_FILE, "one row per case per pass (the pass^k record)"),
        ("renders", "check views the agent produced"),
        ("models", "per-case standalone .g (open with: mged <file>)"),
        ("backups", "restore points from destructive raw commands"),
        ("specs", "the saved build spec verify-by-name reads back"),
    ):
        # Describe what is actually THERE.  Tool mode writes no log (no model
        # calls) and no renders (it builds with views: []), and claiming
        # otherwise sends whoever audits the run looking for a missing file.
        target = Path(run_dir, entry)
        if not target.exists():
            continue
        count = (f" ({sum(1 for p in target.rglob('*') if p.is_file())} files)"
                 if target.is_dir() else "")
        print(f"  {entry + count:<28} {what}")
    print("  (also: evals/runs/latest -> this run)")
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
