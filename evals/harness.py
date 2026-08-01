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

Usage:
    python -m evals.harness                 # tool mode (default)
    python -m evals.harness --mode agent    # needs OPENAI_API_KEY
    python -m evals.harness --case l_bracket
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from brlcad_mcp.server.tools import verify as V
from brlcad_mcp.server.tools.helpers import parse_response
from brlcad_mcp.server.tools.reconstruct import BuildSpec

CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")


# --- cases ----------------------------------------------------------------

@dataclass
class RayCheck:
    """One explicit ray assertion: fired from *start* along *dir*."""

    desc: str
    start: list[float]
    dir: list[float]
    expect: str            # "miss" (a void/hole) or "hit" (material)
    los: float | None = None   # optional expected line-of-sight length (mm)


@dataclass
class Case:
    id: str
    prompt: str
    spec: dict                       # ground truth
    rays: list[RayCheck] = field(default_factory=list)
    bbox: list[float] | None = None  # optional explicit override
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

    @property
    def region(self) -> str:
        return f"{self.spec['name']}.r"

    @property
    def image_path(self) -> str | None:
        return os.path.expanduser(self.image) if self.image else None


def load_cases(path: str = CASES_DIR) -> list[Case]:
    """Load every case YAML, sorted by id."""
    cases: list[Case] = []
    for name in sorted(os.listdir(path)):
        if not name.endswith((".yaml", ".yml")):
            continue
        with open(os.path.join(path, name)) as fh:
            data = yaml.safe_load(fh) or {}
        expect = data.get("expect") or {}
        cases.append(Case(
            id=data["id"], prompt=data["prompt"], spec=data["spec"],
            bbox=expect.get("bbox"), image=data.get("image"),
            dimensions=[str(d) for d in expect.get("dimensions", [])],
            approval=data.get("approval") or Case.approval,
            rays=[RayCheck(**r) for r in expect.get("rays", [])]))
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

    @property
    def summary(self) -> str:
        state = "PASS" if self.passed else "FAIL"
        extra = f" (rounds={self.rounds})" if self.rounds else ""
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


def score_rays(rays: list[RayCheck], region: str, prober) -> list:
    """Run each explicit ray assertion; return (name, ok, detail) triples.

    Scopes the rays to *region* first: ged ``nirt`` fires at the DISPLAYED
    objects, so unrelated geometry would otherwise register false hits.
    """
    checks = []
    if rays:
        prober("zap")
        prober(f"draw {region}")
    for ray in rays:
        cmd = V.ray_cmd(tuple(ray.start), tuple(ray.dir))
        out = parse_response(prober(cmd))
        missed = V.ray_missed(out)
        ok = missed if ray.expect == "miss" else not missed
        detail = f"expected {ray.expect}, got {'miss' if missed else 'hit'}"
        if ok and ray.los is not None and not missed:
            los = _parse_los(out)
            if los is None or abs(los - ray.los) > max(0.5, ray.los * 0.02):
                ok = False
                detail = f"expected LOS {ray.los} mm, got {los}"
            else:
                detail = f"hit, LOS {los} mm"
        checks.append((f"ray:{ray.desc}", ok, detail))
    return checks


def score_dimensions(case: Case, proposal: str):
    """Check the agent's stated dimensions against the drawing's real numbers.

    Separate from the geometry checks on purpose: reading the drawing and
    building from it are different failures, and a model that is internally
    consistent but built from misread numbers is the more dangerous one.
    """
    # Tool mode builds straight from the ground-truth spec, so there is no
    # proposal to read: the check is not applicable rather than failing.
    if not case.dimensions or not proposal:
        return []
    missing = [d for d in case.dimensions if d not in (proposal or "")]
    return [("dimensions", not missing,
             "all stated in the proposal" if not missing
             else f"never stated: {', '.join(missing)}")]


def score_case(case: Case, prober, rounds: int = 0,
               proposal: str = "") -> CaseResult:
    """Score the built region against the case's GROUND TRUTH."""
    spec = BuildSpec.model_validate(case.spec)
    passed, checks = V._verify(spec, prober)
    if not passed and any(name == "exists" and not ok for name, ok, _ in checks):
        # Nothing was built: further bbox/ray failures are noise that hides the
        # real result, which is simply that the region is missing.
        return CaseResult(case.id, False, checks, rounds=rounds)
    if case.bbox is not None:
        actual = V.parse_bb_lengths(parse_response(prober(f"bb {case.region}")))
        ok = V.bbox_matches(tuple(case.bbox), actual)
        checks.append(("bbox:explicit", ok,
                       f"expected {tuple(case.bbox)} mm, got {actual} mm"))
    checks.extend(score_rays(case.rays, case.region, prober))
    checks.extend(score_dimensions(case, proposal))
    passed = all(ok for _, ok, _ in checks)
    return CaseResult(case.id, passed, checks, rounds=rounds)


def report(results: list[CaseResult]) -> str:
    """Per-case lines plus the aggregate reliability number."""
    lines = ["", "=" * 62, "BRL-CAD model-building reliability", "=" * 62]
    for res in results:
        lines.append(res.summary)
        if res.model_file:
            lines.append(f"       model: {res.model_file}")
        if not res.passed:
            for name, ok, detail in res.checks:
                if not ok:
                    lines.append(f"       x {name}: {detail}")
            if res.error:
                lines.append(f"       x error: {res.error}")
    passed = sum(1 for r in results if r.passed)
    total = len(results) or 1
    lines += ["-" * 62,
              f"{passed}/{len(results)} cases passed "
              f"({100.0 * passed / total:.0f}% reliability)"]
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


def export_case(case: Case, prober) -> str | None:
    """Export a case's region to its OWN .g file so it can be opened directly.

    Every case is built in one shared scratch database, which is awkward to
    inspect; ``keep`` copies the region and everything it references into a
    standalone file (openable with ``mged ~/brlcad_eval_models/<case>.g``).
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{case.id}.g")
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

    from brlcad_mcp.server.tools.reconstruct import (
        _accum_name,
        _name_dir,
        _solid_name,
    )

    name = case.spec["name"]
    prober(f"killtree {name}.r")
    for part in case.spec.get("parts", []):
        prober(f"killall {_solid_name(name, part['name'])}")
    for i in range(1, len(case.spec.get("parts", [])) + 1):
        prober(f"killall {_accum_name(name, i)}")
    shutil.rmtree(_name_dir(name), ignore_errors=True)


def run_tool_mode(cases: list[Case]) -> list[CaseResult]:
    """Build each case from its ground-truth spec, then score it."""
    import json

    from brlcad_mcp.server.tools.reconstruct import build_from_spec
    from brlcad_mcp.transport import send_command

    results = []
    for case in cases:
        reset_case(case, send_command)          # each case starts from nothing
        spec = dict(case.spec)
        spec["views"] = []                     # geometry only: skip renders
        out = build_from_spec(json.dumps(spec))
        if out.startswith("Error"):
            results.append(CaseResult(case.id, False, error=out.strip()))
            continue
        result = score_case(case, send_command)
        result.model_file = export_case(case, send_command)
        results.append(result)
    return results


async def run_agent_mode(cases: list[Case]) -> list[CaseResult]:
    """Give the agent each case's prompt, then score what it built."""
    from langchain_core.messages import AIMessage, HumanMessage
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
    registry = SkillRegistry.from_dir()
    # One log for the whole eval run: when a case fails, the reason is readable
    # afterwards instead of needing the run repeated.
    log = open_run_log(stamp=time.strftime("eval_%Y%m%d_%H%M%S"))
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
                            checkpointer=MemorySaver(), log=log)

        async def drive(payload, cfg):
            """Run a turn, auto-answering any authorization halt.

            A workflow that declares an `authorize` step genuinely halts the
            graph, so an unattended run has to answer it -- with the same
            approval a user would give -- rather than mistaking the pause for a
            finished turn.
            """
            state = await graph.ainvoke(payload, cfg)
            for _ in range(4):                      # bounded: never loop forever
                pauses = state.get("__interrupt__") or ()
                if not pauses:
                    break
                log.event("interrupt", question=str(pauses[0]),
                          answer=case.approval)
                state = await graph.ainvoke(
                    Command(resume=case.approval), cfg)
            return state

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

        for case in cases:
            print(f"  running {case.id} ...", flush=True)
            cfg = {"configurable": {"thread_id": case.id},
                   "recursion_limit": 60, "callbacks": log.callbacks()}
            log.start_turn(f"[case {case.id}] {case.prompt}",
                           images=1 if case.image_path else 0)
            try:
                # Turn 1: the agent reads the reference and proposes dimensions.
                state = await drive({"messages": [case_message(case)]}, cfg)
                proposal = all_ai_text(state)
                rounds = state.get("revisions") or 0
                # Turn 2: answer the confirmation the workflow asks for, so the
                # authorize step is exercised rather than bypassed.  Skipped if
                # the agent already built the model in one turn.
                built = any(case.spec["name"] in str(v)
                            for v in (state.get("step_outputs") or {}).values())
                if case.approval and not built:
                    state = await drive(
                        {"messages": [HumanMessage(content=case.approval)]}, cfg)
                    rounds = max(rounds, state.get("revisions") or 0)
                    proposal = all_ai_text(state)
                outcomes.append((case, rounds, None, proposal))
            except Exception as exc:            # a crashed turn is a failure
                outcomes.append((case, 0, f"{type(exc).__name__}: {exc}", ""))

    results = []
    for case, rounds, error, proposal in outcomes:
        if error:
            results.append(CaseResult(case.id, False, error=error, rounds=rounds))
            continue
        result = score_case(case, send_command, rounds=rounds,
                            proposal=proposal)
        result.model_file = export_case(case, send_command)
        results.append(result)
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("tool", "agent"), default="tool")
    ap.add_argument("--case", help="run only this case id")
    args = ap.parse_args()

    cases = load_cases()
    if args.case:
        cases = [c for c in cases if c.id == args.case]
        if not cases:
            sys.exit(f"no such case: {args.case}")

    print(f"{len(cases)} case(s), mode={args.mode}")
    if args.mode == "agent":
        results = asyncio.run(run_agent_mode(cases))
    else:
        results = run_tool_mode(cases)
    print(report(results))
    sys.exit(0 if all(r.passed for r in results) else 1)


if __name__ == "__main__":
    main()
