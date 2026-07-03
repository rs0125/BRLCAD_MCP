"""MCP tool — model health report.

Runs BRL-CAD's own validators over a model and returns ONE readable summary:
structural problems (via `lint`) + geometric interferences (via `gqa`).
A human normally runs these one at a time and eyeballs raw output; this
composes them into a single triaged report the agent can present or act on.

This is the agent/MCP-side, headless counterpart to the interactive Arbalest
V&V GUI — same underlying BRL-CAD checks, different (conversational) surface.
"""

from __future__ import annotations

import re

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.transport import send_command


def _lines(out: str) -> list[str]:
    """Body lines of a listener reply (drop the SUCCESS:/ERROR: status line)."""
    for pre in ("SUCCESS:", "ERROR:"):
        if out.startswith(pre):
            out = out[len(pre):]
    return [ln.rstrip() for ln in out.splitlines() if ln.strip()]


def _lint_findings(obj: str, flag: str) -> list[str]:
    """Run a single lint check; return the finding lines under its header.

    lint reports real findings under a 'Found ...:' header as tab-indented
    lines.  Anything else (e.g. the "Object argument(s) ... do not exist"
    error for a bad name) is NOT a finding — return [] so we never fabricate
    issues out of an error message.
    """
    out = send_command(f"lint {flag} {obj}")
    body = _lines(out)
    if not body or not any(ln.lower().lstrip().startswith("found") for ln in body):
        return []  # no 'Found ...:' header => no findings (or an error msg)
    # collect the tab-indented lines that follow the header
    return [ln.strip() for ln in body if ln.startswith(("\t", "    "))]


def _overlaps(obj: str, grid: float) -> list[tuple[str, str, float]]:
    """gqa overlap pairs (fixed grid — bare gqa hangs on coincident faces)."""
    out = send_command(f"gqa -g {grid} -Ao {obj}")
    pairs = []
    for m in re.finditer(r"(\S+)\s+(\S+)\s+count:\d+\s+dist:(\S+)mm", out):
        pairs.append((m.group(1), m.group(2), float(m.group(3))))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


@mcp.tool()
def model_health_report(
    obj: str = Field(
        ...,
        description="Top object/assembly to audit (e.g. 'havoc', 'all', a "
        "subassembly). Structural checks scan its whole tree.",
    ),
    grid: float = Field(
        default=8.0,
        description="gqa overlap grid spacing in mm — smaller finds finer "
        "overlaps but is much slower; 8 is a fast first pass, drop to 1-2 for "
        "a thorough check.",
    ),
    min_depth: float = Field(
        default=0.1,
        description="Ignore overlaps shallower than this (mm) — filters "
        "coincident-surface noise from genuine interferences.",
    ),
    max_examples: int = Field(
        default=8,
        description="Max example findings to list per category.",
    ),
) -> str:
    """Audit *obj* with BRL-CAD's validators and return a single health report.

    Covers structural issues (cyclic references, missing/dangling objects,
    invalid shapes) via `lint` and geometric interferences via `gqa`.  Reports
    a per-category count with a few examples each, so the model's overall
    health is visible at a glance and the worst issues are surfaced first.
    """
    # tolerate the display annotations that `tops`/`ls` append to names —
    # a trailing '/R' (region) or '/' (comb) is not part of the object name
    obj = obj.strip()
    if obj.endswith("/R"):
        obj = obj[:-2]
    obj = obj.rstrip("/")

    # validate the object exists first — otherwise lint's "does not exist"
    # error text would be misread as findings
    exists = _lines(send_command(f"exists {obj}"))
    if not exists or exists[0].strip() != "1":
        return (f"'{obj}' is not in the database — check the name (try 'tops' "
                f"for the top-level objects, or 'ls' to list everything).")

    cyclic = _lint_findings(obj, "-C")
    missing = _lint_findings(obj, "-M")
    invalid = _lint_findings(obj, "-I")
    overlaps = [p for p in _overlaps(obj, grid) if p[2] >= min_depth]

    total = len(cyclic) + len(missing) + len(invalid) + len(overlaps)

    def mark(n):
        return "[OK]" if n == 0 else f"[!!] {n}"

    def examples(items, fmt, cap):
        out = [f"        - {fmt(x)}" for x in items[:cap]]
        if len(items) > cap:
            out.append(f"        ... and {len(items) - cap} more")
        return out

    L = [
        "=" * 60,
        f" MODEL HEALTH REPORT  —  '{obj}'",
        "=" * 60,
        (" RESULT: no issues found" if total == 0
         else f" RESULT: {total} issue(s) found (details below)"),
        "",
        " STRUCTURAL CHECKS (lint)",
        " " + "-" * 58,
    ]

    # cyclic references — show leaf name and the offending path
    L.append(f"   {mark(len(cyclic))}  cyclic references "
             "(a combination references one of its own ancestors)")
    L += examples(cyclic, lambda s: f"{s.split('/')[-1]:<18} in  {s}",
                  max_examples)

    # missing references
    L.append(f"   {mark(len(missing))}  missing / dangling references "
             "(referenced object not in database)")
    L += examples(missing, lambda s: s, max_examples)

    # invalid shapes — GROUP BY the failure reason in [brackets]
    L.append(f"   {mark(len(invalid))}  invalid shapes "
             "(geometry that fails validity checks)")
    if invalid:
        groups: dict[str, list[str]] = {}
        for f in invalid:
            m = re.search(r"\[([^\]]+)\]", f)
            reason = m.group(1) if m else "unspecified"
            name = f.split("[")[0].strip()
            groups.setdefault(reason, []).append(name)
        for reason in sorted(groups, key=lambda r: -len(groups[r])):
            names = groups[reason]
            L.append(f"        {reason}: {len(names)}")
            per = max(2, max_examples // max(1, len(groups)))
            for nm in names[:per]:
                L.append(f"            - {nm}")
            if len(names) > per:
                L.append(f"            ... and {len(names) - per} more")

    L += [
        "",
        f" GEOMETRIC CHECKS (gqa, grid={grid:g}mm, depth>={min_depth:g}mm)",
        " " + "-" * 58,
        f"   {mark(len(overlaps))}  region overlaps "
        "(two regions occupying the same space)",
    ]
    L += examples(
        overlaps,
        lambda p: f"{p[0].split('/')[-1]} <-> {p[1].split('/')[-1]}  "
                  f"({p[2]:.1f} mm deep)",
        max_examples)
    if overlaps:
        L.append("        (fixable with the separate_overlap / resolve_overlaps tools)")

    L += [
        "",
        " " + "-" * 58,
        f" note: overlap detection is ray-sampled at {grid:g}mm and is an",
        " estimate, not exhaustive — re-run with a finer grid for a more",
        " thorough (slower) geometric check.",
        "=" * 60,
    ]
    return "\n".join(L)
