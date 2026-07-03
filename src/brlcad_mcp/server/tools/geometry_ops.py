"""MCP tool — non-destructive overlap resolution by minimal sliding.

Resolving an interference by *subtracting* one part from another is
destructive: it carves geometry out of a part that should stay whole.
Real CAD fixes a positioning error by *moving* the misplaced part.

Naive approaches fail on real models:
  * a single minimal-offset move can push a part straight into a neighbour;
  * separating *bounding boxes* over-moves badly when a small part sits near
    a large or hollow one (to clear the boxes you fling the part far away).

This tool instead treats gqa as a boolean oracle ("do the actual solids
still intersect?") and BINARY-SEARCHES for the smallest slide, along a
caller-chosen direction, that makes the interference vanish.  The
axis-aligned bounding boxes are used only for a guaranteed-clear upper
bound to bound the search; the answer comes from real geometry, so it does
not over-move and it will not silently trade one overlap for another
(the oracle checks the whole scope).
"""

from __future__ import annotations

import re

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.transport import send_command

_DIRS = {
    "+X": (0, 1.0), "-X": (0, -1.0),
    "+Y": (1, 1.0), "-Y": (1, -1.0),
    "+Z": (2, 1.0), "-Z": (2, -1.0),
}


def _rpp(obj: str):
    out = send_command(f"bb -e {obj}")
    mn = re.search(r"min \{(\S+) (\S+) (\S+)\}", out)
    mx = re.search(r"max \{(\S+) (\S+) (\S+)\}", out)
    if not (mn and mx):
        return None
    return [float(x) for x in mn.groups()], [float(x) for x in mx.groups()]


def _involves(path: str, mover: str) -> bool:
    """True if a gqa path is the mover or a region beneath it.

    gqa reports leaf-region paths like /asm/gun/barrel/r.b; an assembly
    passed as *mover* shows up via its CHILD regions, so match the mover as
    a path component, not just the basename (this was the false-'clean' bug).
    """
    segs = path.strip("/").split("/")
    return mover in segs


def _parents(obj: str) -> list[str]:
    """Combs that reference *obj* (its immediate parents), via dbfind."""
    out = send_command(f"dbfind {obj}")
    # strip the transport status prefix ('SUCCESS:' / 'ERROR:'); the remainder
    # is whitespace-separated parent names (empty if the object is top-level)
    for pre in ("SUCCESS:", "ERROR:"):
        if out.startswith(pre):
            out = out[len(pre):]
    return [n for n in out.split() if n and n != obj]


def _is_bare_solid(obj: str) -> bool:
    """True if *obj* is a raw primitive (not a region/combination).

    Moving a bare solid relocates it inside its region — almost never what
    you want for interference (you want to move the region/assembly).
    A comb/region's `l` lists member ops (u/-/+) or says REGION; a solid
    prints primitive parameters instead.
    """
    out = send_command(f"l {obj}")
    if "REGION" in out or "COMBINATION" in out:
        return False
    for ln in out.splitlines()[1:]:
        if re.match(r"\s*[u+-]\s+\S", ln):  # a boolean member line
            return False
    return True


def _volume(obj: str) -> float:
    """Bounding-box volume of *obj* (mm^3), a cheap size proxy. 0 if unknown."""
    bb = _rpp(obj)
    if not bb:
        return 0.0
    return abs((bb[1][0] - bb[0][0]) * (bb[1][1] - bb[0][1]) * (bb[1][2] - bb[0][2]))


def _overlap_pairs(scope: str, grid: float, min_depth: float):
    """Deduped overlapping part pairs in *scope* (by leaf basename).

    Returns list of (partA, partB, depth) with depth >= min_depth, deepest
    first, excluding self-overlaps.  Uses a fixed grid (never bare gqa).
    """
    out = send_command(f"gqa -g {grid} -Ao {scope}")
    seen, pairs = set(), []
    for m in re.finditer(r"(\S+)\s+(\S+)\s+count:\d+\s+dist:(\S+)mm", out):
        a, b = m.group(1).split("/")[-1], m.group(2).split("/")[-1]
        depth = float(m.group(3))
        if a == b or depth < min_depth:
            continue
        key = frozenset((a, b))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((a, b, depth))
    pairs.sort(key=lambda p: p[2], reverse=True)
    return pairs


def _overlapping(mover: str, scope: str, grid: float) -> bool:
    """Does *mover* still overlap some OTHER part in scope (not itself)?

    Only CROSS-part overlaps count: one side under *mover*, the other not.
    A complex part often has internal self-overlaps between its own
    sub-regions (a barrel inside its housing) that are by-design and never
    resolve by sliding — counting them would make the part look eternally
    stuck.  We resolve interference with *other* parts, so require exactly
    one side of the pair to belong to the mover.

    A FIXED grid (`-g`) is mandatory: bare gqa auto-refines until its volume
    estimate converges, which on coincident faces never happens (it refines
    to billions of cells and hangs).
    """
    out = send_command(f"gqa -g {grid} -Ao {scope}")
    for m in re.finditer(r"(\S+)\s+(\S+)\s+count:\d+\s+dist:", out):
        if _involves(m.group(1), mover) != _involves(m.group(2), mover):
            return True  # exactly one side is the mover -> external interference
    return False


def _min_clear(mover, scope, axis, sign, mv, others, clearance, precision, grid, budget):
    """Minimal offset along (axis,sign) that clears mover, WITHOUT committing.

    Binary-searches the clearing distance, then moves the part back to its
    start.  Returns (distance_or_None, probes_used, reason).  distance is the
    move to apply (incl. clearance); None means blocked or implausibly far.
    """
    # AABB upper bound: definitely-clear distance along this axis.
    hi = 0.0
    for ob in others:
        need = (ob[1][axis] - mv[0][axis]) if sign > 0 else (mv[1][axis] - ob[0][axis])
        hi = max(hi, need)
    hi += clearance + precision
    if hi <= 0:
        return None, 0, "no clearance"

    at = [0.0]
    probes = [0]

    def move_to(off):
        delta = off - at[0]
        if abs(delta) > 1e-9:
            d = [0.0, 0.0, 0.0]
            d[axis] = sign * delta
            send_command(f"otranslate {mover} {d[0]} {d[1]} {d[2]}")
            at[0] = off

    def clear_at(off):
        move_to(off)
        probes[0] += 1
        return not _overlapping(mover, scope, grid)

    if probes[0] >= budget or not clear_at(hi):
        move_to(0.0)
        return None, probes[0], "blocked"

    lo = 0.0
    while hi - lo > precision and probes[0] < budget:
        mid = (lo + hi) / 2.0
        if clear_at(mid):
            hi = mid
        else:
            lo = mid
    move_to(0.0)  # revert; caller applies the winning move
    return hi + clearance, probes[0], "ok"


@mcp.tool()
def separate_overlap(
    mover: str = Field(..., description="Region/assembly to slide out of the way."),
    scope: str = Field(
        ...,
        description="Objects to check overlaps within (space-separated, e.g. "
        "'gun fuselage'). Keep it tight — just the mover and the part(s) it "
        "interferes with — so the gqa oracle stays fast.",
    ),
    direction: str = Field(
        default="auto",
        description="Slide direction: +X -X +Y -Y +Z -Z, or 'auto' (default) "
        "to let the tool pick the axis needing the smallest move. 'auto' costs "
        "more gqa checks, so name a direction on large models if you know it.",
    ),
    clearance: float = Field(
        default=2.0, description="Extra mm gap to leave beyond just-clear."
    ),
    precision: float = Field(
        default=5.0, description="Binary-search resolution in mm."
    ),
    grid: float = Field(
        default=2.0, description="gqa sampling grid (mm); finer = more exact, slower."
    ),
    max_probes: int = Field(
        default=40, description="Cap on total gqa oracle calls (safety)."
    ),
) -> str:
    """Non-destructively resolve interference by sliding *mover* the MINIMAL
    distance that makes gqa report no overlap in *scope*.

    With ``direction='auto'`` the tool ranks the six axes by bounding-box
    clearance and searches the most promising first, returning the smallest
    real move that clears without over-travelling.  Reports the direction and
    distance, or explains why no reasonable move works.
    """
    send_command("units mm")  # bb and otranslate must agree on units

    # Level guard: you must move the meaningful PART, not a bare solid buried
    # inside a region.  gqa reports overlaps at leaf granularity; moving a raw
    # solid (or the wrong leaf) relocates just that piece and tears its parent
    # apart.  Refuse a bare solid and point at the region that owns it.
    if _is_bare_solid(mover):
        owners = _parents(mover)
        hint = f" It is part of {owners}; move that instead." if owners else ""
        return (f"Refusing to move '{mover}': it is a raw solid primitive, not "
                f"a part/assembly.{hint} Moving a bare solid relocates it inside "
                f"its region and can tear the assembly.")

    if not _overlapping(mover, scope, grid):
        return f"'{mover}' has no overlaps in '{scope}' — nothing to do."

    mv = _rpp(mover)
    if mv is None:
        return f"Error: could not read bounding box of '{mover}'."
    others = [b for b in (_rpp(t) for t in scope.split() if t != mover) if b]
    max_extent = max(mv[1][k] - mv[0][k] for k in range(3))
    sane = 3.0 * max_extent  # a real fix backs out ~part-size, not across the model

    # Which directions to try, and in what order.
    if direction == "auto":
        # rank all six axes by AABB clearance (cheap, no gqa) — smallest first
        ranked = []
        for name, (ax, sg) in _DIRS.items():
            c = 0.0
            for ob in others:
                need = (ob[1][ax] - mv[0][ax]) if sg > 0 else (mv[1][ax] - ob[0][ax])
                c = max(c, need)
            ranked.append((c, name))
        candidates = [n for c, n in sorted(ranked) if c > 0]
    elif direction in _DIRS:
        candidates = [direction]
    else:
        return f"Error: direction must be one of {sorted(_DIRS)} or 'auto'."

    budget = [max_probes]
    tried = []
    for name in candidates:
        ax, sg = _DIRS[name]
        dist, used, reason = _min_clear(
            mover, scope, ax, sg, mv, others, clearance, precision, grid, budget[0]
        )
        budget[0] -= used
        if dist is not None and dist <= sane:
            # commit the winning move
            d = [0.0, 0.0, 0.0]
            d[ax] = sg * dist
            send_command(f"otranslate {mover} {d[0]} {d[1]} {d[2]}")
            if _overlapping(mover, scope, grid):
                return (f"Slid '{mover}' {dist:.0f} mm along {name} but gqa "
                        f"still finds an overlap (try finer grid/precision).")
            extra = "" if len(candidates) == 1 else f" (auto-picked from {len(candidates)} axes)"
            owners = _parents(mover)
            note = (f" Note: '{mover}' belongs to {owners} — the whole subtree "
                    f"under '{mover}' moved together; if you meant to move a "
                    f"larger assembly, name the parent instead.") if owners else ""
            return (f"Resolved: slid '{mover}' {dist:.1f} mm along {name} — the "
                    f"minimal move that clears the interference{extra}; gqa "
                    f"confirms 0 overlaps in '{scope}'.{note}")
        tried.append(f"{name}:{reason if dist is None else f'{dist:.0f}mm too far'}")
        if budget[0] <= 0:
            break

    return (f"Could not resolve '{mover}' with a reasonable move. Tried "
            f"[{', '.join(tried)}]. Every direction is either blocked or would "
            f"require sliding it across the model — the interference may need a "
            f"different fix (resize, or move a neighbouring part instead).")


@mcp.tool()
def resolve_overlaps(
    scope: str = Field(
        ...,
        description="Assembly or space-separated objects to scan for overlaps "
        "(e.g. a subassembly). Keep it tight so the gqa scan stays fast.",
    ),
    apply: bool = Field(
        default=False,
        description="False (default) = DRY RUN: report the overlaps and the "
        "planned fixes without changing anything. True = actually resolve them.",
    ),
    min_depth: float = Field(
        default=1.0,
        description="Ignore overlaps shallower than this (mm) — filters out "
        "coincident-surface numerical noise; only genuine interferences.",
    ),
    direction: str = Field(
        default="auto",
        description="Slide direction for each fix (+X..-Z or 'auto').",
    ),
    fixed: str = Field(
        default="",
        description="Space-separated parts to ANCHOR — never moved (e.g. the "
        "chassis/frame/structural body). Their overlapping partner is moved "
        "instead, overriding the smaller-part rule.",
    ),
    clearance: float = Field(default=2.0, description="Gap to leave (mm).")
    ,
    precision: float = Field(default=5.0, description="Binary-search resolution (mm).")
    ,
    grid: float = Field(default=2.0, description="gqa sampling grid (mm).")
    ,
    max_probes: int = Field(default=40, description="gqa probe cap per fix.")
    ,
    max_rounds: int = Field(
        default=5,
        description="Max resolve-then-rescan rounds (apply mode). Fixing one "
        "overlap can shift a part into a new one; the tool re-scans and repeats "
        "until the scope is clean or no progress is made.",
    ),
) -> str:
    """Find and (optionally) resolve ALL interferences in *scope* at once.

    For each overlapping pair it MOVES THE SMALLER part out of the larger
    (small parts are usually the movable/attached ones; large ones are
    structural).  DRY RUN by default: it lists every real overlap and the
    planned smaller-part move so the user can approve before anything changes.

    In apply mode it works ITERATIVELY: resolve the current overlaps, then
    re-scan — because moving a part can shift it into a new interference
    (cascade).  It repeats until the scope is clean, no progress is made, or
    max_rounds is hit, so nested / cascading overlaps converge rather than
    leaving fresh overlaps behind.

    NOTE: it operates on the part names gqa reports.  When those are deep
    leaf regions of a larger assembly, moving them can tear the assembly —
    prefer scoping to, or naming, the meaningful parts.
    """
    send_command("units mm")
    anchored = set(fixed.split())
    # If a fixed axis is given, resolve parts in top-down order along it: the
    # part furthest toward the exit is fixed first, so sliding a nearer part
    # can't ram it into an as-yet-unresolved part ahead of it (avoids the
    # greedy overshoot where a lifted part gets shoved across the model).
    order_axis = _DIRS[direction][0] if direction in _DIRS else None
    order_sign = _DIRS[direction][1] if direction in _DIRS else 1.0

    def _mover_coord(mv):
        bb = _rpp(mv)
        if not bb or order_axis is None:
            return 0.0
        # leading edge along the exit direction
        return order_sign * (bb[1][order_axis] if order_sign > 0 else -bb[0][order_axis])

    def plan_for(pairs):
        out = []
        for a, b, depth in pairs:
            # anchoring overrides smaller-part choice
            a_fix, b_fix = a in anchored, b in anchored
            if a_fix and b_fix:
                continue  # both anchored — can't move either; skip
            if a_fix:
                mover, other = b, a
            elif b_fix:
                mover, other = a, b
            else:
                va, vb = _volume(a), _volume(b)
                mover, other = (a, b) if va <= vb else (b, a)
            out.append((mover, other, depth, _volume(mover), _volume(other)))
        # resolve the part furthest along the exit axis first (top-down)
        if order_axis is not None:
            out.sort(key=lambda p: _mover_coord(p[0]), reverse=True)
        return out

    pairs = _overlap_pairs(scope, grid, min_depth)
    if not pairs:
        return f"No interferences >= {min_depth} mm found in '{scope}'."

    if not apply:
        plan = plan_for(pairs)
        lines = [f"{len(plan)} interference(s) >= {min_depth} mm in '{scope}':"]
        for mv, ot, d, sm, lg in plan:
            lines.append(f"  - {mv} ({sm:.0f} mm^3) overlaps {ot} ({lg:.0f} mm^3), "
                         f"{d:.1f} mm deep -> move {mv} (smaller)")
        return ("DRY RUN — nothing changed.\n" + "\n".join(lines) +
                "\n\nCall again with apply=True to move each smaller part out, "
                "or name a specific part/direction to override.")

    # Apply: resolve, re-scan, repeat until clean / no progress / capped.
    # LIMIT (documented): this is greedy per-pair, not a global optimizer. A
    # locally-minimal move can produce a globally-poor result on cascades
    # (a part boxed between two frozen parts on its only free axis must travel
    # over the far one). A real fix is simultaneous constraint solving
    # (physics-engine scale) — out of scope. See mcp_overlap_tests/CAPABILITIES.md.
    _resolve = getattr(separate_overlap, "fn", separate_overlap)
    rounds = []
    prev_key = None
    for rnd in range(1, max_rounds + 1):
        key = frozenset(frozenset((a, b)) for a, b, _ in pairs)
        if key == prev_key:
            rounds.append(f"round {rnd}: same {len(pairs)} overlap(s) as last "
                          f"round — no progress, stopping.")
            break
        prev_key = key
        applied = []
        for mv, ot, *_ in plan_for(pairs):
            r = _resolve(mover=mv, scope=f"{mv} {ot}", direction=direction,
                         clearance=clearance, precision=precision, grid=grid,
                         max_probes=max_probes)
            applied.append(f"      {mv}: {r.split(chr(10))[0]}")
        rounds.append(f"round {rnd}: resolved {len(applied)} pair(s)\n" +
                      "\n".join(applied))
        pairs = _overlap_pairs(scope, grid, min_depth)  # re-scan for cascades
        if not pairs:
            rounds.append(f"round {rnd}: scope clean — converged.")
            break

    status = ("CLEAN — no interferences remain." if not pairs else
              f"STILL {len(pairs)} overlap(s) after {len(rounds)} round(s): " +
              ", ".join(f"{a}/{b}" for a, b, _ in pairs[:5]) +
              " (may need a different fix or a named direction).")
    return status + "\n\n" + "\n".join(rounds)
