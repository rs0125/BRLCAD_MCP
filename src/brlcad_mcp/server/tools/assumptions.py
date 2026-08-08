"""Declared assumptions: the decisions a build rests on, as data.

A model built from a drawing is only as trustworthy as the readings behind it,
and those readings are usually invisible.  When a drawing is ambiguous -- or
self-contradictory, which real drawings are more often than you would like -- the
agent has to choose, and today that choice lives in a sentence in a chat reply
that scrolls away.  The geometry then looks authoritative while the judgement
that produced it is gone.

So a choice is recorded as a ROW, next to the saved spec it justifies:

    declare_assumption(topic="cavity depth", chose="6.3 mm",
                       over="1.0 mm roof callout",
                       reason="cannot both hold on a 9.6 mm body")

Two things this buys that a sentence does not.

**It can be read back.**  ``promote_draft`` is supposed to report the assumptions
a model was built on; it can now do that from the record rather than from the
agent's memory of what it said several turns ago.

**It can be checked.**  Scoring a free-text declaration meant substring-matching
a transcript, and that produced a FALSE PASS on the first contradictory case we
ran: the two conflicting values both appeared -- in unrelated prose, three
replies apart -- while the agent had in fact resolved the contradiction silently.
``chose`` and ``over`` are exact strings in known fields, so there is nothing to
guess at.

Stored as JSONL beside the specs (see ``_declarations_path``) because both are
the build's RECORD rather than a cache: the render folder can be deleted without
losing anything, this cannot.
"""

from __future__ import annotations

import json
import os
import time

from pydantic import Field

from brlcad_mcp.server.app import mcp
from brlcad_mcp.server.tools.reconstruct import _specs_root

DECLARATIONS_FILE = "assumptions.jsonl"


def _declarations_path() -> str:
    return os.path.join(_specs_root(), DECLARATIONS_FILE)


def read_declarations(region: str = "") -> list[dict]:
    """Every declaration made, oldest first; optionally filtered by *region*.

    Returns [] when nothing has been declared -- an absent file is the normal
    state for a build that raised no questions, not an error.  A malformed line
    is skipped rather than raising: a corrupt record must not be able to break
    a build report or a scoring pass.
    """
    path = _declarations_path()
    if not os.path.isfile(path):
        return []
    out: list[dict] = []
    with open(path) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if not region or row.get("region") in ("", region):
                out.append(row)
    return out


def format_declarations(rows: list[dict]) -> str:
    """The declarations as the lines a person should read in a final report."""
    if not rows:
        return "No assumptions were declared."
    lines = []
    for r in rows:
        line = f"- {r.get('topic', '?')}: {r.get('chose', '?')}"
        if r.get("over"):
            line += f" (over {r['over']})"
        if r.get("reason"):
            line += f" -- {r['reason']}"
        lines.append(line)
    return "\n".join(lines)


@mcp.tool()
def declare_assumption(
    topic: str = Field(
        ...,
        description=("What the decision was ABOUT, in a few words: 'cavity "
                     "depth', 'stud height', 'overall length'."),
    ),
    chose: str = Field(
        ...,
        description=("The reading you went with, as it appears on the drawing "
                     "-- e.g. '6.3 mm'. Give the VALUE, not a description."),
    ),
    over: str = Field(
        default="",
        description=("The reading you rejected, if this was a conflict between "
                     "two printed values -- e.g. '1.0 mm roof callout'. Leave "
                     "empty when the drawing was merely silent rather than "
                     "self-contradictory."),
    ),
    reason: str = Field(
        default="",
        description="Why, in one line. Say what made the alternative untenable.",
    ),
    region: str = Field(
        default="",
        description=("The build name this applies to, if known. Optional: a "
                     "reading is often settled before the region exists."),
    ),
) -> str:
    """Record a decision the model rests on, so it survives the conversation.

    Declare one whenever the reference does not determine an answer and you pick
    one: a dimension that is missing, ambiguous, or contradicted elsewhere on the
    drawing. Prefer declaring too many over too few -- an undeclared assumption
    is indistinguishable from a misread, both to a reviewer and to the record.
    """
    row = {"t": round(time.time(), 3), "topic": topic, "chose": chose,
           "over": over, "reason": reason, "region": region}
    path = _declarations_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row) + "\n")
    conflict = f" over '{over}'" if over else ""
    return (f"Recorded: {topic} = '{chose}'{conflict}. "
            f"{len(read_declarations())} assumption(s) declared for this model.")
