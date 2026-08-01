"""Skill / workflow definitions and their registry (DESIGN.md §5).

A *skill* is a structured, dynamically-loaded definition of one capability --
NOT prose baked into a prompt.  The schema covers I/O, preconditions,
dependencies, steps, examples, cautions, success/abort criteria, recovery
actions, and effects.  Definitions live as YAML under
``client_v2/skills/definitions/`` and are loaded by :class:`SkillRegistry` at
runtime (and reloadable, for on-the-fly debugging).

Progressive disclosure (LangChain "Skills" pattern): the registry exposes a
lean ``catalog()`` (id + kind + description) that is always cheap to show, and a
full ``detail(id)`` that is injected into the worker's prompt only when that
skill is active.  The SkillsMiddleware (next increment) consumes both.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError

SkillKind = Literal["skill", "workflow", "tool-wrapper"]

# Fields that describe HOW a skill executes rather than what its parameters must
# be, so ``planning_view`` leaves them out.  Pinned by a test: adding a field to
# SkillDef forces a conscious decision about whether the planner needs it.
PLANNING_IRRELEVANT_FIELDS = frozenset({
    "id", "version", "kind", "description",   # rendered as the header line
    "io",                                     # inputs shown; outputs are not
    "dependencies", "steps", "success_criteria", "abort_criteria",
    "recovery_actions", "effects",
})
# Fields that constrain parameter VALUES and must reach the planner.
PLANNING_RELEVANT_FIELDS = frozenset({"preconditions", "cautions", "examples"})


class IOParam(BaseModel):
    name: str
    type: str
    unit: str | None = None
    required: bool = False
    constraints: str | None = None


class SkillIO(BaseModel):
    inputs: list[IOParam] = Field(default_factory=list)
    outputs: list[IOParam] = Field(default_factory=list)


class SkillDef(BaseModel):
    """One skill or workflow definition (see module docstring)."""

    id: str
    version: int = 1
    kind: SkillKind = "skill"
    description: str
    io: SkillIO = Field(default_factory=SkillIO)
    preconditions: list[str] = Field(default_factory=list)
    dependencies: list[str] = Field(default_factory=list)
    # Steps/examples stay loosely typed for now: execution semantics land with
    # the worker that runs them.  The registry only needs to load and expose.
    steps: list[Any] = Field(default_factory=list)
    examples: list[Any] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    success_criteria: list[str] = Field(default_factory=list)
    abort_criteria: list[str] = Field(default_factory=list)
    recovery_actions: list[str] = Field(default_factory=list)
    effects: list[str] = Field(default_factory=list)

    def metadata_line(self) -> str:
        """One-line summary for the always-on catalog (progressive disclosure)."""
        return f"- {self.id} ({self.kind}): {self.description.strip()}"

    def _input_lines(self) -> list[str]:
        return [f"{p.name}: {p.type}"
                + (f" [{p.unit}]" if p.unit else "")
                + ("" if p.required else " (optional)") for p in self.io.inputs]

    def planning_view(self) -> str:
        """Everything needed to BIND this skill's parameters correctly.

        The planner fills in a skill's params on the deterministic executor
        path, so anything that constrains those values has to appear here --
        not only in the worker's :meth:`detail`.  That is a real bug we hit: a
        through-hole caution lived in the definition but never reached the
        planner, so it could not influence what got built.

        Deliberately EXCLUDES fields that describe execution rather than
        parameters (steps, dependencies, outputs, success/abort criteria,
        recovery actions, effects) -- see PLANNING_IRRELEVANT_FIELDS, which a
        test pins so a newly added field must be routed consciously instead of
        being silently dropped.
        """
        lines = [f"- {self.id} ({self.kind}): {self.description.strip()}"]
        lines.append(f"    inputs: {'; '.join(self._input_lines()) or '(none)'}")
        for item in self.preconditions:
            lines.append(f"    requires: {item.strip()}")
        for item in self.cautions:
            lines.append(f"    ! {item.strip()}")
        for example in self.examples:
            rendered = json.dumps(example, separators=(",", ":"), default=str)
            lines.append(f"    example: {rendered}")
        return "\n".join(lines)

    def detail(self) -> str:
        """Full, prompt-ready rendering, injected only when this skill is active."""
        lines = [f"SKILL {self.id} (v{self.version}, {self.kind})",
                 self.description.strip()]

        def section(title: str, items: list[str]) -> None:
            if items:
                lines.append(f"\n{title}:")
                lines.extend(f"  - {i}" for i in items)

        ins = self._input_lines()
        outs = [f"{p.name}: {p.type}" for p in self.io.outputs]
        section("Inputs", ins)
        section("Outputs", outs)
        section("Preconditions", self.preconditions)
        section("Dependencies", self.dependencies)
        if self.steps:
            lines.append("\nSteps:")
            lines.extend(f"  {i + 1}. {s}" for i, s in enumerate(self.steps))
        section("Success criteria", self.success_criteria)
        section("Abort criteria", self.abort_criteria)
        section("Recovery actions", self.recovery_actions)
        section("Cautions", self.cautions)
        section("Effects", self.effects)
        return "\n".join(lines)


# The definitions live beside this module, in skills/definitions/.
DEFAULT_SKILLS_DIR = os.path.join(os.path.dirname(__file__), "definitions")


class SkillRegistry:
    """An in-memory, id-keyed collection of skill definitions."""

    def __init__(self, skills: dict[str, SkillDef], source: str | None = None):
        self._skills = skills
        self._source = source

    # --- loading -----------------------------------------------------------

    @staticmethod
    def _load_dir(path: str) -> dict[str, SkillDef]:
        """Parse every ``*.yaml`` / ``*.yml`` under *path* into a dict by id.

        Raises ``ValueError`` with the file name on a malformed definition, so a
        bad edit fails loudly rather than silently dropping a skill.
        """
        skills: dict[str, SkillDef] = {}
        if not os.path.isdir(path):
            return skills
        for name in sorted(os.listdir(path)):
            if not name.endswith((".yaml", ".yml")):
                continue
            with open(os.path.join(path, name)) as fh:
                data = yaml.safe_load(fh) or {}
            try:
                skill = SkillDef.model_validate(data)
            except ValidationError as exc:
                raise ValueError(f"invalid skill definition in {name}: {exc}") \
                    from exc
            if skill.id in skills:
                raise ValueError(f"duplicate skill id '{skill.id}' ({name})")
            skills[skill.id] = skill
        return skills

    @classmethod
    def from_dir(cls, path: str = DEFAULT_SKILLS_DIR) -> SkillRegistry:
        """Load a registry from *path*, remembering it for :meth:`reload`."""
        return cls(cls._load_dir(path), source=path)

    def reload(self) -> str:
        """Re-read the source directory IN PLACE (runtime hot-reload).

        Mutates this registry's contents so anything holding the registry (e.g.
        the SkillsMiddleware) sees the change without a graph rebuild.  A
        malformed edit is reported and the current skills are kept, so a bad
        save can't take the running agent down.  Returns a status line.
        """
        if not self._source:
            return "reload: no source directory to reload from"
        try:
            fresh = self._load_dir(self._source)
        except ValueError as exc:
            return f"reload failed (keeping current skills): {exc}"
        before, after = set(self._skills), set(fresh)
        self._skills = fresh
        parts = [f"reloaded {len(fresh)} skill(s)"]
        if after - before:
            parts.append(f"added: {', '.join(sorted(after - before))}")
        if before - after:
            parts.append(f"removed: {', '.join(sorted(before - after))}")
        return "; ".join(parts)

    # --- access ------------------------------------------------------------

    def ids(self) -> list[str]:
        return sorted(self._skills)

    def get(self, skill_id: str) -> SkillDef | None:
        return self._skills.get(skill_id)

    def by_kind(self, kind: SkillKind) -> list[SkillDef]:
        return [s for s in self._skills.values() if s.kind == kind]

    def missing_dependencies(self, skill_id: str,
                             known_tools: Iterable[str] = ()) -> list[str]:
        """Dependency ids that resolve to neither a skill nor a known tool.

        A skill legitimately depends on MCP TOOLS as well as other skills (that
        is how a tool-wrapper is expressed), so a caller that can enumerate the
        server's tools should pass them -- otherwise every tool dependency looks
        unresolvable.
        """
        skill = self.get(skill_id)
        if skill is None:
            return []
        resolved = set(self._skills) | set(known_tools)
        return [d for d in skill.dependencies if d not in resolved]

    # --- progressive disclosure -------------------------------------------

    def catalog(self) -> str:
        """Lean, always-on listing of available skills (metadata only)."""
        if not self._skills:
            return "(no skills loaded)"
        return "\n".join(self._skills[i].metadata_line() for i in self.ids())

    def detail(self, skill_id: str) -> str | None:
        """Full rendering of one skill, for on-demand prompt injection."""
        skill = self.get(skill_id)
        return skill.detail() if skill else None
