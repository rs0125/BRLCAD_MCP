"""Skill and workflow definitions, and the machinery that serves them.

* ``registry`` -- the ``SkillDef`` schema and the ``SkillRegistry`` that loads
  YAML definitions at runtime (and hot-reloads them).
* ``middleware`` -- the LangChain middleware that injects the catalog, and an
  active skill's full detail, into the worker's prompt.
* ``definitions/`` -- the definitions themselves, one YAML file per skill.

Re-exported here so callers can say ``from client_v2.skills import
SkillRegistry`` without caring which submodule it lives in.
"""

from client_v2.skills.registry import (
    DEFAULT_SKILLS_DIR,
    PLANNING_IRRELEVANT_FIELDS,
    PLANNING_RELEVANT_FIELDS,
    IOParam,
    SkillDef,
    SkillIO,
    SkillKind,
    SkillRegistry,
)

__all__ = [
    "DEFAULT_SKILLS_DIR",
    "PLANNING_IRRELEVANT_FIELDS",
    "PLANNING_RELEVANT_FIELDS",
    "IOParam",
    "SkillDef",
    "SkillIO",
    "SkillKind",
    "SkillRegistry",
]
