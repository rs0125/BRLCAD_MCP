"""client-v2 skill registry: loading, validation, dependency resolution,
and progressive disclosure (metadata catalog vs full detail)."""

import textwrap

import pytest

from client_v2.skills import SkillDef, SkillRegistry


def _write(dir_, name, text):
    (dir_ / name).write_text(textwrap.dedent(text))


def test_loads_the_shipped_example_skills():
    # The real client_v2/skills/ directory should load cleanly.
    reg = SkillRegistry.from_dir()
    ids = reg.ids()
    assert "build_model_spec" in ids
    assert "model_from_dimensioned_sketch" in ids
    assert reg.get("build_model_spec").kind == "skill"
    assert reg.get("model_from_dimensioned_sketch").kind == "workflow"


def _server_tool_names():
    """The MCP tools actually registered by the server."""
    import asyncio

    import brlcad_mcp.server.tools  # noqa: F401  (registers the tools)
    from brlcad_mcp.server.app import mcp
    return {t.name for t in asyncio.run(mcp.list_tools())}


def test_every_dependency_resolves_to_a_skill_or_a_real_tool():
    # A definition is only runnable once everything it names exists.  Checked
    # against the server's real tool list so a typo'd tool name fails here
    # rather than at run time.
    reg = SkillRegistry.from_dir()
    tools = _server_tool_names()
    for skill_id in reg.ids():
        assert reg.missing_dependencies(skill_id, tools) == [], skill_id


def test_missing_dependencies_reports_unauthored_ids(tmp_path):
    _write(tmp_path, "wf.yaml", """
        id: wf
        kind: workflow
        description: needs something that does not exist
        dependencies: [present, absent]
    """)
    _write(tmp_path, "present.yaml", """
        id: present
        kind: skill
        description: exists
    """)
    reg = SkillRegistry.from_dir(str(tmp_path))
    assert reg.missing_dependencies("wf") == ["absent"]


def test_catalog_is_lean_and_detail_is_full():
    reg = SkillRegistry.from_dir()
    catalog = reg.catalog()
    # Progressive disclosure: catalog is one metadata line per skill...
    assert "build_model_spec" in catalog
    assert len(catalog.splitlines()) == len(reg.ids())
    # ...detail carries the heavy content only when asked for.
    detail = reg.detail("build_model_spec")
    assert "Preconditions" in detail
    assert "Steps" in detail
    assert "units == mm" in detail
    assert reg.detail("does_not_exist") is None


def test_from_dir_on_missing_dir_is_empty(tmp_path):
    reg = SkillRegistry.from_dir(str(tmp_path / "nope"))
    assert reg.ids() == []
    assert reg.catalog() == "(no skills loaded)"


def test_malformed_definition_fails_loudly(tmp_path):
    _write(tmp_path, "bad.yaml", """
        id: bad
        # missing required 'description'
        kind: skill
    """)
    with pytest.raises(ValueError, match="invalid skill definition in bad.yaml"):
        SkillRegistry.from_dir(str(tmp_path))


def test_duplicate_ids_rejected(tmp_path):
    for fname in ("a.yaml", "b.yaml"):
        _write(tmp_path, fname, """
            id: dup
            kind: skill
            description: same id twice
        """)
    with pytest.raises(ValueError, match="duplicate skill id 'dup'"):
        SkillRegistry.from_dir(str(tmp_path))


def _skill_yaml(name, desc):
    return f"id: {name}\nkind: skill\ndescription: {desc}\n"


def test_reload_picks_up_added_and_removed_skills(tmp_path):
    (tmp_path / "a.yaml").write_text(_skill_yaml("a", "first"))
    reg = SkillRegistry.from_dir(str(tmp_path))
    assert reg.ids() == ["a"]

    # add a new def and reload -> present, reported as added
    (tmp_path / "b.yaml").write_text(_skill_yaml("b", "second"))
    status = reg.reload()
    assert reg.ids() == ["a", "b"]
    assert "added: b" in status

    # remove a def and reload -> gone, reported as removed
    (tmp_path / "a.yaml").unlink()
    status = reg.reload()
    assert reg.ids() == ["b"]
    assert "removed: a" in status


def test_reload_is_in_place_so_consumers_see_changes(tmp_path):
    # A consumer that captured the registry (like the SkillsMiddleware) must
    # see reloaded skills WITHOUT being rebuilt -- hence in-place mutation.
    (tmp_path / "a.yaml").write_text(_skill_yaml("a", "first"))
    reg = SkillRegistry.from_dir(str(tmp_path))
    captured = reg  # same object the middleware would hold
    (tmp_path / "c.yaml").write_text(_skill_yaml("c", "third"))
    reg.reload()
    assert "c" in captured.catalog()


def test_reload_keeps_current_skills_on_malformed_edit(tmp_path):
    (tmp_path / "a.yaml").write_text(_skill_yaml("a", "first"))
    reg = SkillRegistry.from_dir(str(tmp_path))
    # a broken def should NOT take down the running registry
    (tmp_path / "bad.yaml").write_text("id: bad\nkind: skill\n")  # no description
    status = reg.reload()
    assert "reload failed" in status
    assert reg.ids() == ["a"]   # unchanged


def test_reload_without_source_is_a_noop_message():
    reg = SkillRegistry({"x": SkillDef(id="x", description="d")})
    assert "no source" in reg.reload()


def test_skilldef_detail_renders_optional_and_required_io():
    s = SkillDef.model_validate({
        "id": "x", "description": "d",
        "io": {"inputs": [
            {"name": "spec", "type": "BuildSpec", "unit": "mm", "required": True},
            {"name": "constraints", "type": "text"},
        ]},
    })
    detail = s.detail()
    assert "spec: BuildSpec [mm]" in detail
    assert "constraints: text (optional)" in detail


# --- planning_view: the canonical param-binding rendering -----------------

def _rich_skill():
    return SkillDef.model_validate({
        "id": "build_model_spec", "kind": "tool-wrapper",
        "description": "build a region",
        "io": {"inputs": [{"name": "spec", "type": "BuildSpec", "unit": "mm",
                           "required": True},
                          {"name": "note", "type": "text"}],
               "outputs": [{"name": "region", "type": "string"}]},
        "preconditions": ["units == mm"],
        "cautions": ["a through-hole cutter must start OUTSIDE the material"],
        "examples": [{"input": {"spec": {"name": "washer"}}}],
        "steps": [{"call": "build_from_spec"}],
        "effects": ["creates a region"],
    })


def test_planning_view_carries_everything_that_constrains_params():
    view = _rich_skill().planning_view()
    assert "build_model_spec (tool-wrapper): build a region" in view
    assert "spec: BuildSpec [mm]" in view          # required input + unit
    assert "note: text (optional)" in view
    assert "requires: units == mm" in view         # precondition
    assert "start OUTSIDE the material" in view    # caution
    assert '"washer"' in view                      # worked example


def test_planning_view_omits_execution_only_fields():
    # Keeps the planner's context focused on binding values, not running steps.
    view = _rich_skill().planning_view()
    assert "build_from_spec" not in view           # steps
    assert "creates a region" not in view          # effects


def test_every_skill_field_is_consciously_routed_for_planning():
    # Guard: adding a field to SkillDef must be classified as planning-relevant
    # or not, so a new value-constraining field cannot silently bypass the
    # planner (the bug that hid the through-hole caution).
    from client_v2.skills import (
        PLANNING_IRRELEVANT_FIELDS,
        PLANNING_RELEVANT_FIELDS,
    )
    classified = PLANNING_RELEVANT_FIELDS | PLANNING_IRRELEVANT_FIELDS
    assert set(SkillDef.model_fields) == classified


def test_shipped_build_skill_shows_its_hole_caution_to_the_planner():
    # End-to-end on the real definition: the rule that fixed cored_block.
    view = SkillRegistry.from_dir().get("build_model_spec").planning_view()
    assert "OUTSIDE the material" in view
    assert "blind pocket" in view


def test_every_tool_named_in_steps_is_declared_and_real():
    # Latent gap: the dependency check validated what was LISTED, so a tool used
    # in `steps` but absent from `dependencies` -- or misspelt entirely -- went
    # unnoticed until the executor tried to dispatch it.
    reg = SkillRegistry.from_dir()
    known = _server_tool_names() | set(reg.ids())
    for skill_id in reg.ids():
        skill = reg.get(skill_id)
        called = {s["call"] for s in skill.steps
                  if isinstance(s, dict) and isinstance(s.get("call"), str)}
        assert called <= set(skill.dependencies), (
            f"{skill_id}: steps call {sorted(called - set(skill.dependencies))} "
            f"without declaring them")
        assert called <= known, (
            f"{skill_id}: steps call unknown {sorted(called - known)}")
