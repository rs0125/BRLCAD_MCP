"""client-v2 SkillsMiddleware prompt composition (progressive disclosure)."""

from client_v2.skills import SkillDef, SkillRegistry
from client_v2.skills.middleware import compose_worker_prompt


def _registry():
    return SkillRegistry({
        "build_model_spec": SkillDef(
            id="build_model_spec", description="build a region",
            preconditions=["units == mm"], steps=["call build_from_spec"]),
        "render_checks": SkillDef(
            id="render_checks", description="render check views"),
    })


def test_prompt_always_carries_base_and_catalog():
    prompt = compose_worker_prompt("BASE PROMPT", _registry())
    assert "BASE PROMPT" in prompt
    assert "## Available skills" in prompt
    # catalog metadata (ids + descriptions), both skills present
    assert "build_model_spec" in prompt
    assert "render check views" in prompt
    # no active skill -> detail is NOT injected (kept lean)
    assert "## Active skill" not in prompt
    assert "units == mm" not in prompt


def test_active_skill_injects_full_detail():
    prompt = compose_worker_prompt("BASE", _registry(),
                                   active_skill="build_model_spec")
    assert "## Active skill" in prompt
    assert "Preconditions" in prompt
    assert "units == mm" in prompt


def test_unknown_active_skill_is_ignored_gracefully():
    prompt = compose_worker_prompt("BASE", _registry(), active_skill="ghost")
    assert "## Active skill" not in prompt
    assert "## Available skills" in prompt


# --- the plan reaches the worker -------------------------------------------

_PLAN = {
    "steps": [
        {"skill": "build_model_spec", "params": {"spec": "<spec>"},
         "why": "build the region"},
        {"skill": "render_checks", "params": {}, "why": "then look at it"},
    ],
    "done_when": "the region verifies",
}


def test_plan_and_every_planned_skill_reach_the_worker():
    # THE BUG: the plan was dropped here and only ONE skill's detail was
    # injected -- the plan's terminal step -- so the worker never saw the
    # earlier steps or the parameters the planner had bound.
    prompt = compose_worker_prompt("BASE", _registry(), plan=_PLAN)
    assert "## Plan to follow" in prompt
    assert "1. build_model_spec" in prompt and "2. render_checks" in prompt
    assert "<spec>" in prompt                      # the bound parameters
    assert "build the region" in prompt            # the planner's rationale
    assert "the region verifies" in prompt         # done_when
    # ...and the DEFINITION of every skill named, not just the last one.
    assert "units == mm" in prompt                 # from build_model_spec
    assert prompt.count("SKILL ") == 2


def test_a_plan_supersedes_a_single_active_skill():
    prompt = compose_worker_prompt("BASE", _registry(),
                                   active_skill="render_checks", plan=_PLAN)
    assert "## Plan to follow" in prompt
    assert "## Active skill" not in prompt


def test_active_skill_is_still_the_fallback_without_a_plan():
    for empty in (None, {}, {"steps": []}):
        prompt = compose_worker_prompt("BASE", _registry(),
                                       active_skill="build_model_spec",
                                       plan=empty)
        assert "## Active skill" in prompt
        assert "## Plan to follow" not in prompt


def test_plan_rendering_survives_missing_fields():
    from client_v2.skills.middleware import plan_skill_ids, render_plan
    ragged = {"steps": [{"skill": "a"}, {}, {"skill": "a", "params": {"x": 1}}]}
    assert plan_skill_ids(ragged) == ["a"]          # deduped, order preserved
    assert "1. a" in render_plan(ragged)
    assert render_plan(None) == ""
