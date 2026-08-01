"""client-v2 plan schema: validation against the registry and reply parsing."""

from client_v2.pipeline.plan import Plan, PlanStep, parse_plan, validate_plan
from client_v2.skills import SkillDef, SkillRegistry


def _registry():
    return SkillRegistry({
        "build_model_spec": SkillDef.model_validate({
            "id": "build_model_spec", "description": "build a region",
            "io": {"inputs": [{"name": "spec", "type": "BuildSpec",
                               "required": True}]},
        }),
        "render_orthographic_checks": SkillDef.model_validate({
            "id": "render_orthographic_checks", "description": "render",
            "io": {"inputs": [{"name": "region", "type": "string",
                               "required": True}]},
        }),
    })


def test_validate_accepts_a_well_formed_plan():
    plan = Plan(steps=[
        PlanStep(skill="build_model_spec", params={"spec": "..."}),
        PlanStep(skill="render_orthographic_checks", params={"region": "w.r"}),
    ])
    assert validate_plan(plan, _registry()) == []


def test_validate_flags_unknown_skill_and_missing_required_param():
    plan = Plan(steps=[
        PlanStep(skill="build_model_spec", params={}),        # missing 'spec'
        PlanStep(skill="ghost_skill", params={}),             # unknown
    ])
    errors = validate_plan(plan, _registry())
    assert any("missing required param" in e and "spec" in e for e in errors)
    assert any("unknown skill 'ghost_skill'" in e for e in errors)


def test_validate_flags_empty_plan():
    assert "plan has no steps" in validate_plan(Plan(steps=[]), _registry())


def test_parse_plan_reads_json_even_wrapped_in_prose():
    reply = ('Here is the plan:\n```json\n'
             '{"steps": [{"skill": "build_model_spec", '
             '"params": {"spec": "x"}}], "done_when": "done"}\n```\nthanks!')
    plan, errors = parse_plan(reply, _registry())
    assert plan is not None
    assert errors == []
    assert plan.steps[0].skill == "build_model_spec"
    assert plan.steps[0].params == {"spec": "x"}


def test_parse_plan_returns_errors_for_unusable_reply():
    plan, errors = parse_plan("no json here", _registry())
    assert plan is None
    assert errors


def test_parse_plan_parses_but_reports_validation_errors():
    reply = '{"steps": [{"skill": "nope", "params": {}}]}'
    plan, errors = parse_plan(reply, _registry())
    assert plan is not None            # parsed fine...
    assert any("unknown skill 'nope'" in e for e in errors)   # ...but invalid
