"""Deterministic execution machinery -- no model calls in here.

* ``plan``     -- the ``Plan``/``PlanStep`` schema, validated against the registry.
* ``executor`` -- runs a plan's skills in order, binding ``${step.output}``
  references, for plans whose steps are all directly callable.

Kept separate from :mod:`client_v2.agents` precisely because nothing here asks a
model anything: the same plan always executes the same way.
"""
