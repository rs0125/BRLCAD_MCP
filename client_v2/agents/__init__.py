"""The thin agents -- one module per role, each with a single job.

The old monolithic prompt, decomposed by role:

* ``conversational`` -- intake: is this a work request, or just conversation?
* ``planner``        -- turns a request into an ordered, parameterised plan.
* ``worker``         -- the ONLY agent with tool access; executes the work.
* ``verifier``       -- reads engine-truth results and decides pass or kick-back.
* ``visual``         -- compares renders with a reference image (fidelity only).
* ``formatter``      -- writes the final answer to the user's output contract.

Each is a factory returning a LangGraph node, wired together in
:mod:`client_v2.graph`.  Loop control lives in the graph, not in prose.
"""
