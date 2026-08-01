"""Terminal-facing concerns for the REPL.

* ``attachments`` -- parsing a line of input: slash commands and image
  attachments (``/image``, ``/paste``).
* ``trace``       -- formatting the per-node debug trace (``CLIENT_V2_DEBUG``).

Separated from the agent code so the graph has no opinion about how a human
happens to be talking to it.
"""
