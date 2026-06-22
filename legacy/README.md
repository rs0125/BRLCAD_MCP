# Legacy transport

`listener.tcl` is the **original** transport: a non-blocking Tcl socket
listener loaded into MGED with `source listener.tcl`, which evaluated
incoming commands via `uplevel #0` and framed responses with a
`<<END_OF_RESPONSE>>` sentinel.

It is **superseded** by the native [`libmcpcad`](https://github.com/rs0125/brlcad/tree/libmcpcad)
listener (the `mcp_listen` MGED command), which the project now targets:

- a length-prefixed binary frame protocol instead of a text sentinel,
- execution through the C `ged_exec()` pipeline instead of Tcl `uplevel`,
- explicit `OK` / `ERR <code>` status instead of regex-sniffing output.

It is kept here only for historical reference and for running against
older MGED builds that lack `mcp_listen`. New work should not depend on it.
