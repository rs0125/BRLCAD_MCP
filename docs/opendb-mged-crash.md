# `opendb` crashes a live MGED listener (db-path discovery)

## Symptom

The `render_model` tool discovers which `.g` is open by sending `opendb` (no
args) over the socket. This works against the headless libmcpcad test server
but **segfaults a live MGED GUI** the moment the render tool runs. After the
crash the listener is dead and every following command gets "connection
refused".

Against classic `mged -c` (headless), `opendb` does **not** crash but returns
`OK` with **no filename** — so discovery fails there too ("could not determine
the database path"). Only the standalone test server returns the path.

## Backtrace (the relevant frames)

```
#0 _post_opendb_failed (...)                 src/mged/f_db.c:50
#1 mged_post_opendb_clbk (ac=1, ...)         src/mged/f_db.c:175
#2 ged_clbk_exec (...)                        src/libged/ged.cpp:420
#3 ged_exec (gedp=..., argc=1, argv=...)      src/libged/exec.cpp:122
#4 mcpcad_cmd_exec (...)                       src/libmcpcad/exec.c:57
#5 mcpcad_session_input (...)                  src/libmcpcad/session.c:165
#6 mcp_client_readable (...)                   src/mged/mcp.c:120
   ... Tcl event loop (Tcl_DoOneEvent) ...
```

Crash line, `f_db.c:50`:
```c
int argc = ctx->argc;
const char **argv = ctx->argv;
const char *fname = argv[argc-1];   // <-- NULL deref
```

## Root cause

`opendb` with no filename is a getter. Through `mcp_listen` it goes straight to
`ged_exec`, which fires MGED's registered **post-opendb callback**
(`mged_post_opendb_clbk`). Because no *new* database was opened
(`mctx->old_dbip == gedp->dbip`), the callback routes to `_post_opendb_failed`,
which reads `ctx->argc` / `ctx->argv`.

Those fields are only populated when `opendb` runs through MGED's own
`f_opendb` command wrapper. `mcp_listen` calls `ged_exec` directly and bypasses
that wrapper, so `ctx->argv` is unset (NULL) → `argv[argc-1]` dereferences
garbage → SIGSEGV.

So it isn't `opendb` being destructive. MGED's opendb callback **assumes the
command always came through MGED's own dispatch** (which sets up the context).
Driving the same command via `ged_exec` from the side — exactly what
`mcp_listen` does — fires the callback with an empty context. The headless test
server registers no such callback, which is why it's unaffected. This will bite
anything that runs `opendb` through `ged_exec` outside MGED's command path, not
just this tool.

## Is there another way to get the db path?

Over the socket via ged commands: effectively no. The only commands that return
`dbi_filename` are `opendb` (crashes), `close` (returns it but *closes* the db),
and an incidental path in `mater`. No clean non-destructive getter besides
`opendb`.

Better options (not ged commands):

1. **Explicit path** — `BRLCAD_DB` env / `db_path` arg. Whoever launched the
   listener knows the path; pass it and skip discovery entirely.
2. **libmcpcad-level query** — the listener already holds the `gedp`/`dbip`, so
   it could report `dbip->dbi_filename` at the session layer *without* calling
   `ged_exec` (no MGED callback fires). Small C addition; keeps auto-discovery
   and avoids the crash. Likely the right long-term fix.

## Fixes (independent)

1. **Tool side (unblocks now):** resolve the db path from `db_path` / `BRLCAD_DB`
   and only fall back to `opendb` when neither is set — so a live MGED listener
   never receives `opendb`.
2. **Upstream (real fix):** guard `_post_opendb_failed` against `ctx->argv ==
   NULL` / `argc < 1` before the deref. Genuine bug in mged's callback, worth a
   report/PR.
