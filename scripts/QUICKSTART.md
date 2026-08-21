# Quick start

This archive is the Python half of the system: the MCP tool server, the agent
client, and the evaluation harness. The other half is a BRL-CAD build that
includes `libmcpcad`, which provides the socket listener the tools talk to.

## 1. BRL-CAD with libmcpcad

The library is on the `libmcpcad` branch, proposed upstream as
[PR #238](https://github.com/BRL-CAD/brlcad/pull/238).

```bash
git clone https://github.com/rs0125/brlcad
cd brlcad && git checkout libmcpcad
mkdir build && cd build
cmake .. && make -j"$(nproc)"
```

If your system compiler rejects BRL-CAD's vendored code, build in a container
with an older toolchain. GCC 11 works.

Check the pipeline:

```bash
ctest -R mcpcad --output-on-failure     # mcpcad_parse, mcpcad_exec, mcpcad_session
```

## 2. This archive

```bash
tar xzf brlcad-mcp-rc1.tar.gz
cd brlcad-mcp-rc1

python -m venv .venv
.venv/bin/pip install -r requirements-pinned.txt
.venv/bin/pip install -e .
```

**Install with `-e`.** The configuration loader looks for `.env` relative to the
package location, so a non-editable install puts it somewhere inside the virtual
environment where it will not be found. Editable keeps it here, next to
`pyproject.toml`.

`requirements-pinned.txt` holds the exact versions this was tested against.
Installing without it will resolve newer LangChain releases, which may or may not
still work.

## 3. Configuration

Everything lives in `.env` in this directory. The BRL-CAD settings are already
filled in with defaults. The only value you may need to set is the API key, and
only for the agent:

```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.6-sol
BRLCAD_HOST=127.0.0.1
BRLCAD_PORT=5555
```

Environment variables override `.env` if you prefer to keep the key out of a
file:

```bash
export OPENAI_API_KEY=sk-...
```

## 4. Try it without an API key

The quickest check that the whole chain works, and it costs nothing:

```bash
./evals/run.sh
```

That starts its own throwaway listener on a scratch database, builds 14 cases
from hand-authored ground-truth specs, verifies each with `bb` and `nirt`, and
prints a report. It needs the BRL-CAD build from step 1 and nothing else.

Set `BRLCAD_BUILD` if your build tree is not at `~/dev/brlcad/build`:

```bash
BRLCAD_BUILD=/path/to/brlcad/build ./evals/run.sh
```

The offline test suite needs no BRL-CAD build at all:

```bash
.venv/bin/pytest          # 435 tests, no network, no API key
```

## 5. Try it with the agent

Start a listener on a copy of a model, in one terminal:

```bash
cp ~/models/thing.g /tmp/thing-work.g
/path/to/brlcad/build/src/libmcpcad/tests/mcpcad_test_server 5555 /tmp/thing-work.g
```

Or inside MGED, if you want to watch geometry appear in the live view:

```
mged> mcp_listen 5555
```

Then in a second terminal:

```bash
.venv/bin/brlcad-mcp chat
```

Ask it something that reads the model first, to confirm both halves are talking:

```
> what objects are in this database?
```

Then try the image-to-CAD workflow on one of the drawings in the archive:

```
> /image evals/images/roundbracket.jpg model this bracket, named bracket
```

`/trace` shows each agent as it runs, which is worth turning on the first time.
`/help` lists the rest.

## 6. Score the agent

```bash
./evals/run.sh --mode agent --auto-approve
```

This gives the agent each case's prompt instead of its ground-truth spec, and
scores whatever it built. It needs a key and it costs money; a full pass over the
corpus takes a couple of hours. To try one tier instead:

```bash
./evals/run.sh --mode agent --auto-approve --case '*_guided'
```

Every run writes one directory under `evals/runs/`, with `latest` pointing at the
newest: the verdict per case, every check and its outcome, every model call, the
renders, and a standalone `.g` per case that opens in MGED.

## Notes

- The listener accepts local connections only and has no authentication. Treat it
  as a development tool rather than a service.
- It edits the database file it was given, and saves. Work on a copy.
- The standalone listener serves one client at a time. Do not start two on the
  same port.

Fuller documentation: `README.md` here, and the project report at
https://github.com/rs0125/gsoc-2026-brlcad-mcp
