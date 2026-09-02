#!/usr/bin/env bash
# Run the eval harness against a throwaway headless BRL-CAD listener.
#
# The listener is the fiddly part: mged needs an X display and fails on libXss,
# so the standalone test server is what makes an unattended run possible at all.
# It serves ONE client at a time from a serial accept loop, so this script owns
# it start to finish and kills it on the way out -- an orphan from a previous run
# holds the port and the next run's first build times out with no obvious cause.
#
#   ./evals/run.sh                          # tool mode: no API key, deterministic
#   ./evals/run.sh --mode agent --auto-approve      # the unattended baseline
#   ./evals/run.sh --mode agent --case l_bracket    # one case, scripted approval
#
# Everything a run produces lands in evals/runs/<stamp>_<shape>/, and
# evals/runs/latest points at the most recent one.
set -euo pipefail

cd "$(dirname "$0")/.."

BRLCAD_BUILD="${BRLCAD_BUILD:-$HOME/dev/brlcad/build}"
# Accept either layout: a build tree keeps the harness beside its source, an
# installed or released tree keeps it in bin/.  Only the build tree was checked
# before, so pointing this at a release bundle failed with "no test server".
SERVER=""
for candidate in "$BRLCAD_BUILD/src/libmcpcad/tests/mcpcad_test_server" \
                 "$BRLCAD_BUILD/bin/mcpcad_test_server"; do
    [ -x "$candidate" ] && { SERVER="$candidate"; break; }
done
PORT="${BRLCAD_PORT:-5555}"
DB="${BRLCAD_EVAL_DB:-$(mktemp -u /tmp/brlcad_eval_XXXXXX.g)}"
PY=".venv/bin/python"

[ -n "$SERVER" ] || { echo "no mcpcad_test_server under $BRLCAD_BUILD
looked in src/libmcpcad/tests/ and bin/.  Build it, or set BRLCAD_BUILD to a
BRL-CAD build tree or an unpacked release." >&2; exit 1; }
[ -x "$PY" ] || { echo "no venv at $PY -- create it first." >&2; exit 1; }

# The harness scores with rays, so nirt has to start.  In a build tree it needs
# LD_LIBRARY_PATH (set below); an installed tree resolves its own libraries.
# Warn rather than fail: every other check still runs, and a warning names the
# cause instead of leaving every ray reported as unmeasurable.
if ! LD_LIBRARY_PATH="$BRLCAD_BUILD/lib" "$(dirname "$SERVER")/../bin/nirt" -h \
        >/dev/null 2>&1 \
   && ! LD_LIBRARY_PATH="$BRLCAD_BUILD/lib" "$BRLCAD_BUILD/bin/nirt" -h \
        >/dev/null 2>&1; then
    echo "warning: nirt does not start under $BRLCAD_BUILD -- ray checks will
report as unmeasurable.  Everything else still runs." >&2
fi

# The key normally lives in .env (loaded by the client at import), NOT in the
# shell -- checking only the environment rejected a perfectly good run.  Look in
# both, and never echo the value.
if [[ " $* " == *" agent "* ]]; then
    if [ -z "${OPENAI_API_KEY:-}" ] && ! grep -q '^OPENAI_API_KEY=.' .env 2>/dev/null
    then
        echo "agent mode needs OPENAI_API_KEY, in the environment or in .env" >&2
        echo "(tool mode does not)." >&2
        exit 1
    fi
fi

# A stale listener on this port silently starves the new one; say so rather than
# letting the first case time out.
if ss -ltn 2>/dev/null | grep -q ":$PORT "; then
    echo "port $PORT is already in use -- kill the old listener first:" >&2
    echo "    pkill -f mcpcad_test_server" >&2
    exit 1
fi

SERVER_PID=""
cleanup() {
    [ -n "$SERVER_PID" ] || return 0
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# One listener per pass, on a FRESH database.
#
# Not tidiness -- necessity.  The render path leaks two pipe fds per call (rt's
# stdout and stderr are never closed), so fd numbers climb monotonically in a
# long-lived server.  Once one exceeds 1024, bu_process_pending() does
# FD_SET(fd, &read_set) into a stack fd_set that holds exactly 1024 bits and the
# server dies of __stack_chk_fail mid-run.  That killed a 10-pass job in pass 4.
# Restarting resets the counter, and costs ~4 s a pass.  Both bugs are upstream
# (libbu/libged); this only avoids them.
start_listener() {
    DB="$(mktemp -u /tmp/brlcad_eval_XXXXXX.g)"
    LD_LIBRARY_PATH="$BRLCAD_BUILD/lib" "$SERVER" "$PORT" "$DB" &
    SERVER_PID=$!
    for _ in $(seq 50); do                  # ~5 s for the socket to come up
        ss -ltn 2>/dev/null | grep -q ":$PORT " && return 0
        sleep 0.1
    done
    echo "listener did not come up on :$PORT" >&2
    return 1
}

# --repeat is handled HERE rather than inside the harness, because only this
# script can restart the listener between passes.  The harness gets --repeat 1
# and appends into one run directory, deriving each pass index from the results
# already on disk.
REPEAT=1
RUN_DIR=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --repeat) REPEAT="$2"; shift 2 ;;
        --repeat=*) REPEAT="${1#*=}"; shift ;;
        # Captured, not forwarded: the loop below adds it back for every pass.
        # Forwarding it here too would pass --run-dir twice.  Given explicitly,
        # this RESUMES an earlier job -- the harness continues its pass
        # numbering from the results already in that directory.
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --run-dir=*) RUN_DIR="${1#*=}"; shift ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

echo "listener  : $SERVER :$PORT"
if [ "$REPEAT" -le 1 ]; then
    start_listener
    echo "database  : $DB"
    BRLCAD_PORT="$PORT" BRLCAD_HOST=127.0.0.1 "$PY" -m evals.harness \
        "${ARGS[@]}" ${RUN_DIR:+--run-dir "$RUN_DIR"}
    exit $?
fi

FAILED=0
for pass in $(seq 1 "$REPEAT"); do
    start_listener || { FAILED=$((FAILED + 1)); continue; }
    echo "pass $pass/$REPEAT  database: $DB"
    # A pass that dies (the server abort above, or anything else) must not take
    # the job with it: the passes already written stay valid and the next one
    # starts from a clean process.
    BRLCAD_PORT="$PORT" BRLCAD_HOST=127.0.0.1 "$PY" -m evals.harness \
        "${ARGS[@]}" ${RUN_DIR:+--run-dir "$RUN_DIR"} || FAILED=$((FAILED + 1))
    cleanup
    SERVER_PID=""
    rm -f "$DB"
    [ -n "$RUN_DIR" ] || RUN_DIR="$(readlink -f evals/runs/latest)"
done
echo
echo "$REPEAT pass(es) attempted, $FAILED failed -- results: $RUN_DIR"
