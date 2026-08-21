#!/usr/bin/env bash
# Build a release archive for testing.
#
# Not a compiled binary, deliberately.  The client launches the MCP server as a
# subprocess with `sys.executable -m brlcad_mcp.server`; inside a PyInstaller
# bundle sys.executable is the frozen executable and `-m` does not exist, so the
# server would never start.  This ships source with pinned dependencies instead.
#
#   ./scripts/make-release.sh rc1
#   OPENAI_API_KEY=sk-... ./scripts/make-release.sh rc1     # bake a key into .env
#
# Writes dist/brlcad-mcp-<tag>.tar.gz
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="${1:-rc1}"
NAME="brlcad-mcp-$TAG"
OUT="dist"
STAGE="$OUT/$NAME"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo "no .venv -- create it first" >&2; exit 1; }

# `git archive` below packages HEAD, not the working tree.  Without this check a
# dirty tree ships the last commit while you test something else, which is easy
# to miss because the build succeeds either way.
if ! git diff --quiet HEAD -- ':!dist'; then
    echo "error: uncommitted changes. The archive is built from HEAD, so it" >&2
    echo "       would not contain them. Commit first, or pass --allow-dirty." >&2
    git status --short -- ':!dist' | sed 's/^/       /' >&2
    [ "${2:-}" = "--allow-dirty" ] || exit 1
    echo "       (--allow-dirty given, continuing)" >&2
fi

rm -rf "$STAGE" && mkdir -p "$STAGE"

# Tracked files only.  This is what keeps .env, .venv, evals/runs and every
# __pycache__ out of the archive without maintaining a second exclude list.
git archive HEAD | tar -x -C "$STAGE"

# Pin the dependency set.  The declared deps are ranges (langchain>=1.0), and
# langchain moves fast enough that an archive opened weeks later can resolve to
# something that no longer works.  Pinning what is known good removes that.
.venv/bin/python -m pip freeze --exclude-editable > "$STAGE/requirements-pinned.txt"

# Environment file at the archive ROOT, which is where config.py looks for it
# (PROJECT_ROOT = parents[2] of src/brlcad_mcp/config.py).  A key is written only
# if one is supplied in the environment, so it never lands in a committed file.
sed -E 's/^(OPENAI_API_KEY)=.*/\1=/' .env.example > "$STAGE/.env"
if [ -n "${OPENAI_API_KEY:-}" ]; then
    sed -i -E "s|^OPENAI_API_KEY=.*|OPENAI_API_KEY=${OPENAI_API_KEY}|" "$STAGE/.env"
    echo "note: an API key was baked into $NAME/.env"
    echo "      send this archive privately; do not attach it to a public release"
else
    echo "note: no key baked in. The recipient sets OPENAI_API_KEY in .env"
fi

cp scripts/QUICKSTART.md "$STAGE/QUICKSTART.md"

mkdir -p "$OUT"
tar -czf "$OUT/$NAME.tar.gz" -C "$OUT" "$NAME"
rm -rf "$STAGE"

echo
echo "built $OUT/$NAME.tar.gz  ($(du -h "$OUT/$NAME.tar.gz" | cut -f1))"
tar -tzf "$OUT/$NAME.tar.gz" | sed 's|^|  |' | head -14
echo "  ... $(tar -tzf "$OUT/$NAME.tar.gz" | wc -l) entries total"
