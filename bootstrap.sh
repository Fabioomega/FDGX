#!/usr/bin/env bash
# Start (or restart) fdgx on a fresh instance. Idempotent: run it every time the instance comes back.
#   ./bootstrap.sh [fdgx.yaml] [extra fdgx run args...]
# Everything uv needs (binary, Python interpreters, wheel cache) lives under FDGX_HOME on the
# persistent disk, so a restart re-downloads nothing.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-$HERE/fdgx.yaml}"
shift || true

export FDGX_HOME="${FDGX_HOME:-$HERE/.fdgx-home}"
export UV_INSTALL_DIR="$FDGX_HOME/bin"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$FDGX_HOME/uv-cache}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$FDGX_HOME/python}"
export UV_NO_MODIFY_PATH=1
export PATH="$UV_INSTALL_DIR:$PATH"
export PYTHONUNBUFFERED=1

log() { echo "[bootstrap $(date '+%H:%M:%S')] $*"; }

if ! command -v uv >/dev/null 2>&1; then
  url="https://astral.sh/uv/${FDGX_UV_VERSION:+$FDGX_UV_VERSION/}install.sh"
  log "installing uv into $UV_INSTALL_DIR from $url"
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf "$url" | sh
  else
    wget -qO- "$url" | sh
  fi
fi
log "$(uv --version)"

log "syncing fdgx environment"
if [ -f "$HERE/uv.lock" ]; then
  uv sync --project "$HERE" --frozen --no-dev
else
  uv sync --project "$HERE" --no-dev
fi

log "starting scheduler with $CONFIG"
exec uv run --project "$HERE" --no-sync fdgx run "$CONFIG" "$@"
