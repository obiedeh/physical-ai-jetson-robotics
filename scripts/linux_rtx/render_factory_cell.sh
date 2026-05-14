#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_PYTHON="${ISAAC_PYTHON:-/home/oedeh/isaacsim/python.sh}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim Python not found or not executable: $ISAAC_PYTHON" >&2
  echo "Set ISAAC_PYTHON to the Isaac Sim python.sh path." >&2
  exit 1
fi

"$ISAAC_PYTHON" "$REPO_ROOT/isaac/scripts/render_factory_cell.py" "$@"
