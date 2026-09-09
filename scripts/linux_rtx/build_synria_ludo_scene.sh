#!/usr/bin/env bash
# Build the Synria + Ludo tabletop training scene on the Linux RTX 5090 workstation.
#
# Requires Isaac Sim 5.1 installed via pip in a Python 3.11 venv (see isaac/README.md).
# Override ISAAC_PYTHON if the venv lives somewhere other than ~/.venv/isaacsim5.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$REPO_ROOT/isaac/scripts/build_synria_ludo_scene.py"
ISAAC_PYTHON="${ISAAC_PYTHON:-$HOME/.venv/isaacsim5/bin/python}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim 5.1 Python not found at $ISAAC_PYTHON." >&2
  echo "Activate or override: ISAAC_PYTHON=/path/to/isaacsim5/bin/python bash $0" >&2
  exit 1
fi

if [[ ! -f "$SCRIPT" ]]; then
  echo "Builder script not found: $SCRIPT" >&2
  exit 1
fi

cd "$REPO_ROOT"
"$ISAAC_PYTHON" "$SCRIPT"
