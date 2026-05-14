#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BLENDER="${BLENDER:-blender}"

if ! command -v "$BLENDER" >/dev/null 2>&1; then
  echo "Blender executable not found: $BLENDER" >&2
  echo "Install Blender or set BLENDER to the full executable path." >&2
  exit 1
fi

"$BLENDER" --background --python "$REPO_ROOT/isaac/scripts/render_factory_cell_blender.py" -- "$@"
