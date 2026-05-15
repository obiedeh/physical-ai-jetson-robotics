#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim6/bin/python}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim Python not found: $ISAAC_PYTHON" >&2
  exit 1
fi

NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ -f "$NVIDIA_ICD" ]]; then
  export VK_ICD_FILENAMES="$NVIDIA_ICD"
fi

"$ISAAC_PYTHON" "$REPO_ROOT/isaac/scripts/import_synria_urdf.py" "$@"

ROBOT_USD="$REPO_ROOT/isaac/usd/robots/synria_6dof_arm/synria_6dof_arm.usda"
if [[ -f "$ROBOT_USD" ]]; then
  echo "Import succeeded: $ROBOT_USD"
else
  echo "Warning: expected USD not found at $ROBOT_USD" >&2
fi
