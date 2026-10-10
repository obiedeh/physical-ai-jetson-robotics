#!/usr/bin/env bash
# Launch the read-only console in the operator's dedicated recording environment.
# This script never starts ROS, a driver, a bridge, or teleoperation.
set -euo pipefail
recording_venv="${RECORDING_VENV:-/srv/venvs/synria-d1-py312}"
if [[ ! -f "$recording_venv/bin/activate" ]]; then
  echo "Set RECORDING_VENV to the existing Python 3.12 recording environment." >&2
  exit 1
fi
source "$recording_venv/bin/activate"
exec synria-teleop-console "$@"
