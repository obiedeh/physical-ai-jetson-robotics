#!/usr/bin/env bash
# Launch the read-only console in the operator's dedicated recording environment.
# Source library environment only; never start a ROS node or robot process.
set -euo pipefail
recording_venv="${RECORDING_VENV:-/srv/venvs/synria-d1-py312}"
if [[ -z "${RECORDING_VENV:-}" && ! -f "$recording_venv/bin/activate" ]]; then
  recording_venv="$HOME/venvs/synria-d1-py312"
fi
if [[ ! -f "$recording_venv/bin/activate" ]]; then
  echo "Set RECORDING_VENV to the existing Python 3.12 recording environment." >&2
  exit 1
fi
source "$recording_venv/bin/activate"
console_demo=false
for console_arg in "$@"; do
  if [[ "$console_arg" == "--demo" ]]; then console_demo=true; fi
done
if [[ "$console_demo" == false ]]; then
  ros_setup="${RECORDING_ROS_SETUP:-/opt/ros/jazzy/setup.bash}"
  echo "Launching: preparing recording libraries; no robot services will be started."
  if [[ -f "$ros_setup" ]]; then
    # Distribution setup scripts may reference unset variables; retain strict checks afterward.
    set +u
    source "$ros_setup"
    set -u
  else
    echo "Not ready: ROS environment file is missing: $ros_setup" >&2
    echo "The console will show missing dependencies; hardware services remain manual." >&2
  fi
fi
# Module execution also works when the editable install predates the console entry point.
exec python -m synria_lerobot.console.app "$@"
