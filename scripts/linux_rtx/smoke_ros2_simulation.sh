#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_WS="$REPO_ROOT/ros2_ws"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-8}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/physical_ai_ros2_simulation_smoke_$$}"

if [[ ! -f /opt/ros/jazzy/setup.bash ]]; then
  echo "ROS 2 Jazzy setup file not found at /opt/ros/jazzy/setup.bash." >&2
  exit 1
fi

set +u
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
set -u

if ! command -v colcon >/dev/null 2>&1; then
  echo "colcon is not available. Install ROS 2 build tooling before running this smoke test." >&2
  exit 1
fi

cd "$ROS_WS"
colcon --log-base "$BUILD_ROOT/log" build --symlink-install \
  --build-base "$BUILD_ROOT/build" \
  --install-base "$BUILD_ROOT/install" \
  --packages-select \
  physical_ai_description \
  physical_ai_gazebo \
  rosmaster_m3pro_description \
  synria_arm_description \
  synria_arm_gazebo \
  synria_arm_moveit_config

set +u
# shellcheck disable=SC1091
source "$BUILD_ROOT/install/setup.bash"
set -u

echo "Launching factory-cell Gazebo smoke test for ${TIMEOUT_SECONDS}s."
set +e
timeout -s INT --kill-after=2s "${TIMEOUT_SECONDS}s" \
  ros2 launch physical_ai_gazebo factory_cell.launch.py
status=$?
set -e

if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
  echo "Factory-cell smoke launch failed with exit code $status." >&2
  exit "$status"
fi

echo "Launching Synria mock-control smoke test for ${TIMEOUT_SECONDS}s."
set +e
timeout -s INT --kill-after=2s "${TIMEOUT_SECONDS}s" \
  ros2 launch synria_arm_gazebo control.launch.py
status=$?
set -e

if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
  echo "Synria mock-control smoke launch failed with exit code $status." >&2
  exit "$status"
fi

echo "ROS 2 simulation smoke tests started successfully."
