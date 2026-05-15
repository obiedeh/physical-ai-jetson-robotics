#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROS_WS="$REPO_ROOT/ros2_ws"
BUILD_ROOT="${BUILD_ROOT:-/tmp/physical_ai_robot_description_smoke_$$}"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

if ! command -v xacro >/dev/null 2>&1; then
  echo "xacro is not available. Install/source ROS 2 before running this smoke test." >&2
  exit 1
fi

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
  rosmaster_m3pro_description \
  synria_arm_description

set +u
# shellcheck disable=SC1091
source "$BUILD_ROOT/install/setup.bash"
set -u

declare -a XACROS=(
  "$ROS_WS/src/physical_ai_description/urdf/mobile_manipulator_proxy.urdf.xacro"
  "$ROS_WS/src/rosmaster_m3pro_description/urdf/rosmaster_m3pro.urdf.xacro"
  "$ROS_WS/src/synria_arm_description/urdf/synria_6dof_arm.urdf.xacro"
)

for model in "${XACROS[@]}"; do
  echo "Rendering xacro: ${model#$REPO_ROOT/}"
  xacro "$model" >/tmp/physical_ai_robot_description.urdf

  if command -v check_urdf >/dev/null 2>&1; then
    check_urdf /tmp/physical_ai_robot_description.urdf >/tmp/physical_ai_check_urdf.log
  fi
done

echo "Robot description smoke test passed."
