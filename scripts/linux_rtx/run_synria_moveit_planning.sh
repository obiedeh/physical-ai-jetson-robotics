#!/usr/bin/env bash
# Launch move_group headless for the Synria arm, run the planning campaign, shut the launch down.
# Pass means every request returned a real plan; a timeout or a launch failure is a fail.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="$REPO_ROOT/reports/simulation/synria_moveit_planning.json"
LAUNCH_LOG="${LAUNCH_LOG:-/tmp/synria_moveit_planning_launch.log}"
mkdir -p "$(dirname "$REPORT")"

set +u
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/ros2_ws/install/setup.bash"
set -u

ros2 launch synria_arm_moveit_config planning.launch.py > "$LAUNCH_LOG" 2>&1 &
LAUNCH_PID=$!
trap 'kill -INT $LAUNCH_PID 2>/dev/null || true; wait $LAUNCH_PID 2>/dev/null || true' EXIT

set +e
/usr/bin/python3 "$REPO_ROOT/scripts/linux_rtx/moveit_planning_campaign.py" --out "$REPORT" "$@"
STATUS=$?
set -e
echo "campaign exit $STATUS; launch log $LAUNCH_LOG"
exit $STATUS
