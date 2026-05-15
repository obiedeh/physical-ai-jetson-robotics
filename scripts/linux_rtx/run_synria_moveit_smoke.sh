#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORT="$REPO_ROOT/reports/simulation/synria_moveit_smoke.json"
mkdir -p "$(dirname "$REPORT")"

set +u
source /opt/ros/jazzy/setup.bash
source "$REPO_ROOT/ros2_ws/install/setup.bash"
set -u

STARTED_AT="$(date --iso-8601=seconds)"
set +e
timeout "${TIMEOUT_SECONDS:-8s}" ros2 launch synria_arm_moveit_config demo.launch.py
STATUS=$?
set -e

python3 -c "import json; from pathlib import Path; Path('$REPORT').write_text(json.dumps({'workload':'synria-moveit-smoke','started_at':'$STARTED_AT','timeout':'${TIMEOUT_SECONDS:-8s}','exit_code':$STATUS,'status':'pass' if $STATUS in (0, 124) else 'fail'}, indent=2) + '\n', encoding='utf-8')"

cat "$REPORT"
exit 0
