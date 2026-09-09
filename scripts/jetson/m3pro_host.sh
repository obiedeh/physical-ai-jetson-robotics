#!/usr/bin/env bash
# Launch the ROSMASTER M3 Pro ZMQ host on the Orin. Sources the ROS env
# (domain 30 + UDP-only FastDDS profile for a non-vendor user), verifies the
# board is up, and runs the host with the plugin on PYTHONPATH *prepended*
# to the ROS PYTHONPATH (a bare `PYTHONPATH=... python` would drop rclpy).
#
#   bash scripts/jetson/m3pro_host.sh [--no-base] [--no-home] [--cameras SPEC]
# Default cameras: front:0:640:480:15 (Orbbec RGB /dev/video0). Log: ~/m3pro_host.log
set -u
REPO="$HOME/github/physical-ai-jetson-robotics"
cd "$REPO"
source "$REPO/scripts/jetson/orin_ros_env.sh" >/dev/null 2>&1
if ! m3pro_preflight; then
  echo "[m3pro_host] board not visible — is the micro-ROS agent up? (see docs notes s.9)"; exit 1
fi
PKG="$REPO/robots/lerobot_robot_rosmaster_m3pro"
ARGS=("$@"); [ ${#ARGS[@]} -eq 0 ] && ARGS=(--cameras front:0:640:480:15)
case " ${ARGS[*]} " in *" --cameras "*) : ;; *) ARGS+=(--cameras front:0:640:480:15) ;; esac
exec env PYTHONPATH="$PKG:${PYTHONPATH:-}" /usr/bin/python3 -m lerobot_robot_rosmaster_m3pro.m3pro_host "${ARGS[@]}"
