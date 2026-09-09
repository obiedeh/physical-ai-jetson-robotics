#!/usr/bin/env bash
set -euo pipefail

echo "System:"
uname -a

echo
echo "GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo
echo "Python:"
python3 --version

echo
echo "ROS 2:"
if command -v ros2 >/dev/null 2>&1; then
  ros2 --version || true
else
  echo "ros2 not found"
fi
