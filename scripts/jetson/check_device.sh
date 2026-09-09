#!/usr/bin/env bash
set -euo pipefail

echo "Kernel:"
uname -a

echo
echo "Jetson stats:"
if command -v tegrastats >/dev/null 2>&1; then
  timeout 5s tegrastats || true
else
  echo "tegrastats not found"
fi

echo
echo "NVIDIA runtime:"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
else
  echo "nvidia-smi not available on this Jetson image"
fi

echo
echo "Python:"
python3 --version
