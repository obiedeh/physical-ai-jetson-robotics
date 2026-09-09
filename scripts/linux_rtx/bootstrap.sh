#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
# Run tests with a clean PYTHONPATH so sourced ROS env vars don't
# inject system site-packages and conflict with the venv's pytest plugins.
PYTHONPATH="" pytest -q

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found; install NVIDIA drivers before Isaac/RTX validation."
fi
