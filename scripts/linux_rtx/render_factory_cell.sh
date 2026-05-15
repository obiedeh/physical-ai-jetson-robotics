#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Isaac Sim 6.x pip install (default): ~/.venv/isaacsim6/bin/python
# Isaac Sim 5.x standalone (legacy):  /home/oedeh/isaacsim/python.sh
ISAAC_PYTHON="${ISAAC_PYTHON:-${HOME}/.venv/isaacsim6/bin/python}"

if [[ ! -x "$ISAAC_PYTHON" ]]; then
  echo "Isaac Sim Python not found: $ISAAC_PYTHON" >&2
  echo "For Isaac Sim 5.x: ISAAC_PYTHON=/path/to/isaacsim/python.sh" >&2
  echo "For Isaac Sim 6.x: ISAAC_PYTHON=~/.venv/isaacsim6/bin/python" >&2
  exit 1
fi

# Pin Vulkan to the real NVIDIA ICD only.
# mesa-vulkan-drivers installs gfxstream_vk_icd.json alongside nvidia_icd.json.
# gfxstream re-presents the same physical GPU as a virtual passthrough device,
# causing Vulkan to enumerate two RTX entries with the same UUID. Without this
# fix the crash reporter shows three GPUs (RTX 5090 x2 + Intel iGPU); with it,
# one GPU is enumerated and the duplicate-GPU symptom is eliminated.
# NOTE: Isaac Sim 5.1.0 still crashes after this fix because its bundled RTX
# rendering stack (librtx.scenedb.plugin.so, Kit 107.3.3 / CUDA 12.0) has no
# compiled shaders for sm_120 (Blackwell/RTX 5090). The fix is Isaac Sim 6.0,
# which ships Blackwell support. Keep this ICD pin for the 6.0 install too.
NVIDIA_ICD="/usr/share/vulkan/icd.d/nvidia_icd.json"
if [[ ! -f "$NVIDIA_ICD" ]]; then
  echo "Warning: $NVIDIA_ICD not found; running without VK_ICD_FILENAMES override." >&2
else
  export VK_ICD_FILENAMES="$NVIDIA_ICD"
  echo "Vulkan ICD pinned to: $NVIDIA_ICD"
fi

"$ISAAC_PYTHON" "$REPO_ROOT/isaac/scripts/render_factory_cell.py" "$@"
