#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

.venv/bin/physical-ai-lab rtx-projects
.venv/bin/physical-ai-lab train-synria-reach \
  --device "${TRAIN_DEVICE:-cuda}" \
  --samples "${TRAIN_SAMPLES:-8192}" \
  --epochs "${TRAIN_EPOCHS:-40}" \
  --batch-size "${TRAIN_BATCH_SIZE:-512}"

bash scripts/linux_rtx/render_factory_cell_blender.sh
