#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-linux-rtx}"
OUTPUT="${2:-reports/inventory/linux_rtx.json}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found at $PYTHON. Run scripts/linux_rtx/bootstrap.sh first." >&2
  exit 1
fi

"$PYTHON" -m physical_ai_lab.cli collect-inventory --target "$TARGET" --output "$OUTPUT"
