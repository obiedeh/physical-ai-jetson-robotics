#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-jetson-orin}"
OUTPUT="${2:-reports/inventory/jetson_inventory.json}"
PYTHON="${PYTHON:-.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found at $PYTHON. Create a venv and install the package first." >&2
  exit 1
fi

"$PYTHON" -m physical_ai_lab.cli collect-inventory --target "$TARGET" --output "$OUTPUT"
