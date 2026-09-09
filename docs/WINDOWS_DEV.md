# Windows Development

Use Windows for repo development, tests, docs, and the core CLI.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If the Windows Store Python launcher causes permission issues, use a regular Python.org install or the Codex bundled Python runtime.

## Validate

```powershell
ruff check .
mypy physical_ai_lab
pytest -q
physical-ai-lab demo-telemetry
```

## Collect Inventory Evidence

```powershell
.\scripts\windows\collect_inventory.ps1
```

Output:

```text
reports/inventory/windows_dev.json
```

## Best Windows Tasks

- write and review docs
- run cross-platform package tests
- build agent/RAG prototypes
- generate synthetic telemetry
- prepare GitHub commits and pull requests
- review architecture and business-case materials

## Avoid On Windows

- Jetson-specific TensorRT validation
- ROS 2 hardware bring-up
- Isaac Sim RTX workflows
- real robot control loops

Those belong on Linux RTX or Jetson Linux.
