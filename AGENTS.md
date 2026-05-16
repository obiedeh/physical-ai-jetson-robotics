# AGENTS.md

Repository-level operating instructions for Codex.

For shared engineering standards and skill definitions, read:

```text
https://github.com/obiedeh/obiedeh/tree/main/agent-skills
```

---

# Codex Role

Use Codex for:

- focused patches to Python source and ROS 2 configs
- test generation for `physical_ai_lab/`
- scaffolding new CLI commands following existing patterns in `cli.py`
- updating report templates and evidence structures
- dependency and config changes

Do not use Codex for:

- hardware bring-up decisions
- safety gate waivers
- autonomous robot control logic without explicit Claude Code review
- adding MCP servers or external tool integrations

Default workflow:

```text
Claude Code = architecture review, skill selection, planning
Codex       = implement, patch, test
Claude Code = production-readiness and safety check before merge
```

---

# Skill Selection

- `production-architecture-reviewer`: changes to `physical_ai_lab/`, `cli.py`, service boundaries, or deployment structure
- `repo-hardening-refactor`: dead code, stale docs, duplicate helpers, unnecessary abstractions
- `runtime-stability-debugger`: telemetry pipeline, ops copilot, edge inference latency, GPU memory, long-lived workers
- `edge-ai-deployer`: Dockerfile, Jetson deployment scripts, TensorRT/ONNX conversion, `scripts/jetson/`
- `observability-generator`: metrics, structured logging, health checks, triage signal coverage
- `physical-ai-safety-reviewer`: safety gate logic, fail-safe behavior, human override paths, evidence chains, `ops_copilot.py`, `telemetry.py`
- `ai-ran-workflow-generator`: not primary for this repo; use only if telecom/RAN integration is added

---

# Project Structure

```text
physical_ai_lab/     # Core Python package — CLI, config, telemetry, triage, training
ros2_ws/src/         # ROS 2 packages — URDF, MoveIt, Gazebo, controllers
isaac/               # Isaac Sim scripts and USD scene assets
scripts/             # Platform bootstrap and evidence collection (jetson/, linux_rtx/, windows/)
agents/              # RAG and agentic copilot track (future)
docs/                # Architecture, deployment, hardware, vendor integration docs
reports/             # Mission readiness tracking and evidence templates
tests/               # Pytest suite for physical_ai_lab/
```

---

# Anti-Bloat Rules

Do not create:

- new Python modules without a clear operational role
- duplicate triage or telemetry helpers
- speculative simulation adapters before hardware gates are passed
- vendor-specific configs in public paths (move to private repo)
- oversized READMEs or architecture essays in code comments

Every new file must justify at least one of:

- operational necessity
- reliability improvement
- observability improvement
- deployment-readiness improvement

---

# Safety Rules

This repo controls real robot hardware. Before any change that touches:

- `ops_copilot.py` or `telemetry.py` — run `physical-ai-safety-reviewer` skill
- `TriageThresholds` values — confirm against hardware evidence in `reports/`
- Jetson deployment scripts — run `edge-ai-deployer` skill
- MoveIt or controller configs — validate in simulation before hardware

Do not claim production readiness without evidence in `reports/`.

---

# Output Format

At the end of each task, Codex should report:

1. Files changed and why
2. Tests run
3. Tests not run and why
4. Risks or follow-up work
5. Whether Claude Code safety review is needed
