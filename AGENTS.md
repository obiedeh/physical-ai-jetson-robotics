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

# Evidence Commit Rule

Evidence files are committed when they are produced, not in a cleanup pass
later. This repo's credibility depends on every cited number resolving to a
tracked file; untracked evidence breaks that silently.

Whenever a run produces a report, measurement file, provenance record, chart,
chart script, or evidence media that will be cited anywhere (README, evidence
pages, chart captions, notes, ledgers):

- commit it in the same change as the work that produced it
- do not leave it untracked pending review
- if it is not worth committing, it is not worth citing

Before a change that cites evidence is complete, confirm every cited path is
tracked (`git ls-files --error-unmatch <path>`). Large regenerable outputs
(frame dumps, per-frame metadata, datasets, checkpoints) stay out of git; commit the small
provenance, summary and aggregate files that citations actually point to.

---

# Output Format

At the end of each task, Codex should report:

1. Files changed and why
2. Tests run
3. Tests not run and why
4. Risks or follow-up work
5. Whether Claude Code safety review is needed

# Focus Rule (operator, 2026-09-17)

Every other project is parked until the flagship Synria physical AI
deliverable is in hand. The deliverable is the first line of
`reports/NOT_CLAIMED.md`: autonomous real-arm pick and place, with physical
object-success ground truth, recorded as a committed artifact with device,
date and inputs. Until that artifact exists:

- Work in this repository goes to the Synria pick-and-place lane
  (`docs/SYNRIA_PICK_AND_PLACE_TASK.md`, `docs/SYNRIA_GR00T_LEDGER.md`) and
  whatever it depends on. The rover SLAM lane, the site, the security,
  AI-RAN and safety-observability repositories are parked; they may receive
  fixes only when they block the flagship or when the operator asks.
- Before starting any task, state in one line whether it serves the
  deliverable. If it does not, say so and stop. Being asked to keep the
  operator honest includes declining side quests, including ones the
  operator proposes, with a one-line reason.
- Progress is measured against the ledger's open defects, not against
  volume of commits.

