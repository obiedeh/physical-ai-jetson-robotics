# AGENTS.md

Engineering rules for anyone, human or automated, changing this repository.
This repo drives real robot hardware, so the rules favour evidence and safety
over speed.

---

## Project Structure

```text
physical_ai_lab/   Core Python package: CLI, config, telemetry, triage, training
robots/            Robot adapters (ROSMASTER M3 Pro)
arm_control/       Synria arm kinematics, trajectories and safety limits
ros2_ws/src/       ROS 2 packages: description, control, bringup, MoveIt config
isaac/             Isaac Sim scripts and USD scene assets
lerobot/           Synria arm LeRobot episodes, datasets and policy evaluation
edge_ai/           Jetson inference and benchmarking (ONNX, camera inference)
slam/              SLAM measurement harness and metrics
game_core/         Game adapters (Ludo, checkers, chess) and dice
ludo_engine/       Ludo rules engine
agents/            Operations copilot over robot telemetry
scripts/           Platform bootstrap and evidence collection (jetson/, linux_rtx/, windows/)
reports/           Committed evidence: benchmarks, sessions, corrections
docs/              Architecture, hardware, deployment and flagship ledgers
tests/             Pytest suite
```

---

## Status Labels

Every capability in the README and docs carries one label, and the label must
match the evidence:

- **measured**: a committed artifact with device, date and inputs
- **implemented, unmeasured**: the code runs, no performance artifact yet
- **scaffold**: code or configuration only
- **planned**: not started

Do not claim production readiness without evidence in `reports/`. Anything
not yet established is listed in `reports/NOT_CLAIMED.md`.

---

## Evidence Commit Rule

Evidence is committed when it is produced, not in a cleanup pass later. Every
cited number must resolve to a tracked file.

Whenever a run produces a report, measurement file, provenance record, chart,
chart script, or evidence media that will be cited anywhere (README, evidence
pages, chart captions, notes, ledgers):

- commit it in the same change as the work that produced it
- do not leave it untracked pending review
- if it is not worth committing, it is not worth citing

Before a change that cites evidence is complete, confirm every cited path is
tracked (`git ls-files --error-unmatch <path>`). Large regenerable outputs
(frame dumps, per-frame metadata, datasets, checkpoints) stay out of git;
commit the small provenance, summary and aggregate files that citations
point to.

Corrections stay in the record. A wrong result is corrected in place with the
reason, never silently overwritten.

---

## Safety Rules

Before any change that touches:

- `ops_copilot.py` or `telemetry.py`: a safety review of fail-safe behaviour,
  human override paths and evidence chains
- `TriageThresholds` values: confirm against hardware evidence in `reports/`
- Jetson deployment scripts: verify on the target device before citing results
- MoveIt or controller configs: validate in simulation before hardware

Every motion test on hardware starts with the robot secured, an e-stop within
reach and verified joint and command limits. Copilot and safety outputs are
advisory and operator-reviewed; nothing here acts autonomously on a person's
behalf.

---

## Anti-Bloat Rules

Do not create:

- new Python modules without a clear operational role
- duplicate triage or telemetry helpers
- speculative simulation adapters before hardware gates are passed
- vendor-specific configs or vendor assets in public paths
- oversized READMEs or architecture essays in code comments

Every new file must justify at least one of: operational necessity,
reliability, observability, or deployment readiness.

---

## Change Report

Every change ends with:

1. Files changed and why
2. Tests run
3. Tests not run and why
4. Risks or follow-up work
5. Whether a hardware safety review is needed before it runs on a robot
