# AGENTS.md

## Repository hygiene

Read and follow the [Repository hygiene policy in obiedeh-brand/AGENTS.md](https://github.com/obiedeh/obiedeh-brand/blob/main/AGENTS.md#repository-hygiene-standing-every-repo), the single source of these rules.

- Name boards only: `jetson-agx-thor`, `jetson-orin-nx`, `rtx-5090-workstation`.
- No host names, home-directory paths or LAN addresses in tracked prose, logs, commit messages or branch names; write commands against `~`.
- Never bypass or disable hooks, or change `core.hooksPath` to evade them. Check diffs and commit messages against `~/.config/git/private-terms.txt` without exposing its contents.
- Never rewrite history, force push or move tags. Amend only commits that have never been pushed.
- Keep handoff notes, reviews and plans outside repositories in `~/handoff_notes/`.

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
synria_lerobot/    Synria arm LeRobot episodes, datasets and policy evaluation
edge_ai/           Jetson inference and benchmarking (ONNX, camera inference)
slam/              SLAM measurement harness and metrics
game_core/         Game adapters (Ludo, chess) and dice
ludo_engine/       Ludo rules engine
agents/            Telemetry operations copilot and fail-closed command skill planning
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

## Commit Identity Rule

Every commit is authored and committed by the repository owner, under the
owner's own git identity. Before the first commit in any environment, set
`git config user.name` and `git config user.email` to the owner's identity and
check them. Never commit under a tool or vendor identity, and never add
co-author, session or generated-by trailers to a commit message or a pull
request description. CI rejects a pull request whose commits break this rule
(`scripts/check_repo_hygiene.py --commits`).

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
- handoff, recap, or summary files; session notes belong only in
  `docs/ludo_flagship/ACTIVITY_LOG.md`

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

## Private identifiers in tracked files

- Evidence files, logs and scripts carry board labels (`rtx-5090-workstation`, `bench-thor`, `field-orin`), `<lan-address>` and `<home>`, never host names, addresses or home-directory paths.
- Recorders use `physical_ai_lab.portable`: `board_label()` for the machine, `portable_text` or `portable_json` when writing evidence, `expand_path` or `expand_json` when reading a recorded path back.
- Scripts take addresses from the environment (`THOR_ADDRESS`, `THOR_SSH`, `ORIN_ADDRESS`) or an untracked local config file, never a literal.
- `scripts/check_repo_hygiene.py` enforces paths and addresses in CI and in `tests/test_repo_hygiene.py`. The global git hooks in `~/.config/git/hooks` refuse offending commits and pushes, including machine names kept in a private list outside the repository.
