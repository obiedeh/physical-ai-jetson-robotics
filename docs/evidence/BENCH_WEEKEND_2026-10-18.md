# Bench weekend 2026-10-18: evidence index

Every row below is **implemented, unmeasured** until its evidence file exists
and is linked here. Nothing on this page is a claim until the operator replaces
the placeholder with a path and a number. The report files are written by the
scripts named in each row; dry-run files (`*_dryrun_*`) are never evidence.

| Step | Runbook | Script or surface | Evidence file | Status |
| --- | --- | --- | --- | --- |
| First sim take through the teleop console | [D1 operator runbook](../ludo_flagship/D1_OPERATOR_RUNBOOK.md), [console guide](../ludo_flagship/TELEOP_CONSOLE.md) | simulation console, port 8768 | workspace `smoke-reports/smoke_simulation_*.json` and the session's catalog row; copy the smoke report to `reports/smoke/` | implemented, unmeasured |
| Joint limits verified against the physical stops | [limits verification runbook](../ludo_flagship/LIMITS_VERIFICATION_RUNBOOK.md) | `scripts/verify_limits.py` | `reports/limits/verification_<timestamp>.json` (none yet) | implemented, unmeasured |
| Safety gates proven: joint limit, velocity limit, stale observation, per-cycle step cap | this page, section below | `scripts/prove_safety_gates.py --i-am-the-operator` | `reports/safety/gate_proof_<timestamp>.json` (none yet) | implemented, unmeasured |
| Physical smoke capture through the console | [D1 operator runbook, section 3](../ludo_flagship/D1_OPERATOR_RUNBOOK.md) | physical console, port 8765, Practice run | `reports/smoke/physical_smoke_<timestamp>.json` (none yet) | implemented, unmeasured |
| First D1 session and the D1 count | [D1 operator runbook, sections 4 to 6](../ludo_flagship/D1_OPERATOR_RUNBOOK.md) | physical console; `physical-ai-lab d1-session-summary`; `scripts/d1_status.py` | `reports/ludo_flagship/data/<session>/session_summary.json`, [`D1_dataset_summary.json`](../../reports/ludo_flagship/data/D1_dataset_summary.json) | planned, 0 of 100 |

## Safety gate proof procedure

1. Robot secured, e-stop within reach, nobody inside the arm's reach. The
   follower driver is running read-only and `/joint_states` is live.
2. From the repository root with the ROS environment sourced:

   ```bash
   python scripts/prove_safety_gates.py --i-am-the-operator --operator "Your Name"
   ```

3. Type `arm-clear` when asked. The script reads the follower, runs the four
   probes through the real gate code without creating a publisher, reads the
   follower again, counts publishers on `/joint_commands` and
   `/policy_joint_targets` before and after, and writes the report.
4. Expected: every probe line prints `PASS`, the follower change is 0 degrees
   or close to it, and both publisher counts are 0 before and after.
5. Commit the report and fill the activity log template.

What the proof does and does not show: the joint limit, velocity limit and
stale-observation refusals and the per-cycle step clamp are exercised on live
follower data. The ROS adapter's refusal of a second offer within one command
period needs an authorized motion session and remains covered by unit tests
only; the report says so in its `coverage_note`.

## Physical smoke capture

Use the physical console's Practice run (disposable smoke, 20 seconds). When
the smoke context closes, the console writes
`~/synria-console/smoke-reports/smoke_physical_<timestamp>.json` with the
settings, the measured state rate, and per episode the frame count, duration,
achieved sample rate, expected and dropped frames, and each camera's distinct
frames and delivered rate. Copy it to
`reports/smoke/physical_smoke_<timestamp>.json` and commit it with the activity
log entry.

## When a measurement exists

Replace the row's status with **measured**, add the date and link the file.
The README rows that change are listed in the pull request that lands the
evidence; until then the README stays as it is.
