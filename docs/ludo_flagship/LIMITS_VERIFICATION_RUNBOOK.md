# Joint limits verification runbook

Status: **implemented, unmeasured.** The procedure and the script exist and are
tested with a fake follower. No physical stop has been measured with them yet.
When a measurement exists, its report lives in `reports/limits/` and is linked
from [BENCH_WEEKEND_2026-10-18.md](../evidence/BENCH_WEEKEND_2026-10-18.md).

What this verifies: that every bound in
[`config/synria_limits.yaml`](../../config/synria_limits.yaml) lies at or inside
the real physical stop of the installed arm, so the limit gate refuses a command
before the arm can reach a stop. The file currently carries
`verified_by` and `verified_on` values entered by hand on 2026-10-10 from
whole-degree collision boundaries. This procedure measures those boundaries
against the arm and either confirms them or shows where they are wrong.

## Before you start

- Robot secured, e-stop within reach, nobody inside the arm's reach.
- The follower driver is running read-only and publishing `/joint_states`.
  Check with `ros2 topic hz /joint_states` (expect about 50 Hz).
- Leader arm connected and driving the follower (hardware sync on).
- No console session is recording; the script only subscribes, but keep the
  bench to one activity at a time.
- Know which gripper is installed: `50mm` or `100mm`.

## Procedure

1. From the repository root, with the ROS environment sourced:

   ```bash
   python scripts/verify_limits.py --operator "Your Name" --gripper 50mm
   ```

   The script prints the limits file's current `verified_by` and `verified_on`,
   then walks the fourteen stops in order: Joint1 min, Joint1 max, ... Joint6 max,
   gripper closed, gripper open.

2. For each stop, jog that joint slowly with the leader until the follower
   reaches its physical stop. Keep the other joints near zero where the geometry
   allows it. Watch the live reading the script prints.

3. Press one key:
   - `m` marks the stop: the script takes a fresh reading and records it.
   - `r` re-reads and prints the current angle without recording.
   - `s` skips the stop; that side is recorded as unmeasured and fails.
   - `a` aborts; nothing is written.

4. For the gripper, close it fully and mark, then open it fully and mark.

5. When all fourteen stops are marked the script prints the table below, writes
   `reports/limits/verification_<UTC timestamp>.json`, and, only if every joint
   and the installed gripper passed, asks you to type your name again to write
   `verified_by` and `verified_on` into the limits file. It never writes those
   fields on a partial pass and never from a dry run.

6. Commit the report together with the limits file change, if any, and add the
   activity log entry (template in `ACTIVITY_LOG.md`).

Dry run to check the tooling without the arm (fake follower, report named
`verification_dryrun_*.json`, never evidence):

```bash
python scripts/verify_limits.py --operator "Your Name" --gripper 50mm --dry-run
```

## Pass rule

A joint side passes when the measured physical stop is at or beyond the
configured bound, within `--tolerance-deg` (default 1.0 degree, the recording
tolerance already used by the quality gates). Delta is measured minus
configured, in degrees. A negative delta on the min side or a positive delta on
the max side means the real stop is beyond the configured bound: the bound is
conservative, which passes. A stop that is reached before the bound fails,
because the gate would have allowed a command past the real stop.

The installed gripper passes when its closed reading is within
`--gripper-tolerance-m` (default 2 mm) of 0 and its open reading is within the
same tolerance of the configured stroke.

## Results

Fill this table from the JSON report. Leave it blank until measured.

| joint | yaml min (deg) | yaml max (deg) | measured min (deg) | measured max (deg) | delta min | delta max | pass/fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Joint1 | -155 | 144 | | | | | |
| Joint2 | -87 | 96 | | | | | |
| Joint3 | -34 | 135 | | | | | |
| Joint4 | -112 | 88 | | | | | |
| Joint5 | -69 | 65 | | | | | |
| Joint6 | -171 | 176 | | | | | |

| gripper | yaml min (m) | yaml max (m) | measured closed (m) | measured open (m) | delta closed | delta open | pass/fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50mm | 0.000 | 0.025 | | | | | |
| 100mm | 0.000 | 0.050 | | | | | not installed, unmeasured |

Report file: `reports/limits/verification_<timestamp>.json` (none yet).
Limits file stamped: no.

## If a joint fails

Do not widen the bound in the file to make it pass. Record the failure in the
report, note it in `DECISIONS.md` with the measured value, and change the bound
only in a reviewed commit that cites the report. A bound that is more
conservative than the stop is acceptable; a bound beyond the stop is not.
