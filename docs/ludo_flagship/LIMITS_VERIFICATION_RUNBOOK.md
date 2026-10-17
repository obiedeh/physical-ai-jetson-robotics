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

Measured on 2026-10-17 by a read-only sweep, not yet by `verify_limits.py`:
the operator jogged every joint to both stops while a subscriber on the console
host logged every follower sample (106331 samples over
36 minutes). Minima and maxima are the measured stops. The
yaml columns show the file as it was before the sweep; the pass rule is the one
above. Report: `reports/limits/sweep_20261017T215329Z.json`.

| joint | yaml min (deg) | yaml max (deg) | measured min (deg) | measured max (deg) | delta min | delta max | pass/fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Joint1 | -155 | 144 | -157.32 | 156.27 | -2.3 | 12.3 | pass |
| Joint2 | -87 | 96 | -102.13 | 101.16 | -15.1 | 5.2 | pass |
| Joint3 | -34 | 135 | -34.63 | 139.39 | -0.6 | 4.4 | pass |
| Joint4 | -112 | 88 | -159.08 | 160.75 | -47.1 | 72.8 | pass |
| Joint5 | -69 | 65 | -72.6 | 68.99 | -3.6 | 4.0 | pass |
| Joint6 | -171 | 176 | -179.82 | 179.65 | -8.8 | 3.7 | pass |

| gripper | yaml min (m) | yaml max (m) | measured closed (m) | measured open (m) | delta closed | delta open | pass/fail |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50mm | 0.000 | 0.025 | 0.0000 | 0.0250 | +0.0000 | +0.0000 | pass |
| 100mm | 0.000 | 0.050 | | | | | not installed, unmeasured |

After the sweep the operator widened four sides, each kept inside its measured
stop: Joint2 upper 96 to 100 (stop 101.2; the folded rest reads 100.3), Joint3
lower -34 to -35 (static stop -34.6, recorded motion reached -35.1), Joint5
upper 65 to 68 (stop 69.0), and later the same day Joint5 lower -69 to -72 (stop
-72.6) after three simulation picks pitched the wrist to -71.5 and failed the
limits gate. The pre-sweep file is kept byte for byte as
`config/synria_limits_2026-10-10_reference.yaml`; the working file is
`config/synria_limits.yaml` with `verified_on` 2026-10-17. The 50mm gripper was captured in a second sweep of its own: closed 0.0000 m,
open 0.0250 m, matching the configured stroke. A `verify_limits.py` run with
the interactive marks is still open and would be the formal stamp.

## If a joint fails

Do not widen the bound in the file to make it pass. Record the failure in the
report, note it in `DECISIONS.md` with the measured value, and change the bound
only in a reviewed commit that cites the report. A bound that is more
conservative than the stop is acceptable; a bound beyond the stop is not.
