# Joint limits verification reports

Written by `scripts/verify_limits.py`
([runbook](../../docs/ludo_flagship/LIMITS_VERIFICATION_RUNBOOK.md)).

- `verification_<UTC timestamp>.json`: one operator measurement on hardware.
  Commit it in the same change as any limits file update it justified.
- `verification_dryrun_*.json`: fake follower; tooling check only, never
  evidence. Do not commit these.

- `sweep_<UTC timestamp>.json`: one read-only sweep measurement on hardware,
  logged from the follower feed while the operator jogged each joint to its
  stops; the CSV it summarises lives outside git.

Status: one read-only sweep recorded on 2026-10-17 (`sweep_20261017T215329Z.json`),
which widened Joint2 upper, Joint3 lower and both Joint5 sides, with the 50mm gripper
captured closed 0.0000 m and open 0.0250 m in a second sweep. The interactive
`verify_limits.py` run is still open. **Sweep measured; script run pending.**
