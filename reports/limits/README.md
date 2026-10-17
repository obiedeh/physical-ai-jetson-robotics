# Joint limits verification reports

Written by `scripts/verify_limits.py`
([runbook](../../docs/ludo_flagship/LIMITS_VERIFICATION_RUNBOOK.md)).

- `verification_<UTC timestamp>.json`: one operator measurement on hardware.
  Commit it in the same change as any limits file update it justified.
- `verification_dryrun_*.json`: fake follower; tooling check only, never
  evidence. Do not commit these.

Status: no measurement recorded yet. **Implemented, unmeasured.**
