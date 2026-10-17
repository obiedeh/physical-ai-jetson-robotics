# Safety gate proof reports

Written by `scripts/prove_safety_gates.py`. Each report records four probes run
through the real gate code with the follower feed as witness: joint limit,
velocity limit, stale observation and per-cycle step cap, plus the follower
state before and after and the publisher counts on the guarded command topics.

- `gate_proof_<UTC timestamp>.json`: one operator run on hardware.
- `gate_proof_dryrun_*.json`: fake follower and clock; tooling check only,
  never evidence. Do not commit these.

Status: no hardware run recorded yet. **Implemented, unmeasured.**
