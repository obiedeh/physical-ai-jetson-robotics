# D3 operator runbook

Software: **implemented, unmeasured**. D3: **planned**.

Complete the device binding, hold verification, calibrated-square and policy
preconditions in the [D2 runbook](D2_OPERATOR_RUNBOOK.md). Use the same
[session configuration](../../config/synria_session.json) and a fresh session
id. Confirm the physical initial board agrees with the Ludo engine: four
tokens of each colour in their yard slots, red first. This runner does not
resume a partially completed physical game.

Validate with no adapter loaded, then explicitly enable after authorization:

```bash
python3 -m synria_lerobot.sessions d3 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/turns/$SESSION"
python3 -m synria_lerobot.sessions d3 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/turns/$SESSION" --enable-motion
```

Supply each integer roll through the adapter's operator prompt. The engine
plans the turn; capture returns execute before the moving token. Calibration
maps square ids to task XYZ. All commands must be reachable; otherwise the
whole turn is skipped with no attempt. No-legal-move rolls consume a rules
turn, but do not count as manipulation attempts.

Grade each command using an operator label and timestamped final front still.
The logical board advances only after all commands pass. Retry only after an
explicit scene reset. A failed or unreachable turn ends the session for board
reconciliation. Even if a token moved before failure, no logical success is
inferred. Preserve all attempts and capture-return failures in turns.jsonl.

Review turns, frozen statistics, provenance, end-power state and latency
records. One successful camera-backed operator-graded physical turn is D3's
evidence requirement; attempts and failures do not advance it. Append the
activity and timeline records, then commit the session and cited small stills:

```bash
git add "reports/ludo_flagship/turns/$SESSION" \
  docs/ludo_flagship/ACTIVITY_LOG.md docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria physical turn evidence"
```

Object success is the operator's label plus a camera still; no independent sensor confirms it.
