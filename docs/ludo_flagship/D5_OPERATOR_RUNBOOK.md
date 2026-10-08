# D5 operator runbook

Software: **implemented, unmeasured**. D5: **planned**.

Complete all [D4 checks](D4_OPERATOR_RUNBOOK.md), select the accepted trained
policy, and reset every physical token to the engine's initial yard state.
Set a bounded `max_turns`, `max_attempts`, explicit scene-recovery procedure,
and a fresh session id in the [session config](../../config/synria_session.json).
Every turn uses the physical roll state machine; no random roll fallback is
used in physical sessions.

```bash
python3 -m synria_lerobot.sessions d5 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/games/$SESSION"
python3 -m synria_lerobot.sessions d5 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/games/$SESSION" --enable-motion
```

The runner feeds each stable die value to `game.plan_turn(roll=value)`, grades
every physical command, preserves misses and retries, and ends at the first
winner. Confirm resets before retries. A failed turn, unreadable roll,
unreachable move, operator abort or turn cap ends the run with abort status.
Use the site's verified hold/stop path and reconcile the physical board before
any new session; the runner does not infer where a failed manipulation left a
token.

Both winner and abort paths write turns.jsonl and frozen statistics through the
[shared ludo statistics aggregator](../../isaac/scripts/ludo_stats.py).
The same directory contains provenance, roll events, latency and end-power
state. Review every failed attempt and the final board still, not just the
winner field. Software outputs retain stage status planned; only committed
operator-accepted physical evidence can support a milestone update.

Append activity and timeline entries, then commit the session and cited stills:

```bash
git add "reports/ludo_flagship/games/$SESSION" \
  docs/ludo_flagship/ACTIVITY_LOG.md docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria full-game evidence"
```

Leave the sim-to-real gap table untouched until D5 is established by physical
evidence.

Object success is the operator's label plus a camera still; no independent sensor confirms it.
