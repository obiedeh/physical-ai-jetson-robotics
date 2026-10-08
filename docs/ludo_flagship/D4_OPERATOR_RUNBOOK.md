# D4 operator runbook

Software: **implemented, unmeasured**. D4: **planned**.

Complete [D3 preconditions](D3_OPERATOR_RUNBOOK.md). Print the four configured
DICT_4X4_50 fiducials and fix them at board corners in configured clockwise id
order. Configure the normalized square rectangles, colour ranges and fixed
dump-pad ROI in [perception config](../../config/synria_perception.json).
The checked-in geometry is an example, not physical calibration. Both board
and dump pad must be visible in the supplied front image.

## Measure each perception component

Collect recorded images with operator truth. Each component CSV has columns
`image,truth,conditions`; paths are relative to its CSV. Encode truth as JSON:
board = four marker-centre pixel coordinates in configured id order or null;
tokens = a label per configured square (`red`, `blue`, `green`, `yellow`,
`empty`, or `unlocated`); die = integer 1–6 or null. Quote JSON CSV cells.
Include negative cases, varied lighting, occlusion and the actual camera pose.

Choose fresh output paths and run all three independent harnesses:

```bash
python3 -m synria_lerobot.perception board --truth "$BOARD_TRUTH" \
  --config "$PERCEPTION_CONFIG" --output "$BOARD_REPORT" --data-kind recorded --operator "$OPERATOR"
python3 -m synria_lerobot.perception tokens --truth "$TOKEN_TRUTH" \
  --config "$PERCEPTION_CONFIG" --output "$TOKEN_REPORT" --data-kind recorded --operator "$OPERATOR"
python3 -m synria_lerobot.perception die --truth "$DIE_TRUTH" \
  --config "$PERCEPTION_CONFIG" --output "$DIE_REPORT" --data-kind recorded --operator "$OPERATOR"
git add "$PERCEPTION_CONFIG" "$BOARD_REPORT" "$TOKEN_REPORT" "$DIE_REPORT"
git commit -m "Record physical perception accuracy"
```

Set the session's `accuracy_reports` map and `perception_config`. Each report
must be committed, unchanged, tied to that configuration, and meet its own
configured accuracy and image-count thresholds. Synthetic reports in
[software fixtures](../../reports/ludo_flagship/perception/) cannot enable
physical use. Keep raw image corpora outside git with the recorded hashes.

## Physical roll sequence

Fill and verify [roll parameters](../../config/synria_roll.json) on the real
rig. Each phase supplies task, target XYZ metres, maximum policy steps and
step period. The physical sequence is pick die, drop into cup, shake, invert,
then read the fixed dump pad. The template contains no executable motion
parameters. Configure the number and age of independent stable die frames.
The adapter must verify phase completion and provide a timestamped still for
each read. Motion actions pass through the same limits and delta gates.

```bash
python3 -m synria_lerobot.sessions d4 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/rolls/$SESSION"
python3 -m synria_lerobot.sessions d4 --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/rolls/$SESSION" --enable-motion
```

Review roll-state events linked to turn numbers, every camera-backed operator
grade, latency and frozen statistics. D4 needs three consecutive successful
physical-roll turns. Any failed attempt, including one later retried, resets
the consecutive count. Skips do not advance the count. Stop using the adapter's
operator-abort control after the required sequence; preserve the full ledger.
Read failure requests hold and aborts. Append activity/timeline entries and
commit the session with its cited stills:

```bash
git add "reports/ludo_flagship/rolls/$SESSION" \
  docs/ludo_flagship/ACTIVITY_LOG.md docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria physical roll evidence"
```

Object success is the operator's label plus a camera still; no independent sensor confirms it.
