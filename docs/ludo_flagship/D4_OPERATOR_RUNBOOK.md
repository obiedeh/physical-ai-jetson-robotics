# D4 operator runbook

Software: **implemented, unmeasured**. D4: **planned**.

The current executable path is the standalone three-skill roll below. It does
not require a Ludo board, board calibration or token perception, and does not
qualify for D4. Token turns and goal conditioning remain deferred. The board
perception workflow in this document is retained for the later D4 gate.

For that later gate, complete [D3 preconditions](D3_OPERATOR_RUNBOOK.md). Print the four configured
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

## Standalone fixed-scene roll

Use [session config](../../config/synria_session.json)'s `roll_skills` entries
for three separately trained and served checkpoints, in order: `die_into_cup`,
`roll_and_dump`, `cup_return`. Each requires its local checkpoint directory,
trainer completion receipt, endpoint, recorded per-policy response timeout and
an operator-chosen positive `max_steps` command budget. No physical budget is
supplied by the template. All three share one command period, at least the
verified bridge move time, one source adapter and one guarded command sink.
The task registry, contract, checkpoint hash, image shape, cadence and timeout
must match each endpoint's metadata before sources open. Metadata matching is
not network authentication or a hardware safety interlock.

Follow the [adapter safety prerequisites](D2_OPERATOR_RUNBOOK.md) and
[local serving guide](ACT_TRAINING_RUNBOOK.md). Keep the marked cup location,
die start zone and dump tray fixed. Motion remains opt-in and requires verified
limits and policy speed settings, leader sync OFF, read-only source preflight
and manual arming. Never use the historical XYZ phase configuration with these
task-specific policies.

```bash
synria-roll-skills --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/rolls/${SESSION}-preflight"
synria-roll-skills --config "$SESSION_CONFIG" \
  --output "reports/ludo_flagship/rolls/$SESSION" --enable-motion
```

At each boundary the arm is offered controlled hold before an explicit readiness
confirmation. Do not manually reset the cup between chained skills. The policy
runs only its declared command budget, with a full command-period pause before
the first target and after every offered target. Budget completion is not
success. After hold, inspect the native-resolution front still and give one
task-specific success/failure label. Optional funnel diagnostics do not add pass
conditions. Any failure, declined confirmation or abort is ledgered and stops
the sequence before another skill. Ctrl-C/EOF are supported during execution;
use manual disarm and the independently verified site/controller stop procedure
on a fault. Neither a hold offer nor disarm proves an in-flight trajectory has
stopped, and neither requests torque-off.

The default die source is the operator's integer 1–6 linked to a front still.
Optional `fixed_roll.read.source=perception` requires the committed die accuracy
report, explicit stable-read bounds and `image_shape` matching the model's CHW
front dimensions. Its input remains stored-size RGB converted to the harness's
BGR convention; native still pixels do not replace the calibrated image grid.
Every analyzed image's dimensions must match. Legacy reports without dimension
evidence must be remeasured and committed for this new roll mode; they are not
silently upgraded. No synthetic report can enable physical perception.

Review `skill_attempts.jsonl`, per-policy latency files, `roll_result.json` and
all stills, including failed/aborted attempts. D4 still needs three consecutive
successful physical-roll **and token-move** turns; this roll-only command never
increments that count, invokes Ludo or advances D5. Append activity/timeline
entries and commit the small session artifacts with cited stills:

```bash
git add "reports/ludo_flagship/rolls/$SESSION" \
  docs/ludo_flagship/ACTIVITY_LOG.md docs/ludo_flagship/timeline.jsonl
git commit -m "Record Synria physical roll evidence"
```

Object success is the operator's label plus a camera still; no independent sensor confirms it.
