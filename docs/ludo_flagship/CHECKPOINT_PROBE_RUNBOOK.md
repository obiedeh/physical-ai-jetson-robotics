# Frozen checkpoint diagnostics

Software: **implemented, unmeasured**. D1–D5 remain **planned**. These probes
are diagnostic only: they are neither D2 trials nor a way to select a policy
for D2. Pre-register the D2 policy-selection rule separately, before examining
probe results; do not choose a checkpoint retrospectively from these curves.
No physical probe or training run has been performed by this change.

## Freeze before training

Use a finalized physical dataset outside git and a separate recording/training
environment with upstream LeRobot 0.6 on Python 3.12. Preserve the dataset's
physical contract, action source, lookahead and requested rate. Finish the D1
quality/visual review before training. Do not append to or resume this dataset
after freezing its content hash.

Install this repository into that separate environment, not the working
teleoperation environment: `python3 -m pip install -e '.[robot-learning,vision]'`.

Copy the [probe template](../../config/synria_checkpoint_probes.json) to a new
repository-relative campaign file, named by `PROBE_SET`. Fill its identifier,
operator/date, dataset content hash, nonempty held-out episode indices and their
exact frame counts, and fixed physical trials (`trial_id`, `start_square`,
`target_square`, `token`). Set explicit capture `fps`, `max_duration_s` and
`shutdown_timeout_s`. No operator probe choices are supplied by the template.
For the exact hash and authoritative finalized episode counts, use the
[evaluation module](../../synria_lerobot/checkpoint_eval.py)'s
`strict_content_hash(Path(DATASET_ROOT))` and `read_episode_metadata(...)`.
The hash includes relative paths and file sizes as well as bytes; it is not
interchangeable with a different hashing convention.

Training episodes must be exactly all actual dataset episodes except the
frozen held-out set. Both sets must be nonempty. Neither unknown indices nor
silently unused episodes are allowed. Then register and commit the probe file
before constructing a trainer:

```bash
python3 -m synria_lerobot.checkpoint_eval register "$PROBE_SET"
git add "$PROBE_SET"
git commit -m "Freeze diagnostic checkpoint probes"
```

Save the printed hash as `PROBE_HASH`. The loader requires both committed bytes
and that hash. Changing the split, trials or capture settings requires a new
prospective probe campaign, not editing the old evidence.

## Every saved checkpoint

The training integration must construct `CheckpointEvaluator` before training
and call its `evaluate(...)` hook after **every** finalized checkpoint save.
The [upstream ACT wrapper](ACT_TRAINING_RUNBOOK.md) performs this integration.
Other callers must supply observation-only
prediction inputs, physical-unit `B×T×7` predictions/targets, boolean padding
masks, and every held-out episode/frame index exactly once. Normalization must
come from training episodes alone. Do not apply the dataset lookahead again.

Checkpoint directories, datasets and raw media stay outside git. Put small
evaluation records in a separate repository report directory, never within
the hashed dataset or checkpoint; pass `docs/ludo_flagship/timeline.jsonl` as
the timeline. The hook records policy id, training step, UTC time, physical
contract, fixed split and dataset/checkpoint/probe hashes. It reports MAE,
RMSE and maximum error per joint in radians and separately for the gripper in
metres. Padding contributes neither error nor count. Action prediction error
is not physical object success.

Each checkpoint record is immutable. Prediction failures retain a failed
record and a `checkpoint_eval` timeline event, then raise; do not suppress the
failure or continue training without reconciling it. Empty/missing checkpoint
roots are invalid saves and do not receive a plausible hash. Full input hashes
are checked before and after evaluation. Their cost is unmeasured. Keep
checkpoint roots immutable and give every step its own directory.

## Optional physical diagnostics

Only a future, physically present operator may run this section after the
[D2 safety and adapter prerequisites](D2_OPERATOR_RUNBOOK.md) are accepted.
Use the existing verified controller/bridge setup; this command starts no
driver, bridge or teleoperation. The bridge begins disarmed, leader sync is
confirmed OFF, and manual arming follows read-only preflight. A hold offer or
an observed arming command is not proof of physical stopping; the separate
operator-verified controller/emergency procedure remains necessary.

Explicitly choose one successful checkpoint-evaluation record as
`CHECKPOINT_EVALUATION`, a completed `SESSION_CONFIG`, the absolute repository
root as `REPOSITORY`, and a new `PROBE_OUTPUT` for each run. The session's
policy id/checkpoint hash and physical contract must match the record. At this
stage the endpoint's checkpoint identity is **operator-declared**, not attested
by the server. Checkpoint serving/attestation is a separate implementation step.
The chosen checkpoint is for diagnostic inspection only, never automatic D2
selection. `MEDIA_ROOT` must be outside git, the dataset and the checkpoint.

```bash
python3 -m synria_lerobot.checkpoint_eval probe \
  --probe-set "$PROBE_SET" --probe-hash "$PROBE_HASH" \
  --checkpoint-evaluation "$CHECKPOINT_EVALUATION" --dataset-root "$DATASET_ROOT" \
  --session-config "$SESSION_CONFIG" --repository "$REPOSITORY" \
  --output "$PROBE_OUTPUT" --media-root "$MEDIA_ROOT"
```

Without `--enable-motion`, this performs read-only adapter preflight and writes
a read-only report without a command sink or trial clips. Review it; choose
another new `PROBE_OUTPUT` before repeating the same command with explicit
`--enable-motion`. The physical trial list is the frozen list, not D2's scene
schedule. Grade success/failure and each funnel stage from the front still;
confirm reset before a retry, or type `abort` at an operator prompt.

An independent bounded worker samples both latest RGB camera views throughout
each attempt, including the grading pause, and takes an end frame. Each attempt
gets unique external wrist/front WebM files, a timestamp manifest, achieved
sample rate, end-frame source hashes, and a nearest-frame link to the label's
front still. WebM playback uses configured FPS; consult the timestamps and
achieved rate for actual timing. Do not claim continuous capture at an assumed
rate. OpenCV needs a working VP8/WebM encoder; missing codec support fails the
attempt rather than producing substitute media.

All attempts, failures and aborts remain recorded. Both finalized, nonempty,
hashed clips are required for capture completion. A missing clip fails the
diagnostic attempt while preserving any operator label. Capture faults refuse
subsequent normal offers and request the same eligible guarded hold; they
cannot interrupt a blocked terminal prompt, so the operator must abort or use
the site's stop procedure. Shutdown is bounded. A live capture worker keeps
its sources/writers owned and reports the unresolved cleanup; do not reuse
that process or delete partial evidence.

Object success is the operator's label plus a camera still; no independent sensor confirms it.

## Replay and retain evidence

Pass every checkpoint record, including failed evaluations, to the
[playback script](../../scripts/playback_synria_training.py). Physical runs are
optional; unavailable clips remain explicit. For example, with paths selected
by the operator (the shell expands the evaluation list):

```bash
python3 scripts/playback_synria_training.py \
  --evaluations "$EVALUATION_DIRECTORY"/*.json \
  --physical-runs "$PROBE_OUTPUT/probe_run.json" --output "$PLAYBACK_HTML"
```

Open the generated HTML locally. It contains per-step or UTC metric curves,
all checkpoint statuses, a step/trial selector, and first-versus-selected
checkpoint clips side by side. It verifies local media/still hashes before
rendering, escapes labels/errors, and embeds no remote assets. Raw videos are
referenced by absolute local file URL and must remain available at those paths;
some browsers restrict local media playback. The page is not a portable video
bundle. Missing physical probes are not silently treated as successes.

Review and commit the small evaluation/attempt/run records, linked stills,
HTML, activity entry and timeline alongside each cited result. Preserve raw
videos, timestamp manifests, datasets and checkpoint files outside git. No
diagnostic result automatically changes D1–D5, the D2 scorer or policy ledger.
