# Physical-contract ACT training

Software: **implemented, unmeasured**. D1–D5 remain **planned**. Validation uses
temporary synthetic datasets and tiny CPU updates, not a trained physical
policy or a physical evaluation.

## Fixed-scene task eligibility

This ACT policy consumes **state and images only**. It ignores task text. The
existing move executor expresses the planned target XYZ and token identity in
that text, and this dataset contract declares no visual goal cue or other
goal input. Changing a planned target while keeping the observations identical
therefore cannot condition this model on the new target.

These checkpoints are **not cleared for varying-target D2, D3, D4 or D5 motion**.
Run and checkpoint manifests explicitly record the input features,
`task_text_conditioning: false`, no declared visual goal cue, and
`varying_target_motion_eligible: false`. The prospective 2026-10-08 decision
allows the three [fixed-scene roll skills](../../config/synria_tasks.json).
Each new checkpoint records `fixed_scene_motion_eligible: true` and
`eligible_task_ids` containing only its trained task. Use a separate dataset
and checkpoint for each skill. Token moves and goal-conditioning design remain
deferred until the roll works. Fixed-scene eligibility does not verify limits,
the safety bridge or physical motion. Do not treat
finite predictions, low held-out error or a successful local reload as resolving
this limitation.

Choose `TASK_ID` as `die_into_cup`, `roll_and_dump` or `cup_return` for the
workflow's paths and identifiers. Training does not accept a task override:
`expected_contract` must exactly match that dataset's full task snapshot/hash
and operator-configured episode window. All checked-in real windows remain
null; first time the physical skill and record a prospective choice in
[DECISIONS.md](DECISIONS.md), then collect and review its D1 data. Keep the
marked cup position, die start zone and fixed tray from its
[collection protocol](D1_DATASET_PROTOCOL.md).

Serve a finalized checkpoint with its trainer-written completion receipt:

```bash
python scripts/serve_synria_policy.py \
  --checkpoint "$CHECKPOINT_ROOT" \
  --checkpoint-record "$CHECKPOINT_RECORD" \
  --task-registry "$TASK_REGISTRY" --device cpu
```

The default address is `127.0.0.1:8080`. Set these variables to the chosen
checkpoint directory, its committed `checkpoint-step-<step>.json` evidence and
the operator-configured registry. The server verifies the receipt, content
hash, task snapshot and saved cadence, then loads only local model weights and
saved normalizers. Its `evaluated` receipt status confirms completed diagnostic
capture, not a metric threshold or policy selection. Missing/partial or changed
checkpoints are refused. Loading accepts only the wrapper's default registered
ACT processor steps with tensor state confined to the hashed checkpoint;
custom processor imports and pretrained-backbone downloads are refused before
model construction. Each request must carry the exact trained task text
and contract. Reset at each trial; chunk slots follow the saved command-period
stride. `/metadata` exposes the serving identity for inspection. This command
starts no ROS, camera or command path.

## Inputs and isolation

Use a separate Python 3.12 environment and install this repository with
`python3 -m pip install -e '.[robot-learning,vision]'`. The optional dependency
is upstream LeRobot 0.6's `training` extra, including dataset and trainer
dependencies; it is not a vendor fork. Do not modify the working teleoperation
environment. This wrapper opens no robot interface and starts no ROS process,
camera, driver, bridge or teleoperation.

Checkpoint compatibility is tested with released upstream 0.6.1 and the local
0.6.2 development checkout; these are distinct installations, not two claimed
package-index releases. The wrapper unwraps the model and inspects the installed
save function, supplying `accelerator` only when that named keyword is supported.
Save errors are not retried under another signature. The installed package
version is recorded in every run and checkpoint manifest. The existing optional
dependency remains `lerobot[training]>=0.6,<0.7` on Python 3.12 or newer.

Finish the D1 quality and visual review, then follow the
[frozen probe runbook](CHECKPOINT_PROBE_RUNBOOK.md) **before training**. Preserve
the finalized dataset outside git. The committed probe set specifies held-out
episodes and physical trial identities; training uses exactly the remaining
episodes. Both sets must be nonempty. Diagnostic probe results must never be
used for D2 policy selection; register the D2 selection rule separately before
viewing them. No operator dataset, held-out IDs or physical probes are supplied
by this software.

Complete an operator-owned copy of the
[training configuration](../../config/synria_act_training.json):

- `policy_id`: a new identifier absent from the policy ledger; `dataset_repo_id`
  identifies the existing local dataset.
- `expected_contract`: copy the complete dataset `physical_contract.json`
  object, including task ID/definition/hash/window, recording purpose, action
  source, velocity flag, gripper, lookahead and rate. Smoke cannot train.
- `repository`, committed `probe_set` and `probe_sha256`; new
  `evidence_directory` inside the repository; separate existing `policy_ledger`
  and `timeline` destinations. Relative config paths resolve under `repository`.
- Required `command_period_s`, `chunk_size` and this policy's
  `response_timeout_s`; explicit batch size, save frequency, optimizer settings,
  architecture, and `device` (`cpu` or `cuda`). Template recipe values are
  unmeasured starting points, not a recommended physical policy.

The command period multiplied by dataset FPS must be an integer stride `q`
of at least one, and `chunk_size >= q`. Recorded action lookahead is preserved,
not shifted again. The serving convention is absolute targets offered now and
held until the next command tick, using chunk slots `0,q,2q,...`; this does not
guarantee reaching a pose at the end of a period. The future physical session
must also honor the bridge-duration floor and verified motion limits. A
training configuration never authorizes motion.

## Train locally

From the repository root, choose an explicit seed and positive step count,
an absolute dataset path, its contract path, and a **new external** raw-output
directory. Existing output/evidence directories are refused, not overwritten.
For a CPU run, hide accelerators before starting the interpreter:

```bash
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 WANDB_MODE=disabled \
  python3 scripts/train_synria_act.py \
  --dataset-root "$DATASET_ROOT" --contract "$DATASET_ROOT/physical_contract.json" \
  --output "$TRAINING_OUTPUT" --config "$TRAINING_CONFIG" \
  --seed "$TRAINING_SEED" --steps "$TRAINING_STEPS"
```

The wrapper also disables remote downloads/uploads and experiment-service
logging. Backbones start with `pretrained_backbone_weights=None`; no pretrained
weights are fetched. Supported execution is single-process, full-precision
training. This initial path starts a new run; it does not resume partially
failed optimization.

The [wrapper](../../synria_lerobot/act_training.py) calls upstream ACT,
`update_policy` and `save_checkpoint`, and saves both normalization pipelines.
Normalization aggregates only the training episodes' persisted statistics;
the dataset-global statistics are not used. Camera features must match the
D1 named CHW metadata, with equal stored wrist/front sizes; loaded tensors are
CHW RGB. The saved model state is seven positions/gripper values or thirteen
values including six velocities, exactly as declared by the physical contract.

Every periodic save and the final unique step invoke the frozen held-out
evaluation hook. Raw checkpoint directories include an immutable physical
manifest; separate small checkpoint records hold the complete content hash.
Dataset, contract, split/probe hash, step, recipe, seed, host, git SHA and
upstream version are recorded. A full content hash is checked again after each
evaluation; hashing and training performance are unmeasured.

## Simulation lane

Takes recorded by the simulation console carry the `simulation` purpose, and the
physical lane above refuses them. A policy for the Isaac scene trains in the
simulation lane instead: set `"lane": "simulation"` in the training
configuration and `"lane": "simulation"` in the probe set. The lane changes four
things and nothing else:

- the dataset contract must carry the `simulation` purpose, and a qualifying
  dataset is refused, so the two lanes can never be mixed by mistake;
- the probe set holds out episodes but carries no physical trials
  (`physical_trials` is `[]`); closed-loop trials in the scene are the
  simulation runner's business;
- curated exclusions are kept: an excluded take is left out of training (it may
  not be held out), and `training_episode_limit` may name how many of the
  remaining takes to train on, in recording order, so each larger subset
  contains every smaller one. Every omitted index is listed in the run and
  checkpoint manifests as `omitted_episode_indices`; nothing goes unused silently;
- the policy ledger must be the [simulation ledger](SIM_POLICY_LEDGER.md), whose
  first line names the lane. Rows carry `lane=simulation` and
  `N of M episodes trained`.

Every manifest, checkpoint record and evaluation record from this lane says
`"lane": "simulation"`, `"physical_eligible": false`,
`"fixed_scene_motion_eligible": false` and `"simulation_motion_eligible": true`.
The policy server loads such a checkpoint only with `--lane simulation`, and a
physical server refuses it; the physical probe runner refuses simulation probes.
A configuration or probe set without a `lane` field is physical.

The data-scaling study trains the same dataset at several
`training_episode_limit` values with the same held-out set, seed and step count,
and compares the held-out action error per checkpoint and the closed-loop
success rate in the scene against the trained episode count.

### Closed-loop trials in the scene

`scripts/run_sim_policy_trials.py` runs scored trials against a server started
with `--lane simulation`, with the Isaac scene in its policy launch mode
(`SKIP_DRIVER=1 bash scripts/run_alicia_stack.sh policy` in the teleop
repository), which mirrors `/isaac/policy_joint_targets` instead of the real
arm's feedback. Before every trial the runner asks the scene for a Randomize
and reset, walks the arm from wherever it is to the start pose (the dataset's
mean first frame unless `--start-pose` is given) within the per-command step
bound, then resets the policy and steps it at the command period. The score
follows the served task and comes from the scene's own poses. `pick_cup`:
success when the cup has been `--lift-m` (3 cm) above its rest height for
`--hold-s` (1 s) without a break. `cup_pick_and_place`: success when the cup,
lifted at least once, rests within `--place-radius-m` (3 cm) of the scene's
mark (`/isaac/pick_objects/place_target/pose`) and within `--place-height-m`
(1 cm) of its rest height with the gripper centre `--gripper-clear-m` (6 cm)
away, for `--hold-s`. Any other task is refused. Every trial records where the
cup stood after the reset; the trial otherwise ends at `--max-trial-s`. Speed
bounds come from
[`config/synria_sim_policy.json`](../../config/synria_sim_policy.json), a
simulation-lane file the physical command path refuses. The runner never
publishes on `/joint_states`, `/joint_commands`, `/policy_joint_targets` or
`/policy_arm`, and refuses a server or contract outside the simulation lane.
The report, written once under `reports/sim_policy/`, carries the rule, every
trial, the success rate and `physical_evidence: false`.

### Augmentation, temporal ensembling and checkpoint averaging

Three optional levers, each recorded in the manifest and refused when malformed:

- `image_transforms` in the training configuration names upstream transforms
  (brightness, contrast, saturation, hue, sharpness, affine) applied to training
  frames only, never to held-out frames, with the upstream default ranges.
  Augmentation runs on the CPU inside the data loader, so set
  `dataloader_workers` (default 0, every frame decoded and augmented in the
  training process) to a few worker processes when it is enabled. The first
  augmented run on the simulation study host trained about seven times slower
  per step than the plain baseline with 0 workers. The count is part of the
  recorded configuration; held-out evaluation always uses 0.
- `temporal_ensemble_coeff` makes the policy server predict a chunk on every
  request and return, for the current step, the exp-weighted mean of what every
  still-overlapping chunk predicted for it (weight `exp(-coeff * age)`, oldest
  chunk age 0, as in the ACT paper). Null keeps the one-prediction-per-chunk queue.
- `scripts/average_synria_checkpoints.py --evidence <run> --last 4` averages the
  float weights of a completed run's last evaluated checkpoints into one
  checkpoint beside them, evaluates it on the frozen held-out set with the run's
  evaluator, and writes a receipt the policy server accepts. The manifest names
  the averaged steps. Nothing trains.

### The data-scaling study

`scripts/run_sim_data_scaling_study.py --spec <study>.json` trains the frozen
dataset at each size in `subset_sizes` (nested by recording order, so each
run contains every smaller one) with one seed, step count and held-out set,
through the simulation-lane trainer, so every run has its own evidence
directory, checkpoint hash and ledger row. The template is
[`config/synria_sim_data_scaling.json`](../../config/synria_sim_data_scaling.json).
With `--trials N` the final checkpoint of each run is served on a local port
and the trial campaign above runs against the scene before the next size
trains. The study record `study.json` and the table `STUDY.md` are written
into the evidence root after every row, so a failed study leaves what it
measured marked failed. The table lists, per episode count, the held-out
joint error of the final checkpoint and the closed-loop success rate; its
caveats (recording order is not random, one seed, simulation only) are part
of the document.

## Failures and evidence

Review the evidence directory's start/run manifests, checkpoint records and
evaluation records, plus the appended timeline events. Each run that begins
optimization appends one policy-ledger row with its required response timeout;
checkpoints do not add extra rows. Failure rows retain the last finalized
checkpoint hash, or explicit `N/A` when none exists. Incomplete saves are listed
as partial paths without a qualifying hash; never load or evaluate them.
Do not reinterpret a failed optimizer step as a valid checkpoint.

Evaluation errors retain their failed record and stop training. Manifest and
ledger finalization are both attempted; secondary write errors do not replace
the primary training/evaluation failure. Reconcile any reported evidence-write
error manually before another run, using a new policy id/output destination;
do not erase negative results or suppress failed diagnostics.

Commit the small evidence, ledger, activity entry and timeline with the run.
The trainer creates exactly one policy-ledger row for this optimization run;
checkpoint saves and later D2 sessions must not create duplicate rows for it.
Keep raw datasets and checkpoints outside git. Use the
[playback script](../../scripts/playback_synria_training.py) for diagnostic
curves; neither its metrics nor the latest checkpoint automatically select a
D2 policy or advance a stage. Physical probe execution and server attestation
remain separate, explicitly authorized workflows.

For the [standalone three-skill roll](D4_OPERATOR_RUNBOOK.md#standalone-fixed-scene-roll),
repeat collection, frozen probes and training separately for all three task IDs.
Serve three finalized task-bound checkpoints on distinct local ports, each with
its matching completion receipt and task registry:

```bash
python3 scripts/serve_synria_policy.py \
  --checkpoint "$DIE_INTO_CUP_CHECKPOINT" --checkpoint-record "$DIE_INTO_CUP_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8080
python3 scripts/serve_synria_policy.py \
  --checkpoint "$ROLL_AND_DUMP_CHECKPOINT" --checkpoint-record "$ROLL_AND_DUMP_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8081
python3 scripts/serve_synria_policy.py \
  --checkpoint "$CUP_RETURN_CHECKPOINT" --checkpoint-record "$CUP_RETURN_RECEIPT" \
  --task-registry "$TASK_REGISTRY" --device cpu --host 127.0.0.1 --port 8082
```

Run each server in its own terminal after setting the six explicit checkpoint
and receipt variables; these are separate long-running commands, not a shell
pipeline. Set the corresponding session `roll_skills` endpoints to the three
addresses. Their command periods and stored image sizes must match the shared
adapter; retain each policy's own trained response timeout and command budget.
Metadata refuses a substituted task, checkpoint or cadence. Serving makes no
arm offers; use D2's disarmed preflight/manual arming procedure before the roll
runner. For D2, serve only the prospectively selected `die_into_cup` checkpoint;
probe metrics and the latest save never make that selection automatically.
